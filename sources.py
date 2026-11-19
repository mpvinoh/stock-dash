"""進階資料來源：籌碼（三大法人、融資）、基本面（月營收、財報）、估值（本益比）、新聞。

台股資料來自 FinMind 免費 API（https://finmindtrade.com），美股資料來自 Yahoo Finance。
內建本機硬碟快取（data_cache/），抓過一次後直接從硬碟讀取並只做差額更新，大幅節省 API 額度與等待時間。
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import time

import pandas as pd
import requests
import yfinance as yf

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
HISTORY_START = "2015-01-01"  # 籌碼、估值、基本面抓這天以後的資料（回測用）
LIMIT_HIT = {"value": False}  # FinMind 是否回報「已達查詢上限」
CACHE_DIR = pathlib.Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def is_taiwan(ticker: str) -> bool:
    return ticker.endswith(".TW") or ticker.endswith(".TWO")


def stock_id(ticker: str) -> str:
    return ticker.split(".")[0]


US_ETFS = {
    "SPY", "VOO", "VTI", "IVV", "QQQ", "QQQM", "DIA", "IWM", "SOXX", "SMH",
    "SCHD", "TLT", "GLD", "SLV", "ARKK", "VGT", "VT", "EWT", "XLK", "XLF", "XLE", "XLV", "XLI",
}


def is_etf(ticker: str) -> bool:
    """台股 ETF（00 開頭）或常見美股 ETF。"""
    sid = stock_id(ticker).upper()
    return (is_taiwan(ticker) and sid.startswith("00")) or (not is_taiwan(ticker) and sid in US_ETFS)


def _finmind_raw(dataset: str, sid: str, start: str, token: str = "", end: str | None = None) -> pd.DataFrame | None:
    params = {"dataset": dataset, "data_id": sid, "start_date": start}
    if end:
        params["end_date"] = end
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=45)
        j = r.json()
    except Exception:
        return None
    if j.get("status") == 402:
        LIMIT_HIT["value"] = True
        return None
    if j.get("status") != 200:
        return None
    return pd.DataFrame(j.get("data") or [])


def _finmind(dataset: str, sid: str, start: str, token: str = "", ttl_hours: float = 6.0) -> pd.DataFrame:
    """含本機硬碟快取與差額更新的 FinMind 查詢。"""
    cache_file = CACHE_DIR / f"fm_{dataset}_{sid}.parquet"
    cached: pd.DataFrame | None = None
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
        except Exception:
            cached = None
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cached is not None and not cached.empty and age_hours < ttl_hours:
            return cached

    # 若已有舊快取且有 date 欄位，只抓最後日期前 10 天之後的差額資料
    fetch_start = start
    if cached is not None and not cached.empty and "date" in cached.columns:
        try:
            max_dt = pd.to_datetime(cached["date"]).max()
            fetch_start = (max_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        except Exception:
            fetch_start = start

    fresh = _finmind_raw(dataset, sid, fetch_start, token)
    if fresh is None:
        return cached if cached is not None else pd.DataFrame()
    if fresh.empty:
        return cached if cached is not None else pd.DataFrame()

    if cached is not None and not cached.empty and fetch_start != start:
        combined = pd.concat([cached, fresh], ignore_index=True).drop_duplicates()
    else:
        combined = fresh
    try:
        combined.to_parquet(cache_file, index=False)
    except Exception:
        pass
    return combined


# ---------------------------------------------------------------------------
# 籌碼面
# ---------------------------------------------------------------------------
def fetch_chips(ticker: str, token: str = "") -> pd.DataFrame:
    """每日三大法人買賣超（股）與融資餘額（張）。index 為日期。"""
    if not is_taiwan(ticker):
        return pd.DataFrame()
    sid = stock_id(ticker)
    inst = _finmind("TaiwanStockInstitutionalInvestorsBuySell", sid, HISTORY_START, token)
    out = pd.DataFrame()
    if not inst.empty:
        inst = inst.copy()
        inst["net"] = inst["buy"] - inst["sell"]
        group = {
            "Foreign_Investor": "foreign",
            "Foreign_Dealer_Self": "foreign",
            "Investment_Trust": "trust",
            "Dealer_self": "dealer",
            "Dealer_Hedging": "dealer",
        }
        inst["who"] = inst["name"].map(group)
        out = inst.dropna(subset=["who"]).pivot_table(
            index="date", columns="who", values="net", aggfunc="sum"
        )
    margin = _finmind("TaiwanStockMarginPurchaseShortSale", sid, HISTORY_START, token)
    if not margin.empty:
        m = margin.drop_duplicates(subset=["date"], keep="last").set_index("date")[
            ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]
        ]
        m.columns = ["margin_balance", "short_balance"]
        out = out.join(m, how="outer") if not out.empty else m
    # 證交所官方成交量（股）：Yahoo 的台股成交量偏低約 10–20%，算法人買賣超比例時用這個
    price = _finmind("TaiwanStockPrice", sid, HISTORY_START, token)
    if not price.empty and not out.empty:
        p = price.drop_duplicates(subset=["date"], keep="last").set_index("date")[["Trading_Volume"]].rename(
            columns={"Trading_Volume": "volume"}
        )
        out = out.join(p, how="left")
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index)
    for col in ["foreign", "trust", "dealer", "margin_balance", "short_balance", "volume"]:
        if col not in out:
            out[col] = float("nan")
    return out.sort_index().astype(float)


# ---------------------------------------------------------------------------
# 基本面
# ---------------------------------------------------------------------------
def fetch_revenue(ticker: str, token: str = "") -> pd.DataFrame:
    """月營收。欄位：period（營收月份）、revenue、available（公布日）。"""
    if not is_taiwan(ticker) or is_etf(ticker):
        return pd.DataFrame()
    df = _finmind("TaiwanStockMonthRevenue", stock_id(ticker), HISTORY_START, token, ttl_hours=12.0)
    if df.empty:
        return df
    df = df.copy()
    df["period"] = pd.to_datetime(
        df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).astype(str) + "-01"
    )
    fallback = pd.to_datetime(df["date"]) + pd.Timedelta(days=10)
    created = pd.to_datetime(df.get("create_time"), errors="coerce") if "create_time" in df else fallback
    df["available"] = created.fillna(fallback)
    df = df.sort_values("period").drop_duplicates("period", keep="last")
    return df[["period", "revenue", "available"]].reset_index(drop=True)


def fetch_financials(ticker: str, token: str = "") -> pd.DataFrame:
    """季度財報：EPS、營收、毛利。index 為季底日，另附 available（推估公布日）。"""
    if not is_taiwan(ticker) or is_etf(ticker):
        return pd.DataFrame()
    df = _finmind("TaiwanStockFinancialStatements", stock_id(ticker), HISTORY_START, token, ttl_hours=24.0)
    if df.empty:
        return df
    df = df[df["type"].isin(["EPS", "Revenue", "GrossProfit"])]
    q = df.pivot_table(index="date", columns="type", values="value", aggfunc="last")
    q.index = pd.to_datetime(q.index)
    q = q.sort_index()
    q["available"] = [d + pd.Timedelta(days=90 if d.month == 12 else 45) for d in q.index]
    if {"GrossProfit", "Revenue"} <= set(q.columns):
        q["gross_margin"] = q["GrossProfit"] / q["Revenue"]
    return q


def _safe_float(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def fetch_us_fundamentals(ticker: str, ttl_hours: float = 12.0) -> dict:
    """美股：用 Yahoo Finance 抓取季報時間序列、估值、機構持股與空單比例（含硬碟快取）。"""
    cache_file = CACHE_DIR / f"us_fund_{ticker.replace('.', '_')}.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < ttl_hours:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    try:
        t = yf.Ticker(ticker)
        q = t.quarterly_income_stmt
        info = t.get_info() or {}
    except Exception:
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    out: dict = {}
    quarters: list[dict] = []
    try:
        if q is not None and not q.empty:
            qt = q.T.sort_index()
            rev = qt["Total Revenue"].astype(float) if "Total Revenue" in qt else pd.Series(dtype=float)
            eps = qt["Diluted EPS"].astype(float) if "Diluted EPS" in qt else pd.Series(dtype=float)
            gp = qt["Gross Profit"].astype(float) if "Gross Profit" in qt else pd.Series(dtype=float)
            gm_s = (gp / rev.replace(0, float("nan"))) if (len(gp) and len(rev)) else pd.Series(dtype=float)
            yoy_s = (rev / rev.shift(4) - 1) if len(rev) >= 5 else pd.Series(dtype=float)

            for dt_idx in qt.index[-8:]:
                r_val = _safe_float(rev.get(dt_idx)) if len(rev) else None
                e_val = _safe_float(eps.get(dt_idx)) if len(eps) else None
                g_val = _safe_float(gm_s.get(dt_idx)) if len(gm_s) else None
                y_val = _safe_float(yoy_s.get(dt_idx)) if len(yoy_s) else None
                if any(x is not None for x in (r_val, e_val, g_val)):
                    quarters.append({
                        "date": f"{pd.Timestamp(dt_idx):%Y-%m-%d}",
                        "revenue": r_val,
                        "rev_yoy": y_val,
                        "eps": e_val,
                        "gross_margin": g_val,
                    })

            if len(rev.dropna()) >= 5:
                r_clean = rev.dropna()
                out["rev_yoy"] = float(r_clean.iloc[-1] / r_clean.iloc[-5] - 1)
            if len(eps.dropna()) >= 8:
                e_clean = eps.dropna()
                ttm, prev = float(e_clean.iloc[-4:].sum()), float(e_clean.iloc[-8:-4].sum())
                out["eps_ttm"] = ttm
                out["eps_growth"] = (ttm / prev - 1) if prev > 0 else None
            elif len(eps.dropna()) >= 4:
                out["eps_ttm"] = float(eps.dropna().iloc[-4:].sum())
            if len(gm_s.dropna()) >= 5:
                g_clean = gm_s.dropna()
                out["gm"] = float(g_clean.iloc[-1])
                out["gm_change"] = float(g_clean.iloc[-1] - g_clean.iloc[-5])
            elif len(gm_s.dropna()) >= 1:
                out["gm"] = float(gm_s.dropna().iloc[-1])
    except Exception:
        pass

    if quarters:
        out["quarters"] = quarters

    # 從 info 補齊估值、機構持股、空單比例與分析師目標價
    field_map = {
        "pe": "trailingPE",
        "forward_pe": "forwardPE",
        "peg": "trailingPegRatio",
        "pb": "priceToBook",
        "market_cap": "marketCap",
        "inst_pct": "heldPercentInstitutions",
        "insider_pct": "heldPercentInsiders",
        "short_float": "shortPercentOfFloat",
        "short_ratio": "shortRatio",
        "target_mean": "targetMeanPrice",
    }
    for k, src_k in field_map.items():
        val = _safe_float(info.get(src_k))
        if val is not None:
            out[k] = val
    if "peg" not in out and _safe_float(info.get("pegRatio")) is not None:
        out["peg"] = _safe_float(info.get("pegRatio"))
    # 若季報缺最新營收年增或毛利率，改用 info 的 summary 補齊
    if out.get("rev_yoy") is None and _safe_float(info.get("revenueGrowth")) is not None:
        out["rev_yoy"] = _safe_float(info.get("revenueGrowth"))
    if out.get("gm") is None and _safe_float(info.get("grossMargins")) is not None:
        out["gm"] = _safe_float(info.get("grossMargins"))
    if info.get("sector"):
        out["sector"] = str(info["sector"])
    if info.get("recommendationKey"):
        out["rec_key"] = str(info["recommendationKey"])
    if info.get("quoteType"):
        out["quote_type"] = str(info["quoteType"])

    if out:
        try:
            cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# 估值
# ---------------------------------------------------------------------------
def fetch_per(ticker: str, token: str = "") -> pd.Series:
    """每日本益比（台股個股；ETF 沒有）。"""
    if not is_taiwan(ticker) or is_etf(ticker):
        return pd.Series(dtype=float)
    df = _finmind("TaiwanStockPER", stock_id(ticker), HISTORY_START, token)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.drop_duplicates(subset=["date"], keep="last")
    s = df.set_index(pd.to_datetime(df["date"]))["PER"].astype(float)
    return s[s > 0].sort_index()


# ---------------------------------------------------------------------------
# 新聞（含硬碟快取，減少重複逐日查詢）
# ---------------------------------------------------------------------------
def fetch_news(ticker: str, token: str = "", days: int = 4, limit: int = 20) -> list[dict]:
    """近幾天的新聞標題。每則：date、title、source、link。"""
    cache_file = CACHE_DIR / f"news_{ticker.replace('.', '_')}.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 3.0:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    items: list[dict] = []
    if is_taiwan(ticker):
        today = dt.date.today()
        for i in range(days):
            day = (today - dt.timedelta(days=i)).isoformat()
            df = _finmind_raw("TaiwanStockNews", stock_id(ticker), day, token)
            if df is None:
                break
            for _, r in df.iterrows():
                items.append(
                    {"date": str(r["date"])[:16], "title": r["title"], "source": r.get("source", ""), "link": r.get("link", "")}
                )
    else:
        try:
            for n in yf.Ticker(ticker).news or []:
                c = n.get("content", n)
                url = (c.get("canonicalUrl") or {}).get("url") or c.get("link", "")
                items.append(
                    {
                        "date": str(c.get("pubDate") or c.get("providerPublishTime") or "")[:16].replace("T", " "),
                        "title": c.get("title", ""),
                        "source": (c.get("provider") or {}).get("displayName", "") if isinstance(c.get("provider"), dict) else c.get("publisher", ""),
                        "link": url,
                    }
                )
        except Exception:
            pass

    seen, unique = set(), []
    for it in sorted(items, key=lambda x: x["date"], reverse=True):
        key = it["title"].split(" - ")[0].strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    result = unique[:limit]
    if result:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    elif cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return result
