"""🚀 起漲雷達：掃描台股（上市櫃成交值前 N 大）與美股（Nasdaq 100／S&P 500 熱門權值與 AI 股），找出「剛啟動」「準備啟動」「拉回買點」。

訊號（和個股分析頁同一套規則）：
- 🕯️ 中線有效K（日線有效紅K + 年線向上）：等待確認＝準備啟動；5 天內起漲確認＝剛啟動
- 🧘 長線有效K（週線有效紅K + 40 週線向上）：同上（2 週內確認）
- 🎯 拉回買點：技術與波浪「買進時機」分數達門檻且已止跌（掃描時不含台股籌碼濾網）
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from data import CACHE_DIR, NAME_MAP
from keybar import analyze_keybars
from model import add_indicators, label_for, score_frame
from waves import analyze_waves

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
NAMES_FILE = CACHE_DIR / "tw_names.json"
QUOTES_FILE = CACHE_DIR / "tw_quotes.parquet"
COMMON = re.compile(r"^[1-9]\d{3}$")  # 4 碼、非 0 開頭 = 台股普通股（排除 ETF、權證、特別股）

# 美股熱門清單（去重）：AI／半導體／科技精選 40 檔 + S&P 500 / Nasdaq 100 權值共 ~100 檔
US_AI_40 = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "TSM", "AMD",
    "ASML", "ARM", "MU", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL",
    "SMCI", "PLTR", "ORCL", "CRM", "ADBE", "NFLX", "NOW", "PANW", "CRWD", "SNOW",
    "APP", "VRT", "ANET", "DELL", "IBM", "CSCO", "UBER", "CEG", "VST", "COIN",
]
US_EXTRA_60 = [
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA", "AXP",
    "PYPL", "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG", "AMGN",
    "GILD", "PFE", "COST", "WMT", "HD", "LOW", "MCD", "SBUX", "PG", "KO",
    "PEP", "NKE", "DIS", "CMCSA", "T", "VZ", "XOM", "CVX", "COP", "SLB",
    "GE", "CAT", "DE", "HON", "UNP", "UPS", "BA", "LMT", "RTX", "NOC",
    "ETN", "NEE", "LIN", "FCX", "NEM", "SHOP", "SPOT", "ABNB", "MELI", "INTU",
    "ADI", "NXPI", "SNPS", "CDNS", "MSTR",
]
US_HOT_100 = list(dict.fromkeys(US_AI_40 + US_EXTRA_60))

PRIORITY = {"confirmed": 3, "waiting": 2}


def _scan_files(market: str):
    tag = "us" if market == "us" else "tw"
    return CACHE_DIR / f"scan_{tag}_result.parquet", CACHE_DIR / f"scan_{tag}_meta.json"


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


def _roc_date(s: str) -> pd.Timestamp | None:
    s = str(s).strip()
    if len(s) == 7 and s.isdigit():
        return pd.Timestamp(int(s[:3]) + 1911, int(s[3:5]), int(s[5:7]))
    return None


def fetch_quotes(max_age_hours: float = 6) -> pd.DataFrame:
    """全部上市櫃股票最新一日行情（含名稱、成交值）。有本機快取。"""
    if QUOTES_FILE.exists() and (time.time() - QUOTES_FILE.stat().st_mtime) / 3600 < max_age_hours:
        return pd.read_parquet(QUOTES_FILE)
    rows = []
    try:
        for x in requests.get(TWSE_URL, timeout=30).json():
            rows.append({
                "code": x["Code"], "name": x["Name"], "market": "TW", "date": _roc_date(x["Date"]),
                "open": _num(x["OpeningPrice"]), "high": _num(x["HighestPrice"]), "low": _num(x["LowestPrice"]),
                "close": _num(x["ClosingPrice"]), "volume": _num(x["TradeVolume"]), "value": _num(x["TradeValue"]),
            })
    except Exception:
        pass
    try:
        for x in requests.get(TPEX_URL, timeout=30).json():
            rows.append({
                "code": x["SecuritiesCompanyCode"], "name": x["CompanyName"], "market": "TWO",
                "date": _roc_date(x["Date"]), "open": _num(x["Open"]), "high": _num(x["High"]),
                "low": _num(x["Low"]), "close": _num(x["Close"]), "volume": _num(x["TradingShares"]),
                "value": _num(x["TransactionAmount"]),
            })
    except Exception:
        pass
    if not rows:
        if QUOTES_FILE.exists():
            return pd.read_parquet(QUOTES_FILE)
        return pd.DataFrame()
    q = pd.DataFrame(rows)
    q.to_parquet(QUOTES_FILE)
    NAMES_FILE.write_text(
        json.dumps({r.code: [r.name, r.market] for r in q.itertuples()}, ensure_ascii=False), encoding="utf-8"
    )
    return q


def universe(n: int = 150) -> pd.DataFrame:
    q = fetch_quotes()
    if q.empty:
        return q
    q = q[q["code"].str.match(COMMON) & (q["close"] > 0)]
    return q.sort_values("value", ascending=False).head(n).reset_index(drop=True)


def _download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
        try:
            raw = yf.download(
                chunk, period="2y", interval="1d", auto_adjust=True, group_by="ticker",
                threads=True, progress=False,
            )
        except Exception:
            continue
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df):
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    out[t] = df
            except Exception:
                continue
    return out


def _patch(df: pd.DataFrame, q: pd.Series) -> pd.DataFrame:
    """Yahoo 落後一天時，用證交所 / 櫃買最新行情補上最後一根。"""
    if q["date"] is None or pd.isna(q["date"]) or not (q["close"] > 0):
        return df
    if df.index[-1] >= q["date"]:
        return df
    bar = pd.DataFrame(
        {"Open": [q["open"]], "High": [q["high"]], "Low": [q["low"]], "Close": [q["close"]], "Volume": [q["volume"]]},
        index=[q["date"]],
    )
    return pd.concat([df, bar])


def _evaluate_one(df: pd.DataFrame, code: str, name: str, turnover_yi: float, threshold: int) -> dict | None:
    """用同一套規則評估單一標的（台股／美股共用）。"""
    if df is None or len(df) < 260:
        return None
    try:
        d = add_indicators(df)
        tech = score_frame(d)
        waves = analyze_waves(d, lookback=250)
        kb = analyze_keybars(d, waves, with_bt=False)
    except Exception:
        return None
    mid, lng = kb["mid"], kb["long"]
    s = tech["score"].iloc[-1]
    lab = label_for(s, threshold, tech["reversal"].iloc[-1]) if pd.notna(s) else None
    pull = bool(lab and lab[0] == "buy")
    signals, prio = [], 0
    if lng["code"] in PRIORITY:
        signals.append("🧘 長線" + ("起漲確認" if lng["code"] == "confirmed" else "等待確認"))
        prio += PRIORITY[lng["code"]] + 1
    if mid["code"] in PRIORITY:
        signals.append("🚶 中線" + ("起漲確認" if mid["code"] == "confirmed" else "等待確認"))
        prio += PRIORITY[mid["code"]]
    if pull:
        signals.append("🎯 拉回買點")
        prio += 2
    if not signals:
        return None
    if pull and (mid["code"] == "confirmed" or lng["code"] == "confirmed"):
        prio += 2
    act = next((x for x in (mid, lng) if x["code"] in PRIORITY), None)
    close = float(d["Close"].iloc[-1])
    stage = "剛啟動" if "confirmed" in (mid["code"], lng["code"]) else "準備啟動" if act else "拉回買點"
    return {
        "代號": code,
        "名稱": name,
        "類型": stage,
        "訊號": "、".join(signals),
        "收盤": close,
        "漲跌%": (close / float(d["Close"].iloc[-2]) - 1) * 100,
        "成交值(億)": turnover_yi,
        "K高": act["high"] if act else np.nan,
        "K低(停損)": act["low"] if act else np.nan,
        "距K低%": (act["low"] / close - 1) * 100 if act else np.nan,
        "買進時機": float(s) if pd.notna(s) else np.nan,
        "波浪階段": waves.get("wave_stage", ""),
        "波浪情境": (act or {}).get("wave", "") if act else "",
        "說明": (act or {}).get("desc", "").replace("**", "") if act else "",
        "_prio": prio,
        "資料日期": d.index[-1],
    }


def scan(n: int = 150, threshold: int = 70, progress=None) -> tuple[pd.DataFrame, dict]:
    """掃描台股上市櫃普通股成交值前 N 大。"""
    uni = universe(n)
    if uni.empty:
        raise RuntimeError("無法取得證交所 / 櫃買中心的股票清單，請稍後再試")
    tickers = [f"{c}.{m}" for c, m in zip(uni["code"], uni["market"])]
    if progress:
        progress(0.1, f"下載 {len(tickers)} 檔台股近 2 年股價…")
    prices = _download(tickers)
    rows = []
    for k, (t, qrow) in enumerate(zip(tickers, uni.itertuples(index=False))):
        if progress and k % 10 == 0:
            progress(0.3 + 0.65 * k / len(tickers), f"分析台股中…（{k}/{len(tickers)}）")
        df = prices.get(t)
        if df is None:
            continue
        df = _patch(df, pd.Series(qrow._asdict()))
        r = _evaluate_one(df, qrow.code, qrow.name, qrow.value / 1e8, threshold)
        if r:
            rows.append(r)
    res = pd.DataFrame(rows)
    if len(res):
        res = res.sort_values(["_prio", "成交值(億)"], ascending=False).reset_index(drop=True)
    meta = {
        "market": "tw",
        "label": f"台股成交值前 {len(tickers)} 大",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scanned": len(prices),
        "universe": len(tickers),
        "threshold": threshold,
        "data_date": f"{res['資料日期'].max():%Y-%m-%d}" if len(res) else "",
    }
    sf, mf = _scan_files("tw")
    res.to_parquet(sf)
    mf.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return res, meta


def scan_us(scope: str = "hot100", threshold: int = 70, progress=None) -> tuple[pd.DataFrame, dict]:
    """掃描美股熱門清單（hot100 = 約 105 檔權值熱門股；ai40 = AI／半導體／科技 40 檔）。"""
    tickers = US_AI_40 if scope == "ai40" else US_HOT_100
    label = f"美股 AI／科技精選 {len(tickers)} 檔" if scope == "ai40" else f"美股熱門權值 {len(tickers)} 檔"
    if progress:
        progress(0.1, f"下載 {len(tickers)} 檔美股近 2 年股價…")
    prices = _download(tickers)
    rows = []
    for k, t in enumerate(tickers):
        if progress and k % 8 == 0:
            progress(0.3 + 0.65 * k / len(tickers), f"分析美股中…（{k}/{len(tickers)}）")
        df = prices.get(t)
        if df is None or len(df) < 260:
            continue
        turnover_yi = float(df["Close"].iloc[-1] * df["Volume"].iloc[-1]) / 1e8  # 億美元
        r = _evaluate_one(df, t, NAME_MAP.get(t, t), turnover_yi, threshold)
        if r:
            rows.append(r)
    res = pd.DataFrame(rows)
    if len(res):
        res = res.sort_values(["_prio", "成交值(億)"], ascending=False).reset_index(drop=True)
    meta = {
        "market": "us",
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scanned": len(prices),
        "universe": len(tickers),
        "threshold": threshold,
        "data_date": f"{res['資料日期'].max():%Y-%m-%d}" if len(res) else "",
    }
    sf, mf = _scan_files("us")
    res.to_parquet(sf)
    mf.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return res, meta


def load_last_scan(market: str = "tw", max_age_hours: float = 12) -> tuple[pd.DataFrame | None, dict | None]:
    sf, mf = _scan_files(market)
    # 相容舊版台股快取檔名
    if market == "tw" and not sf.exists() and (CACHE_DIR / "scan_result.parquet").exists():
        sf, mf = CACHE_DIR / "scan_result.parquet", CACHE_DIR / "scan_meta.json"
    if not (sf.exists() and mf.exists()):
        return None, None
    if (time.time() - sf.stat().st_mtime) / 3600 > max_age_hours:
        return None, None
    try:
        return pd.read_parquet(sf), json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None, None


def tw_names() -> dict[str, list[str]]:
    """代號 → [名稱, 市場]（全部上市櫃，從證交所 / 櫃買清單快取）。"""
    if not NAMES_FILE.exists():
        try:
            fetch_quotes()
        except Exception:
            return {}
    try:
        return json.loads(NAMES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
