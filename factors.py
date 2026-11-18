"""多因子評分：籌碼面（含轉折與融資清洗）、基本面、估值、消息面、壓力 / 支撐位、ATR 風控部位試算，以及綜合評分。

每個面向都產生「每日分數」時間序列（消息面除外），讓綜合評分也能回測。
資料一律依「公布日」對齊，避免用到當時還不知道的資訊（前視偏誤）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import MIN_REVERSAL, NO_REVERSAL_CAP

# 各面向滿分（= 權重）
WEIGHTS = {"tech": 30, "chips": 25, "fund": 25, "value": 10, "news": 10}
FACTOR_NAMES = {
    "tech": "📈 技術與波浪",
    "chips": "🏦 籌碼面",
    "fund": "📊 基本面",
    "value": "💰 估值",
    "news": "📰 消息面",
}


# ---------------------------------------------------------------------------
# 籌碼面（25）：結合 20 日累積 + 近 3～5 日轉折（投信認養 / 融資斷頭清洗）
# ---------------------------------------------------------------------------
def chip_frame(d: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    """回傳每日籌碼指標與分數。d 是含 Volume 的股價資料。"""
    if chips.empty:
        return pd.DataFrame(index=d.index)
    c = chips.reindex(d.index)
    volume = c["volume"].fillna(d["Volume"]) if "volume" in c else d["Volume"]
    vol20 = volume.rolling(20).sum().replace(0, np.nan)
    vol5 = volume.rolling(5).sum().replace(0, np.nan)
    vol3 = volume.rolling(3).sum().replace(0, np.nan)

    f = pd.DataFrame(index=d.index)
    f["foreign20"] = c["foreign"].rolling(20, min_periods=15).sum() / vol20
    f["trust20"] = c["trust"].rolling(20, min_periods=15).sum() / vol20
    f["inst5"] = (c["foreign"] + c["trust"]).rolling(5, min_periods=4).sum() / vol5
    f["inst3"] = (c["foreign"] + c["trust"]).rolling(3, min_periods=2).sum() / vol3
    f["trust3"] = c["trust"].rolling(3, min_periods=2).sum() / vol3

    # 投信連續買超天數 & 投信剛轉買（前 10 日賣超或無動作，近 2～3 日轉買）
    buy = (c["trust"] > 0).astype(int)
    f["trust_streak"] = buy.groupby((buy == 0).cumsum()).cumsum()
    mb = c["margin_balance"].ffill()
    f["margin20"] = mb / mb.shift(20) - 1
    f["margin5"] = mb / mb.shift(5) - 1

    s_foreign = np.select(
        [f["foreign20"] >= 0.04, f["foreign20"] >= 0.015, f["foreign20"] >= 0, f["foreign20"] >= -0.02],
        [7, 5, 4, 2],
        default=0,
    )
    s_trust = np.select(
        [f["trust20"] >= 0.015, f["trust20"] >= 0.003, f["trust20"] >= 0], [6, 4, 2], default=0
    ) + np.where(f["trust_streak"] >= 2, 2, np.where(f["trust3"] > 0, 1, 0))
    s_trust = np.minimum(s_trust, 7)

    # 近期法人轉折（近 3～5 日外資或投信轉買）：在底部反轉時最關鍵
    s_turn = np.where((f["inst5"] > 0) & (f["inst3"] > 0), 5, np.where((f["inst3"] > 0) | (f["trust3"] > 0), 3, 0))

    # 融資沉澱與清洗：20 日減少或近 5 日急跌斷頭大減加分
    s_margin = np.select(
        [(f["margin20"] <= -0.05) | (f["margin5"] <= -0.03), f["margin20"] <= 0, f["margin20"] <= 0.05],
        [6, 4, 2],
        default=0,
    )
    f["score"] = np.clip(s_foreign + s_trust + s_turn + s_margin, 0, 25).astype(float)
    valid = f["foreign20"].notna() & f["margin20"].notna()
    f.loc[~valid, "score"] = np.nan
    return f


def explain_chips(f: pd.DataFrame, chips: pd.DataFrame) -> list[str]:
    if f.empty or f["score"].dropna().empty:
        return ["沒有籌碼資料（美股沒有公開的三大法人與融資資料）"]
    r = f.iloc[-1]
    last = chips.dropna(how="all").iloc[-1] if not chips.empty else None
    lines = [
        f"外資近 20 日{'買超' if r['foreign20'] >= 0 else '賣超'}，約占成交量 {abs(r['foreign20']):.1%}",
        f"投信近 20 日{'買超' if r['trust20'] >= 0 else '賣超'}，約占成交量 {abs(r['trust20']):.1%}"
        + (f"（🔥 已連買 {int(r['trust_streak'])} 天）" if r["trust_streak"] >= 2 else ""),
        f"近 3～5 日法人動向：{'外資/投信轉買進駐 ✅' if r.get('inst3', 0) > 0 or r.get('inst5', 0) > 0 else '法人仍在調節賣超'}",
        f"融資餘額：近 20 日 {r['margin20']:+.1%}、近 5 日 {r.get('margin5', 0):+.1%}"
        + (
            "（🧹 散戶退場、籌碼有效清洗 👍）"
            if (r["margin20"] <= -0.05 or r.get("margin5", 0) <= -0.03)
            else "（⚠️ 散戶追價或接刀、籌碼變亂）"
            if r["margin20"] > 0.05
            else ""
        ),
    ]
    if last is not None and pd.notna(last.get("foreign")):
        lines.append(
            f"最新一天：外資 {last['foreign'] / 1000:+,.0f} 張、投信 {last['trust'] / 1000:+,.0f} 張、自營商 {last['dealer'] / 1000:+,.0f} 張"
        )
    return lines


# ---------------------------------------------------------------------------
# 基本面（25）
# ---------------------------------------------------------------------------
def fundamental_frame(index: pd.DatetimeIndex, revenue: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """每日基本面分數（依公布日往後填）。"""
    f = pd.DataFrame(index=index)
    if revenue.empty and fin.empty:
        return f
    parts = []
    if not revenue.empty:
        r = revenue.copy()
        r["yoy"] = r["revenue"] / r["revenue"].shift(12) - 1
        r["yoy3"] = r["yoy"].rolling(3).mean()
        r["is_high"] = r["revenue"] >= r["revenue"].rolling(12).max()
        r["accel"] = r["revenue"].rolling(3).mean() > r["revenue"].shift(3).rolling(3).mean()
        s_yoy = np.select(
            [r["yoy3"] >= 0.30, r["yoy3"] >= 0.15, r["yoy3"] >= 0.05, r["yoy3"] >= 0, r["yoy3"] >= -0.10],
            [10, 8, 6, 4, 2], default=0,
        )
        s_high = np.where(r["is_high"], 5, np.where(r["accel"], 3, 0))
        r["rev_score"] = np.where(r["yoy3"].notna(), s_yoy + s_high, np.nan)
        daily = r.set_index("available")[["yoy", "yoy3", "is_high", "rev_score", "revenue"]].sort_index()
        daily = daily[~daily.index.duplicated(keep="last")]
        parts.append(daily.reindex(index, method="ffill"))
    if not fin.empty and "EPS" in fin:
        q = fin.copy()
        q["eps_ttm"] = q["EPS"].rolling(4).sum()
        q["eps_growth"] = np.where(q["eps_ttm"].shift(4) > 0, q["eps_ttm"] / q["eps_ttm"].shift(4) - 1, np.nan)
        s_eps = np.select(
            [q["eps_ttm"] <= 0, q["eps_growth"] >= 0.30, q["eps_growth"] >= 0.10, q["eps_growth"] >= 0],
            [0, 6, 5, 3], default=0,
        )
        if "gross_margin" in q:
            q["gm_change"] = q["gross_margin"] - q["gross_margin"].shift(4)
            s_gm = np.select([q["gm_change"] > 0.01, q["gm_change"] >= -0.01], [4, 2], default=0)
        else:
            q["gm_change"] = np.nan
            s_gm = 0
        q["fin_score"] = np.where(q["eps_ttm"].notna(), s_eps + s_gm, np.nan)
        cols = ["eps_ttm", "eps_growth", "gross_margin", "gm_change", "fin_score"]
        cols = [c for c in cols if c in q]
        daily = q.set_index("available")[cols].sort_index()
        daily = daily[~daily.index.duplicated(keep="last")]
        parts.append(daily.reindex(index, method="ffill"))
    f = pd.concat(parts, axis=1)
    rev_s = f["rev_score"] if "rev_score" in f else pd.Series(np.nan, index=index)
    fin_s = f["fin_score"] if "fin_score" in f else pd.Series(np.nan, index=index)
    have_rev, have_fin = rev_s.notna(), fin_s.notna()
    total = rev_s.fillna(0) + fin_s.fillna(0)
    max_pts = have_rev * 15 + have_fin * 10
    f["score"] = np.where(max_pts > 0, total / max_pts.replace(0, np.nan) * 25, np.nan)
    return f


def explain_fund(f: pd.DataFrame, us: dict | None = None) -> list[str]:
    if us:
        lines = []
        if us.get("rev_yoy") is not None:
            lines.append(f"最新一季營收年增 {us['rev_yoy']:+.1%}")
        if us.get("eps_growth") is not None:
            lines.append(f"近四季 EPS {us['eps_ttm']:.2f}，年增 {us['eps_growth']:+.1%}")
        if us.get("gm") is not None:
            lines.append(f"毛利率 {us['gm']:.1%}（比去年同季 {us['gm_change'] * 100:+.1f} 個百分點）")
        return lines or ["沒有基本面資料"]
    if f.empty or "score" not in f or f["score"].dropna().empty:
        return ["沒有基本面資料（ETF 不適用）"]
    r = f.iloc[-1]
    lines = []
    if pd.notna(r.get("yoy3")):
        lines.append(f"近 3 個月營收平均年增 {r['yoy3']:+.1%}（最新一個月 {r['yoy']:+.1%}）")
        if r.get("is_high"):
            lines.append("最新月營收創 12 個月新高 🔥")
    if pd.notna(r.get("eps_ttm")):
        g = r.get("eps_growth")
        lines.append(f"近四季 EPS 合計 {r['eps_ttm']:.2f} 元" + (f"，年增 {g:+.1%}" if pd.notna(g) else ""))
    if pd.notna(r.get("gross_margin")):
        lines.append(
            f"最新一季毛利率 {r['gross_margin']:.1%}"
            + (f"（比去年同季 {r['gm_change'] * 100:+.1f} 個百分點）" if pd.notna(r.get("gm_change")) else "")
        )
    return lines


def us_fund_score(us: dict) -> float:
    if not us:
        return np.nan
    pts, mx = 0.0, 0.0
    if us.get("rev_yoy") is not None:
        y = us["rev_yoy"]
        pts += 10 if y >= 0.30 else 8 if y >= 0.15 else 6 if y >= 0.05 else 4 if y >= 0 else 2 if y >= -0.10 else 0
        mx += 10
    if us.get("eps_growth") is not None:
        g = us["eps_growth"]
        pts += 0 if us.get("eps_ttm", 0) <= 0 else 6 if g >= 0.30 else 5 if g >= 0.10 else 3 if g >= 0 else 0
        mx += 6
    if us.get("gm_change") is not None:
        pts += 4 if us["gm_change"] > 0.01 else 2 if us["gm_change"] >= -0.01 else 0
        mx += 4
    return pts / mx * 25 if mx else np.nan


# ---------------------------------------------------------------------------
# 估值（10）：本益比在近 5 年的百分位
# ---------------------------------------------------------------------------
def value_frame(index: pd.DatetimeIndex, per: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    if per.empty:
        return f
    p = per.reindex(index, method="ffill", limit=10)
    f["per"] = p
    f["pct"] = p.rolling(1250, min_periods=250).rank(pct=True)
    f["score"] = np.select(
        [f["pct"] <= 0.2, f["pct"] <= 0.4, f["pct"] <= 0.6, f["pct"] <= 0.8], [10, 8, 5, 3], default=0
    ).astype(float)
    f.loc[f["pct"].isna(), "score"] = np.nan
    return f


def explain_value(f: pd.DataFrame, us: dict | None = None) -> list[str]:
    if us and us.get("pe"):
        return [f"本益比約 {us['pe']:.1f} 倍（美股沒有歷史本益比資料，不列入評分）"]
    if f.empty or "score" not in f or f["score"].dropna().empty:
        return ["沒有本益比資料（ETF 不適用）"]
    r = f.iloc[-1]
    level = "偏低 👍" if r["pct"] <= 0.3 else "偏高 ⚠️" if r["pct"] >= 0.8 else "中等"
    return [f"目前本益比 {r['per']:.1f} 倍，在近 5 年的 {r['pct']:.0%} 位置（{level}）"]


# ---------------------------------------------------------------------------
# 壓力 / 支撐位 + ATR 動態風控計畫
# ---------------------------------------------------------------------------
def support_resistance(d: pd.DataFrame, lookback: int = 250) -> dict:
    """綜合分價量表、前波高低點、均線，找出最近的壓力與支撐，並計算 ATR 停損停利價。"""
    w = d.iloc[-lookback:]
    close = float(d["Close"].iloc[-1])
    atr = float(d["ATR14"].iloc[-1]) if "ATR14" in d and pd.notna(d["ATR14"].iloc[-1]) else close * 0.025
    levels: list[tuple[float, str, float]] = []

    # 1. 分價量表（近 120 日）：成交量最集中的價位
    vp = d.iloc[-120:]
    lo, hi = vp["Low"].min(), vp["High"].max()
    bins = np.linspace(lo, hi, 41)
    typical = (vp["High"] + vp["Low"] + vp["Close"]) / 3
    hist, edges = np.histogram(typical, bins=bins, weights=vp["Volume"])
    centers = (edges[:-1] + edges[1:]) / 2
    profile = pd.DataFrame({"price": centers, "volume": hist})
    share = hist / hist.sum() if hist.sum() else hist
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and share[i] >= 1.5 / len(hist):
            levels.append((centers[i], "成交密集區", 1 + share[i] * 20))

    # 2. 前波高低點（左右各 10 天的轉折）
    k = 10
    highs, lows = w["High"].to_numpy(), w["Low"].to_numpy()
    for i in range(k, len(w) - k):
        if highs[i] == highs[i - k : i + k + 1].max():
            levels.append((highs[i], "前波高點", 1.5))
        if lows[i] == lows[i - k : i + k + 1].min():
            levels.append((lows[i], "前波低點", 1.5))

    # 3. 均線
    for ma, name in [("MA20", "月線"), ("MA60", "季線"), ("MA240", "年線")]:
        v = d[ma].iloc[-1]
        if pd.notna(v):
            levels.append((float(v), name, 1.0))

    # 合併距離 1.5% 以內的價位
    levels.sort(key=lambda x: x[0])
    clusters: list[dict] = []
    for price, src, strength in levels:
        if clusters and abs(price / clusters[-1]["price"] - 1) <= 0.015:
            c = clusters[-1]
            c["prices"].append(price)
            c["sources"].add(src)
            c["strength"] += strength
            c["price"] = float(np.average(c["prices"]))
        else:
            clusters.append({"price": price, "prices": [price], "sources": {src}, "strength": strength})

    def fmt(c):
        return {"price": c["price"], "sources": "、".join(sorted(c["sources"])), "strength": c["strength"]}

    above = [fmt(c) for c in clusters if c["price"] > close * 1.005]
    below = [fmt(c) for c in clusters if c["price"] < close * 0.995]
    resist = sorted(above, key=lambda x: x["price"])[:3]
    support = sorted(below, key=lambda x: -x["price"])[:3]

    rr = None
    if resist and support:
        rr = (resist[0]["price"] - close) / (close - support[0]["price"])

    # ATR 動態停損與移動停利建議
    atr_stop = round(close - 2.0 * atr, 2)
    sup_stop = round(support[0]["price"] * 0.99, 2) if support else atr_stop
    suggest_stop = round(max(atr_stop, sup_stop) if sup_stop < close * 0.985 else atr_stop, 2)
    stop_pct = (suggest_stop / close - 1) * 100
    chandelier = round(float(d["Chandelier_Exit"].iloc[-1]), 2) if "Chandelier_Exit" in d else atr_stop

    return {
        "close": close,
        "atr": round(atr, 2),
        "atr_pct": round(atr / close * 100, 2),
        "suggest_stop": suggest_stop,
        "stop_pct": round(stop_pct, 2),
        "chandelier": chandelier,
        "resistance": resist,
        "support": support,
        "rr": rr,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# 綜合評分
# ---------------------------------------------------------------------------
def composite(tech: pd.DataFrame, parts: dict[str, pd.Series], news_score: float | None = None) -> pd.DataFrame:
    """把各面向（已換算成各自滿分）合併成 0–100 分。缺少的面向不計，按比例放大。"""
    out = pd.DataFrame(index=tech.index)
    out["tech"] = tech["score"] / 100 * WEIGHTS["tech"]
    for k in ["chips", "fund", "value"]:
        out[k] = parts.get(k, pd.Series(np.nan, index=tech.index)).reindex(tech.index)
    out["news"] = np.nan
    if news_score is not None and len(out):
        out.iloc[-1, out.columns.get_loc("news")] = news_score

    def combine(cols: list[str]) -> pd.Series:
        pts = out[cols].sum(axis=1, min_count=1)
        mx = sum(out[c].notna() * WEIGHTS[c] for c in cols)
        raw = pts / mx.replace(0, np.nan) * 100
        capped = raw.where(tech["reversal"] >= MIN_REVERSAL, raw.clip(upper=NO_REVERSAL_CAP))
        return capped.where(out["tech"].notna())

    out["score_bt"] = combine(["tech", "chips", "fund", "value"])  # 回測用（不含消息面）
    out["score"] = combine(["tech", "chips", "fund", "value", "news"])
    return out
