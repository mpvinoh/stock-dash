"""把所有資料來源、波浪黃金分割、歷史相似K線、大盤氣象站與多因子評分整合在一起。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import display_name, fetch_history
from factors import (
    FACTOR_NAMES,
    WEIGHTS,
    chip_frame,
    composite,
    explain_chips,
    explain_fund,
    explain_value,
    fundamental_frame,
    support_resistance,
    us_fund_score,
    value_frame,
)
from model import LABELS, add_indicators, explain, label_for, score_frame
from sources import (
    LIMIT_HIT,
    fetch_chips,
    fetch_financials,
    fetch_news,
    fetch_per,
    fetch_revenue,
    fetch_us_fundamentals,
    is_etf,
    is_taiwan,
)
from waves import analyze_waves, match_historical_patterns
from keybar import analyze_keybars

RR_MIN = 1.3  # 風險報酬比低於這個值，「符合買進」降一級
CHIPS_MIN = 10  # 籌碼濾網：籌碼分數至少 10 / 25（法人沒在倒貨、融資沒暴增）


def chips_ok_series(res: dict) -> pd.Series | None:
    """籌碼濾網是否通過（每日）。沒有籌碼資料（美股）時回傳 None。"""
    f = res.get("chips")
    if f is None or "score" not in f or f["score"].dropna().empty:
        return None
    return f["score"] >= CHIPS_MIN


def timing_series(res: dict, threshold: float) -> pd.Series:
    """買進時機分數（回測與訊號用）：技術與波浪分數，籌碼濾網沒過時壓在門檻以下。"""
    tech = res["tech"]["score"]
    ok = chips_ok_series(res)
    if ok is None:
        return tech
    chips_score = res["chips"]["score"]
    capped = tech.where(ok, np.minimum(tech, threshold - 1))
    return capped.where(chips_score.notna())


def market_regime_and_rs(d: pd.DataFrame, ticker: str) -> dict:
    """計算大盤氣象站（0050 或 SPY）與個股相對大盤強弱（RS）。"""
    bm_code = "0050" if is_taiwan(ticker) else "SPY"
    try:
        _, bm_raw = fetch_history(bm_code)
        bm = add_indicators(bm_raw)
    except Exception:
        return {"regime": "未知", "regime_desc": "", "rs_score": 50, "rs_desc": "無法取得大盤基準"}

    row = bm.iloc[-1]
    c = float(row["Close"])
    long_ma = float(row["LongMA"]) if pd.notna(row["LongMA"]) else c
    slope = float(row["LongMA_slope"]) if pd.notna(row["LongMA_slope"]) else 0.0
    dd = float(row["Drawdown"]) if pd.notna(row["Drawdown"]) else 0.0
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    vol_ratio = float(row["Vol20"] / row["Vol252"]) if pd.notna(row["Vol252"]) and row["Vol252"] > 0 else 1.0

    bm_name = "台股大盤（0050）" if is_taiwan(ticker) else "美股大盤（S&P 500）"
    if slope > 0 and (dd <= -0.10 or rsi < 35 or vol_ratio > 1.4):
        regime = "⛈️ 長多急殺洗盤"
        regime_desc = f"{bm_name}年線仍向上，但自高點回檔 {dd * 100:.1f}%、RSI={rsi:.0f}，屬於恐慌清洗階段，切忌急接刀，宜等止跌訊號分批佈局"
    elif slope > 0 and c >= row["MA60"]:
        regime = "🌞 多頭順風"
        regime_desc = f"{bm_name}位於季線與年線之上，多頭結構穩健，拉回支撐可順勢操作"
    elif slope > 0 and c >= long_ma * 0.97:
        regime = "🌤️ 多頭中期整理"
        regime_desc = f"{bm_name}守在年線附近整理（距高點 {dd * 100:.1f}%），選股重於選市"
    else:
        regime = "❄️ 空頭警戒"
        regime_desc = f"{bm_name}跌破年線且趨勢轉弱，建議提高買進標準並縮小部位"

    # 計算相對強弱 RS（比較近 20 日與 60 日超額報酬）
    common = d.index.intersection(bm.index)
    if len(common) < 65:
        return {"regime": regime, "regime_desc": regime_desc, "rs_score": 50, "rs_desc": "上市時間較短"}

    sc, bc = d.loc[common, "Close"], bm.loc[common, "Close"]
    ex5 = (sc.iloc[-1] / sc.iloc[-6] - 1) - (bc.iloc[-1] / bc.iloc[-6] - 1)
    ex20 = (sc.iloc[-1] / sc.iloc[-21] - 1) - (bc.iloc[-1] / bc.iloc[-21] - 1)
    ex60 = (sc.iloc[-1] / sc.iloc[-61] - 1) - (bc.iloc[-1] / bc.iloc[-61] - 1)
    raw_rs = 50 + ex20 * 180 + ex60 * 120
    rs_score = int(np.clip(round(raw_rs), 1, 99))

    bm_down5 = (bc.iloc[-1] / bc.iloc[-6] - 1) < -0.025
    resilient = bm_down5 and ex5 > 0.015
    if resilient:
        rs_desc = f"💪 逆勢抗跌領頭羊（RS={rs_score}）：近 5 日贏大盤 {ex5 * 100:+.1f}%、近 60 日贏大盤 {ex60 * 100:+.1f}%"
    elif rs_score >= 65:
        rs_desc = f"🔥 強於大盤（RS={rs_score}）：近 20 日相對大盤 {ex20 * 100:+.1f}%、近 60 日 {ex60 * 100:+.1f}%"
    elif rs_score <= 35:
        rs_desc = f"🐢 弱於大盤（RS={rs_score}）：近 20 日相對大盤 {ex20 * 100:+.1f}%、近 60 日 {ex60 * 100:+.1f}%"
    else:
        rs_desc = f"⚖️ 與大盤同步（RS={rs_score}）：近 20 日相對大盤 {ex20 * 100:+.1f}%"

    return {
        "regime": regime,
        "regime_desc": regime_desc,
        "rs_score": rs_score,
        "rs_desc": rs_desc,
        "resilient": resilient,
    }


def analyze(user_input: str, token: str = "") -> dict:
    """抓資料並計算所有面向（含波浪、黃金分割、歷史相似K線、大盤氣象站）。"""
    LIMIT_HIT["value"] = False
    ticker, raw = fetch_history(user_input)
    d = add_indicators(raw)
    tech = score_frame(d)
    res = {"ticker": ticker, "name": display_name(ticker), "d": d, "tech": tech, "us": None}

    chips = fetch_chips(ticker, token)
    res["chips_raw"] = chips
    res["chips"] = chip_frame(d, chips)

    if is_taiwan(ticker):
        res["revenue"] = fetch_revenue(ticker, token)
        res["fin"] = fetch_financials(ticker, token)
        res["fund"] = fundamental_frame(d.index, res["revenue"], res["fin"])
        res["value"] = value_frame(d.index, fetch_per(ticker, token))
    else:
        us = fetch_us_fundamentals(ticker)
        res["us"] = us
        res["revenue"], res["fin"] = pd.DataFrame(), pd.DataFrame()
        fund = pd.DataFrame(index=d.index)
        fund["score"] = np.nan
        if us:
            fund.iloc[-1, 0] = us_fund_score(us)
        res["fund"] = fund
        res["value"] = pd.DataFrame(index=d.index)

    res["news"] = fetch_news(ticker, token)
    res["sr"] = support_resistance(d)
    res["waves"] = analyze_waves(d, lookback=250)
    res["keybar"] = analyze_keybars(d, res["waves"])
    res["patterns"] = match_historical_patterns(d, window=20, top_k=10)
    res["market"] = market_regime_and_rs(d, ticker)
    res["finmind_limited"] = LIMIT_HIT["value"]
    return res


def parts_of(res: dict) -> dict[str, pd.Series]:
    out = {}
    for k in ["chips", "fund", "value"]:
        f = res.get(k)
        if f is not None and "score" in f:
            out[k] = f["score"]
    return out


def build_composite(res: dict, news_score: float | None) -> pd.DataFrame:
    return composite(res["tech"], parts_of(res), news_score)


def final_label(res: dict, threshold: float) -> tuple[tuple, list[str]]:
    """買進建議：技術與波浪分數 + 止跌閘門 + 籌碼濾網 + 風險報酬比 + 第5波頂背離警戒。"""
    tech = res["tech"]
    score, reversal = tech["score"].iloc[-1], tech["reversal"].iloc[-1]
    label = label_for(score, threshold, reversal)
    notes: list[str] = []
    ok = chips_ok_series(res)
    if label == LABELS[0] and ok is not None and not bool(ok.iloc[-1]):
        label = LABELS[1]
        notes.append(
            f"技術與波浪達標，但籌碼尚未轉好（籌碼分數 {res['chips']['score'].iloc[-1]:.0f} < {CHIPS_MIN}），所以降一級"
        )
    div = (res.get("waves") or {}).get("divergence")
    if div and div["type"] == "bearish" and label == LABELS[0]:
        label = LABELS[1]
        notes.append("出現第 5 波高檔頂背離警訊，降一級避免追高")
    rr = res["sr"]["rr"]
    if label == LABELS[0] and rr is not None and rr < RR_MIN:
        label = LABELS[1]
        notes.append(f"分數達標，但上方壓力較近（風險報酬比 {rr:.1f} < {RR_MIN}），所以降一級")
    return label, notes


def factor_summary(comp: pd.DataFrame) -> dict[str, float | None]:
    """最新一天各面向的得分率（0–1）。"""
    row = comp.iloc[-1]
    return {k: (None if pd.isna(row[k]) else float(row[k]) / WEIGHTS[k]) for k in WEIGHTS}


def build_context(res: dict, comp: pd.DataFrame, label: str, notes: list[str], news: list[dict], bt: pd.DataFrame | None) -> dict:
    """整理給 AI 分析師的數據（包含波浪理論標示、黃金分割位、歷史相似走勢與風控價位）。"""
    d, tech = res["d"], res["tech"]
    info = explain(d, tech)
    row = d.iloc[-1]
    sr = res["sr"]
    wv = res["waves"]
    mkt = res["market"]
    pat = res["patterns"].get("summary")
    fs = factor_summary(comp)
    ok = chips_ok_series(res)

    ctx = {
        "股票": res["name"],
        "類型": "台股ETF" if is_etf(res["ticker"]) else "台股" if is_taiwan(res["ticker"]) else "美股",
        "資料日期": f"{d.index[-1]:%Y-%m-%d}",
        "收盤價": round(float(row["Close"]), 2),
        "近5日漲跌": f"{row['Close'] / d['Close'].iloc[-6] - 1:+.1%}",
        "距一年高點": f"{row['Drawdown']:+.1%}",
        "大盤環境與相對強弱": {
            "大盤氣象": mkt["regime"],
            "大盤說明": mkt["regime_desc"],
            "個股相對強弱RS": mkt["rs_desc"],
        },
        "買進時機分數（技術與波浪，0-100）": round(float(tech["score"].iloc[-1]), 0),
        "籌碼濾網": None if ok is None else ("通過" if bool(ok.iloc[-1]) else "未通過"),
        "買進建議": label,
        "建議備註": notes,
        "綜合評估分數（五面向，0-100，僅供參考）": round(float(comp["score"].iloc[-1]), 0),
        "各面向得分率": {
            FACTOR_NAMES[k]: (None if v is None else f"{v:.0%}") for k, v in fs.items()
        },
        "波浪理論與黃金分割": {
            "目前波浪階段": wv["wave_stage"],
            "近期波段轉折序列": wv["path_str"],
            "黃金分割分析": wv["fib_str"],
            "頂底背離訊號": wv["divergence"]["desc"] if wv.get("divergence") else "無明顯背離",
        },
        "歷史相似K線比對（前10次最相似走勢）": (
            {
                "1個月後勝率": f"{pat['win_20d']:.0f}%",
                "1個月平均報酬": f"{pat['avg_20d']:+.1f}%",
                "3個月後勝率": f"{pat['win_60d']:.0f}%",
                "3個月平均報酬": f"{pat['avg_60d']:+.1f}%",
                "1個月內中位數最深再跌": f"{pat['worst_20d_mae']:+.1f}%",
            }
            if pat
            else "資料不足"
        ),
        "技術面": {
            "情境": info["scenario"],
            "細節": sum(info["parts"].values(), []),
            "扣分": info["penalty"],
        },
        "籌碼面": explain_chips(res["chips"], res["chips_raw"]),
        "基本面": explain_fund(res["fund"], res["us"]),
        "估值": explain_value(res["value"], res["us"]),
        "壓力支撐與ATR風控": {
            "最近壓力": [f"{x['price']:.2f}（{x['sources']}）" for x in sr["resistance"]],
            "最近支撐": [f"{x['price']:.2f}（{x['sources']}）" for x in sr["support"]],
            "風險報酬比": None if sr["rr"] is None else round(sr["rr"], 2),
            "14日ATR波動": f"{sr['atr']}（{sr['atr_pct']:.1f}%）",
            "建議初始停損價": f"{sr['suggest_stop']}（{sr['stop_pct']:+.1f}%）",
            "Chandelier移動停利價": sr["chandelier"],
        },
        "近期新聞": [
            {"日期": n["date"][:10], "標題": n["title"], "情緒": n.get("sentiment")} for n in news[:12]
        ],
    }
    if bt is not None and len(bt):
        r = bt.iloc[1]
        ctx["買進訊號歷史回測（持有3個月）"] = {
            "訊號次數": int(r["訊號次數"]),
            "訊號勝率": None if pd.isna(r["訊號勝率"]) else f"{r['訊號勝率']:.0f}%",
            "隨便買勝率": None if pd.isna(r["隨便哪天買的勝率"]) else f"{r['隨便哪天買的勝率']:.0f}%",
        }
    kb = res.get("keybar")
    if kb:
        ctx["有效K棒起漲訊號（第二套獨立訊號，和買進時機分數分開）"] = keybar_context(kb)
    return ctx


def keybar_summary_row(stats: pd.DataFrame, idx: int) -> str:
    if stats is None or len(stats) <= idx:
        return "資料不足"
    r = stats.iloc[idx]
    if r["訊號次數"] == 0 or pd.isna(r["訊號勝率"]):
        return "沒有歷史訊號"
    small = "（⚠️ 樣本太少，參考價值低）" if r["訊號次數"] < 5 else ""
    return (
        f"{int(r['訊號次數'])} 次，勝率 {r['訊號勝率']:.0f}%（隨便買 {r['隨便哪天買的勝率']:.0f}%），"
        f"平均 {r['訊號平均報酬']:+.1f}%（隨便買 {r['隨便哪天買的平均報酬']:+.1f}%）{small}"
    )


def keybar_context(kb: dict) -> dict:
    """整理給 AI 的有效 K 棒資訊。"""

    def st(x: dict) -> dict:
        out = {"狀態": x["title"]}
        if x.get("desc"):
            out["說明"] = x["desc"].replace("**", "")
        if x.get("wave"):
            out["波浪位置"] = x["wave"].replace("**", "")
        return out

    black = kb.get("black")
    return {
        "短線（日線，僅供停損停利價位參考；回測沒有統計優勢，不給買進建議）": st(kb["short"]),
        "中線（日線有效紅K + 年線向上，確認後為買點）": {
            **st(kb["mid"]),
            "這檔歷史回測（持有3個月）": keybar_summary_row(kb["bt_mid"][0], 1),
        },
        "長線（週線有效紅K + 40週線向上，確認後為買點）": {
            **st(kb["long"]),
            "這檔歷史回測（持有1季）": keybar_summary_row(kb["bt_long"][0], 1),
            "本週K棒是否尚未收完": bool(kb["weekly_partial"]),
        },
        "近10日有效黑K（短線風險提示，回測只在1週內略偏弱，不是賣出訊號）": (
            f"{black['date']:%Y-%m-%d} 放量長黑 {black['ret']:+.1f}%"
            + ("，且收在月線之下" if black["below_ma20"] else "")
            if black
            else "無"
        ),
    }
