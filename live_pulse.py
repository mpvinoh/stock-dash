"""🤖 盤中即時量價巡邏引擎（0-Token 第一關篩選）：支援台股與美股。

設計原則：
1. 第一關完全由 Python 計算（0 Token），只在篩出明顯量價異動或趨勢逼近突破時，才交給 AI 寫簡短口語播報。
2. 盤中依已開盤時間比例估算全天預估量（避免早上剛開盤量比失真），找出「疑似大資金放量點火」「突破 20 日高」「逼近突破」等訊號。
3. 休市時段自動休眠不浪費資源；使用者也可手動按「立即巡邏」查看最新交易日的盤面熱點。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from data import NAME_MAP, tw_names
from sources import is_taiwan

# 4 段互動個性與自動巡邏頻率
PROFILES: dict[str, dict] = {
    "enthusiastic": {
        "label": "🔥 積極熱情",
        "interval_sec": 300,
        "vol_min": 1.45,
        "ret_min": 0.015,
        "near_pct": 0.020,
        "desc": "每 5 分鐘巡邏・稍有放量或逼近突破就熱情提醒、主動拋問題互動",
        "tone": "活潑熱情、像一起盯盤的好戰友，語氣有朝氣（適度用 1–2 個表情符號），結尾主動拋出一個輕鬆具體的問題邀使用者互動（例如問要不要切過去看圖或比較風險）。",
    },
    "normal": {
        "label": "🙂 普通",
        "interval_sec": 900,
        "vol_min": 1.90,
        "ret_min": 0.022,
        "near_pct": 0.010,
        "desc": "每 15 分鐘巡邏・明顯放量點火或突破前高時提醒（推薦）",
        "tone": "親切自然、重點俐落，像專業又好聊的盤中播報員，簡短說明為什麼值得留意並給一個溫和的下一步建議。",
    },
    "calm": {
        "label": "🧘 沉穩",
        "interval_sec": 1800,
        "vol_min": 2.50,
        "ret_min": 0.032,
        "near_pct": 0.003,
        "desc": "每 30 分鐘巡邏・只報重大放量突破或觀察名單風險警訊",
        "tone": "沉穩冷靜、像資深風控顧問，文字精煉克制，只陳述關鍵突破或停損警訊，絕不鼓吹追價。",
    },
    "off": {
        "label": "🔇 不主動提示",
        "interval_sec": None,
        "vol_min": 99.0,
        "ret_min": 99.0,
        "near_pct": 0.0,
        "desc": "關閉自動巡邏，零背景呼叫與 Token 消耗",
        "tone": "客觀簡潔。",
    },
}

# 精選高流動性熱門巡邏池（各約 25 檔，一次批次下載只要 3–5 秒）
HOT_TW = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW",
    "3231.TW", "2327.TW", "6669.TW", "2345.TW", "3017.TW",
    "2603.TW", "2618.TW", "2303.TW", "3711.TW", "2881.TW",
    "2891.TW", "2357.TW", "3034.TW", "2379.TW", "3661.TW",
    "2408.TW", "2376.TW", "3037.TW", "2368.TW", "1519.TW",
]
HOT_US = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "TSM", "AMD",
    "PLTR", "MU", "ARM", "ASML", "ORCL",
    "NFLX", "CRWD", "PANW", "VRT", "ANET",
    "LLY", "JPM", "COST", "COIN", "MSTR",
]


def market_status_now() -> dict:
    """判斷目前台股與美股是否正處於盤中交易時段，並計算已開盤時間比例（供預估量使用）。"""
    now_tw = dt.datetime.now(ZoneInfo("Asia/Taipei"))
    now_us = dt.datetime.now(ZoneInfo("America/New_York"))

    tw_open, tw_frac = False, 1.0
    if now_tw.weekday() < 5:
        mins = now_tw.hour * 60 + now_tw.minute
        if 9 * 60 <= mins <= 13 * 60 + 35:
            tw_open = True
            tw_frac = float(np.clip((mins - 9 * 60) / 270.0, 0.15, 1.0))

    us_open, us_frac = False, 1.0
    if now_us.weekday() < 5:
        mins = now_us.hour * 60 + now_us.minute
        if 9 * 60 + 30 <= mins <= 16 * 60 + 5:
            us_open = True
            us_frac = float(np.clip((mins - (9 * 60 + 30)) / 390.0, 0.15, 1.0))

    if tw_open and us_open:
        label = "🟢 台美股皆盤中"
    elif tw_open:
        label = f"🟢 台股盤中（{now_tw:%H:%M}）"
    elif us_open:
        label = f"🟢 美股盤中（美東 {now_us:%H:%M}）"
    else:
        label = f"🌙 目前非交易時段（台北 {now_tw:%H:%M}）"

    return {
        "tw_open": tw_open,
        "us_open": us_open,
        "any_open": tw_open or us_open,
        "tw_frac": tw_frac,
        "us_frac": us_frac,
        "label": label,
        "time_str": now_tw.strftime("%H:%M"),
        "tw_date": now_tw.date(),
        "us_date": now_us.date(),
    }


def _short_name(ticker: str) -> str:
    code = ticker.split(".")[0]
    if code in NAME_MAP:
        return f"{NAME_MAP[code]}（{code}）"
    tw = tw_names().get(code)
    if tw:
        return f"{tw[0]}（{code}）"
    return code


def _batch_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    unique = list(dict.fromkeys(tickers))
    if not unique:
        return out
    try:
        raw = yf.download(
            unique, period="3mo", interval="1d", auto_adjust=True, group_by="ticker",
            threads=True, progress=False,
        )
    except Exception:
        return out
    for t in unique:
        try:
            df = raw[t] if len(unique) > 1 else raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if len(df) >= 25:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[t] = df
        except Exception:
            continue
    return out


def scan_live_pulse(
    watch_tickers: list[str],
    profile_key: str = "normal",
    scope: str = "auto",
) -> dict:
    """執行一次 0-Token 盤中／最新盤面異動巡邏，回傳排序後的前幾筆異動事件。

    scope: 'auto'（盤中優先掃開盤中的市場；休市時台美熱門一起掃）、'tw'、'us'、'all'
    """
    prof = PROFILES.get(profile_key, PROFILES["normal"])
    st = market_status_now()
    watch_set = set(watch_tickers)

    pool = list(watch_tickers)
    if scope == "tw" or (scope == "auto" and st["tw_open"] and not st["us_open"]):
        pool += HOT_TW
    elif scope == "us" or (scope == "auto" and st["us_open"] and not st["tw_open"]):
        pool += HOT_US
    else:
        pool += HOT_TW[:18] + HOT_US[:18]

    prices = _batch_download(pool)
    events: list[dict] = []

    for t, df in prices.items():
        if len(df) < 25:
            continue
        is_tw = is_taiwan(t)
        c = df["Close"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        h = df["High"].to_numpy(float)
        lo = df["Low"].to_numpy(float)
        v = df["Volume"].to_numpy(float)

        close, prev_c, opn, hi, low, vol = c[-1], c[-2], o[-1], h[-1], lo[-1], v[-1]
        if prev_c <= 0:
            continue
        ret = close / prev_c - 1
        vol_ma20 = float(np.nanmean(v[-21:-1])) if len(v) >= 21 else float(np.nanmean(v[:-1]))
        if vol_ma20 <= 0:
            continue

        # 若該市場正處於盤中且最後一根 K 是今天，依開盤時間比例推估全天成交量
        last_dt = df.index[-1].date()
        is_today_bar = (is_tw and st["tw_open"] and last_dt == st["tw_date"]) or (
            (not is_tw) and st["us_open"] and last_dt == st["us_date"]
        )
        frac = st["tw_frac"] if is_tw else st["us_frac"]
        est_vol = (vol / frac) if is_today_bar else vol
        vol_ratio = est_vol / vol_ma20

        rng = hi - low
        body_ratio = (close - opn) / rng if rng > 0 else 0.0
        ma10 = float(np.mean(c[-10:]))
        ma20 = float(np.mean(c[-20:]))
        ma60 = float(np.mean(c[-60:])) if len(c) >= 60 else ma20
        prev_hi20 = float(np.max(h[-21:-1]))
        dist_hi20 = close / prev_hi20 - 1  # >=0 代表突破 20 日高
        in_watch = t in watch_set

        tags: list[str] = []
        prio = 0
        detail_parts: list[str] = []

        # 1. 疑似大資金放量點火
        if close > opn and body_ratio >= 0.45 and vol_ratio >= prof["vol_min"] and ret >= prof["ret_min"]:
            tags.append("🔥 疑似大資金放量點火")
            prio += 5
            detail_parts.append(f"{'預估' if is_today_bar else ''}量能放大至均量 {vol_ratio:.1f} 倍、帶量收紅 {ret * 100:+.1f}%")

        # 2. 趨勢突破確立（站上 20 日新高）
        if dist_hi20 >= 0 and close > opn and vol_ratio >= 1.25 and close > ma60:
            tags.append("🚀 突破20日新高")
            prio += 4
            detail_parts.append(f"站上近20日高點 {prev_hi20:,.2f}（量比 {vol_ratio:.1f} 倍）")
        # 3. 蓄勢逼近突破（距離 20 日高很近且多頭排列）
        elif -prof["near_pct"] <= dist_hi20 < 0 and close >= ma20 and close > ma60 and vol_ratio >= 1.15 and ret > 0:
            tags.append("👀 逼近前高準備啟動")
            prio += 3
            detail_parts.append(f"距20日高點 {prev_hi20:,.2f} 僅差 {abs(dist_hi20) * 100:.1f}%，量比 {vol_ratio:.1f} 倍")

        # 4. 長多拉回止跌轉強（積極／普通模式才主動提）
        if not tags and profile_key in ("enthusiastic", "normal"):
            dd20 = close / prev_hi20 - 1
            if close > ma60 and -0.15 <= dd20 <= -0.05 and close > opn and close > ma10 and ret >= 0.015 and vol_ratio >= 1.1:
                tags.append("🎯 回檔轉強跡象")
                prio += 2
                detail_parts.append(f"守在季線之上回檔後收紅 {ret * 100:+.1f}%，站回10日線")

        # 5. 觀察名單風險警訊（放量重挫或跌破月線）
        if in_watch and ret <= -0.03 and vol_ratio >= 1.4:
            tags.append("⚠️ 帶量急跌警戒")
            prio += 5
            detail_parts.append(f"跌幅 {ret * 100:+.1f}% 且量比 {vol_ratio:.1f} 倍，留意停損防線")

        if not tags:
            continue
        if in_watch:
            prio += 2  # 觀察名單標的優先排前面

        code = t.split(".")[0]
        events.append({
            "ticker": t,
            "code": code,
            "name": _short_name(t),
            "market": "tw" if is_tw else "us",
            "flag": "🇹🇼" if is_tw else "🇺🇸",
            "in_watch": in_watch,
            "tags": tags,
            "tag_str": "・".join(tags[:2]),
            "close": round(close, 2),
            "ret_pct": round(ret * 100, 2),
            "vol_ratio": round(vol_ratio, 2),
            "prev_hi20": round(prev_hi20, 2),
            "is_intraday": is_today_bar,
            "date": f"{df.index[-1]:%Y-%m-%d}",
            "detail": "；".join(detail_parts),
            "prio": prio,
        })

    events.sort(key=lambda x: (x["prio"], x["vol_ratio"], x["ret_pct"]), reverse=True)
    top = events[:5]

    # 產生指紋（用於快取：同一組異動與等級不重複消耗 AI Token）
    sig_raw = json.dumps(
        [(e["ticker"], e["tag_str"], round(e["ret_pct"], 0), round(e["vol_ratio"], 1)) for e in top],
        ensure_ascii=False,
    )
    sig = hashlib.md5(f"{profile_key}:{sig_raw}".encode()).hexdigest()[:12]

    return {
        "status": st,
        "profile": profile_key,
        "scanned": len(prices),
        "events": top,
        "signature": sig,
        "checked_at": st["time_str"],
    }
