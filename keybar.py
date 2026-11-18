"""有效 K 棒（放量長紅／長黑）、起漲確認與三週期買點雷達（第二套獨立訊號）。

邏輯：
1. 有效紅 K：收紅、漲幅 ≥ max(門檻, ATR 倍數)、實體飽滿、成交量放大。
2. 每根有效 K 產生 3 個關鍵價位：K 高（突破確認）、實體 1/2（多空分界）、K 低（失效／停損）。
3. 起漲確認（有效 K 出現後 N 根內）：
   - 收盤突破 K 高且收紅 → 「突破K高」確認
   - 拉回觸及實體 1/2 但收盤守住 K 低，之後收紅且收盤 > 前一根高點 → 「回測1/2不破」確認
   - 收盤跌破 K 低 → 失效；超過 N 根仍未確認 → 過期
4. 三個週期：
   - 短線（日線，不加趨勢條件）：只提供關鍵價位當停損停利參考（回測沒有統計優勢）
   - 中線（日線 + 年線向上）：起漲確認為買點
   - 長線（週線 + 40 週線向上）：起漲確認為買點
5. 有效黑 K：放量長黑，作為賣出／減碼警示（是否顯示由回測決定）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAILY = {"min_ret": 0.03, "atr_k": 1.2, "body_min": 0.60, "vol_k": 1.5, "wait": 10, "cooldown": 5}
WEEKLY = {"min_ret": 0.05, "atr_k": 1.0, "body_min": 0.55, "vol_k": 1.4, "wait": 6, "cooldown": 2}

DAILY_H = {20: "1個月", 60: "3個月"}
WEEKLY_H = {4: "1個月", 13: "1季", 26: "半年"}
PERIODS = [("2016–2020", "2016-01-01", "2020-12-31"), ("2021 至今", "2021-01-01", "2100-01-01")]


# ---------------------------------------------------------------------------
# 基本工具
# ---------------------------------------------------------------------------
def _atr_pct(d: pd.DataFrame) -> pd.Series:
    c, h, lo = d["Close"], d["High"], d["Low"]
    prev = c.shift(1)
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c


def effective_bars(d: pd.DataFrame, cfg: dict, side: str = "bull") -> pd.Series:
    """有效紅 K（bull）或有效黑 K（bear）。"""
    c, o, h, lo, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
    ret = c / c.shift(1) - 1
    rng = (h - lo).replace(0, np.nan)
    vol_ma = v.rolling(20, min_periods=10).mean().shift(1)
    thr = np.maximum(cfg["min_ret"], cfg["atr_k"] * _atr_pct(d).shift(1))
    big_vol = v >= cfg["vol_k"] * vol_ma
    if side == "bull":
        m = (c > o) & (ret >= thr) & ((c - o) / rng >= cfg["body_min"]) & big_vol
    else:
        m = (c < o) & (ret <= -thr) & ((o - c) / rng >= cfg["body_min"]) & big_vol
    return m.fillna(False).astype(bool)


def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = (
        d[["Open", "High", "Low", "Close", "Volume"]]
        .resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    w["MA40"] = w["Close"].rolling(40).mean()
    w["trend"] = w["MA40"] > w["MA40"].shift(4)
    return w


# ---------------------------------------------------------------------------
# 事件追蹤（有效 K → 等待 → 確認／失效／過期）
# ---------------------------------------------------------------------------
def track_events(d: pd.DataFrame, mask: pd.Series, cfg: dict) -> list[dict]:
    c, o, h, lo = (d[k].to_numpy(float) for k in ["Close", "Open", "High", "Low"])
    v = d["Volume"].to_numpy(float)
    vol_ma = d["Volume"].rolling(20, min_periods=10).mean().shift(1).to_numpy(float)
    idx = d.index
    m = mask.reindex(d.index).fillna(False).to_numpy(bool)
    events: list[dict] = []
    cur: dict | None = None
    cool_until = -1

    def start(j: int) -> dict:
        ev = {
            "idx": j,
            "date": idx[j],
            "high": float(h[j]),
            "low": float(lo[j]),
            "mid": float((o[j] + c[j]) / 2),
            "close": float(c[j]),
            "ret": float(c[j] / c[j - 1] - 1) if j > 0 else np.nan,
            "vol_ratio": float(v[j] / vol_ma[j]) if vol_ma[j] else np.nan,
            "status": "waiting",
            "how": "",
            "touched": False,
            "confirm_idx": None,
            "end_idx": None,
        }
        events.append(ev)
        return ev

    for j in range(len(c)):
        if cur is not None and j > cur["idx"]:
            if c[j] < cur["low"]:
                cur.update(status="failed", end_idx=j)
                cur = None
            elif c[j] > cur["high"] and c[j] > o[j]:
                cur.update(status="confirmed", how="突破K高", confirm_idx=j, end_idx=j)
                cur, cool_until = None, j + cfg["cooldown"]
            elif lo[j] <= cur["mid"]:
                cur["touched"] = True
            elif cur["touched"] and c[j] > o[j] and c[j] > h[j - 1]:
                cur.update(status="confirmed", how="回測1/2不破", confirm_idx=j, end_idx=j)
                cur, cool_until = None, j + cfg["cooldown"]
            if cur is not None and j - cur["idx"] >= cfg["wait"]:
                cur.update(status="expired", end_idx=j)
                cur = None
        if m[j] and j > cool_until:
            if cur is not None:  # 等待中又出現新的有效 K：以新的為準
                cur.update(status="replaced", end_idx=j)
            cur = start(j)

    # 起漲確認之後：有沒有收盤跌破 K 低（停損）
    for ev in events:
        if ev["status"] != "confirmed":
            continue
        ci = ev["confirm_idx"]
        ev["entry"] = float(c[ci])
        ev["confirm_date"] = idx[ci]
        after = np.flatnonzero(c[ci + 1 :] < ev["low"])
        ev["stop_idx"] = int(ci + 1 + after[0]) if len(after) else None
    return events


# ---------------------------------------------------------------------------
# 回測
# ---------------------------------------------------------------------------
def backtest_events(
    d: pd.DataFrame, events: list[dict], horizons: dict[int, str], valid: pd.Series, stop_window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳（各持有期間統計、分時段統計、訊號明細）。"""
    c = d["Close"]
    lo = d["Low"].to_numpy(float)
    vmask = valid.reindex(d.index).fillna(False).to_numpy(bool)
    # 只統計回測期間內（valid）的訊號，和「隨便哪天買」用同一段期間比較
    conf = [e for e in events if e["status"] == "confirmed" and vmask[e["confirm_idx"]]]
    ent = np.array([e["confirm_idx"] for e in conf], dtype=int)
    fwd = {hz: (c.shift(-hz) / c - 1).to_numpy() for hz in horizons}

    def one(sel_ent: np.ndarray, sel_valid: np.ndarray) -> list[dict]:
        rows = []
        for hz, name in horizons.items():
            f = fwd[hz]
            s = f[sel_ent] if len(sel_ent) else np.array([])
            s = s[~np.isnan(s)]
            b = f[sel_valid & ~np.isnan(f)]
            rows.append(
                {
                    "持有期間": name,
                    "訊號次數": len(s),
                    "訊號勝率": (s > 0).mean() * 100 if len(s) else np.nan,
                    "訊號平均報酬": s.mean() * 100 if len(s) else np.nan,
                    "隨便哪天買的勝率": (b > 0).mean() * 100 if len(b) else np.nan,
                    "隨便哪天買的平均報酬": b.mean() * 100 if len(b) else np.nan,
                }
            )
        return rows

    stats = pd.DataFrame(one(ent, vmask))

    period_rows = []
    main_h = list(horizons)[1] if len(horizons) > 1 else list(horizons)[0]
    for pname, s0, s1 in PERIODS:
        in_p = (d.index >= s0) & (d.index <= s1)
        sel = np.array([i for i in ent if in_p[i]], dtype=int)
        r = [x for x in one(sel, vmask & in_p) if x["持有期間"] == horizons[main_h]][0]
        period_rows.append({"期間": pname, **r})
    by_period = pd.DataFrame(period_rows)

    rows = []
    for e in conf:
        ci = e["confirm_idx"]
        hit = None
        if ci + stop_window < len(lo):
            hit = bool(lo[ci + 1 : ci + 1 + stop_window].min() < e["low"])
        rows.append(
            {
                "有效K日期": e["date"],
                "確認日期": e["confirm_date"],
                "確認方式": e["how"],
                "進場價": e["entry"],
                "K低停損": e["low"],
                "停損距離": (e["low"] / e["entry"] - 1) * 100,
                **{f"{n}後報酬": fwd[hz][ci] * 100 for hz, n in horizons.items()},
                "期間內跌破K低": hit,
            }
        )
    sigs = pd.DataFrame(rows)
    return stats, by_period, sigs


# ---------------------------------------------------------------------------
# 波浪情境標註（只用來解說，不加分）
# ---------------------------------------------------------------------------
def wave_context(ev: dict, waves: dict | None) -> str:
    if not waves or not waves.get("pivots"):
        return ""
    ev_date, ev_close = ev["date"], ev["close"]
    pivots = waves["pivots"]
    prior = [p for p in pivots if p["date"] <= ev_date and p.get("label")]
    if not prior:
        return ""
    last = prior[-1]
    lab = last["label"]
    div = waves.get("divergence")

    def prev_high(label: str):
        hs = [p for p in prior if p.get("label") == label]
        return hs[-1]["price"] if hs else None

    if lab in ("②", "④"):
        nxt, ref = ("③", "①") if lab == "②" else ("⑤", "③")
        hp = prev_high(ref)
        if hp and ev_close > hp:
            return f"🚀 出現在第 {lab} 波回檔之後，且收盤已突破 {ref} 波高點 {hp:,.2f}：屬於**第 {nxt} 波推進**的攻擊K"
        tail = f"，若站上 {ref} 波高點 {hp:,.2f} 可視為**第 {nxt} 波起漲確認**" if hp else ""
        return f"🚀 出現在第 {lab} 波回檔低點之後{tail}"
    if lab in ("ⓒ", "[0] 起漲"):
        return "🌱 出現在修正浪 ⓒ／起漲低點之後：可能是**新一輪第 ① 波**，但尚未確立，部位宜小"
    if lab == "ⓐ":
        return "⚠️ 出現在 ⓐ 波下跌之後：也可能只是 ⓑ 波反彈，需站穩前高才算轉強"
    if lab == "⑤" or (div and div.get("type") == "bearish" and lab in ("③", "⑤")):
        return "⚠️ 出現在第 ⑤ 波／高檔背離附近：可能是末升段長紅，**不宜追高**"
    if lab in ("①", "③", "ⓑ"):
        return f"📈 出現在 {lab} 高點之後的整理段：突破 {lab} 高點 {last['price']:,.2f} 代表波浪續攻"
    return ""


# ---------------------------------------------------------------------------
# 目前狀態描述
# ---------------------------------------------------------------------------
def _state(events: list[dict], d: pd.DataFrame, cfg: dict, fresh: int, recent: int, unit: str) -> dict:
    """依最後一個事件判斷目前狀態。fresh = 確認後幾根內算「新鮮買點」；recent = 多久以前的事件就不顯示。"""
    n = len(d)
    close = float(d["Close"].iloc[-1])
    evs = [e for e in events if e["status"] != "replaced"]
    if not evs:
        return {"code": "none", "title": "➖ 近期沒有有效K棒", "desc": "", "event": None}
    ev = evs[-1]
    age = n - 1 - ev["idx"]
    base = {"event": ev, "high": ev["high"], "mid": ev["mid"], "low": ev["low"], "date": ev["date"]}
    lv = f"K高 {ev['high']:,.2f}／實體1/2 {ev['mid']:,.2f}／K低 {ev['low']:,.2f}"

    if ev["status"] == "waiting":
        left = cfg["wait"] - age
        return {
            **base,
            "code": "waiting",
            "title": "⏳ 有效K已出現，等待確認",
            "desc": (
                f"{ev['date']:%Y/%m/%d} 出現有效紅K（{lv}）。收盤站上 **{ev['high']:,.2f}**，"
                f"或拉回 **{ev['mid']:,.2f}** 附近守住再轉強，即確認起漲；收盤跌破 **{ev['low']:,.2f}** 則失效"
                f"（還剩約 {left} {unit}）。"
            ),
        }
    if ev["status"] == "confirmed":
        since = n - 1 - ev["confirm_idx"]
        stop_idx = ev.get("stop_idx")
        if stop_idx is not None:
            if n - 1 - stop_idx > recent:
                return {"code": "none", "title": "➖ 近期沒有有效K棒", "desc": "", "event": None}
            return {
                **base,
                "code": "stopped",
                "title": "✕ 起漲後跌破K低（停損出場）",
                "desc": f"{ev['confirm_date']:%Y/%m/%d} 以 {ev['how']} 確認起漲，之後收盤跌破 K低 {ev['low']:,.2f}，訊號結束。",
            }
        gain = (close / ev["entry"] - 1) * 100
        if since <= fresh:
            return {
                **base,
                "code": "confirmed",
                "title": f"✅ 起漲確認（{ev['how']}）",
                "desc": (
                    f"{ev['confirm_date']:%Y/%m/%d} 確認起漲（確認價 {ev['entry']:,.2f}，目前 {gain:+.1f}%）。"
                    f"停損參考 K低 **{ev['low']:,.2f}**（距現價 {(ev['low'] / close - 1) * 100:+.1f}%）。"
                ),
            }
        if since <= recent:
            return {
                **base,
                "code": "running",
                "title": f"📈 起漲後續行中（已 {since} {unit}）",
                "desc": (
                    f"{ev['confirm_date']:%Y/%m/%d} 確認起漲，距確認價 {gain:+.1f}%。不建議追高，"
                    f"已持有者可用 K低 {ev['low']:,.2f} 當最後防線，等下一根有效K再加碼。"
                ),
            }
        return {"code": "none", "title": "➖ 近期沒有新的有效K棒", "desc": "", "event": None}
    if age > recent:
        return {"code": "none", "title": "➖ 近期沒有有效K棒", "desc": "", "event": None}
    if ev["status"] == "failed":
        return {
            **base,
            "code": "failed",
            "title": "✕ 有效K失效",
            "desc": f"{ev['date']:%Y/%m/%d} 的有效紅K（{lv}）之後收盤跌破 K低，假突破，等下一根。",
        }
    return {
        **base,
        "code": "expired",
        "title": "⌛ 有效K過期（未確認）",
        "desc": f"{ev['date']:%Y/%m/%d} 的有效紅K（{lv}）在 {cfg['wait']} {unit}內沒有確認起漲。",
    }


def analyze_keybars(d: pd.DataFrame, waves: dict | None = None, with_bt: bool = True) -> dict:
    """三週期有效 K 分析。d 需含 add_indicators 的欄位（LongMA_slope、LongMA）。with_bt=False 時略過回測（掃描用）。"""
    valid = d["LongMA"].notna()
    bull = effective_bars(d, DAILY)
    trend = d["LongMA_slope"] > 0

    # 短線：日線、不限趨勢，只給價位參考
    ev_short = track_events(d, bull, DAILY)
    short = _state(ev_short, d, DAILY, fresh=5, recent=20, unit="天")

    # 中線：日線 + 年線向上
    ev_mid = track_events(d, bull & trend & valid, DAILY)
    mid = _state(ev_mid, d, DAILY, fresh=5, recent=30, unit="天")
    bt_mid = (
        backtest_events(d, ev_mid, DAILY_H, valid & (d.index >= "2016-01-01"), stop_window=20) if with_bt else None
    )

    # 長線：週線 + 40 週線向上
    w = to_weekly(d)
    wbull = effective_bars(w, WEEKLY)
    ev_long = track_events(w, wbull & w["trend"].fillna(False), WEEKLY)
    long_ = _state(ev_long, w, WEEKLY, fresh=2, recent=13, unit="週")
    bt_long = (
        backtest_events(w, ev_long, WEEKLY_H, w["MA40"].notna() & (w.index >= "2016-01-01"), stop_window=4)
        if with_bt
        else None
    )
    partial = d.index[-1] < w.index[-1]

    for st_, evs in ((short, ev_short), (mid, ev_mid), (long_, ev_long)):
        st_["wave"] = wave_context(st_["event"], waves) if st_.get("event") else ""

    # 有效黑 K（近 10 天）
    bear = effective_bars(d, DAILY, side="bear")
    bear_dates = list(d.index[bear])
    recent_bear = [x for x in bear_dates if x >= d.index[max(0, len(d) - 10)]]
    black = None
    if recent_bear:
        bd = recent_bear[-1]
        r = d.loc[bd]
        below_ma20 = bool(r["Close"] < r["MA20"]) if pd.notna(r.get("MA20")) else False
        black = {
            "date": bd,
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "ret": float(d["Close"].pct_change().loc[bd] * 100),
            "below_ma20": below_ma20,
        }

    return {
        "short": short,
        "mid": mid,
        "long": long_,
        "weekly": w,
        "weekly_partial": partial,
        "events_short": ev_short,
        "events_mid": ev_mid,
        "events_long": ev_long,
        "bt_mid": bt_mid,
        "bt_long": bt_long,
        "black": black,
        "black_dates": bear_dates,
    }
