"""波浪理論（Elliott Wave）、費波那契黃金分割率（Fibonacci）與歷史相似 K 線比對。

功能：
1. ATR 動態 ZigZag 波段轉折偵測（自動過濾雜訊，抓出波段高低點）
2. 客觀波浪標示（[0] 起漲 → ① → ② → ③ → ④ → ⑤ 與 ⓐ → ⓑ → ⓒ 修正浪）
3. 費波那契黃金分割回撤位（23.6%、38.2%、50%、61.8%、78.6%）與延伸目標價（127.2%、161.8%）
4. 第 5 波高檔頂背離 / C 波末端底背離偵測
5. 歷史最相似 10 次 K 線走勢比對（Pattern Matching）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIB_RETRACE = [
    (0.236, "23.6%（極強勢淺回檔）"),
    (0.382, "38.2%（強勢黃金分割・常見第4波）"),
    (0.500, "50.0%（中關支撐）"),
    (0.618, "61.8%（黃金分割防線・常見第2波）"),
    (0.786, "78.6%（起漲最後防線）"),
]
FIB_EXTEND = [
    (1.272, "127.2%（第一波段目標）"),
    (1.618, "161.8%（黃金延伸目標）"),
]


# ---------------------------------------------------------------------------
# 1. ZigZag 波段高低點偵測
# ---------------------------------------------------------------------------
def detect_pivots(d: pd.DataFrame, lookback: int = 250) -> list[dict]:
    """用 ATR 動態門檻找出最近 lookback 天內的波段高低點（嚴格高低交替）。"""
    w = d.iloc[-lookback:].copy()
    if len(w) < 20:
        return []

    highs = w["High"].to_numpy(dtype=float)
    lows = w["Low"].to_numpy(dtype=float)
    closes = w["Close"].to_numpy(dtype=float)
    rsi = w["RSI"].to_numpy(dtype=float) if "RSI" in w else np.full(len(w), np.nan)
    dates = w.index

    # 依據該股票自身的波動度決定轉折門檻（約 2.2 倍日 ATR，介於 4.5%～12%）
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr_pct = float(np.nanmedian(tr / closes[1:])) if len(tr) else 0.02
    dev = float(np.clip(atr_pct * 2.4, 0.045, 0.12))

    pivots: list[dict] = []
    # 初始化：先從前 15 天找起點
    init_n = min(15, len(w))
    hi_idx = int(np.argmax(highs[:init_n]))
    lo_idx = int(np.argmin(lows[:init_n]))
    if lo_idx <= hi_idx:
        state = "up"
        pivots.append({"idx": lo_idx, "date": dates[lo_idx], "price": lows[lo_idx], "kind": "L", "rsi": rsi[lo_idx]})
        cur_idx, cur_val = hi_idx, highs[hi_idx]
    else:
        state = "down"
        pivots.append({"idx": hi_idx, "date": dates[hi_idx], "price": highs[hi_idx], "kind": "H", "rsi": rsi[hi_idx]})
        cur_idx, cur_val = lo_idx, lows[lo_idx]

    for i in range(max(hi_idx, lo_idx) + 1, len(w)):
        if state == "up":
            if highs[i] >= cur_val:
                cur_idx, cur_val = i, highs[i]
            elif lows[i] <= cur_val * (1 - dev) and (i - cur_idx) >= 3:
                pivots.append({"idx": cur_idx, "date": dates[cur_idx], "price": cur_val, "kind": "H", "rsi": rsi[cur_idx]})
                state = "down"
                cur_idx, cur_val = i, lows[i]
        else:
            if lows[i] <= cur_val:
                cur_idx, cur_val = i, lows[i]
            elif highs[i] >= cur_val * (1 + dev) and (i - cur_idx) >= 3:
                pivots.append({"idx": cur_idx, "date": dates[cur_idx], "price": cur_val, "kind": "L", "rsi": rsi[cur_idx]})
                state = "up"
                cur_idx, cur_val = i, highs[i]

    # 加入最後一個已成型但尚未反轉的極端點（若與最後一筆距離夠遠）
    if cur_idx > pivots[-1]["idx"] + 2:
        kind = "H" if state == "up" else "L"
        pivots.append({"idx": cur_idx, "date": dates[cur_idx], "price": cur_val, "kind": kind, "rsi": rsi[cur_idx]})

    return pivots


# ---------------------------------------------------------------------------
# 2. 波浪標示與黃金分割分析
# ---------------------------------------------------------------------------
def analyze_waves(d: pd.DataFrame, lookback: int = 250) -> dict:
    """分析目前所處的艾略特波浪位置、黃金分割支撐/目標位，以及頂底背離。"""
    pivots = detect_pivots(d, lookback=lookback)
    close = float(d["Close"].iloc[-1])
    last_date = d.index[-1]
    last_rsi = float(d["RSI"].iloc[-1]) if "RSI" in d and pd.notna(d["RSI"].iloc[-1]) else np.nan

    if len(pivots) < 2:
        return {
            "pivots": pivots,
            "wave_stage": "資料不足",
            "wave_quality": "unknown",
            "path_str": "歷史轉折點不足，無法標示波浪",
            "fib_str": "資料不足，無法計算黃金分割",
            "fib": None,
            "divergence": None,
        }

    # 找出最近一波主要循環：先定位主要波段最高點 peak_k，再往左找起漲最低點 [0]
    high_pivots = [k for k, p in enumerate(pivots) if p["kind"] == "H"]
    low_pivots = [k for k, p in enumerate(pivots) if p["kind"] == "L"]
    peak_k = max(high_pivots, key=lambda k: pivots[k]["price"]) if high_pivots else len(pivots) - 1

    lows_before_peak = [k for k in low_pivots if k < peak_k]
    if lows_before_peak:
        # 若距離 peak_k 超過 7 個轉折，從 peak_k 往前最多 7 個轉折內找起漲低點，確保聚焦於近半年至一年的主波段
        cand_lows = [k for k in lows_before_peak if peak_k - k <= 7] or lows_before_peak
        anchor_k = min(cand_lows, key=lambda k: pivots[k]["price"])
    else:
        anchor_k = low_pivots[0] if low_pivots else 0

    for p in pivots:
        p["label"] = ""

    pivots[anchor_k]["label"] = "[0] 起漲"
    origin_price = pivots[anchor_k]["price"]
    wave_num = 0
    in_correction = False
    corr_step = 0
    peak_w1 = peak_w3 = peak_w5 = None
    low_w2 = low_w4 = None

    for k in range(anchor_k + 1, len(pivots)):
        p = pivots[k]
        if not in_correction:
            if p["kind"] == "H":
                if wave_num == 0:
                    wave_num = 1
                    p["label"] = "①"
                    peak_w1 = p
                elif wave_num == 2:
                    if p["price"] > (peak_w1["price"] * 0.995 if peak_w1 else 0):
                        wave_num = 3
                        p["label"] = "③"
                        peak_w3 = p
                    else:
                        in_correction = True
                        corr_step = 2
                        p["label"] = "ⓑ"
                elif wave_num == 4:
                    if p["price"] > (peak_w3["price"] * 0.995 if peak_w3 else 0):
                        wave_num = 5
                        p["label"] = "⑤"
                        peak_w5 = p
                    else:
                        in_correction = True
                        corr_step = 2
                        p["label"] = "ⓑ"
            else:  # kind == "L"
                if wave_num == 1:
                    if p["price"] >= origin_price * 0.98 or k > peak_k:
                        if p["price"] >= origin_price * 0.98:
                            wave_num = 2
                            p["label"] = "②"
                            low_w2 = p
                        else:
                            in_correction = True
                            corr_step = 1
                            p["label"] = "ⓐ"
                    else:
                        # 在主升段展開前跌破起漲點，重設起漲點
                        p["label"] = "[0] 起漲"
                        origin_price = p["price"]
                        wave_num = 0
                elif wave_num == 3:
                    ref_floor = low_w2["price"] * 0.96 if low_w2 else origin_price
                    if p["price"] >= ref_floor and k < peak_k:
                        wave_num = 4
                        p["label"] = "④"
                        low_w4 = p
                    elif p["price"] >= origin_price + 0.35 * (peak_w3["price"] - origin_price):
                        wave_num = 4
                        p["label"] = "④"
                        low_w4 = p
                    else:
                        in_correction = True
                        corr_step = 1
                        p["label"] = "ⓐ"
                elif wave_num == 5:
                    in_correction = True
                    corr_step = 1
                    p["label"] = "ⓐ"
        else:
            if p["kind"] == "L" and corr_step in (0, 2):
                corr_step = 3 if corr_step == 2 else 1
                p["label"] = "ⓒ" if corr_step == 3 else "ⓐ"
            elif p["kind"] == "H" and corr_step == 1:
                corr_step = 2
                p["label"] = "ⓑ"
            elif p["kind"] == "H" and corr_step == 3:
                # 修正浪結束後再創高
                wave_num = 1
                in_correction = False
                p["label"] = "①"
                peak_w1 = p

    # 決定用來計算黃金分割的「主參考上漲波段 (L0 -> H1)」
    high_indices = [k for k, p in enumerate(pivots) if p["kind"] == "H"]
    low_indices = [k for k, p in enumerate(pivots) if p["kind"] == "L"]
    fib = None
    if high_indices and low_indices:
        # 取主波段最高點 peak_k 作為 H1
        h_k = peak_k
        h_piv = pivots[h_k]
        prior_lows = [k for k in low_indices if k < h_k]
        if prior_lows:
            last_l_k = prior_lows[-1]
            # 若目前收盤價仍在最近一次子波段低點之上，且該子波段漲幅 >= 15%，用子波段低點；
            # 若股價已跌破最近子波段低點（代表正在修正整個大波段），則改用主循環起漲點 [0] 或近一年最低點
            if close >= pivots[last_l_k]["price"] and (h_piv["price"] / pivots[last_l_k]["price"] - 1 >= 0.15):
                l_piv = pivots[last_l_k]
            elif close >= pivots[anchor_k]["price"] * 0.95:
                l_piv = pivots[anchor_k]
            else:
                l_piv = min((pivots[k] for k in prior_lows), key=lambda x: x["price"])
        else:
            l_piv = pivots[low_indices[0]]

        lo_p, hi_p = float(l_piv["price"]), float(h_piv["price"])
        span = hi_p - lo_p
        if span > 0:
            retrace_pct = (hi_p - close) / span * 100
            retrace_levels = [
                {"ratio": r, "name": name, "price": round(hi_p - r * span, 2)} for r, name in FIB_RETRACE
            ]
            extend_levels = [
                {"ratio": r, "name": name, "price": round(lo_p + r * span, 2)} for r, name in FIB_EXTEND
            ]
            fib = {
                "low_date": f"{l_piv['date']:%Y-%m-%d}",
                "low_price": round(lo_p, 2),
                "low_label": l_piv.get("label") or "波段低點",
                "high_date": f"{h_piv['date']:%Y-%m-%d}",
                "high_price": round(hi_p, 2),
                "high_label": h_piv.get("label") or "波段高點",
                "gain_pct": round((hi_p / lo_p - 1) * 100, 1),
                "retrace_pct": round(retrace_pct, 1),
                "retrace_levels": retrace_levels,
                "extend_levels": extend_levels,
                "zone_382": round(hi_p - 0.382 * span, 2),
                "zone_500": round(hi_p - 0.500 * span, 2),
                "zone_618": round(hi_p - 0.618 * span, 2),
                "zone_786": round(hi_p - 0.786 * span, 2),
                "target_1272": round(lo_p + 1.272 * span, 2),
                "target_1618": round(lo_p + 1.618 * span, 2),
            }

    # 偵測頂背離（第 5 波警訊）與底背離（C 波 / 第 2、4 波落底訊號）
    divergence = None
    h_pivs = [p for p in pivots if p["kind"] == "H" and pd.notna(p.get("rsi"))]
    l_pivs = [p for p in pivots if p["kind"] == "L" and pd.notna(p.get("rsi"))]
    if len(h_pivs) >= 2:
        h1, h2 = h_pivs[-2], h_pivs[-1]
        if h2["price"] > h1["price"] * 1.01 and h2["rsi"] < h1["rsi"] - 3.5 and close >= h2["price"] * 0.92:
            divergence = {
                "type": "bearish",
                "title": "⚠️ 高檔頂背離（第 5 波末升段警訊）",
                "desc": f"波段高點從 {h1['price']:,.1f} 創高到 {h2['price']:,.1f}，但 RSI 從 {h1['rsi']:.0f} 降到 {h2['rsi']:.0f}，上攻動能減弱",
            }
    if divergence is None and len(l_pivs) >= 1 and pd.notna(last_rsi):
        prev_l = l_pivs[-1]
        if close <= prev_l["price"] * 1.03 and last_rsi > prev_l["rsi"] + 3.0 and last_rsi < 48:
            divergence = {
                "type": "bullish",
                "title": "✨ 低檔底背離（下殺力道衰竭）",
                "desc": f"股價回測低點（{close:,.1f} vs 前低 {prev_l['price']:,.1f}），但 RSI 未破底（{last_rsi:.0f} > {prev_l['rsi']:.0f}）",
            }

    # 根據最後一個轉折與目前股價位置，綜合判定目前的「波浪階段」
    last_p = pivots[-1]
    ret = fib["retrace_pct"] if fib else 0.0
    if fib and ret > 85:
        wave_stage = "📉 A-B-C 修正浪（跌破起漲防線）"
        wave_quality = "weak"
    elif in_correction:
        if corr_step >= 2 and close < last_p["price"]:
            wave_stage = f"📉 ⓒ 波下殺修正中（已回吐前波漲幅 {ret:.0f}%）"
        else:
            wave_stage = f"📉 A-B-C 修正整理中（已回吐前波漲幅 {ret:.0f}%）"
        wave_quality = "Wait"
    elif last_p["kind"] == "H":
        # 從波段高點拉回中
        next_w = 2 if wave_num == 1 else 4 if wave_num == 3 else None
        w_name = f"第 {next_w} 波" if next_w else "波段"
        if 30 <= ret <= 66:
            wave_stage = f"🌊 {w_name}黃金分割回檔中（已回吐 {ret:.0f}%，位於 38.2%～61.8% 甜蜜區）"
            wave_quality = "sweet"
        elif 15 <= ret < 30:
            wave_stage = f"🌊 {w_name}淺幅強勢回檔（已回吐 {ret:.0f}%，接近 23.6%～38.2%）"
            wave_quality = "shallow"
        elif 66 < ret <= 85:
            wave_stage = f"🌊 {w_name}深幅回檔（已回吐 {ret:.0f}%，測試 61.8%～78.6% 最後防線）"
            wave_quality = "deep"
        else:
            wave_stage = f"🌊 剛自 {last_p['label'] or '波段'} 高點小幅拉回（回吐 {ret:.0f}%）"
            wave_quality = "high"
    else:
        # 從波段低點往上推進中
        cur_w = 3 if wave_num == 2 else 5 if wave_num == 4 else 1
        if 25 <= ret <= 68:
            wave_stage = f"🌊 第 {wave_num} 波黃金分割止跌、啟動第 {cur_w} 波上攻（回檔 {ret:.0f}% 後轉強）"
            wave_quality = "sweet_turn"
        elif cur_w == 5 and divergence and divergence["type"] == "bearish":
            wave_stage = "⚠️ 第 5 波末升段且出現頂背離（切勿追高）"
            wave_quality = "top_risk"
        else:
            wave_stage = f"🌊 第 {cur_w} 波多頭推進中"
            wave_quality = "impulse"

    # 產生給使用者與 AI 分析師閱讀的波浪路徑摘要
    labeled = [p for p in pivots[anchor_k:] if p.get("label")]
    path_str = " → ".join(f"{p['label']} {p['price']:,.2f}（{p['date']:%m/%d}）" for p in labeled[-6:])
    if fib:
        fib_str = (
            f"參考波段：{fib['low_label']} {fib['low_price']:,.2f}（{fib['low_date']}）"
            f" 漲至 {fib['high_label']} {fib['high_price']:,.2f}（{fib['high_date']}，漲幅 +{fib['gain_pct']:.1f}%），"
            f"目前收盤 {close:,.2f}，相當於回吐這波漲幅的 **{fib['retrace_pct']:.1f}%**。"
            f"黃金分割支撐：38.2% 在 {fib['zone_382']:,.2f}、50% 在 {fib['zone_500']:,.2f}、61.8% 在 {fib['zone_618']:,.2f}；"
            f"若守穩再創高，波段延伸目標價：127.2% 在 {fib['target_1272']:,.2f}、161.8% 在 {fib['target_1618']:,.2f}。"
        )
    else:
        fib_str = "目前無明確單一上漲波段可供計算黃金分割。"

    return {
        "pivots": pivots,
        "wave_stage": wave_stage,
        "wave_quality": wave_quality,
        "path_str": path_str,
        "fib_str": fib_str,
        "fib": fib,
        "divergence": divergence,
    }


# ---------------------------------------------------------------------------
# 3. 歷史最相似 10 次走勢比對（Historical Pattern Matching）
# ---------------------------------------------------------------------------
def match_historical_patterns(d: pd.DataFrame, window: int = 20, top_k: int = 10) -> dict:
    """在該股票過去 15 年歷史中，找出與最近 20 日「K 線走勢型態 + RSI + 季線乖離」最相似的前 10 次。"""
    if len(d) < 320:
        return {"matches": pd.DataFrame(), "summary": None, "paths": None}

    closes = d["Close"].to_numpy(dtype=float)
    rsi = d["RSI"].to_numpy(dtype=float)
    bias60 = d["Bias60"].to_numpy(dtype=float)
    n = len(d)

    # 目前最新 20 日的正規化走勢曲線（以第 1 天為基準的累積漲跌 %）
    cur_win = closes[n - window : n]
    cur_curve = (cur_win / cur_win[0] - 1) * 100
    cur_std = np.std(cur_curve) or 1.0
    cur_norm = (cur_curve - np.mean(cur_curve)) / cur_std
    cur_ret20 = cur_curve[-1]
    cur_rsi = rsi[-1]
    cur_bias = bias60[-1] * 100

    candidates: list[tuple[float, int]] = []
    # 只搜尋距離今天至少 65 個交易日以前的歷史，確保都有完整的未來 60 天走勢
    for end_i in range(max(240, window), n - 65):
        if np.isnan(rsi[end_i]) or np.isnan(bias60[end_i]):
            continue
        hist_win = closes[end_i - window + 1 : end_i + 1]
        hist_curve = (hist_win / hist_win[0] - 1) * 100
        hist_std = np.std(hist_curve) or 1.0
        hist_norm = (hist_curve - np.mean(hist_curve)) / hist_std

        # 綜合距離：曲線形狀相似度 + 20日總漲跌幅差距 + RSI 差距 + 季線乖離差距
        shape_dist = float(np.sqrt(np.mean((cur_norm - hist_norm) ** 2)))
        ret_dist = abs(cur_ret20 - hist_curve[-1]) / 8.0
        rsi_dist = abs(cur_rsi - rsi[end_i]) / 15.0
        bias_dist = abs(cur_bias - bias60[end_i] * 100) / 10.0
        dist = shape_dist * 0.45 + ret_dist * 0.25 + rsi_dist * 0.15 + bias_dist * 0.15
        candidates.append((dist, end_i))

    candidates.sort(key=lambda x: x[0])
    chosen: list[tuple[float, int]] = []
    for dist, idx in candidates:
        if all(abs(idx - prev_idx) >= 20 for _, prev_idx in chosen):
            chosen.append((dist, idx))
            if len(chosen) >= top_k:
                break

    if not chosen:
        return {"matches": pd.DataFrame(), "summary": None, "paths": None}

    rows = []
    fwd_paths = []
    for rank, (dist, idx) in enumerate(chosen, 1):
        base_p = closes[idx]
        p20 = (closes[idx + 20] / base_p - 1) * 100
        p60 = (closes[idx + 60] / base_p - 1) * 100
        min20 = (np.min(closes[idx + 1 : idx + 21]) / base_p - 1) * 100
        max60 = (np.max(closes[idx + 1 : idx + 61]) / base_p - 1) * 100
        sim_score = max(0.0, min(99.0, 100.0 * np.exp(-dist * 0.55)))
        rows.append(
            {
                "排名": rank,
                "歷史日期": d.index[idx],
                "相似度": sim_score,
                "當時RSI": rsi[idx],
                "1個月後報酬": p20,
                "3個月後報酬": p60,
                "1個月內最深再跌": min20,
                "3個月內最高漲幅": max60,
            }
        )
        fwd_paths.append((closes[idx : idx + 61] / base_p - 1) * 100)

    df_m = pd.DataFrame(rows)
    mat = np.vstack(fwd_paths)
    summary = {
        "count": len(df_m),
        "win_20d": float((df_m["1個月後報酬"] > 0).mean() * 100),
        "avg_20d": float(df_m["1個月後報酬"].mean()),
        "win_60d": float((df_m["3個月後報酬"] > 0).mean() * 100),
        "avg_60d": float(df_m["3個月後報酬"].mean()),
        "worst_20d_mae": float(df_m["1個月內最深再跌"].median()),
        "best_60d_mfe": float(df_m["3個月內最高漲幅"].median()),
    }
    paths = pd.DataFrame(
        {
            "day": np.arange(0, 61),
            "平均路徑": np.mean(mat, axis=0),
            "中位數路徑": np.median(mat, axis=0),
            "樂觀前25%": np.percentile(mat, 75, axis=0),
            "保守後25%": np.percentile(mat, 25, axis=0),
        }
    ).set_index("day")
    return {"matches": df_m, "summary": summary, "paths": paths}
