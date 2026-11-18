"""中長期智慧波段模型（v3）：結合趨勢、波浪黃金分割回撤、先超賣後止跌序列、ATR 動態風控與回測。

四大評分面向（各 25 分，滿分 100）：
1. 📈 長期趨勢與波浪結構（25 分）：年線斜率向上、股價守穩年線、季線多頭排列
2. 🌊 波浪黃金分割回檔位置（25 分）：回吐前波漲幅的 30%～65%（涵蓋 38.2%、50%、61.8% 黃金分割區）且未跌破起漲防線
3. 🌡️ 超賣蓄能（近 10 日序列記憶，25 分）：近 10 日內曾進入 RSI 超賣或負乖離區（解決「止跌當天 RSI 已反彈」的計分矛盾）
4. 🛑 止跌轉強訊號（25 分）：站上短均線（10MA/20MA）、MACD 動能翻揚、RSI 自低檔勾頭向上、紅 K 實體或長下影線止穩
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_REVERSAL = 10  # 止跌訊號至少要有 10 分才解鎖買進訊號
NO_REVERSAL_CAP = 60  # 未止跌時的總分上限

LABELS = [
    ("buy", "✅ 符合買進條件", "波浪回檔到位且出現止跌轉強訊號，可考慮分批進場"),
    ("light", "🟡 可小量分批", "條件接近，可小量試單或等待更明確轉強"),
    ("neutral", "⏸️ 中性觀望", "目前沒有明顯優勢，建議持續觀察"),
    ("avoid", "⛔ 暫不建議進場", "趨勢偏弱、浪型破壞或高檔過熱，風險較高"),
]
WAIT_LABEL = ("wait", "👀 列入觀察", "已進入黃金分割回檔/超賣區，但尚未出現止跌訊號，等止跌再進場")


# ---------------------------------------------------------------------------
# 技術指標與波浪/ATR 特徵
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, lo = d["Close"], d["High"], d["Low"]

    for n in (5, 10, 20, 60, 120, 240):
        d[f"MA{n}"] = c.rolling(n).mean()
    # 長期均線：優先用年線（240 日），資料不足時改用半年線（120 日）
    d["LongMA"] = d["MA240"].fillna(d["MA120"])
    d["LongMA_slope"] = d["LongMA"] / d["LongMA"].shift(20) - 1

    # ATR（14 日平均真實波幅）與動態停損線
    prev_c = c.shift(1)
    tr = pd.concat([(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["ATR_pct"] = d["ATR14"] / c
    d["ATR_Stop"] = c - 2.0 * d["ATR14"]
    d["Chandelier_Exit"] = h.rolling(22, min_periods=5).max() - 2.5 * d["ATR14"]

    # RSI（14 日，Wilder 平滑）與近 10 日最低 RSI（序列記憶）
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)
    d.loc[(loss == 0) & gain.notna(), "RSI"] = 100.0
    d["RSI_min10"] = d["RSI"].rolling(10, min_periods=1).min()

    # MACD（12, 26, 9）
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # 高低點回檔與波浪費波那契回撤比（Fibonacci Retracement Ratio）
    d["High252"] = c.rolling(252, min_periods=60).max()
    d["Drawdown"] = c / d["High252"] - 1
    d["Bias60"] = c / d["MA60"] - 1
    d["Bias60_min10"] = d["Bias60"].rolling(10, min_periods=1).min()

    # 中期波段起漲低點（近 90 日低點）與波段高點（近 60 日高點）→ 計算前波漲幅回吐比例
    d["SwingLow90"] = lo.rolling(90, min_periods=30).min()
    d["SwingHigh60"] = h.rolling(60, min_periods=20).max()
    wave_span = (d["SwingHigh60"] - d["SwingLow90"]).replace(0, np.nan)
    d["WaveGain"] = d["SwingHigh60"] / d["SwingLow90"] - 1
    d["FibRetrace"] = ((d["SwingHigh60"] - c) / wave_span).clip(0, 1.5)

    # 年化波動度：近 20 日 vs 近一年
    ret = c.pct_change()
    d["Vol20"] = ret.rolling(20).std() * np.sqrt(252)
    d["Vol252"] = ret.rolling(252, min_periods=60).std() * np.sqrt(252)
    return d


# ---------------------------------------------------------------------------
# 評分（向量化，可同時算出每一天的分數，回測用，絕無前視偏誤）
# ---------------------------------------------------------------------------
def score_frame(d: pd.DataFrame) -> pd.DataFrame:
    c = d["Close"]
    long_ma, slope = d["LongMA"], d["LongMA_slope"]
    rsi, rsi_min10 = d["RSI"], d["RSI_min10"]
    bias, bias_min10 = d["Bias60"], d["Bias60_min10"]
    dd, fib_ret = d["Drawdown"], d["FibRetrace"]
    hist = d["MACD_hist"]

    # 1. 長期趨勢與波浪結構（25）：年線向上 + 股價在年線之上 + 季線在年線之上
    trend = (
        np.where(slope > 0, 10, 0)
        + np.where(c > long_ma, 10, np.where(c > long_ma * 0.95, 5, 0))
        + np.where(d["MA60"] > long_ma, 5, 0)
    )

    # 2. 波浪與黃金分割回檔位置（25）：
    #    結合「黃金分割回撤 30%～68%（涵蓋 38.2%、50%、61.8%）」與「距高點適度回檔」
    #    解決高波動股前高錨定一整年的問題，同時排除跌破 78.6% 浪型破壞的空頭走勢
    fib_score = np.select(
        [
            (fib_ret >= 0.32) & (fib_ret <= 0.68),  # 標準 38.2%～61.8% 黃金分割買區
            (fib_ret >= 0.22) & (fib_ret < 0.32),   # 強勢第 4 波淺回檔（23.6%～38.2%）
            (fib_ret > 0.68) & (fib_ret <= 0.79),   # 深回檔測試 78.6% 防線
        ],
        [15, 10, 8],
        default=3,
    )
    dd_score = np.select(
        [dd > -0.05, dd > -0.10, dd > -0.28, dd > -0.40],
        [2, 6, 10, 6],
        default=2,
    )
    pullback = np.clip(fib_score + dd_score, 0, 25)

    # 3. 超賣蓄能（25）：同時看當日與「近 10 日內最深超賣程度」（先超賣、後止跌不矛盾）
    eff_rsi = np.minimum(rsi, rsi_min10 + 4)
    eff_bias = np.minimum(bias, bias_min10 + 0.02)
    oversold = np.select(
        [eff_rsi < 32, eff_rsi < 40, eff_rsi < 48, eff_rsi < 56, eff_rsi < 65],
        [18, 14, 10, 6, 2],
        default=0,
    ) + np.select([eff_bias < -0.08, eff_bias < -0.04, eff_bias < 0], [7, 5, 3], default=0)

    # 4. 止跌轉強訊號（25）：站回月線或 10 日線 + MACD 動能翻揚 + 近 5 日轉漲且 RSI 勾頭向上
    ma_reclaim = np.where(c > d["MA20"], 10, np.where((c > d["MA10"]) & (c > d["Open"]), 6, 0))
    macd_turn = np.where(hist > hist.shift(3), 8, 0)
    price_turn = np.where((c > c.shift(5)) | ((rsi >= rsi_min10 + 5) & (c > d["MA5"])), 7, 0)
    reversal = ma_reclaim + macd_turn + price_turn

    # 扣分：短線過熱、第 5 波高檔頂背離、恐慌性波動放大
    overheat = np.where((rsi > 75) & (bias > 0.15), -10, 0)
    rsi_max30 = rsi.rolling(30).max()
    top_div = np.where((fib_ret < 0.12) & (rsi < rsi_max30 - 6) & (hist < hist.shift(3)), -5, 0)
    panic = np.where(d["Vol20"] > d["Vol252"] * 1.5, -5, 0)

    out = pd.DataFrame(
        {
            "trend": trend,
            "pullback": pullback,
            "oversold": oversold,
            "reversal": reversal,
            "penalty": overheat + top_div + panic,
        },
        index=d.index,
    ).astype(float)
    raw_total = out.sum(axis=1).clip(0, 100)
    # 止跌閘門：止跌訊號不足時，總分最高只給 NO_REVERSAL_CAP，避免「接刀」
    out["score"] = raw_total.where(out["reversal"] >= MIN_REVERSAL, raw_total.clip(upper=NO_REVERSAL_CAP))
    out["capped"] = (raw_total > out["score"]).astype(float)

    valid = long_ma.notna() & rsi.notna() & dd.notna() & d["MA60"].notna()
    out.loc[~valid, :] = np.nan
    return out


def label_for(score: float, threshold: float, reversal: float | None = None) -> tuple[str, str, str]:
    if score >= threshold:
        return LABELS[0]
    if score >= threshold - 15:
        if reversal is not None and reversal < MIN_REVERSAL:
            return WAIT_LABEL
        return LABELS[1]
    if score >= threshold - 30:
        return LABELS[2]
    return LABELS[3]


# ---------------------------------------------------------------------------
# 白話解說
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def explain(d: pd.DataFrame, comp: pd.DataFrame) -> dict:
    """針對最新一天，產生情境判斷與每個面向的白話說明。"""
    row, s = d.iloc[-1], comp.iloc[-1]
    c = row["Close"]
    long_name = "年線" if pd.notna(row["MA240"]) else "半年線（年線資料不足）"
    up = row["LongMA_slope"] > 0

    trend_txt = [
        f"{long_name}{'向上 ↑' if up else '向下 ↓'}（近 20 日 {_pct(row['LongMA_slope'])}）",
        f"股價{'在' if c > row['LongMA'] else '跌破'}{long_name}之{'上' if c > row['LongMA'] else '下'}"
        f"（距離 {_pct(c / row['LongMA'] - 1)}）",
        f"季線{'高於' if row['MA60'] > row['LongMA'] else '低於'}{long_name}",
    ]
    fib_r = row.get("FibRetrace", np.nan)
    fib_note = f"波段回撤 {fib_r * 100:.0f}%（黃金分割 38.2%～61.8% 最理想）" if pd.notna(fib_r) else ""
    pullback_txt = [
        f"距離一年內高點 {_pct(row['Drawdown'])}",
        *([fib_note] if fib_note else []),
    ]
    oversold_txt = [
        f"目前 RSI = {row['RSI']:.0f}（近 10 日最低 {row['RSI_min10']:.0f}；30 以下超賣）",
        f"季線乖離率 {_pct(row['Bias60'])}（近 10 日最深 {_pct(row['Bias60_min10'])}）",
    ]
    reversal_txt = [
        f"股價{'已站上月線 ✅' if c > row['MA20'] else '已站上10日線' if c > row['MA10'] else '仍在月線與10日線之下'}",
        f"MACD 柱狀體{'改善中 ↑' if row['MACD_hist'] > d['MACD_hist'].iloc[-4] else '仍在轉弱 ↓'}",
        f"近 5 日股價{'上漲' if c > d['Close'].iloc[-6] else '下跌'}",
    ]
    penalty_txt = []
    if s.get("capped", 0) > 0:
        penalty_txt.append(
            f"🛑 尚未出現止跌訊號（止跌分數 {s['reversal']:.0f} < {MIN_REVERSAL}），"
            f"總分上限 {NO_REVERSAL_CAP} 分，避免接到還在下跌的刀子"
        )
    if row["RSI"] > 75 and row["Bias60"] > 0.15:
        penalty_txt.append("⚠️ 短線過熱（RSI > 75 且季線乖離 > 15%），扣 10 分")
    if row["Vol20"] > row["Vol252"] * 1.5:
        penalty_txt.append(
            f"⚠️ 近期波動劇烈（近 20 日年化波動 {row['Vol20']:.0%}，"
            f"是平常的 {row['Vol20'] / row['Vol252']:.1f} 倍），扣 5 分，建議分批"
        )

    trend_ok = s["trend"] >= 15
    in_pullback = (row["Drawdown"] <= -0.06) or (0.25 <= fib_r <= 0.75)
    reversing = s["reversal"] >= MIN_REVERSAL
    if trend_ok and in_pullback and reversing:
        scenario = "📗 長多黃金分割回檔、出現止跌轉強：最符合本模型的買點型態"
    elif trend_ok and in_pullback:
        scenario = "📙 長多回檔進入觀察區、尚未止跌：等止跌訊號出現再進場"
    elif trend_ok and row["RSI"] > 70:
        scenario = "📕 多頭但短線過熱：留意第 5 波末升段風險，等拉回"
    elif trend_ok:
        scenario = "📘 多頭延續中：趨勢健康，但尚未拉回到便宜區間"
    elif row["Drawdown"] <= -0.20:
        scenario = "📓 趨勢轉弱、處於修正浪：等待底部背離或重新站回均線"
    else:
        scenario = "📒 盤整或偏弱：方向不明確，建議觀望"

    return {
        "scenario": scenario,
        "parts": {
            "trend": trend_txt,
            "pullback": pullback_txt,
            "oversold": oversold_txt,
            "reversal": reversal_txt,
        },
        "penalty": penalty_txt,
    }


# ---------------------------------------------------------------------------
# 回測：固定持有期間 + ATR 動態停損停利模擬
# ---------------------------------------------------------------------------
HORIZONS = {20: "1 個月", 60: "3 個月", 120: "6 個月"}


def backtest(
    d: pd.DataFrame, score: pd.Series, threshold: float, gap: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = d["Close"]
    cross = ((score >= threshold) & (score.shift(1) < threshold)).to_numpy()

    idx, last = [], -(10**9)
    for i in np.flatnonzero(cross):
        if i - last >= gap:
            idx.append(i)
            last = i

    fwd = {h: c.shift(-h) / c - 1 for h in HORIZONS}
    valid_days = score.notna()

    stats = []
    for h, name in HORIZONS.items():
        sig = fwd[h].iloc[idx].dropna()
        base = fwd[h][valid_days].dropna()
        stats.append(
            {
                "持有期間": name,
                "訊號次數": len(sig),
                "訊號勝率": sig.gt(0).mean() * 100 if len(sig) else np.nan,
                "訊號平均報酬": sig.mean() * 100 if len(sig) else np.nan,
                "訊號報酬中位數": sig.median() * 100 if len(sig) else np.nan,
                "隨便哪天買的勝率": base.gt(0).mean() * 100 if len(base) else np.nan,
                "隨便哪天買的平均報酬": base.mean() * 100 if len(base) else np.nan,
            }
        )

    # 計算每個訊號若搭配「初始 2.2×ATR 停損 + 2.8×ATR 移動停利（最多抱 60 日）」的出場表現
    closes = c.to_numpy(dtype=float)
    lows = d["Low"].to_numpy(dtype=float)
    highs = d["High"].to_numpy(dtype=float)
    atrs = d["ATR14"].bfill().to_numpy(dtype=float)
    exit_rets = []
    for i in idx:
        if i + 1 >= len(closes):
            exit_rets.append(np.nan)
            continue
        entry = closes[i]
        atr = atrs[i] if pd.notna(atrs[i]) and atrs[i] > 0 else entry * 0.025
        init_stop = entry - 2.2 * atr
        peak = entry
        end_j = min(i + 60, len(closes) - 1)
        ret_val = closes[end_j] / entry - 1
        for j in range(i + 1, end_j + 1):
            peak = max(peak, highs[j])
            # 獲利達 1.5 倍 ATR 後啟動移動停利線
            trail = peak - 2.8 * atr if peak >= entry + 1.5 * atr else init_stop
            stop_px = max(init_stop, trail)
            if lows[j] <= stop_px:
                ret_val = min(closes[j], stop_px) / entry - 1
                break
        exit_rets.append(ret_val * 100)

    signals = pd.DataFrame(
        {
            "日期": d.index[idx],
            "買進價": c.iloc[idx].to_numpy(),
            "分數": score.iloc[idx].to_numpy(),
            **{f"{name}後報酬": fwd[h].iloc[idx].to_numpy() * 100 for h, name in HORIZONS.items()},
            "動態停損停利報酬(60日內)": exit_rets,
        }
    )
    return pd.DataFrame(stats), signals


# ---------------------------------------------------------------------------
# 績效指標
# ---------------------------------------------------------------------------
def performance(close: pd.Series) -> dict:
    close = close.dropna()
    n = len(close)
    if n < 2:
        return {}
    total = close.iloc[-1] / close.iloc[0] - 1
    years = n / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = close.pct_change().std() * np.sqrt(252)
    mdd = (close / close.cummax() - 1).min()
    return {
        "總報酬": total * 100,
        "年化報酬": cagr * 100,
        "年化波動度": vol * 100,
        "最大回撤": mdd * 100,
        "報酬/風險比": cagr / vol if vol else np.nan,
    }
