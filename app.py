"""股票分析小幫手（Streamlit 介面）

版面：最上方「搜尋／觀察名單」→ 個股走勢圖（看圖優先）→ 細節分頁；另有「🚀 起漲雷達」掃描名單外的股票。
啟動方式：uv run streamlit run app.py
"""

from __future__ import annotations

import hmac
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai import BEST_MODEL, analyst_comment, classify_news, news_to_score, rule_based_comment
from analysis import (
    CHIPS_MIN,
    RR_MIN,
    analyze,
    build_composite,
    build_context,
    chips_ok_series,
    factor_summary,
    final_label,
    keybar_summary_row,
    timing_series,
)
from data import parse_inputs
from factors import FACTOR_NAMES, WEIGHTS, explain_chips, explain_fund, explain_value
from keybar import DAILY, WEEKLY
from model import HORIZONS, MIN_REVERSAL, NO_REVERSAL_CAP, backtest, explain, performance
from scanner import fetch_quotes, load_last_scan, scan

# 台股習慣：紅漲綠跌
UP, DOWN = "#e5484d", "#30a46c"
st.set_page_config(page_title="股票分析小幫手", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>.block-container{padding-top:1.4rem;padding-bottom:2rem}"
    "div[data-testid='stPills'] button{font-size:0.95rem}"
    "@media (max-width:640px){.block-container{padding-left:0.7rem;padding-right:0.7rem;padding-top:1rem}}"
    "</style>",
    unsafe_allow_html=True,
)


def secret(name: str) -> str:
    v = ""
    try:
        v = str(st.secrets.get(name, "")).strip()
    except Exception:
        pass
    if not v:
        v = os.environ.get(name, "").strip()
    return "" if (not v or "請把" in v or "請填入" in v) else v


APP_PASSWORD = secret("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("cloud_authed"):
    st.markdown("## 🔒 股票分析小幫手")
    pwd = st.text_input("請輸入存取密碼", type="password", placeholder="輸入密碼後按 Enter")
    if pwd:
        if hmac.compare_digest(pwd, APP_PASSWORD):
            st.session_state["cloud_authed"] = True
            st.rerun()
        else:
            st.error("密碼不正確，請重新輸入。")
    st.stop()

DEFAULT_GEMINI_KEY = secret("GEMINI_API_KEY")
GEMINI_MODEL = secret("GEMINI_MODEL") or BEST_MODEL  # 可在 secrets.toml 指定模型
FINMIND_TOKEN = secret("FINMIND_TOKEN")
DEFAULT_LIST = ["0050", "00981A", "2330", "2327"]
PERIODS = {"3個月": 3, "6個月": 6, "1年": 12, "2年": 24, "5年": 60}


# ---------------------------------------------------------------------------
# 資料載入（快取）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load(user_input: str, _token: str, version: str = "v3.3"):
    return analyze(user_input, _token)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def news_sentiment(_key: str, ticker: str, name: str, news_json: str, model: str):
    return classify_news(_key, name, json.loads(news_json), model)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ai_comment(_key: str, ticker: str, date: str, threshold: int, context_json: str, model: str):
    return analyst_comment(_key, json.loads(context_json), model)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def ensure_names() -> bool:
    """下載全部上市櫃股票清單（讓中文名稱搜尋、名稱顯示涵蓋所有台股）。"""
    try:
        return not fetch_quotes(max_age_hours=12).empty
    except Exception:
        return False


def model_note(used: str) -> str:
    if used == GEMINI_MODEL:
        tier = "（免費層級）" if used != BEST_MODEL else ""
        return f"🧠 使用模型：`{used}`{tier}"
    return (
        f"⚠️ 使用模型：`{used}`（備援）。設定的 `{GEMINI_MODEL}` 目前無法使用，"
        "可能是暫時忙碌，或 API 金鑰還沒啟用付費層級。"
    )


# ---------------------------------------------------------------------------
# 觀察名單狀態（存在網址裡，加入書籤就能保留名單）
# ---------------------------------------------------------------------------
ss = st.session_state
ensure_names()
if "watchlist" not in ss:
    qp = st.query_params.get("s", "")
    ss.watchlist = parse_inputs(qp) if qp else DEFAULT_LIST.copy()
ss.setdefault("threshold", 70)
ss.setdefault("period", "1年")


def add_items(items: list[str]) -> None:
    for it in items:
        if it not in ss.watchlist:
            ss.watchlist.append(it)
    if items:
        ss.focus_pill = items[-1]


def on_add_box() -> None:
    add_items(parse_inputs(ss.get("add_box", "")))
    ss.add_box = ""


def remove_item(item: str) -> None:
    if item in ss.watchlist and len(ss.watchlist) > 1:
        ss.watchlist.remove(item)
        ss.focus_pill = ss.watchlist[0]


def on_manage(key: str) -> None:
    keep = ss.get(key) or ss.watchlist[:1]
    ss.watchlist = [x for x in ss.watchlist if x in keep]


# ---------------------------------------------------------------------------
# 最上方：標題 + 搜尋框 + 設定
# ---------------------------------------------------------------------------
t1, t2 = st.columns([5, 1], vertical_alignment="bottom")
t1.markdown("## 📈 股票分析小幫手")
with t2.popover("⚙️ 設定", width="stretch"):
    st.slider(
        "買進門檻分數", min_value=65, max_value=90, step=5, key="threshold",
        help="「買進時機」（技術與波浪 + 籌碼濾網）達到門檻才算「符合買進條件」。門檻越高，訊號越少但越嚴格。",
    )
    mkey = "manage_" + "_".join(ss.watchlist)
    st.multiselect(
        "管理觀察名單（點 ✕ 移除）", options=ss.watchlist, default=ss.watchlist, key=mkey,
        on_change=on_manage, args=(mkey,),
    )
    if st.button("🔄 重新抓取最新資料", width="stretch"):
        st.cache_data.clear()
    st.text_input(
        "🔑 自備 Gemini 金鑰（選填）",
        type="password",
        key="user_gemini_key",
        placeholder="留空則使用系統預設金鑰",
        help="雲端多人共用導致免費額度暫時滿載時，可在此貼上自己的 Gemini API Key（只保存在目前瀏覽器連線，關掉即清除）。",
    )
    GEMINI_KEY = ss.get("user_gemini_key", "").strip() or DEFAULT_GEMINI_KEY
    st.caption(f"🤖 AI 分析：{'✅ 已啟用（' + GEMINI_MODEL + '）' if GEMINI_KEY else '⚠️ 未設定金鑰（改用制式摘要）'}")
    st.caption(
        "資料來源：Yahoo Finance（股價）、證交所／櫃買中心（股票清單、最新行情）、"
        "FinMind（台股籌碼、營收、財報、本益比、新聞）。顏色：台股習慣 🔴 紅漲 🟢 綠跌。"
    )

st.text_input(
    "新增股票",
    key="add_box",
    on_change=on_add_box,
    placeholder="🔍 輸入股票代號或名稱，按 Enter 加入觀察名單（例如 2330、台積電、長榮航、NVDA；可一次輸入多檔，用空白分隔）",
    label_visibility="collapsed",
)

threshold = int(ss.threshold)
st.query_params["s"] = ",".join(ss.watchlist)

# ---------------------------------------------------------------------------
# 載入觀察名單
# ---------------------------------------------------------------------------
stocks: dict[str, dict] = {}
errors: list[str] = []
seen_tickers: dict[str, str] = {}
progress = st.progress(0.0, text="正在抓取資料…")
for i, item in enumerate(list(ss.watchlist)):
    progress.progress(i / max(len(ss.watchlist), 1), text=f"正在分析 {item} 的股價、波浪、有效K、籌碼、財報與新聞…")
    try:
        r = load(item, FINMIND_TOKEN)
    except Exception as e:
        errors.append(f"找不到「{item}」或暫時抓不到資料（{str(e)[:60]}），已從名單移除。")
        ss.watchlist.remove(item)
        continue
    if r["ticker"] in seen_tickers:  # 例如同時輸入「台積電」和「2330」
        ss.watchlist.remove(item)
        if ss.get("focus_pill") == item:
            ss.focus_pill = seen_tickers[r["ticker"]]
        continue
    seen_tickers[r["ticker"]] = item
    stocks[item] = r
progress.progress(0.9, text="AI 正在判讀新聞情緒…" if GEMINI_KEY else "整理中…")

news_models: set[str] = set()
for item, res in stocks.items():
    news_json = json.dumps(res["news"], ensure_ascii=False)
    try:
        news, overall, status = news_sentiment(GEMINI_KEY, res["ticker"], res["name"], news_json, GEMINI_MODEL)
    except Exception as e:
        news, overall, status = res["news"], None, f"新聞情緒判讀失敗：{e}"
    if overall is not None:
        news_models.add(status)
        status = model_note(status)
    res["news_scored"], res["news_overall"], res["news_status"] = news, overall, status
    res["comp"] = build_composite(res, news_to_score(overall))
    res["timing"] = timing_series(res, threshold)
    res["bt_stats"], res["bt_sigs"] = backtest(res["d"], res["timing"], threshold)
progress.empty()
st.query_params["s"] = ",".join(ss.watchlist)

for msg in errors:
    st.warning(msg)
limited = [r["name"] for r in stocks.values() if r.get("finmind_limited")]
if limited:
    st.warning(
        f"⚠️ FinMind 免費查詢次數已達每小時上限，{'、'.join(limited)} 的部分籌碼 / 財報 / 新聞改用本機快取或暫缺。"
        "請約 1 小時後到「⚙️ 設定」按「🔄 重新抓取最新資料」，或到 finmindtrade.com 註冊免費帳號，"
        "把 token 填進 secrets.toml 的 FINMIND_TOKEN（上限會提高）。"
    )
if news_models and news_models != {GEMINI_MODEL}:
    st.caption(f"⚠️ AI 目前用備援模型（{'、'.join(sorted(news_models))}），設定的 {GEMINI_MODEL} 暫時無法使用。")
if not stocks:
    st.info("👆 請在上面的搜尋框輸入至少一個股票代號")
    st.stop()


def current(res: dict):
    """回傳（買進時機分數、建議、備註、綜合評估分數）。"""
    timing = res["tech"]["score"].iloc[-1]
    overall = res["comp"]["score"].iloc[-1]
    if pd.isna(timing):
        return timing, None, [], overall
    label, notes = final_label(res, threshold)
    return timing, label, notes, overall


def pill_label(item: str) -> str:
    r = stocks[item]
    _, lab, _, _ = current(r)
    emoji = lab[1].split(" ")[0] if lab else "▫️"
    c = r["d"]["Close"]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    kb = r["keybar"]
    fire = " 🕯️" if "confirmed" in (kb["mid"]["code"], kb["long"]["code"]) else ""
    return f"{emoji} {r['name']} {chg:+.1f}%{fire}"


# ---------------------------------------------------------------------------
# 觀察名單（點一下切換）
# ---------------------------------------------------------------------------
opts = list(stocks)
if ss.get("focus_pill") not in opts:
    ss.focus_pill = ss.get("focus_last") if ss.get("focus_last") in opts else opts[0]
picked = st.pills(
    "觀察名單（點一下切換）", opts, format_func=pill_label, key="focus_pill", selection_mode="single",
    label_visibility="collapsed",
)
focus = picked if picked in opts else ss.get("focus_last") if ss.get("focus_last") in opts else opts[0]
ss.focus_last = focus
st.caption("圖示：✅ 符合買進　🟡 可小量分批　👀 列入觀察　⏸️ 中性　⛔ 暫不建議　🕯️ 有效K已確認起漲　｜　每天的漲跌幅依台股習慣：紅漲綠跌")


def window(df: pd.DataFrame) -> pd.DataFrame:
    months = PERIODS.get(ss.get("period") or "1年", 12)
    return df[df.index >= df.index.max() - pd.DateOffset(months=months)]


def radar(fs: dict, height: int = 210) -> go.Figure:
    keys = [k for k in WEIGHTS if fs[k] is not None]
    vals = [fs[k] * 100 for k in keys]
    labels = [FACTOR_NAMES[k].split(" ")[1] for k in keys]
    fig = go.Figure(
        go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself", line_color="#ff8c00")
    )
    fig.update_layout(
        height=height, margin=dict(t=20, b=20, l=30, r=30), showlegend=False,
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
    )
    return fig


res = stocks[focus]
d, tech, comp = res["d"], res["tech"], res["comp"]
timing, label, notes, overall = current(res)
fs = factor_summary(comp)
wv = res["waves"]
mkt = res["market"]
pat = res["patterns"]
kb = res["keybar"]
sr = res["sr"]

main_tabs = st.tabs(["🔍 個股分析", "🚀 起漲雷達（名單外）", "🏠 名單總覽與比較", "📖 說明"])
tab_stock, tab_scan, tab_home, tab_help = main_tabs

# ===========================================================================
# 個股分析：最上面是圖，細節在下面
# ===========================================================================
with tab_stock:
    # ---- 一行重點 ----
    close, prev = d["Close"].iloc[-1], d["Close"].iloc[-2]
    h1, h2, h3, h4, h5 = st.columns([1.3, 1.2, 1.6, 1.6, 0.6], vertical_alignment="center")
    h1.metric(f"{res['name']}（{d.index[-1]:%m/%d}）", f"{close:,.2f}", f"{(close / prev - 1) * 100:+.2f}%", delta_color="inverse")
    if pd.isna(timing):
        h2.info("上市太短，無法評分")
    else:
        h2.markdown(f"**🎯 買進時機 {timing:.0f}**  \n{label[1]}")
    h3.markdown(f"**🌊 波浪**  \n{wv['wave_stage']}")
    h4.markdown(f"**🕯️ 有效K**  \n中線：{kb['mid']['title']}  \n長線：{kb['long']['title']}")
    if len(ss.watchlist) > 1:
        h5.button("🗑️ 移除", key=f"rm_{focus}", on_click=remove_item, args=(focus,), help="從觀察名單移除這檔")

    # ---- 圖表控制 ----
    g0, g1, g2, g3, g4 = st.columns([2.2, 1, 1, 1, 1.3], vertical_alignment="center")
    g0.segmented_control("期間", list(PERIODS), key="period", label_visibility="collapsed")
    show_wave = g1.checkbox("🌊 波浪", value=True)
    show_fib = g2.checkbox("📐 黃金分割", value=True)
    show_sr = g3.checkbox("🧱 支撐壓力", value=False)
    show_kb = g4.checkbox("🕯️ 有效K棒", value=True)

    w = window(d)
    wc = comp.loc[w.index]
    sigs = res["bt_sigs"]
    sigs = sigs[sigs["日期"] >= w.index[0]]
    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.88, 0.12], row_heights=[0.66, 0.13, 0.21],
        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
        horizontal_spacing=0.01, vertical_spacing=0.04,
        subplot_titles=("", "分價量", "成交量", "買進時機分數"),
    )
    fig.add_trace(go.Candlestick(
        x=w.index, open=w["Open"], high=w["High"], low=w["Low"], close=w["Close"], name="K 線",
        increasing_line_color=UP, decreasing_line_color=DOWN, increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
    ), row=1, col=1)
    for ma, color, lab in [("MA20", "#f5a623", "月線"), ("MA60", "#4a90e2", "季線"), ("MA240", "#9b59b6", "年線")]:
        fig.add_trace(go.Scatter(x=w.index, y=w[ma], name=lab, line=dict(width=1.3, color=color)), row=1, col=1)

    if show_wave and wv.get("pivots"):
        vis = [p for p in wv["pivots"] if p["date"] >= w.index[0]]
        if vis:
            fig.add_trace(go.Scatter(
                x=[p["date"] for p in vis] + [w.index[-1]], y=[p["price"] for p in vis] + [float(w["Close"].iloc[-1])],
                mode="lines", name="波浪折線", line=dict(color="#6366f1", width=2.2, dash="dashdot"),
            ), row=1, col=1)
            for p in vis:
                is_high = p["kind"] == "H"
                fig.add_annotation(
                    x=p["date"], y=p["price"], text=f"<b>{p.get('label') or ('高' if is_high else '低')}</b><br>{p['price']:,.1f}",
                    showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.5,
                    arrowcolor="#db2777" if is_high else "#4f46e5", ax=0, ay=-32 if is_high else 32,
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#db2777" if is_high else "#4f46e5", borderwidth=1.2,
                    font=dict(size=11, color="#1e1b4b"), row=1, col=1,
                )

    if show_fib and wv.get("fib"):
        fib = wv["fib"]
        for f_price, f_name, f_col in [
            (fib["zone_382"], "黃金 38.2%", "#0284c7"), (fib["zone_500"], "黃金 50%", "#d97706"),
            (fib["zone_618"], "黃金 61.8%", "#059669"), (fib["target_1272"], "延伸 127.2%", "#7c3aed"),
        ]:
            fig.add_hline(y=f_price, line_dash="dash", line_color=f_col, line_width=1.2, opacity=0.8, row=1, col=1,
                          annotation_text=f"{f_name} ({f_price:,.1f})", annotation_position="top right",
                          annotation_font_color=f_col)

    if show_sr:
        for x in sr["resistance"]:
            fig.add_hline(y=x["price"], line_dash="dot", line_color=UP, opacity=0.65, row=1, col=1,
                          annotation_text=f"壓力 {x['price']:,.1f}", annotation_position="top left", annotation_font_color=UP)
        for x in sr["support"]:
            fig.add_hline(y=x["price"], line_dash="dot", line_color=DOWN, opacity=0.65, row=1, col=1,
                          annotation_text=f"支撐 {x['price']:,.1f}", annotation_position="bottom left",
                          annotation_font_color=DOWN)

    if not sigs.empty:
        fig.add_trace(go.Scatter(
            x=sigs["日期"], y=d.loc[sigs["日期"], "Low"] * 0.97, mode="markers", name="拉回買進訊號",
            marker=dict(symbol="triangle-up", size=12, color="#ff8c00"),
        ), row=1, col=1)

    if show_kb:
        w0 = w.index[0]

        def add_marks(dates, ys, name, symbol, color, size=11, text=None):
            if len(dates):
                fig.add_trace(go.Scatter(
                    x=dates, y=ys, mode="markers", name=name, hovertext=text,
                    marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color="#333")),
                ), row=1, col=1)

        ek = [e for e in kb["events_short"] if e["date"] >= w0]
        add_marks([e["date"] for e in ek], [e["high"] * 1.02 for e in ek], "有效紅K", "star", "#facc15", 13,
                  [f"有效紅K {e['ret'] * 100:+.1f}%、量 {e['vol_ratio']:.1f} 倍" for e in ek])
        fails = [e for e in ek if e["status"] == "failed"]
        add_marks([d.index[e["end_idx"]] for e in fails], [d["Close"].iloc[e["end_idx"]] for e in fails],
                  "有效K失效", "x", "#6b7280", 9)
        conf = [e for e in kb["events_mid"] if e["status"] == "confirmed" and e["confirm_date"] >= w0]
        add_marks([e["confirm_date"] for e in conf], [d.loc[e["confirm_date"], "Low"] * 0.95 for e in conf],
                  "中線起漲確認", "diamond", "#9333ea", 12, [e["how"] for e in conf])
        lconf = []
        for e in kb["events_long"]:
            if e["status"] == "confirmed":
                dd_ = d.index[d.index <= e["confirm_date"]]
                if len(dd_) and dd_[-1] >= w0:
                    lconf.append(dd_[-1])
        add_marks(lconf, [d.loc[x, "Low"] * 0.92 for x in lconf], "長線週K起漲確認", "hexagram", "#b91c1c", 15)
        bk = [x for x in kb["black_dates"] if x >= w0]
        add_marks(bk, [d.loc[x, "High"] * 1.03 for x in bk], "有效黑K", "triangle-down", DOWN, 10)
        act = next((s_ for s_ in (kb["mid"], kb["short"]) if s_["code"] in ("waiting", "confirmed", "running")), None)
        if act:
            for price, nm, colr in [(act["high"], "K高", "#ea580c"), (act["mid"], "K實體1/2", "#a3a3a3"),
                                    (act["low"], "K低", "#1f2937")]:
                fig.add_hline(y=price, line_dash="longdash", line_color=colr, line_width=1.1, opacity=0.85, row=1, col=1,
                              annotation_text=f"{nm} {price:,.1f}", annotation_position="bottom right",
                              annotation_font_color=colr)

    prof = sr["profile"]
    fig.add_trace(go.Bar(x=prof["volume"], y=prof["price"], orientation="h", name="分價量",
                         marker_color="rgba(120,120,120,0.45)", showlegend=False), row=1, col=2)
    vol_colors = np.where(w["Close"] >= w["Open"], UP, DOWN)
    fig.add_trace(go.Bar(x=w.index, y=w["Volume"], marker_color=vol_colors, name="成交量", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=res["timing"].reindex(w.index), name="買進時機",
                             line=dict(color="#ff8c00")), row=3, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=wc["score_bt"], name="綜合評估", line=dict(color="#999", width=1, dash="dot")),
                  row=3, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff8c00", row=3, col=1,
                  annotation_text=f"門檻 {threshold}", annotation_position="top left")
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, font=dict(size=11)), margin=dict(t=40, b=10, l=10, r=10),
                      bargap=0.05)
    fig.layout.yaxis2.matches = "y"
    fig.layout.yaxis2.showticklabels = False
    fig.layout.xaxis2.showticklabels = False
    fig.layout.xaxis3.matches = "x"
    fig.layout.xaxis4.matches = "x"
    lo, hi = w["Low"].min(), w["High"].max()
    if show_fib and wv.get("fib"):
        hi = max(hi, min(wv["fib"]["target_1272"], hi * 1.25) * 0.98)
    fig.layout.yaxis.range = [lo * (0.90 if show_kb else 0.94), hi * 1.05]
    fig.layout.yaxis4.range = [0, 100]
    for ax in ("xaxis", "xaxis3", "xaxis4"):
        fig.layout[ax].rangebreaks = [dict(bounds=["sat", "mon"])]
    st.plotly_chart(fig, width="stretch")

    # ---- 看圖重點 ----
    with st.container(border=True):
        st.markdown("**📌 看圖重點**")
        pts = []
        if not pd.isna(timing):
            pts.append(f"🎯 **{label[1]}**（買進時機 {timing:.0f} 分）：{explain(d, tech)['scenario']}")
        if wv.get("fib"):
            fb = wv["fib"]
            pts.append(
                f"🌊 前波 {fb['low_price']:,.1f} → {fb['high_price']:,.1f}，目前回吐 **{fb['retrace_pct']:.0f}%**；"
                f"黃金分割 38.2% {fb['zone_382']:,.1f}／50% {fb['zone_500']:,.1f}／61.8% {fb['zone_618']:,.1f}"
            )
        if kb["mid"]["code"] != "none":
            pts.append(f"🕯️ 中線：{kb['mid']['title']} — {kb['mid'].get('desc', '')}")
        if kb["long"]["code"] != "none":
            pts.append(f"🕯️ 長線：{kb['long']['title']} — {kb['long'].get('desc', '')}")
        sup = f"{sr['support'][0]['price']:,.1f}" if sr["support"] else "無明顯支撐"
        resi = f"{sr['resistance'][0]['price']:,.1f}" if sr["resistance"] else "無明顯壓力"
        pts.append(f"🧱 最近支撐 **{sup}**、最近壓力 **{resi}**；ATR 建議停損 **{sr['suggest_stop']:,.2f}**（{sr['stop_pct']:+.1f}%）")
        if kb["black"]:
            pts.append(f"⚠️ {kb['black']['date']:%m/%d} 出現有效黑K（{kb['black']['ret']:+.1f}%），短線風險提示")
        pts.append(f"🌦️ {mkt['regime']}｜{mkt['rs_desc']}")
        for p_ in pts:
            st.markdown(f"- {p_}")
        st.caption(
            "圖例：🔺 橘三角＝拉回買進訊號　⭐ 黃星＝有效紅K　◆ 紫菱形＝中線起漲確認　✡ 紅六芒星＝長線週K起漲確認　"
            "✕＝有效K失效　▼ 綠三角＝有效黑K　｜　①②③④⑤／ⓐⓑⓒ＝波浪轉折　橫虛線＝黃金分割、K高／1/2／K低"
        )

    # ---- 細節分頁 ----
    sub = st.tabs(["📊 分析重點與 AI", "🌊 波浪與相似K線", "🏦 籌碼", "📑 基本面", "📰 消息", "🧪 回測"])
    tab_one, tab_wave, tab_chip, tab_fund, tab_news, tab_bt = sub

    # -----------------------------------------------------------------------
    with tab_one:
        if pd.isna(timing):
            st.info("這檔上市時間太短（少於約半年），還無法計算評分。")
        else:
            info = explain(d, tech)
            ok = chips_ok_series(res)
            c1, c2, c3, c4 = st.columns([1.05, 1.35, 1.25, 1.35])
            with c1.container(border=True):
                st.metric("🎯 買進時機", f"{timing:.0f} / 100")
                st.markdown(f"### {label[1]}")
                st.caption(label[2])
            with c2.container(border=True):
                st.markdown("**🌊 波浪與黃金分割定位**")
                st.markdown(f"**{wv['wave_stage']}**")
                if wv.get("fib"):
                    fib = wv["fib"]
                    st.caption(f"前波 {fib['low_price']:,.1f} → {fib['high_price']:,.1f}（+{fib['gain_pct']:.1f}%），"
                               f"目前已回吐 **{fib['retrace_pct']:.1f}%**")
                    st.caption(f"黃金分割 38.2%：**{fib['zone_382']:,.1f}** ｜ 61.8%：**{fib['zone_618']:,.1f}**")
                if wv.get("divergence"):
                    st.caption(wv["divergence"]["title"])
            with c3.container(border=True):
                st.markdown("**🌦️ 大盤氣象與相對強弱**")
                st.markdown(f"**{mkt['regime']}**")
                st.caption(mkt["rs_desc"])
                if ok is not None:
                    st.caption(f"籌碼濾網：{'✅ 通過' if bool(ok.iloc[-1]) else '❌ 未通過'}（≥ {CHIPS_MIN} 分）")
                else:
                    st.caption("籌碼濾網：不適用（美股無公開三大法人）")
            with c4.container(border=True):
                st.metric("🏅 綜合評估（體質參考）", f"{overall:.0f} / 100")
                st.caption(info["scenario"])
                for p in info["penalty"] + notes:
                    st.caption(f"• {p}")

            # 三週期買點雷達
            st.markdown("##### 🎯 三週期買點雷達（有效K棒・第二套獨立訊號：突破追強型）")
            mid_c, long_c = kb["mid"]["code"], kb["long"]["code"]
            if label[0] in ("buy", "light") and "confirmed" in (mid_c, long_c):
                st.success("🔥 **兩套訊號同時偏多**：「拉回買」的買進時機與「突破追強」的有效K起漲確認同時出現，可信度相對較高。")
            elif label[0] == "avoid" and "confirmed" in (mid_c, long_c):
                st.info("⚖️ **兩套訊號看法不同**：拉回買模型認為不宜進場，但有效K已確認起漲（屬於追強勢型態）。"
                        "若要參與，部位宜小，並嚴守 K低 停損。")

            def keybar_card(col, title: str, note: str, stt: dict, extra: list[str]):
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(note)
                    st.markdown(f"#### {stt['title']}")
                    if stt.get("desc"):
                        st.markdown(stt["desc"])
                    if stt.get("wave"):
                        st.markdown(f"🌊 {stt['wave']}")
                    for x in extra:
                        st.caption(x)

            k1, k2, k3 = st.columns(3)
            short_extra = ["📌 回測顯示短線（1 週）沒有統計優勢，這裡只提供價位當停損停利參考，不給買進建議。"]
            if kb["black"]:
                b = kb["black"]
                short_extra.insert(0, f"⚠️ {b['date']:%m/%d} 出現有效黑K（{b['ret']:+.1f}%"
                                      f"{'，收在月線下' if b['below_ma20'] else ''}）：短線風險提示（不是賣出訊號）。")
            keybar_card(k1, "🏃 短線（幾天～2週）", "日線有效紅K（不限趨勢）", kb["short"], short_extra)
            keybar_card(k2, "🚶 中線（1～3個月）", "日線有效紅K + 年線向上 → 起漲確認為買點，停損 K低", kb["mid"],
                        [f"📊 這檔歷史（持有3個月）：{keybar_summary_row(kb['bt_mid'][0], 1)}"])
            long_extra = [f"📊 這檔歷史（持有1季）：{keybar_summary_row(kb['bt_long'][0], 1)}"]
            if kb["weekly_partial"]:
                long_extra.insert(0, "⏱️ 本週K棒尚未收完，以目前價格暫估，週五收盤後才算數。")
            keybar_card(k3, "🧘 長線（1季～半年）", "週線有效紅K + 40週線向上（回測最穩定）", kb["long"], long_extra)

            # AI 分析師
            with st.container(border=True):
                st.markdown("### 🤖 AI 分析師（波浪解讀 × 有效K × 多空辯論 × 實戰劇本）")
                ctx = build_context(res, comp, label[1], notes, res["news_scored"], res["bt_stats"])
                ctx_json = json.dumps(ctx, ensure_ascii=False, default=str)
                state_key = f"ai_v3k_{res['ticker']}_{d.index[-1]:%Y%m%d}_{threshold}"
                if GEMINI_KEY:
                    bcol, _ = st.columns([2, 5])
                    again = state_key in ss
                    if bcol.button("🔄 重新請 AI 分析" if again else "✨ 請 AI 解讀波浪與買賣劇本",
                                   type="secondary" if again else "primary", width="stretch"):
                        if again:
                            ai_comment.clear()
                        with st.spinner("AI 正在分析波浪位置、有效K、黃金分割、籌碼與多空觀點…（約 5–30 秒）"):
                            try:
                                ss[state_key] = ai_comment(GEMINI_KEY, res["ticker"], f"{d.index[-1]:%Y%m%d}",
                                                           threshold, ctx_json, GEMINI_MODEL)
                            except Exception as e:
                                st.error(f"AI 暫時無法回應，請稍後再試：{str(e)[:150]}")
                    if state_key in ss:
                        text, used = ss[state_key]
                        st.markdown(text)
                        st.caption(model_note(used))
                        st.caption("AI 只根據上面的數據整理，可能有誤解，請自行判斷。")
                    else:
                        st.caption("按下按鈕後，AI 會用白話說明目前走第幾波、黃金分割與防線、短中長線有效K狀態，"
                                   "並列出 🐂 多方 vs. 🐻 空方辯論與 🎯 If-Then 操作劇本。")
                else:
                    st.markdown(rule_based_comment(ctx))

            st.markdown("##### 🏅 綜合評估：五個面向")
            explain_map = {
                "tech": sum(info["parts"].values(), []),
                "chips": explain_chips(res["chips"], res["chips_raw"]),
                "fund": explain_fund(res["fund"], res["us"]),
                "value": explain_value(res["value"], res["us"]),
                "news": [
                    f"近期 {len(res['news_scored'])} 則新聞，整體情緒 {res['news_overall']:+.2f}（−1 偏負面 ～ +1 偏正面）"
                    if res["news_overall"] is not None else res["news_status"]
                ],
            }
            pcols = st.columns(5)
            for col, k in zip(pcols, WEIGHTS):
                with col.container(border=True):
                    st.markdown(f"**{FACTOR_NAMES[k]}**")
                    v = comp[k].iloc[-1]
                    if pd.isna(v):
                        st.caption("不適用 / 無資料（不計入）")
                    else:
                        st.progress(float(v) / WEIGHTS[k], text=f"{v:.1f} / {WEIGHTS[k]}")
                    for line in explain_map[k]:
                        st.caption(f"• {line}")

        st.markdown("##### 🧱 壓力支撐 & 🛡️ ATR 動態停損停利風控")
        c1, c2, c3, c4 = st.columns(4)
        with c1.container(border=True):
            st.markdown("**上方壓力**（由近到遠）")
            for x in sr["resistance"] or []:
                st.write(f"🔺 **{x['price']:,.2f}**（{x['price'] / sr['close'] - 1:+.1%}）— {x['sources']}")
            if not sr["resistance"]:
                st.write("上方沒有明顯壓力（接近歷史高點）")
        with c2.container(border=True):
            st.markdown("**下方支撐**（由近到遠）")
            for x in sr["support"] or []:
                st.write(f"🔻 **{x['price']:,.2f}**（{x['price'] / sr['close'] - 1:+.1%}）— {x['sources']}")
            if not sr["support"]:
                st.write("下方沒有明顯支撐")
        with c3.container(border=True):
            st.markdown("**風險報酬比**")
            if sr["rr"] is not None:
                st.metric("到壓力的漲幅 ÷ 到支撐的跌幅", f"{sr['rr']:.2f}")
                st.caption("大於 1.3～1.5 比較划算；小於 1 代表上漲空間比下跌風險小。")
            else:
                st.write("無法計算")
        with c4.container(border=True):
            st.markdown("**🛡️ ATR 動態停損與移動停利**")
            st.write(f"• 建議初始停損價：**{sr['suggest_stop']:,.2f}**（{sr['stop_pct']:+.1f}%）")
            st.write(f"• Chandelier 移動停利：**{sr['chandelier']:,.2f}**")
            st.caption(f"14 日 ATR = {sr['atr']:,.2f}（日波動約 {sr['atr_pct']:.1f}%），依波動度自動調整防線。")

        with st.expander("🧮 單筆風險部位計算機（如果現在買，該買幾股才不會受重傷？）"):
            rc1, rc2, rc3 = st.columns([1, 1, 2])
            cap_wan = rc1.number_input("預計投資總資金（萬元）", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)
            risk_pct = rc2.slider("單筆最大可承受虧損（占總資金 %）", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            max_loss_amt = cap_wan * 10000 * (risk_pct / 100)
            risk_per_share = max(sr["close"] - sr["suggest_stop"], sr["close"] * 0.01)
            rec_shares = min(int(max_loss_amt // risk_per_share), int((cap_wan * 10000) // sr["close"]))
            rec_cost = rec_shares * sr["close"]
            rc3.markdown(
                f"- 若以現價 **{sr['close']:,.2f}** 進場、跌破 **{sr['suggest_stop']:,.2f}**（每股風險 {risk_per_share:,.2f}）停損：\n"
                f"- 建議最多買進 **{rec_shares:,} 股**（約 **{rec_shares / 1000:.2f} 張**，投入約 **{rec_cost / 10000:,.1f} 萬元**，"
                f"占總資金 {rec_cost / (cap_wan * 10000) * 100:.0f}%）。\n"
                f"- 就算觸及停損，最大虧損約 **{rec_shares * risk_per_share:,.0f} 元**（不超過總資金的 {risk_pct:.1f}%）。"
            )

    # -----------------------------------------------------------------------
    with tab_wave:
        with st.container(border=True):
            st.markdown(f"#### 📐 目前波浪階段：{wv['wave_stage']}")
            st.markdown(f"- **近期波段轉折路徑**：{wv['path_str']}")
            st.markdown(f"- **費波那契黃金分割解析**：{wv['fib_str']}")
            if wv.get("divergence"):
                div = wv["divergence"]
                (st.warning if div["type"] == "bearish" else st.success)(f"**{div['title']}**：{div['desc']}")
            else:
                st.caption("• 目前無明顯 RSI 頂背離或底背離訊號。")

        if wv.get("fib"):
            fib = wv["fib"]
            close_now = float(d["Close"].iloc[-1])
            fc1, fc2 = st.columns([1.3, 1])
            with fc1.container(border=True):
                st.markdown(f"**📉 黃金分割回撤位**（{fib['low_date']} 低點 `{fib['low_price']:,.2f}` → "
                            f"{fib['high_date']} 高點 `{fib['high_price']:,.2f}`，目前回吐 **{fib['retrace_pct']:.1f}%**）")
                st.dataframe(
                    pd.DataFrame([{
                        "黃金分割比例": lv["name"], "對應價位": lv["price"], "距現價幅度": (lv["price"] / close_now - 1) * 100,
                        "目前狀態": "✅ 目前守穩其上" if close_now >= lv["price"] else "⚠️ 已跌破（轉為上方壓力）",
                    } for lv in fib["retrace_levels"]]),
                    hide_index=True, width="stretch",
                    column_config={"對應價位": st.column_config.NumberColumn(format="%.2f"),
                                   "距現價幅度": st.column_config.NumberColumn(format="%+.1f%%")},
                )
            with fc2.container(border=True):
                st.markdown("**🚀 波段延伸目標價**（守穩支撐並突破前高後的測幅目標）")
                st.dataframe(
                    pd.DataFrame([{"延伸比例": lv["name"], "目標價位": lv["price"],
                                   "潛在空間": (lv["price"] / close_now - 1) * 100} for lv in fib["extend_levels"]]),
                    hide_index=True, width="stretch",
                    column_config={"目標價位": st.column_config.NumberColumn(format="%.2f"),
                                   "潛在空間": st.column_config.NumberColumn(format="%+.1f%%")},
                )
                st.caption("💡 **口訣**：第 2 波常回吐 **50%～61.8%**（不能跌破起漲點）；第 4 波常在 **23.6%～38.2%** 止穩；"
                           "第 3 波目標常達 **127.2%～161.8%**。")

        if wv.get("pivots"):
            with st.expander("📋 近一年波段高低轉折點明細"):
                st.dataframe(
                    pd.DataFrame([{
                        "日期": p["date"], "類型": "🔺 波段高點" if p["kind"] == "H" else "🔻 波段低點",
                        "波浪標記": p.get("label") or "-", "轉折價位": p["price"], "當時 RSI": p.get("rsi"),
                    } for p in reversed(wv["pivots"])]),
                    hide_index=True, width="stretch",
                    column_config={"日期": st.column_config.DateColumn(format="YYYY/MM/DD"),
                                   "轉折價位": st.column_config.NumberColumn(format="%.2f"),
                                   "當時 RSI": st.column_config.NumberColumn(format="%.1f")},
                )

        st.divider()
        st.markdown("#### 🔍 歷史最相似 10 次 K 線走勢比對")
        st.caption("在過去 15 年中，找出與最近 20 個交易日「走勢形狀 + RSI + 季線乖離」最相似的 10 個時期，統計之後的真實表現。")
        if not pat or pat.get("summary") is None:
            st.info("這檔股票歷史資料不足，無法比對。")
        else:
            ps = pat["summary"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("1 個月後勝率", f"{ps['win_20d']:.0f}%", f"平均 {ps['avg_20d']:+.1f}%", delta_color="inverse")
            mc2.metric("3 個月後勝率", f"{ps['win_60d']:.0f}%", f"平均 {ps['avg_60d']:+.1f}%", delta_color="inverse")
            mc3.metric("1 個月內最深再跌（中位數）", f"{ps['worst_20d_mae']:+.1f}%")
            mc4.metric("3 個月內最高漲幅（中位數）", f"{ps['best_60d_mfe']:+.1f}%")
            paths = pat["paths"]
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=paths.index, y=paths["樂觀前25%"], mode="lines", line=dict(width=0),
                                       showlegend=False, hoverinfo="skip"))
            fig_p.add_trace(go.Scatter(x=paths.index, y=paths["保守後25%"], mode="lines", line=dict(width=0), fill="tonexty",
                                       fillcolor="rgba(255,140,0,0.18)", name="25%～75% 常態區間"))
            fig_p.add_trace(go.Scatter(x=paths.index, y=paths["平均路徑"], mode="lines", name="平均路徑",
                                       line=dict(color=UP, width=2.5)))
            fig_p.add_trace(go.Scatter(x=paths.index, y=paths["中位數路徑"], mode="lines", name="中位數路徑",
                                       line=dict(color="#4a90e2", width=2, dash="dash")))
            fig_p.add_hline(y=0, line_dash="dot", line_color="#888")
            fig_p.update_layout(title="前 10 次相似走勢在之後 60 個交易日的表現", xaxis_title="往後交易日數",
                                yaxis_title="累積漲跌幅 (%)", height=360, hovermode="x unified",
                                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_p, width="stretch")
            pct_col = st.column_config.NumberColumn(format="%+.1f%%")
            st.dataframe(pat["matches"], hide_index=True, width="stretch", column_config={
                "歷史日期": st.column_config.DateColumn(format="YYYY/MM/DD"),
                "相似度": st.column_config.NumberColumn(format="%.0f%%"),
                "當時RSI": st.column_config.NumberColumn(format="%.0f"),
                "1個月後報酬": pct_col, "3個月後報酬": pct_col, "1個月內最深再跌": pct_col, "3個月內最高漲幅": pct_col,
            })

    # -----------------------------------------------------------------------
    with tab_chip:
        chips = res["chips_raw"]
        if chips.empty:
            st.info("這檔股票沒有籌碼資料。美股沒有公開的三大法人與融資資料。")
        else:
            for line in explain_chips(res["chips"], chips):
                st.write(f"• {line}")
            cw = window(chips.reindex(d.index)).dropna(how="all")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                subplot_titles=("三大法人每日買賣超（張）", "融資餘額（張）"))
            for col, name, color in [("foreign", "外資", "#4a90e2"), ("trust", "投信", "#ff8c00"), ("dealer", "自營商", "#9b59b6")]:
                fig.add_trace(go.Bar(x=cw.index, y=cw[col] / 1000, name=name, marker_color=color), row=1, col=1)
            fig.add_trace(go.Scatter(x=cw.index, y=cw["margin_balance"], name="融資餘額", line=dict(color="#e5484d")),
                          row=2, col=1)
            fig.update_layout(height=520, barmode="relative", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, width="stretch")
            st.caption("怎麼看：投信連買或近 3 日法人由賣轉買、同時近 5 日融資大減（散戶退場），代表籌碼沉澱到法人手中；"
                       "反過來，法人賣、融資增加，代表散戶在接，籌碼較亂。")
            st.info("💡 券商分點（主力大戶）資料需要 FinMind 付費方案，目前用「三大法人 + 投信轉折 + 融資」代替。")

    # -----------------------------------------------------------------------
    with tab_fund:
        for line in explain_fund(res["fund"], res["us"]) + explain_value(res["value"], res["us"]):
            st.write(f"• {line}")
        rev, fin, val = res["revenue"], res["fin"], res["value"]
        if not rev.empty:
            r = rev.copy()
            r["yoy"] = (r["revenue"] / r["revenue"].shift(12) - 1) * 100
            r = r.iloc[-36:]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=r["period"], y=r["revenue"] / 1e8, name="月營收（億元）", marker_color="#4a90e2"))
            fig.add_trace(go.Scatter(x=r["period"], y=r["yoy"], name="年增率 %", line=dict(color="#ff8c00")), secondary_y=True)
            fig.update_layout(title="月營收與年增率（近 3 年）", height=360, hovermode="x unified", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
        if not fin.empty and "EPS" in fin:
            q = fin.iloc[-12:]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=q.index, y=q["EPS"], name="單季 EPS（元）", marker_color="#30a46c"))
            if "gross_margin" in q:
                fig.add_trace(go.Scatter(x=q.index, y=q["gross_margin"] * 100, name="毛利率 %", line=dict(color="#9b59b6")),
                              secondary_y=True)
            fig.update_layout(title="單季 EPS 與毛利率（近 3 年）", height=340, hovermode="x unified", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
        if not val.empty and "per" in val and val["per"].notna().any():
            v = val.dropna(subset=["per"]).iloc[-1250:]
            fig = go.Figure(go.Scatter(x=v.index, y=v["per"], name="本益比", line=dict(color="#333")))
            for q_, color in [(0.2, DOWN), (0.5, "#999"), (0.8, UP)]:
                lvl = v["per"].quantile(q_)
                fig.add_hline(y=lvl, line_dash="dot", line_color=color, annotation_text=f"{int(q_ * 100)}% 分位 {lvl:.1f} 倍")
            fig.update_layout(title="本益比走勢（近 5 年）：靠近下緣代表相對便宜", height=320)
            st.plotly_chart(fig, width="stretch")
        if rev.empty and fin.empty and not res["us"]:
            st.info("ETF 沒有營收、財報與本益比，基本面和估值不列入評分。")

    # -----------------------------------------------------------------------
    with tab_news:
        news = res["news_scored"]
        if res["news_overall"] is not None:
            pos = sum(1 for n in news if n.get("sentiment") == 1)
            neg = sum(1 for n in news if n.get("sentiment") == -1)
            st.metric("AI 判讀的整體情緒", f"{res['news_overall']:+.2f}", f"正面 {pos} 則 / 負面 {neg} 則", delta_color="off")
        else:
            st.caption(res["news_status"])
        for n in news:
            emo = {1: "🟥 偏多", -1: "🟩 偏空", 0: "⬜ 中性"}.get(n.get("sentiment"), "")
            reason = f"　_{n['reason']}_" if n.get("reason") else ""
            title = f"[{n['title']}]({n['link']})" if n.get("link") else n["title"]
            st.markdown(f"{emo} `{n['date'][:10]}` {title}{reason}")
        if not news:
            st.info("近幾天沒有找到相關新聞。")
        st.caption("顏色依台股習慣：紅色偏多、綠色偏空。AI 只看標題判斷，可能誤判；消息面不列入回測。")

    # -----------------------------------------------------------------------
    with tab_bt:
        st.markdown("#### 🎯 第一套：拉回買（買進時機分數）")
        st.caption("統計過去每次分數**突破門檻**後，持有 1／3／6 個月的報酬，並和「隨便哪天買」比較。")
        timing_s = res["timing"]
        span = comp["score_bt"].dropna()
        if not timing_s.dropna().empty:
            span = span[span.index >= timing_s.dropna().index[0]]
        pct = st.column_config.NumberColumn(format="%.1f%%")
        if span.empty:
            st.info("資料不足，無法回測。")
        else:
            start = span.index[0]
            stats_t, _ = backtest(d, tech["score"].where(tech.index >= start), threshold)
            stats_m, sigs_m = res["bt_stats"], res["bt_sigs"]
            stats_c, _ = backtest(d, comp["score_bt"].where(comp.index >= start), threshold)
            st.caption(f"回測期間：{start:%Y/%m/%d} ～ {span.index[-1]:%Y/%m/%d}（約 {len(span) / 252:.1f} 年），門檻 {threshold} 分")
            cfg = {k: pct for k in ["訊號勝率", "訊號平均報酬", "訊號報酬中位數", "隨便哪天買的勝率", "隨便哪天買的平均報酬"]}
            cols3 = ["持有期間", "訊號次數", "訊號勝率", "訊號平均報酬", "隨便哪天買的勝率", "隨便哪天買的平均報酬"]
            has_chips = chips_ok_series(res) is not None
            st.markdown("**⭐ 目前使用：技術與波浪 + 籌碼濾網**" if has_chips else "**⭐ 目前使用：技術與波浪**（這檔沒有籌碼資料）")
            st.dataframe(stats_m[cols3], hide_index=True, width="stretch", column_config=cfg)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**對照 ①：只看技術與波浪**")
                st.dataframe(stats_t[cols3], hide_index=True, width="stretch", column_config=cfg)
            with c2:
                st.markdown("**對照 ②：五面向加權合併（綜合評估）**")
                st.dataframe(stats_c[cols3], hide_index=True, width="stretch", column_config=cfg)

            def verdict(s: pd.DataFrame) -> str:
                r = s.iloc[1]
                if r["訊號次數"] < 5:
                    return f"樣本太少（{int(r['訊號次數'])} 次）"
                better = r["訊號平均報酬"] > r["隨便哪天買的平均報酬"] and r["訊號勝率"] > r["隨便哪天買的勝率"]
                return "✅ 優於隨便買" if better else "➖ 沒有明顯優於隨便買"

            st.markdown(f"**3 個月結論**：⭐ 目前使用 {verdict(stats_m)}；① 只看技術與波浪 {verdict(stats_t)}；② 加權合併 {verdict(stats_c)}")
            if not sigs_m.empty and "動態停損停利報酬(60日內)" in sigs_m:
                dyn_s = sigs_m["動態停損停利報酬(60日內)"].dropna()
                if len(dyn_s):
                    st.caption(f"🛡️ 若搭配 ATR 動態風控出場（初始 2.2×ATR 停損 + 獲利後 2.8×ATR 移動停利，最長 60 日）："
                               f"共 {len(dyn_s)} 次，勝率 **{(dyn_s > 0).mean() * 100:.1f}%**、平均報酬 **{dyn_s.mean():+.1f}%**。")
            with st.expander(f"📋 歷次拉回買進訊號明細（共 {len(sigs_m)} 次）"):
                st.dataframe(sigs_m.sort_values("日期", ascending=False), hide_index=True, width="stretch", column_config={
                    "日期": st.column_config.DateColumn(format="YYYY/MM/DD"),
                    "買進價": st.column_config.NumberColumn(format="%.2f"),
                    "分數": st.column_config.NumberColumn(format="%.0f"),
                    **{f"{n}後報酬": pct for n in HORIZONS.values()},
                    "動態停損停利報酬(60日內)": pct,
                })

        st.divider()
        st.markdown("#### 🕯️ 第二套：有效K棒起漲確認（2016 年起）")
        st.caption("進場日 = 起漲確認那天（突破K高，或回測實體1/2不破再轉強）的收盤價；和同一段期間「隨便哪天買」比較。")
        kcfg = {k: pct for k in ["訊號勝率", "訊號平均報酬", "隨便哪天買的勝率", "隨便哪天買的平均報酬", "停損距離"]}
        for title, (k_stats, k_per, k_sigs), win_name in [
            ("🚶 中線：日線有效紅K + 年線向上", kb["bt_mid"], "20 日內跌破K低"),
            ("🧘 長線：週線有效紅K + 40週線向上", kb["bt_long"], "4 週內跌破K低"),
        ]:
            st.markdown(f"**{title}**")
            bc1, bc2 = st.columns([3, 2])
            bc1.dataframe(k_stats, hide_index=True, width="stretch", column_config=kcfg)
            bc2.dataframe(k_per[["期間", "持有期間", "訊號次數", "訊號平均報酬", "隨便哪天買的平均報酬"]],
                          hide_index=True, width="stretch", column_config=kcfg)
            if len(k_sigs):
                hit = k_sigs["期間內跌破K低"].dropna()
                st.caption(f"確認後 {win_name} 的比例：{hit.mean() * 100:.0f}%；平均停損距離（進場價到 K低）：{k_sigs['停損距離'].mean():.1f}%")
                with st.expander(f"📋 有效K起漲確認明細（共 {len(k_sigs)} 次）"):
                    st.dataframe(k_sigs.sort_values("確認日期", ascending=False), hide_index=True, width="stretch", column_config={
                        "有效K日期": st.column_config.DateColumn(format="YYYY/MM/DD"),
                        "確認日期": st.column_config.DateColumn(format="YYYY/MM/DD"),
                        "進場價": st.column_config.NumberColumn(format="%.2f"),
                        "K低停損": st.column_config.NumberColumn(format="%.2f"),
                        **{c_: pct for c_ in k_sigs.columns if c_.endswith("後報酬") or c_ == "停損距離"},
                    })
        st.info(
            "📚 **14 檔台股大型股、2016 年起的整體研究**（超額 = 訊號報酬 − 同一檔隨便哪天買）\n\n"
            "- **長線（週線）最穩定**：持有 1 季 120 次，勝率 69.2%（隨便買 63.5%），平均超額 +8.5%；2016–2020 +5.2%、2021 年後 +10.3%。\n"
            "- **中線（日線）不穩定**：持有 3 個月平均超額 +3.9%，但勝率沒有比隨便買高；2016–2020 只有 +0.6%。\n"
            "- **一定要等確認**：有效K當天就買，約 55% 會在 20 天內跌破K低；等確認後降到約 38%。\n"
            "- **短線（1 週）沒有優勢**；**有效黑K不是賣出訊號**（1–3 個月沒有一致轉弱）。\n\n"
            "⚠️ 研究比較了約 15 種組合，挑最好的有事後挑選偏誤；回測沒有計入手續費、稅與滑價。"
        )

# ===========================================================================
# 🚀 起漲雷達
# ===========================================================================
with tab_scan:
    st.markdown("#### 🚀 起漲雷達：現在「準備啟動」或「剛啟動」的股票（不用先加入觀察名單）")
    sc1, sc2 = st.columns([3, 1.2], vertical_alignment="bottom")
    n_uni = sc2.selectbox("掃描範圍", [100, 200, 300], index=1, format_func=lambda n: f"成交值前 {n} 大", key="scan_n")
    scan_df, meta = load_last_scan()
    run = sc2.button("🔄 立即重新掃描" if scan_df is not None else "🔍 開始掃描", type="primary", width="stretch")
    if run:
        bar = st.progress(0.0, text="取得股票清單…")
        try:
            scan_df, meta = scan(n_uni, threshold, progress=lambda p, m: bar.progress(min(float(p), 1.0), text=m))
        except Exception as e:
            st.error(f"掃描失敗：{str(e)[:150]}")
        bar.empty()
    if scan_df is None or meta is None:
        sc1.info("還沒有今天的掃描結果。按右邊「🔍 開始掃描」，約 20–60 秒。")
    else:
        sc1.caption(
            f"上次掃描：{meta['time']}｜資料日期 {meta['data_date']}｜掃描上市櫃普通股成交值前 {meta['universe']} 大，"
            f"找到 **{len(scan_df)}** 檔｜拉回買門檻 {meta['threshold']} 分"
        )
        if len(scan_df) == 0:
            st.info("目前沒有股票符合條件（大盤急跌時很常見）。")
        else:
            filt = st.segmented_control("篩選", ["全部", "剛啟動", "準備啟動", "拉回買點"], default="全部",
                                        key="scan_filter", label_visibility="collapsed")
            view = scan_df if filt in (None, "全部") else scan_df[scan_df["類型"] == filt]
            view = view.copy()
            in_list = {stocks[k]["ticker"].split(".")[0] for k in stocks} | set(ss.watchlist)
            view.insert(0, "名單", np.where(view["代號"].isin(in_list), "✓", ""))
            show_cols = ["名單", "代號", "名稱", "類型", "訊號", "收盤", "漲跌%", "成交值(億)", "K高", "K低(停損)", "距K低%",
                         "買進時機", "波浪階段", "波浪情境"]
            event = st.dataframe(
                view[show_cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                key="scan_table",
                column_config={
                    "收盤": st.column_config.NumberColumn(format="%.2f"),
                    "漲跌%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "成交值(億)": st.column_config.NumberColumn(format="%.1f"),
                    "K高": st.column_config.NumberColumn(format="%.2f"),
                    "K低(停損)": st.column_config.NumberColumn(format="%.2f"),
                    "距K低%": st.column_config.NumberColumn(format="%+.1f%%"),
                    "買進時機": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
                },
            )
            rows_sel = event.selection.rows if event and event.selection else []
            codes = view.iloc[rows_sel]["代號"].tolist() if rows_sel else []
            def add_from_scan(cs: list[str]) -> None:
                add_items(cs)
                st.toast(f"已加入 {'、'.join(cs)}，切到「🔍 個股分析」就能看")

            st.button(
                f"➕ 把勾選的 {len(codes)} 檔加入觀察名單" if codes else "👆 先勾選表格左側的方框，再按這裡加入觀察名單",
                disabled=not codes, type="primary", on_click=add_from_scan, args=(codes,),
            )
    st.info(
        "**怎麼分類**（和個股分析頁同一套規則）\n\n"
        "- **剛啟動**：🚶 中線（日線有效紅K + 年線向上）5 天內起漲確認，或 🧘 長線（週線有效紅K + 40 週線向上）2 週內起漲確認\n"
        "- **準備啟動**：有效紅K已出現，正在等待確認（站上 K高，或回測 1/2 不破再轉強）；收盤跌破 K低 就失效\n"
        "- **拉回買點**：技術與波浪「買進時機」達門檻且已止跌\n\n"
        "排序：長線訊號、兩套訊號同時出現的排前面，再依成交值。\n\n"
        "⚠️ 掃描時沒有抓籌碼資料（FinMind 免費額度不夠掃幾百檔），所以拉回買點**未含籌碼濾網**；加入觀察名單後會完整分析。"
        "回測顯示長線訊號最穩定、中線不穩定，這份名單是「值得研究的候選」，不是買進建議。"
    )

# ===========================================================================
# 🏠 名單總覽與比較
# ===========================================================================
with tab_home:
    if mkt.get("regime_desc"):
        st.info(f"**🌦️ 大盤氣象站 — {mkt['regime']}**：{mkt['regime_desc']}")
    cols = st.columns(min(len(stocks), 4))
    for i, (item, r) in enumerate(stocks.items()):
        dd = r["d"]
        t_, lab, nts, ov = current(r)
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"#### {r['name']}")
            c_, p_ = dd["Close"].iloc[-1], dd["Close"].iloc[-2]
            st.metric(f"收盤（{dd.index[-1]:%m/%d}）", f"{c_:,.2f}", f"{(c_ / p_ - 1) * 100:+.2f}%", delta_color="inverse")
            if pd.isna(t_):
                st.info("上市時間太短，資料不足以評分")
                continue
            st.progress(t_ / 100, text=f"🎯 買進時機 **{t_:.0f}** / 100")
            st.markdown(f"**{lab[1]}**")
            for n in nts:
                st.caption(f"↳ {n}")
            st.caption(f"🌊 {r['waves']['wave_stage']}")
            st.caption(f"🕯️ 中線：{r['keybar']['mid']['title']}｜長線：{r['keybar']['long']['title']}")
            st.caption(r["market"]["rs_desc"])
            st.caption(f"🏅 綜合評估 {ov:.0f} / 100")
            st.plotly_chart(radar(factor_summary(r["comp"])), width="stretch", key=f"radar_{item}")

    if len(stocks) >= 2:
        st.divider()
        st.markdown("##### ⚖️ 各檔比較")
        rows = {}
        for item, r in stocks.items():
            f = factor_summary(r["comp"])
            t_, lab, _, ov = current(r)
            rows[r["name"]] = {
                "買進時機": t_, "建議": lab[1] if lab else "-", "波浪階段": r["waves"]["wave_stage"],
                "中線有效K": r["keybar"]["mid"]["title"], "長線有效K": r["keybar"]["long"]["title"],
                "相對強弱RS": r["market"]["rs_score"], "綜合評估": ov,
                **{FACTOR_NAMES[k]: (None if v is None else v * 100) for k, v in f.items()},
            }
        st.dataframe(pd.DataFrame(rows).T, width="stretch", column_config={
            "買進時機": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            "相對強弱RS": st.column_config.NumberColumn(format="%d"),
            "綜合評估": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            **{FACTOR_NAMES[k]: st.column_config.NumberColumn(format="%.0f%%") for k in WEIGHTS},
        })
        closes = pd.DataFrame({r["name"]: window(r["d"])["Close"] for r in stocks.values()})
        norm = closes.apply(lambda s: s / s.dropna().iloc[0] * 100)
        fig = px.line(norm, labels={"value": "累積走勢（起點 = 100）", "index": "", "variable": ""})
        fig.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.markdown(f"##### 近 {ss.get('period') or '1年'} 走勢比較（起點都換算成 100）")
        st.plotly_chart(fig, width="stretch")
        perf = pd.DataFrame({n: performance(closes[n]) for n in closes}).T
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(perf, width="stretch", column_config={
            "總報酬": pct, "年化報酬": pct, "年化波動度": pct, "最大回撤": pct,
            "報酬/風險比": st.column_config.NumberColumn(format="%.2f"),
        })
        corr = closes.pct_change().corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=120 + 60 * len(corr), title="相關性（越接近 1，漲跌越同步，分散風險效果越差）")
        st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 📖 說明
# ===========================================================================
with tab_help:
    st.markdown(f"""
### 怎麼使用
1. **新增股票**：在最上面的搜尋框輸入代號或名稱（例如 `2330`、`台積電`、`長榮航`、`NVDA`），按 Enter。
2. **切換股票**：點搜尋框下面的股票按鈕。網址會記住你的名單，加入書籤下次打開還在。
3. **移除股票**：個股標題列的「🗑️ 移除」，或「⚙️ 設定」裡的名單管理。
4. **先看圖、再看細節**：K 線圖下方的「📌 看圖重點」整理了最重要的資訊；細節在下面的分頁。
5. **找名單外的機會**：「🚀 起漲雷達」掃描成交值前 100～300 大的上市櫃股票。

### 兩套買點訊號
| 訊號 | 邏輯 | 用途 |
|---|---|---|
| 🎯 **買進時機**（第一套） | 長多**拉回**、黃金分割回檔、先超賣後止跌 + 籌碼濾網 | 喜歡買在相對低檔 |
| 🕯️ **有效K起漲**（第二套） | **放量長紅**後確認起漲（突破追強） | 喜歡等轉強才進場 |

另有 🏅 **綜合評估**（技術、籌碼、基本面、估值、消息五面向）用來看公司體質與風險，不決定買進建議。

### 第一套：買進建議怎麼決定
1. **技術與波浪分數**達到門檻（目前 {threshold} 分）：年線向上、回吐前波漲幅 32%～68%（黃金分割區）、近 10 日曾超賣、今天止跌轉強
2. **止跌閘門**：止跌訊號不到 {MIN_REVERSAL} 分時，分數最高 {NO_REVERSAL_CAP} 分，避免接刀
3. **籌碼濾網**（台股）：籌碼分數至少 {CHIPS_MIN} / 25，沒過就降一級
4. **第 5 波頂背離或風險報酬比 < {RR_MIN}**：降一級避免追高

| 分數 | 建議 |
|---|---|
| ≥ {threshold} | ✅ 符合買進條件 |
| {threshold - 15}–{threshold - 1} | 🟡 可小量分批（還沒止跌則顯示 👀 列入觀察） |
| {threshold - 30}–{threshold - 16} | ⏸️ 中性觀望 |
| < {threshold - 30} | ⛔ 暫不建議進場 |

### 第二套：有效K棒起漲確認
| 條件 | 日線（短中線） | 週線（長線） |
|---|---|---|
| 收紅 K | 收盤 > 開盤 | 同左 |
| 漲幅夠大 | ≥ {DAILY['min_ret']:.0%}，且 ≥ {DAILY['atr_k']} 倍平常波動（ATR） | ≥ {WEEKLY['min_ret']:.0%}，且 ≥ {WEEKLY['atr_k']} 倍週 ATR |
| 實體飽滿 | 實體 ≥ 高低差的 {DAILY['body_min']:.0%} | ≥ {WEEKLY['body_min']:.0%} |
| 有量 | 成交量 ≥ 20 日均量 {DAILY['vol_k']} 倍 | ≥ 20 週均量 {WEEKLY['vol_k']} 倍 |

**三個關鍵價位**：K高（突破＝確認起漲）、實體 1/2（回測買點）、K低（收盤跌破＝失效／停損）。
**起漲確認**（日線 {DAILY['wait']} 天、週線 {WEEKLY['wait']} 週內）：收盤突破 K高，或拉回實體 1/2 守住 K低、再收紅並過前一根高點。

| 週期 | 訊號 | 用途 |
|---|---|---|
| 🏃 短線 | 日線有效紅K | **只當停損停利價位參考**（回測沒有統計優勢） |
| 🚶 中線 | 日線有效紅K + 年線向上 → 起漲確認 | 買點，停損 K低（優勢不穩定） |
| 🧘 長線 | 週線有效紅K + 40 週線向上 → 起漲確認 | 買點，回測最穩定 |

**波浪理論**只用來標註有效K出現的情境（例如第 ② 波後站上 ① 波高點＝第 ③ 波起漲），不加分。
**有效黑K**只是短線風險提示，不是賣出訊號。

### 🚀 起漲雷達
每次掃描上市櫃普通股中**當日成交值前 100～300 大**（排除 ETF、權證），用和個股頁相同的規則找出「剛啟動」「準備啟動」「拉回買點」。
為了速度，掃描不抓籌碼資料，所以拉回買點未含籌碼濾網。

### 🤖 AI 分析師
把所有數據整理成 JSON 交給 Gemini，請它用白話寫出結論、波浪解讀、短中長線有效K、多空辯論和 If-Then 劇本。
AI 被要求**只能引用提供的數據**；目前模型 `{GEMINI_MODEL}`，無法使用時會自動退回其他 Flash 模型。

### 限制
- 券商分點（主力）資料需要付費，目前用三大法人 + 融資代替。
- 模型無法預測突發事件；回測沒有計入手續費、稅和滑價；過去有效不代表未來有效。
- ⚠️ 僅供學習參考，不構成投資建議。
""")
