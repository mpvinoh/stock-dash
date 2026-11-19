"""股票分析小幫手 v3.4（台股 × 美股通用版・看圖優先介面）

版面：最上方「搜尋／快選／觀察名單」→ 個股狀態卡與 K 線主圖 → 30 秒看懂結論 → 細節分頁；另有「🚀 起漲雷達（台股／美股）」。
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
from factors import FACTOR_NAMES, WEIGHTS, explain_chips, explain_fund, explain_us_flow, explain_value
from keybar import DAILY, WEEKLY
from model import HORIZONS, MIN_REVERSAL, NO_REVERSAL_CAP, backtest, explain, performance
from scanner import fetch_quotes, load_last_scan, scan, scan_us
from sources import is_taiwan

st.set_page_config(page_title="股票分析小幫手", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    .block-container{padding-top:1.1rem;padding-bottom:2.2rem;max-width:1360px}
    div[data-testid='stPills'] button{font-size:0.92rem;border-radius:999px}
    div[data-testid='stMetricValue']{font-size:1.55rem}
    .stTabs [data-baseweb="tab-list"]{gap:0.35rem}
    .stTabs [data-baseweb="tab"]{padding:0.45rem 0.9rem;border-radius:8px 8px 0 0}
    @media (max-width:640px){
        .block-container{padding-left:0.65rem;padding-right:0.65rem;padding-top:0.8rem}
        div[data-testid='stMetricValue']{font-size:1.25rem}
    }
    </style>""",
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
DEFAULT_LIST = ["0050", "2330", "NVDA", "AAPL"]
PRESETS = {
    "🌐 台美精選": ["0050", "2330", "NVDA", "AAPL"],
    "🇹🇼 台股預設": ["0050", "00981A", "2330", "2327"],
    "🇺🇸 美股科技": ["NVDA", "AAPL", "MSFT", "TSLA", "SPY"],
}
QUICK_TW = [("0050", "0050 元大台灣50"), ("2330", "2330 台積電"), ("2317", "2317 鴻海"),
            ("2454", "2454 聯發科"), ("2308", "2308 台達電"), ("2327", "2327 國巨"),
            ("00981A", "00981A 主動統一"), ("2603", "2603 長榮")]
QUICK_US = [("NVDA", "NVDA 輝達"), ("AAPL", "AAPL 蘋果"), ("MSFT", "MSFT 微軟"),
            ("GOOGL", "GOOGL Google"), ("AMZN", "AMZN 亞馬遜"), ("META", "META Meta"),
            ("AVGO", "AVGO 博通"), ("TSLA", "TSLA 特斯拉"), ("TSM", "TSM 台積電ADR"),
            ("PLTR", "PLTR Palantir"), ("SPY", "SPY 標普500"), ("QQQ", "QQQ 那斯達克")]
PERIODS = {"3個月": 3, "6個月": 6, "1年": 12, "2年": 24, "5年": 60}


# ---------------------------------------------------------------------------
# 資料載入（快取）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load(user_input: str, _token: str, version: str = "v3.4"):
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
# 狀態管理（名單存在網址裡，加入書籤可保留）
# ---------------------------------------------------------------------------
ss = st.session_state
ensure_names()
if "watchlist" not in ss:
    qp = st.query_params.get("s", "")
    ss.watchlist = parse_inputs(qp) if qp else DEFAULT_LIST.copy()
ss.setdefault("threshold", 70)
ss.setdefault("period", "1年")
ss.setdefault("color_scheme", "tw")


def add_items(items: list[str]) -> None:
    for it in items:
        if it not in ss.watchlist:
            ss.watchlist.append(it)
    if items:
        ss.focus_pill = items[-1]


def apply_preset(items: list[str]) -> None:
    ss.watchlist = items.copy()
    ss.focus_pill = items[0]


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
# 頂部導覽列：標題 + 熱門快選 + 設定
# ---------------------------------------------------------------------------
h_left, h_quick, h_set = st.columns([4.4, 1.3, 1.0], vertical_alignment="bottom")
h_left.markdown("## 📈 股票分析小幫手　`🇹🇼 台股 × 🇺🇸 美股`")

with h_quick.popover("⚡ 熱門快選", width="stretch"):
    st.markdown("**一鍵切換預設組合**")
    pc = st.columns(len(PRESETS))
    for col, (pname, plist) in zip(pc, PRESETS.items()):
        col.button(pname, key=f"preset_{pname}", on_click=apply_preset, args=(plist,), width="stretch")
    st.divider()
    st.markdown("**🇹🇼 點一下加入台股**")
    qc_tw = st.columns(2)
    for idx, (code, lab) in enumerate(QUICK_TW):
        qc_tw[idx % 2].button(lab, key=f"q_tw_{code}", on_click=add_items, args=([code],), width="stretch")
    st.markdown("**🇺🇸 點一下加入美股**")
    qc_us = st.columns(2)
    for idx, (code, lab) in enumerate(QUICK_US):
        qc_us[idx % 2].button(lab, key=f"q_us_{code}", on_click=add_items, args=([code],), width="stretch")

with h_set.popover("⚙️ 設定", width="stretch"):
    st.slider(
        "買進門檻分數", min_value=65, max_value=90, step=5, key="threshold",
        help="「買進時機」（技術與波浪 + 籌碼濾網）達到門檻才算「符合買進條件」。",
    )
    st.radio(
        "漲跌配色習慣",
        options=["tw", "us"],
        format_func=lambda x: "🔴 紅漲綠跌（台股習慣）" if x == "tw" else "🟢 綠漲紅跌（美股習慣）",
        key="color_scheme",
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
        help="雲端多人共用導致免費額度暫時滿載時，可貼上自己的 Gemini API Key（只存在本次瀏覽器連線）。",
    )
    GEMINI_KEY = ss.get("user_gemini_key", "").strip() or DEFAULT_GEMINI_KEY
    st.caption(f"🤖 AI 分析：{'✅ 已啟用（' + GEMINI_MODEL + '）' if GEMINI_KEY else '⚠️ 未設定金鑰（改用制式摘要）'}")
    st.caption("資料來源：Yahoo Finance（台美股價、美股季報與新聞）、證交所／櫃買、FinMind（台股籌碼與財報）。")

# 依配色設定決定漲跌顏色
if ss.color_scheme == "us":
    UP, DOWN, DELTA_COLOR = "#16a34a", "#dc2626", "normal"
    POS_ICON, NEG_ICON = "🟩 偏多", "🟥 偏空"
else:
    UP, DOWN, DELTA_COLOR = "#e5484d", "#30a46c", "inverse"
    POS_ICON, NEG_ICON = "🟥 偏多", "🟩 偏空"

st.text_input(
    "新增股票",
    key="add_box",
    on_change=on_add_box,
    placeholder="🔍 輸入台股或美股代號／中文名稱，按 Enter 加入（例：2330、台積電、NVDA、輝達、AAPL、SPY；可空白分隔多檔）",
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
    progress.progress(i / max(len(ss.watchlist), 1), text=f"正在分析 {item}（股價、波浪、有效K、籌碼／財報與新聞）…")
    try:
        r = load(item, FINMIND_TOKEN)
    except Exception as e:
        errors.append(f"找不到「{item}」或暫時抓不到資料（{str(e)[:60]}），已從名單移除。")
        ss.watchlist.remove(item)
        continue
    if r["ticker"] in seen_tickers:
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
        f"⚠️ FinMind 免費查詢次數已達每小時上限，{'、'.join(limited)} 的部分籌碼／財報／新聞改用快取或暫缺。"
        "可稍後至「⚙️ 設定」重新抓取，或填入免費的 FINMIND_TOKEN。"
    )
if news_models and news_models != {GEMINI_MODEL}:
    st.caption(f"⚠️ AI 目前用備援模型（{'、'.join(sorted(news_models))}），設定的 {GEMINI_MODEL} 暫時無法使用。")
if not stocks:
    st.info("👆 請在上面的搜尋框輸入股票代號，或點右上角「⚡ 熱門快選」加入股票。")
    st.stop()


def current(res: dict):
    """回傳（買進時機分數、建議、備註、綜合評估分數）。"""
    timing = res["tech"]["score"].iloc[-1]
    overall = res["comp"]["score"].iloc[-1]
    if pd.isna(timing):
        return timing, None, [], overall
    label, notes = final_label(res, threshold)
    return timing, label, notes, overall


def flag_of(ticker: str) -> str:
    return "🇹🇼" if is_taiwan(ticker) else "🇺🇸"


def cur_unit(ticker: str) -> str:
    return "元" if is_taiwan(ticker) else "美元"


def pill_label(item: str) -> str:
    r = stocks[item]
    _, lab, _, _ = current(r)
    emoji = lab[1].split(" ")[0] if lab else "▫️"
    c = r["d"]["Close"]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    kb = r["keybar"]
    fire = " 🕯️" if "confirmed" in (kb["mid"]["code"], kb["long"]["code"]) else ""
    return f"{flag_of(r['ticker'])}{emoji} {r['name']} {chg:+.1f}%{fire}"


# ---------------------------------------------------------------------------
# 觀察名單藥丸列（點一下即切換）
# ---------------------------------------------------------------------------
opts = list(stocks)
if ss.get("focus_pill") not in opts:
    ss.focus_pill = ss.get("focus_last") if ss.get("focus_last") in opts else opts[0]
picked = st.pills(
    "觀察名單", opts, format_func=pill_label, key="focus_pill", selection_mode="single", label_visibility="collapsed",
)
focus = picked if picked in opts else ss.get("focus_last") if ss.get("focus_last") in opts else opts[0]
ss.focus_last = focus
st.caption("燈號：✅ 符合買進　🟡 小量分批　👀 觀察等止跌　⏸️ 中性　⛔ 暫不建議　🕯️ 有效K已確認起漲　｜　網址會自動記住你的名單")


def window(df: pd.DataFrame) -> pd.DataFrame:
    months = PERIODS.get(ss.get("period") or "1年", 12)
    return df[df.index >= df.index.max() - pd.DateOffset(months=months)]


def radar(fs: dict, height: int = 200) -> go.Figure:
    keys = [k for k in WEIGHTS if fs[k] is not None]
    vals = [fs[k] * 100 for k in keys]
    labels = [FACTOR_NAMES[k].split(" ")[1] for k in keys]
    fig = go.Figure(
        go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself", line_color="#ff8c00")
    )
    fig.update_layout(
        height=height, margin=dict(t=18, b=18, l=28, r=28), showlegend=False,
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
is_tw = is_taiwan(res["ticker"])
unit = cur_unit(res["ticker"])

main_tabs = st.tabs(["🔍 個股分析", "🚀 起漲雷達（台股／美股）", "🏠 名單總覽與比較", "📖 使用說明"])
tab_stock, tab_scan, tab_home, tab_help = main_tabs

# ===========================================================================
# 🔍 個股分析：先看狀態卡與 K 線主圖，再看 30 秒結論與分層細節
# ===========================================================================
with tab_stock:
    close, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
    chg_pct = (close / prev - 1) * 100

    # ---- 1. 精簡清爽的個股標題與 4 張重點卡 ----
    with st.container(border=True):
        top_l, top_r = st.columns([6, 1], vertical_alignment="center")
        mkt_tag = "🇹🇼 台股" if is_tw else "🇺🇸 美股"
        top_l.markdown(f"### {flag_of(res['ticker'])} {res['name']}　<span style='font-size:0.9rem;color:#666'>{mkt_tag}・幣別：{unit}・資料日 {d.index[-1]:%Y/%m/%d}</span>", unsafe_allow_html=True)
        if len(ss.watchlist) > 1:
            top_r.button("🗑️ 移除", key=f"rm_{focus}", on_click=remove_item, args=(focus,), width="stretch")

        m1, m2, m3, m4 = st.columns([1.1, 1.25, 1.55, 1.45])
        m1.metric("最新收盤", f"{close:,.2f} {unit}", f"{chg_pct:+.2f}%", delta_color=DELTA_COLOR)
        if pd.isna(timing):
            m2.info("上市太短，尚無法評分")
        else:
            m2.metric("🎯 拉回買（買進時機）", f"{timing:.0f} / 100", label[1], delta_color="off")
        m3.markdown(f"**🕯️ 有效K（突破型）**  \n中線：{kb['mid']['title']}  \n長線：{kb['long']['title']}")
        fib_mini = f"（回吐 {wv['fib']['retrace_pct']:.0f}%）" if wv.get("fib") else ""
        m4.markdown(f"**🌊 波浪 & 大盤**  \n{wv['wave_stage']}  \n{mkt['regime']}｜RS {mkt['rs_score']}{fib_mini}")

    # ---- 2. K 線主圖（最明顯、看圖優先） ----
    g0, g1, g2, g3, g4 = st.columns([2.4, 1, 1, 1, 1.2], vertical_alignment="center")
    g0.segmented_control("期間", list(PERIODS), key="period", label_visibility="collapsed")
    show_wave = g1.checkbox("🌊 波浪標記", value=True)
    show_fib = g2.checkbox("📐 黃金分割", value=True)
    show_kb = g3.checkbox("🕯️ 有效K棒", value=True)
    show_sr = g4.checkbox("🧱 支撐壓力線", value=False)

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
                mode="lines", name="波浪折線", line=dict(color="#6366f1", width=2.0, dash="dashdot"),
            ), row=1, col=1)
            for p in vis:
                is_high = p["kind"] == "H"
                fig.add_annotation(
                    x=p["date"], y=p["price"], text=f"<b>{p.get('label') or ('高' if is_high else '低')}</b><br>{p['price']:,.1f}",
                    showarrow=True, arrowhead=2, arrowsize=0.85, arrowwidth=1.4,
                    arrowcolor="#db2777" if is_high else "#4f46e5", ax=0, ay=-30 if is_high else 30,
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#db2777" if is_high else "#4f46e5", borderwidth=1.1,
                    font=dict(size=10.5, color="#1e1b4b"), row=1, col=1,
                )

    if show_fib and wv.get("fib"):
        fib = wv["fib"]
        for f_price, f_name, f_col in [
            (fib["zone_382"], "黃金 38.2%", "#0284c7"), (fib["zone_500"], "黃金 50%", "#d97706"),
            (fib["zone_618"], "黃金 61.8%", "#059669"), (fib["target_1272"], "延伸 127.2%", "#7c3aed"),
        ]:
            fig.add_hline(y=f_price, line_dash="dash", line_color=f_col, line_width=1.1, opacity=0.78, row=1, col=1,
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
            marker=dict(symbol="triangle-up", size=11, color="#ff8c00"),
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
        add_marks([e["date"] for e in ek], [e["high"] * 1.02 for e in ek], "有效紅K", "star", "#facc15", 12,
                  [f"有效紅K {e['ret'] * 100:+.1f}%、量 {e['vol_ratio']:.1f} 倍" for e in ek])
        fails = [e for e in ek if e["status"] == "failed"]
        add_marks([d.index[e["end_idx"]] for e in fails], [d["Close"].iloc[e["end_idx"]] for e in fails],
                  "有效K失效", "x", "#6b7280", 9)
        conf = [e for e in kb["events_mid"] if e["status"] == "confirmed" and e["confirm_date"] >= w0]
        add_marks([e["confirm_date"] for e in conf], [d.loc[e["confirm_date"], "Low"] * 0.95 for e in conf],
                  "中線起漲確認", "diamond", "#9333ea", 11, [e["how"] for e in conf])
        lconf = []
        for e in kb["events_long"]:
            if e["status"] == "confirmed":
                dd_ = d.index[d.index <= e["confirm_date"]]
                if len(dd_) and dd_[-1] >= w0:
                    lconf.append(dd_[-1])
        add_marks(lconf, [d.loc[x, "Low"] * 0.92 for x in lconf], "長線週K起漲確認", "hexagram", "#b91c1c", 14)
        bk = [x for x in kb["black_dates"] if x >= w0]
        add_marks(bk, [d.loc[x, "High"] * 1.03 for x in bk], "有效黑K", "triangle-down", DOWN, 9)
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
    fig.update_layout(height=690, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, font=dict(size=11)), margin=dict(t=36, b=8, l=8, r=8),
                      bargap=0.05)
    fig.layout.yaxis2.matches = "y"
    fig.layout.yaxis2.showticklabels = False
    fig.layout.xaxis2.showticklabels = False
    fig.layout.xaxis3.matches = "x"
    fig.layout.xaxis4.matches = "x"
    lo, hi = w["Low"].min(), w["High"].max()
    if show_fib and wv.get("fib"):
        hi = max(hi, min(wv["fib"]["target_1272"], hi * 1.25) * 0.98)
    fig.layout.yaxis.range = [lo * (0.91 if show_kb else 0.94), hi * 1.05]
    fig.layout.yaxis4.range = [0, 100]
    for ax in ("xaxis", "xaxis3", "xaxis4"):
        fig.layout[ax].rangebreaks = [dict(bounds=["sat", "mon"])]
    st.plotly_chart(fig, width="stretch")

    # ---- 3. 30 秒看懂結論（3 欄卡片，一眼掃完） ----
    s1, s2, s3 = st.columns(3)
    with s1.container(border=True):
        st.markdown("**🎯 訊號與位置**")
        if not pd.isna(timing):
            st.caption(f"• {explain(d, tech)['scenario']}")
        if wv.get("fib"):
            fb = wv["fib"]
            st.caption(f"• 前波 {fb['low_price']:,.1f} → {fb['high_price']:,.1f}（已回吐 **{fb['retrace_pct']:.0f}%**）")
            st.caption(f"• 黃金支撐 38.2% `{fb['zone_382']:,.1f}` ／ 61.8% `{fb['zone_618']:,.1f}`")
        if kb["mid"]["code"] != "none":
            st.caption(f"• 中線有效K：{kb['mid']['title']}")
    with s2.container(border=True):
        st.markdown("**🧱 關鍵價位與停損**")
        sup = f"{sr['support'][0]['price']:,.2f}" if sr["support"] else "無明顯支撐"
        resi = f"{sr['resistance'][0]['price']:,.2f}" if sr["resistance"] else "無明顯壓力"
        rr_txt = f"{sr['rr']:.2f}" if sr["rr"] is not None else "-"
        st.caption(f"• 最近支撐：**{sup}**　｜　最近壓力：**{resi}**（風報比 `{rr_txt}`）")
        st.caption(f"• ATR 初始停損：**{sr['suggest_stop']:,.2f} {unit}**（{sr['stop_pct']:+.1f}%）")
        st.caption(f"• Chandelier 移動停利：**{sr['chandelier']:,.2f} {unit}**")
    with s3.container(border=True):
        st.markdown("**🌦️ 大盤環境與風險提醒**")
        st.caption(f"• {mkt['regime']}｜{mkt['rs_desc']}")
        warns = (explain(d, tech)["penalty"] if not pd.isna(timing) else []) + notes
        if kb["black"]:
            warns.append(f"{kb['black']['date']:%m/%d} 有效黑K（{kb['black']['ret']:+.1f}%），短線風險提示")
        if wv.get("divergence"):
            warns.append(wv["divergence"]["title"])
        for w_ in warns[:3]:
            st.caption(f"• {w_}")
        if not warns:
            st.caption("• 目前無特殊扣分或過熱警訊")

    st.caption(
        "圖例：🔺 拉回買進訊號　⭐ 有效紅K　◆ 中線起漲確認　✡ 長線週K起漲確認　✕ 失效　▼ 有效黑K　｜　①②③④⑤／ⓐⓑⓒ 波浪轉折"
    )

    # ---- 4. 分層細節分頁（最受歡迎的 AI 與三週期雷達放第一頁） ----
    sub = st.tabs([
        "🤖 AI 分析 & 買點雷達",
        "📊 五面向 & 支撐壓力",
        "🌊 波浪與相似K線",
        "🏦 籌碼與資金流",
        "📑 基本面與估值",
        "📰 近期新聞",
        "🧪 歷史回測",
    ])
    tab_ai, tab_score, tab_wave, tab_chip, tab_fund, tab_news, tab_bt = sub

    # -----------------------------------------------------------------------
    # 子頁 1：AI 分析師 + 三週期買點雷達 + 風險部位計算機
    # -----------------------------------------------------------------------
    with tab_ai:
        if pd.isna(timing):
            st.info("這檔上市時間太短（少於約半年），暫時無法啟用 AI 完整評分分析。")
        else:
            with st.container(border=True):
                st.markdown("### 🤖 AI 分析師（波浪解讀 × 有效K × 多空辯論 × 實戰劇本）")
                ctx = build_context(res, comp, label[1], notes, res["news_scored"], res["bt_stats"])
                ctx_json = json.dumps(ctx, ensure_ascii=False, default=str)
                state_key = f"ai_v34_{res['ticker']}_{d.index[-1]:%Y%m%d}_{threshold}"
                if GEMINI_KEY:
                    bcol, _ = st.columns([2.2, 5])
                    again = state_key in ss
                    if bcol.button(
                        "🔄 重新請 AI 分析" if again else "✨ 請 AI 用白話解讀這檔股票",
                        type="secondary" if again else "primary",
                        width="stretch",
                    ):
                        if again:
                            ai_comment.clear()
                        with st.spinner("AI 正在整理波浪位置、有效K、籌碼／基本面與多空觀點…（約 5–30 秒）"):
                            try:
                                ss[state_key] = ai_comment(
                                    GEMINI_KEY, res["ticker"], f"{d.index[-1]:%Y%m%d}", threshold, ctx_json, GEMINI_MODEL
                                )
                            except Exception as e:
                                st.error(f"AI 暫時無法回應，請稍後再試：{str(e)[:150]}")
                    if state_key in ss:
                        text, used = ss[state_key]
                        st.markdown(text)
                        st.caption(model_note(used))
                        st.caption("AI 只根據系統數據整理，可能有誤解，請自行判斷。")
                    else:
                        st.caption("按一下按鈕，AI 會幫你白話拆解：目前走第幾波、黃金分割防線、短中長線有效K、多空辯論與明日操作劇本。")
                else:
                    st.markdown(rule_based_comment(ctx))

            st.markdown("##### 🎯 三週期買點雷達（有效K棒・第二套獨立訊號：突破追強型）")
            mid_c, long_c = kb["mid"]["code"], kb["long"]["code"]
            if label[0] in ("buy", "light") and "confirmed" in (mid_c, long_c):
                st.success("🔥 **兩套訊號同時偏多**：「拉回買」與「突破追強」的有效K起漲確認同時出現，共振可信度較高。")
            elif label[0] == "avoid" and "confirmed" in (mid_c, long_c):
                st.info("⚖️ **兩套訊號看法不同**：拉回買模型偏保守，但有效K已確認起漲。若想追強，建議縮小部位並嚴守 K低停損。")

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
            short_extra = ["📌 回測顯示短線（1 週）沒有統計優勢，僅提供價位當停損停利參考，不給買進建議。"]
            if kb["black"]:
                b = kb["black"]
                short_extra.insert(0, f"⚠️ {b['date']:%m/%d} 有效黑K（{b['ret']:+.1f}%"
                                      f"{'，收月線下' if b['below_ma20'] else ''}）：短線風險提示（非賣出訊號）。")
            keybar_card(k1, "🏃 短線（幾天～2週）", "日線有效紅K（不限趨勢）", kb["short"], short_extra)
            keybar_card(k2, "🚶 中線（1～3個月）", "日線有效紅K + 年線向上 → 確認為買點，停損 K低", kb["mid"],
                        [f"📊 這檔歷史（持有3個月）：{keybar_summary_row(kb['bt_mid'][0], 1)}"])
            long_extra = [f"📊 這檔歷史（持有1季）：{keybar_summary_row(kb['bt_long'][0], 1)}"]
            if kb["weekly_partial"]:
                long_extra.insert(0, "⏱️ 本週K棒尚未收完，以目前價格暫估，週五收盤後才算數。")
            keybar_card(k3, "🧘 長線（1季～半年）", "週線有效紅K + 40週線向上", kb["long"], long_extra)

        with st.container(border=True):
            st.markdown("##### 🧮 單筆風險部位計算機（如果現在買，該買多少才不會受重傷？）")
            rc1, rc2, rc3 = st.columns([1, 1, 2.2])
            if is_tw:
                cap_wan = rc1.number_input("預計投資總資金（萬台幣）", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)
                cap_amt = cap_wan * 10000
            else:
                cap_amt = rc1.number_input("預計投資總資金（美元 USD）", min_value=1000.0, max_value=10000000.0, value=30000.0, step=1000.0)
            risk_pct = rc2.slider("單筆最大可承受虧損（%）", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            max_loss_amt = cap_amt * (risk_pct / 100)
            risk_per_share = max(sr["close"] - sr["suggest_stop"], sr["close"] * 0.01)
            rec_shares = min(int(max_loss_amt // risk_per_share), int(cap_amt // sr["close"]))
            rec_cost = rec_shares * sr["close"]
            if is_tw:
                rc3.markdown(
                    f"- 現價 **{sr['close']:,.2f} 元**進場、跌破 **{sr['suggest_stop']:,.2f} 元**停損（每股風險 {risk_per_share:,.2f} 元）：\n"
                    f"- 建議最多買 **{rec_shares:,} 股**（約 **{rec_shares / 1000:.2f} 張**，投入 **{rec_cost / 10000:,.1f} 萬元**，占資金 {rec_cost / cap_amt * 100:.0f}%）。\n"
                    f"- 觸及停損時最大虧損約 **{rec_shares * risk_per_share:,.0f} 元**（≤ 總資金 {risk_pct:.1f}%）。"
                )
            else:
                rc3.markdown(
                    f"- 現價 **${sr['close']:,.2f}** 進場、跌破 **${sr['suggest_stop']:,.2f}** 停損（每股風險 ${risk_per_share:,.2f}）：\n"
                    f"- 建議最多買 **{rec_shares:,} 股**（投入約 **${rec_cost:,.0f} USD**，占資金 {rec_cost / cap_amt * 100:.0f}%）。\n"
                    f"- 觸及停損時最大虧損約 **${rec_shares * risk_per_share:,.0f} USD**（≤ 總資金 {risk_pct:.1f}%）。"
                )

    # -----------------------------------------------------------------------
    # 子頁 2：五面向體質評分 & 壓力支撐
    # -----------------------------------------------------------------------
    with tab_score:
        if not pd.isna(timing):
            info = explain(d, tech)
            ok = chips_ok_series(res)
            st.markdown(f"##### 🏅 綜合評估：{overall:.0f} / 100（五面向體質參考）")
            if ok is not None:
                st.caption(f"台股籌碼濾網：{'✅ 通過' if bool(ok.iloc[-1]) else '❌ 未通過'}（籌碼分數 ≥ {CHIPS_MIN}）")
            else:
                st.caption("美股沒有每日三大法人資料，籌碼濾網不適用，五面向將依可用項目自動按比例配分。")
            explain_map = {
                "tech": sum(info["parts"].values(), []),
                "chips": explain_chips(res["chips"], res["chips_raw"]) if is_tw else explain_us_flow(res.get("us_flow", pd.DataFrame()), res.get("us")),
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
                        st.caption("不計入評分（下方供參考）")
                    else:
                        st.progress(float(v) / WEIGHTS[k], text=f"{v:.1f} / {WEIGHTS[k]}")
                    for line in explain_map[k]:
                        st.caption(f"• {line}")

        st.markdown("##### 🧱 壓力支撐 & 🛡️ ATR 動態風控明細")
        c1, c2, c3, c4 = st.columns(4)
        with c1.container(border=True):
            st.markdown("**上方壓力**（由近到遠）")
            for x in sr["resistance"] or []:
                st.write(f"🔺 **{x['price']:,.2f}**（{x['price'] / sr['close'] - 1:+.1%}）— {x['sources']}")
            if not sr["resistance"]:
                st.write("上方沒有明顯壓力（接近高點）")
        with c2.container(border=True):
            st.markdown("**下方支撐**（由近到遠）")
            for x in sr["support"] or []:
                st.write(f"🔻 **{x['price']:,.2f}**（{x['price'] / sr['close'] - 1:+.1%}）— {x['sources']}")
            if not sr["support"]:
                st.write("下方沒有明顯支撐")
        with c3.container(border=True):
            st.markdown("**風險報酬比**")
            if sr["rr"] is not None:
                st.metric("到壓力漲幅 ÷ 到支撐跌幅", f"{sr['rr']:.2f}")
                st.caption("大於 1.3～1.5 較划算；小於 1 代表上檔空間比風險小。")
            else:
                st.write("無法計算")
        with c4.container(border=True):
            st.markdown("**🛡️ ATR 動態風控**")
            st.write(f"• 建議初始停損：**{sr['suggest_stop']:,.2f} {unit}**（{sr['stop_pct']:+.1f}%）")
            st.write(f"• Chandelier 移動停利：**{sr['chandelier']:,.2f} {unit}**")
            st.caption(f"14 日 ATR = {sr['atr']:,.2f}（日波動約 {sr['atr_pct']:.1f}%）")

    # -----------------------------------------------------------------------
    # 子頁 3：波浪與相似K線
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
            fc1, fc2 = st.columns([1.3, 1])
            with fc1.container(border=True):
                st.markdown(f"**📉 黃金分割回撤位**（{fib['low_date']} 低點 `{fib['low_price']:,.2f}` → "
                            f"{fib['high_date']} 高點 `{fib['high_price']:,.2f}`，已回吐 **{fib['retrace_pct']:.1f}%**）")
                st.dataframe(
                    pd.DataFrame([{
                        "黃金分割比例": lv["name"], "對應價位": lv["price"], "距現價幅度": (lv["price"] / close - 1) * 100,
                        "目前狀態": "✅ 守穩其上" if close >= lv["price"] else "⚠️ 已跌破（轉壓力）",
                    } for lv in fib["retrace_levels"]]),
                    hide_index=True, width="stretch",
                    column_config={"對應價位": st.column_config.NumberColumn(format="%.2f"),
                                   "距現價幅度": st.column_config.NumberColumn(format="%+.1f%%")},
                )
            with fc2.container(border=True):
                st.markdown("**🚀 波段延伸目標價**（守穩支撐再創高後的測幅目標）")
                st.dataframe(
                    pd.DataFrame([{"延伸比例": lv["name"], "目標價位": lv["price"],
                                   "潛在空間": (lv["price"] / close - 1) * 100} for lv in fib["extend_levels"]]),
                    hide_index=True, width="stretch",
                    column_config={"目標價位": st.column_config.NumberColumn(format="%.2f"),
                                   "潛在空間": st.column_config.NumberColumn(format="%+.1f%%")},
                )
                st.caption("💡 第 2 波常回吐 **50%～61.8%**；第 4 波常在 **23.6%～38.2%** 止穩；第 3 波目標常達 **127.2%～161.8%**。")

        if wv.get("pivots"):
            with st.expander("📋 近一年波段高低轉折點明細"):
                st.dataframe(
                    pd.DataFrame([{
                        "日期": p["date"], "類型": "🔺 高點" if p["kind"] == "H" else "🔻 低點",
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
            mc1.metric("1 個月後勝率", f"{ps['win_20d']:.0f}%", f"平均 {ps['avg_20d']:+.1f}%", delta_color=DELTA_COLOR)
            mc2.metric("3 個月後勝率", f"{ps['win_60d']:.0f}%", f"平均 {ps['avg_60d']:+.1f}%", delta_color=DELTA_COLOR)
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
                                yaxis_title="累積漲跌幅 (%)", height=350, hovermode="x unified",
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
    # 子頁 4：籌碼與資金流（台股：三大法人＋融資；美股：機構持股＋空單比例＋OBV資金流）
    # -----------------------------------------------------------------------
    with tab_chip:
        if is_tw:
            chips = res["chips_raw"]
            if chips.empty:
                st.info("目前抓不到這檔台股的法人與融資資料。")
            else:
                for line in explain_chips(res["chips"], chips):
                    st.write(f"• {line}")
                cw = window(chips.reindex(d.index)).dropna(how="all")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                    subplot_titles=("三大法人每日買賣超（張）", "融資餘額（張）"))
                for col, name, color in [("foreign", "外資", "#4a90e2"), ("trust", "投信", "#ff8c00"), ("dealer", "自營商", "#9b59b6")]:
                    fig.add_trace(go.Bar(x=cw.index, y=cw[col] / 1000, name=name, marker_color=color), row=1, col=1)
                fig.add_trace(go.Scatter(x=cw.index, y=cw["margin_balance"], name="融資餘額", line=dict(color=UP)),
                              row=2, col=1)
                fig.update_layout(height=500, barmode="relative", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig, width="stretch")
                st.caption("怎麼看：投信連買或近 3 日法人由賣轉買、同時近 5 日融資大減，代表籌碼沉澱到法人手中；反之散戶融資暴增時籌碼較亂。")
                st.info("💡 券商分點（主力大戶）資料需要 FinMind 付費方案，目前用「三大法人 + 投信轉折 + 融資」代替。")
        else:
            us = res.get("us") or {}
            flow = res.get("us_flow", pd.DataFrame())
            st.markdown("##### 🇺🇸 美股機構籌碼與價量資金流（替代三大法人）")
            uc1, uc2, uc3, uc4 = st.columns(4)
            uc1.metric("機構持股比例", f"{us['inst_pct']:.1%}" if us.get("inst_pct") is not None else "-")
            uc2.metric("內部人持股比例", f"{us['insider_pct']:.1%}" if us.get("insider_pct") is not None else "-")
            uc3.metric("空單佔流通股", f"{us['short_float']:.1%}" if us.get("short_float") is not None else "-")
            uc4.metric("空單回補天數", f"{us['short_ratio']:.1f} 天" if us.get("short_ratio") is not None else "-")
            for line in explain_us_flow(flow, us):
                st.write(f"• {line}")
            if not flow.empty:
                fw = window(flow)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                    subplot_titles=("OBV 能量潮（累積資金動能）vs 20 日均線", "近 20 日紅K成交量佔比（>55% 偏買盤）"))
                fig.add_trace(go.Scatter(x=fw.index, y=fw["obv"], name="OBV 能量潮", line=dict(color="#4a90e2", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=fw.index, y=fw["obv_ma20"], name="OBV 20日線", line=dict(color="#ff8c00", dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=fw.index, y=fw["up_vol_ratio"] * 100, name="紅K量佔比 %", line=dict(color=UP)), row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="#888", row=2, col=1)
                fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", y=1.1))
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig, width="stretch")
            st.info("💡 美股制度不公布每日三大法人買賣超與融資，改以「機構／空單比例（定期揭露）＋ OBV 價量能量潮」作為籌碼參考，不作為強制濾網。")

    # -----------------------------------------------------------------------
    # 子頁 5：基本面與估值（台股月營收/季報/PE；美股季報圖表＋華爾街估值目標價）
    # -----------------------------------------------------------------------
    with tab_fund:
        for line in explain_fund(res["fund"], res["us"]) + explain_value(res["value"], res["us"]):
            st.write(f"• {line}")
        if is_tw:
            rev, fin, val = res["revenue"], res["fin"], res["value"]
            if not rev.empty:
                r = rev.copy()
                r["yoy"] = (r["revenue"] / r["revenue"].shift(12) - 1) * 100
                r = r.iloc[-36:]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=r["period"], y=r["revenue"] / 1e8, name="月營收（億元）", marker_color="#4a90e2"))
                fig.add_trace(go.Scatter(x=r["period"], y=r["yoy"], name="年增率 %", line=dict(color="#ff8c00")), secondary_y=True)
                fig.update_layout(title="月營收與年增率（近 3 年）", height=350, hovermode="x unified", legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
            if not fin.empty and "EPS" in fin:
                q = fin.iloc[-12:]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=q.index, y=q["EPS"], name="單季 EPS（元）", marker_color="#30a46c"))
                if "gross_margin" in q:
                    fig.add_trace(go.Scatter(x=q.index, y=q["gross_margin"] * 100, name="毛利率 %", line=dict(color="#9b59b6")),
                                  secondary_y=True)
                fig.update_layout(title="單季 EPS 與毛利率（近 3 年）", height=330, hovermode="x unified", legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
            if not val.empty and "per" in val and val["per"].notna().any():
                v = val.dropna(subset=["per"]).iloc[-1250:]
                fig = go.Figure(go.Scatter(x=v.index, y=v["per"], name="本益比", line=dict(color="#333")))
                for q_, color in [(0.2, DOWN), (0.5, "#999"), (0.8, UP)]:
                    lvl = v["per"].quantile(q_)
                    fig.add_hline(y=lvl, line_dash="dot", line_color=color, annotation_text=f"{int(q_ * 100)}% 分位 {lvl:.1f} 倍")
                fig.update_layout(title="本益比走勢（近 5 年）：靠近下緣代表相對便宜", height=310)
                st.plotly_chart(fig, width="stretch")
            if rev.empty and fin.empty:
                st.info("ETF 沒有營收、財報與本益比，基本面和估值不列入評分。")
        else:
            us = res.get("us") or {}
            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("過去本益比 (Trailing PE)", f"{us['pe']:.1f}x" if us.get("pe") else "-")
            vc2.metric("預估本益比 (Forward PE)", f"{us['forward_pe']:.1f}x" if us.get("forward_pe") else "-")
            vc3.metric("PEG 本益成長比", f"{us['peg']:.2f}" if us.get("peg") else "-")
            if us.get("target_mean"):
                upside = (us["target_mean"] / close - 1) * 100
                vc4.metric("華爾街平均目標價", f"${us['target_mean']:,.2f}", f"{upside:+.1f}%", delta_color=DELTA_COLOR)
            else:
                vc4.metric("華爾街平均目標價", "-")

            quarters = us.get("quarters") or []
            if quarters:
                qdf = pd.DataFrame(quarters)
                if qdf["revenue"].notna().any():
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=qdf["date"], y=qdf["revenue"] / 1e8, name="單季營收（億美元）", marker_color="#4a90e2"))
                    if qdf["rev_yoy"].notna().any():
                        fig.add_trace(go.Scatter(x=qdf["date"], y=qdf["rev_yoy"] * 100, name="年增率 %", line=dict(color="#ff8c00")), secondary_y=True)
                    fig.update_layout(title="美股單季營收與年增率", height=340, hovermode="x unified", legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, width="stretch")
                if qdf["eps"].notna().any() or qdf["gross_margin"].notna().any():
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    if qdf["eps"].notna().any():
                        fig.add_trace(go.Bar(x=qdf["date"], y=qdf["eps"], name="單季稀釋 EPS（美元）", marker_color="#30a46c"))
                    if qdf["gross_margin"].notna().any():
                        fig.add_trace(go.Scatter(x=qdf["date"], y=qdf["gross_margin"] * 100, name="毛利率 %", line=dict(color="#9b59b6")), secondary_y=True)
                    fig.update_layout(title="美股單季 EPS 與毛利率", height=330, hovermode="x unified", legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, width="stretch")
            elif not us.get("pe"):
                st.info("這檔美股（或 ETF）沒有公開季度財報圖表資料。")

    # -----------------------------------------------------------------------
    # 子頁 6：近期新聞
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
            emo = {1: POS_ICON, -1: NEG_ICON, 0: "⬜ 中性"}.get(n.get("sentiment"), "")
            reason = f"　_{n['reason']}_" if n.get("reason") else ""
            title = f"[{n['title']}]({n['link']})" if n.get("link") else n["title"]
            st.markdown(f"{emo} `{n['date'][:10]}` {title}{reason}")
        if not news:
            st.info("近幾天沒有找到相關新聞。")
        st.caption("AI 只看標題判斷，可能誤判；消息面不列入回測。")

    # -----------------------------------------------------------------------
    # 子頁 7：歷史回測（自動對應台股／美股實測研究）
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
            st.markdown("**⭐ 目前使用：技術與波浪 + 籌碼濾網**" if has_chips else "**⭐ 目前使用：技術與波浪**（美股／無籌碼資料）")
            st.dataframe(stats_m[cols3], hide_index=True, width="stretch", column_config=cfg)
            if has_chips:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**對照 ①：只看技術與波浪**")
                    st.dataframe(stats_t[cols3], hide_index=True, width="stretch", column_config=cfg)
                with c2:
                    st.markdown("**對照 ②：五面向加權合併（綜合評估）**")
                    st.dataframe(stats_c[cols3], hide_index=True, width="stretch", column_config=cfg)

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
        if is_tw:
            st.info(
                "📚 **🇹🇼 14 檔台股大型股、2016 年起整體研究**（超額 = 訊號報酬 − 同一檔隨便哪天買）\n\n"
                "- **長線（週線有效K）最穩定**：持有 1 季 120 次，勝率 69.2%（隨便買 63.5%），平均超額 +8.5%；2016–2020 +5.2%、2021 年後 +10.3%。\n"
                "- **中線（日線有效K）不穩定**：持有 3 個月平均超額 +3.9%，但勝率與隨便買打平；2016–2020 只有 +0.6%。\n"
                "- **一定要等確認**：有效K當天就買，約 55% 會在 20 天內跌破K低；等確認後降到約 38%。\n"
                "- **短線（1 週）沒有優勢**；**有效黑K不是賣出訊號**。"
            )
        else:
            st.info(
                "📚 **🇺🇸 15 檔美股龍頭股、2016 年起實測研究**（NVDA、AAPL、MSFT、AMZN、GOOGL、META、AVGO、TSLA、AMD、JPM、UNH、COST、LLY、XOM、WMT）\n\n"
                "- **第一套拉回買（門檻 70）**：持有 3 個月 714 次，勝率 **67.9%**、平均報酬 **+8.0%**（與隨便買 68.1%／+8.0% 打平，15 檔中 9 檔略勝）。\n"
                "- **第二套有效K（中線日線／長線週線）**：持有 3 個月勝率約 65%（平均 +6%～+7.6%），持有半年週線勝率 **76.8%**（平均 +18.2%），"
                "但因為美股權值股長期上漲趨勢極強、且財報日跳空後常短線拉回，**平均超額報酬低於「隨便哪天買」（約 −1.7%～−3.2%）**。\n"
                "- **實戰解讀**：在美股上，這兩套訊號的主要價值在於「**提供明確的止跌條件、K低／ATR 停損紀律，避免追高與接刀**」，"
                "而不是保證打敗長期定期定額持有。"
            )

# ===========================================================================
# 🚀 起漲雷達（同時支援 🇹🇼 台股 & 🇺🇸 美股）
# ===========================================================================
with tab_scan:
    st.markdown("#### 🚀 起漲雷達：掃描名單外「剛啟動」「準備啟動」「拉回買點」的股票")
    m_col, s_col, b_col = st.columns([1.8, 1.8, 1.2], vertical_alignment="bottom")
    radar_mkt = m_col.segmented_control(
        "選擇市場",
        options=["tw", "us"],
        default="tw",
        format_func=lambda x: "🇹🇼 台股雷達" if x == "tw" else "🇺🇸 美股雷達",
        key="radar_mkt",
        label_visibility="collapsed",
    ) or "tw"

    if radar_mkt == "tw":
        scan_opt = s_col.selectbox("掃描範圍", [100, 200, 300], index=1, format_func=lambda n: f"成交值前 {n} 大台股", key="scan_n_tw")
        val_col_name = "成交值(億台幣)"
    else:
        scan_opt = s_col.selectbox(
            "掃描範圍",
            ["hot100", "ai40"],
            index=0,
            format_func=lambda x: "🔥 美股熱門權值（約 105 檔）" if x == "hot100" else "🤖 AI／半導體／科技精選（40 檔）",
            key="scan_n_us",
        )
        val_col_name = "成交值(億美元)"

    scan_df, meta = load_last_scan(market=radar_mkt)
    run = b_col.button("🔄 重新掃描" if scan_df is not None else "🔍 開始掃描", type="primary", width="stretch", key=f"run_scan_{radar_mkt}")
    if run:
        bar = st.progress(0.0, text="準備掃描…")
        try:
            if radar_mkt == "tw":
                scan_df, meta = scan(scan_opt, threshold, progress=lambda p, m: bar.progress(min(float(p), 1.0), text=m))
            else:
                scan_df, meta = scan_us(scan_opt, threshold, progress=lambda p, m: bar.progress(min(float(p), 1.0), text=m))
        except Exception as e:
            st.error(f"掃描失敗：{str(e)[:150]}")
        bar.empty()

    if scan_df is None or meta is None:
        st.info(f"還沒有今天的{'台股' if radar_mkt == 'tw' else '美股'}掃描結果。按右上角「🔍 開始掃描」，約 15–45 秒即可完成。")
    else:
        lbl = meta.get("label") or f"前 {meta['universe']} 大"
        st.caption(
            f"上次掃描：{meta['time']}｜資料日 {meta['data_date']}｜範圍：{lbl}｜找到 **{len(scan_df)}** 檔｜拉回買門檻 {meta['threshold']} 分"
        )
        if len(scan_df) == 0:
            st.info("目前沒有股票符合條件（大盤急跌時很常見）。")
        else:
            filt = st.segmented_control("篩選", ["全部", "剛啟動", "準備啟動", "拉回買點"], default="全部",
                                        key=f"scan_filter_{radar_mkt}", label_visibility="collapsed")
            view = scan_df if filt in (None, "全部") else scan_df[scan_df["類型"] == filt]
            view = view.copy()
            in_list = {stocks[k]["ticker"].split(".")[0] for k in stocks} | set(ss.watchlist)
            view.insert(0, "名單", np.where(view["代號"].isin(in_list), "✓", ""))
            view = view.rename(columns={"成交值(億)": val_col_name})
            show_cols = ["名單", "代號", "名稱", "類型", "訊號", "收盤", "漲跌%", val_col_name, "K高", "K低(停損)", "距K低%",
                         "買進時機", "波浪階段", "波浪情境"]
            event = st.dataframe(
                view[show_cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                key=f"scan_table_{radar_mkt}",
                column_config={
                    "收盤": st.column_config.NumberColumn(format="%.2f"),
                    "漲跌%": st.column_config.NumberColumn(format="%+.2f%%"),
                    val_col_name: st.column_config.NumberColumn(format="%.1f"),
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
                st.toast(f"已加入 {'、'.join(cs)}，切回「🔍 個股分析」就能看圖")

            st.button(
                f"➕ 把勾選的 {len(codes)} 檔加入觀察名單" if codes else "👆 先勾選表格左側的方框，再按這裡加入觀察名單",
                disabled=not codes, type="primary", on_click=add_from_scan, args=(codes,), key=f"add_scan_{radar_mkt}",
            )
    st.info(
        "**三種分類說明**（台股與美股共用同一套規則）\n\n"
        "- **剛啟動**：中線（日線有效紅K + 年線向上）5 天內確認起漲，或長線（週線有效紅K + 40 週線向上）2 週內確認起漲。\n"
        "- **準備啟動**：有效紅K已出現，等待站上 K高 或回測實體 1/2 不破轉強；跌破 K低 即失效。\n"
        "- **拉回買點**：買進時機達門檻且已通過止跌閘門（台股掃描時未含籌碼濾網，加入名單後會完整補查）。"
    )

# ===========================================================================
# 🏠 名單總覽與比較
# ===========================================================================
with tab_home:
    # 依名單中包含的市場，分別顯示台股與美股大盤氣象站
    shown_regimes: set[str] = set()
    for r in stocks.values():
        key_m = "tw" if is_taiwan(r["ticker"]) else "us"
        if key_m not in shown_regimes and r["market"].get("regime_desc"):
            shown_regimes.add(key_m)
            st.info(f"**🌦️ {'🇹🇼 台股' if key_m == 'tw' else '🇺🇸 美股'}大盤氣象站 — {r['market']['regime']}**：{r['market']['regime_desc']}")

    cols = st.columns(min(len(stocks), 4))
    for i, (item, r) in enumerate(stocks.items()):
        dd = r["d"]
        t_, lab, nts, ov = current(r)
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"#### {flag_of(r['ticker'])} {r['name']}")
            c_, p_ = dd["Close"].iloc[-1], dd["Close"].iloc[-2]
            st.metric(f"收盤（{dd.index[-1]:%m/%d}）", f"{c_:,.2f} {cur_unit(r['ticker'])}", f"{(c_ / p_ - 1) * 100:+.2f}%", delta_color=DELTA_COLOR)
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
            st.plotly_chart(radar(factor_summary(r["comp"])), width="stretch", key=f"radar_{item}")

    if len(stocks) >= 2:
        st.divider()
        st.markdown("##### ⚖️ 各檔橫向比較")
        rows = {}
        for item, r in stocks.items():
            f = factor_summary(r["comp"])
            t_, lab, _, ov = current(r)
            rows[f"{flag_of(r['ticker'])} {r['name']}"] = {
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
        closes = pd.DataFrame({f"{flag_of(r['ticker'])} {r['name']}": window(r["d"])["Close"] for r in stocks.values()})
        norm = closes.ffill().apply(lambda s: s / s.dropna().iloc[0] * 100)
        fig = px.line(norm, labels={"value": "累積走勢（起點 = 100）", "index": "", "variable": ""})
        fig.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.markdown(f"##### 近 {ss.get('period') or '1年'} 走勢比較（起點都換算成 100）")
        st.plotly_chart(fig, width="stretch")
        perf = pd.DataFrame({n: performance(closes[n]) for n in closes}).T
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(perf, width="stretch", column_config={
            "總報酬": pct, "年化報酬": pct, "年化波動度": pct, "最大回撤": pct,
            "報酬/風險比": st.column_config.NumberColumn(format="%.2f"),
        })
        corr = closes.ffill().pct_change().corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=120 + 60 * len(corr), title="相關性（越接近 1，漲跌越同步，分散風險效果越差）")
        st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 📖 使用說明
# ===========================================================================
with tab_help:
    st.markdown(f"""
### 怎麼快速上手
1. **加入股票（台股＋美股通用）**：在最上方搜尋框輸入代號或中文名稱（如 `2330`、`台積電`、`NVDA`、`輝達`、`SPY`）按 Enter，或點右上角 **「⚡ 熱門快選」** 一鍵加入。
2. **切換股票**：點搜尋框下方的藥丸按鈕，國旗會標示 🇹🇼 台股／🇺🇸 美股，網址會自動保存名單。
3. **先看圖、30 秒看懂結論**：一進入就是「狀態卡 → K 線主圖 → 3 欄重點結論」，細節分層收在下方子分頁；**🤖 AI 分析師**放在第一個子分頁。
4. **找名單外的機會**：到 **「🚀 起漲雷達」**，可切換掃描 **🇹🇼 台股（成交值前 100–300 大）** 或 **🇺🇸 美股（熱門權值 105 檔／AI 科技 40 檔）**。

### 兩套獨立買點訊號
| 訊號 | 邏輯 | 適合誰 |
|---|---|---|
| 🎯 **買進時機（拉回買）** | 長多趨勢 ＋ 黃金分割回檔（32%～68%）＋ 近 10 日曾超賣 ＋ 今天止跌（未止跌最高鎖 {NO_REVERSAL_CAP} 分）；台股另加籌碼濾網（≥ {CHIPS_MIN} 分） | 喜歡等拉回買在相對低檔 |
| 🕯️ **有效K起漲（突破追強）** | 多頭下出現放量飽滿紅K（日線 ≥ {DAILY['min_ret']:.0%}、週線 ≥ {WEEKLY['min_ret']:.0%}），並在期限內確認（突破 K高 或回測實體 1/2 不破轉強） | 喜歡等轉強確認、守 K低 停損 |

- **買進時機分數不是愈高愈穩**：請把它當作及格門檻（預設 **{threshold} 分**），搭配籌碼／資金流、風險報酬比（≥ {RR_MIN}）與大盤環境綜合看；兩套訊號同時偏多時共振更有參考價值。
- **台股 vs. 美股差異**：
  - **大盤基準**：台股對比 `0050`，美股對比 `SPY`（S&P 500）。
  - **籌碼面**：台股有每日三大法人與融資（含籌碼濾網）；美股制度無此資料，改顯示**機構持股 %、空單比例與 OBV 資金流**（參考用）。
  - **基本面與部位計算機**：台股看月營收／季報／台幣張數；美股看季報圖表／預估本益比／華爾街目標價／美元股數。

### 🤖 AI 分析師
整合所有技術波浪、有效K、籌碼／資金流、財報估值、新聞與風控價位交給 Gemini（目前 `{GEMINI_MODEL}`，失敗自動退回其他 Flash 模型），產出白話結論、多空辯論與 If-Then 操作劇本。

⚠️ 過去回測表現不代表未來，且不計手續費、稅與滑價。本工具僅供學習參考，不構成投資建議。
""")
