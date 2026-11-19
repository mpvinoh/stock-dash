"""AI 功能（Gemini）：新聞情緒判讀、AI 分析師口語化建議（含波浪理論解讀、多空辯論、If-Then 劇本）。

沒有 API 金鑰或呼叫失敗時，改用規則產生的摘要，不會讓程式中斷。
"""

from __future__ import annotations

import json
import re

# 優先使用最強的 gemini-4-argon（需要付費層級）；失敗時才退回 Flash 系列
# 實測（免費層級）：argon 回 429 額度不足；3.8 / 3.7 Flash 常逾時；3.6 / 3.5 Flash 2–3 秒
BEST_MODEL = "gemini-4-argon"
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
TIMEOUT_MS = 35_000
_dead_models: set[str] = set()  # 這次執行中確定不能用的模型，之後跳過

SYSTEM_PROMPT = """你是一位說話親切、務實、精通「艾略特波浪理論與黃金分割」與「多因子量化風控」的台股與美股分析師，用繁體中文、口語化的方式向一般投資人解說。
規則：
1. 只能根據使用者提供的數據說話，不可以編造任何數字、新聞或事件。數據裡沒有的就說「資料沒有顯示」。注意台股用台幣、美股用美元，美股沒有三大法人與融資，改引用機構持股、空單比例與 OBV 資金流。
2. 必須清楚解讀「波浪理論與黃金分割」數據：用白話告訴使用者目前股價處於第幾波（例如第 2 波或第 4 波黃金分割回檔、第 3 波主升段、第 5 波末升段或 A-B-C 修正浪），前一波從哪裡漲到哪裡、目前回吐了幾 %，以及黃金分割 38.2%、50%、61.8% 防線與 127.2%、161.8% 目標價在哪裡。
3. 採用「多空辯論（Bull vs. Bear）」視角：同時呈現多方加分理由與空方風險挑刺，幫助投資人客觀判斷，不要一面倒。
4. 不要保證獲利，也不要說「一定會漲 / 跌」。
5. 價位建議只能引用提供的支撐、壓力、黃金分割價位與 ATR 停損價位。
6. 「買進建議」來自經過回測驗證的規則（技術與波浪時機 + 籌碼濾網），請以它為主軸；「綜合評估」和基本面、估值、新聞是幫助判斷公司體質與風險的背景資訊。
7. 「有效K棒起漲訊號」是另一套獨立的「突破追強」訊號（和「拉回買」的買進時機分數邏輯不同），請分開說明，不要混為一談：
   - 短線只提供 K高／實體1/2／K低 價位當停損停利參考，回測沒有統計優勢，不可以據此建議短線買進。
   - 中線、長線在「起漲確認」後才算買點，停損參考 K低；長線（週線）是回測最穩定的。
   - 有效黑K只是短線風險提示，回測顯示 1–3 個月並沒有明顯轉弱，不可以說成賣出訊號。
   - 引用個股回測時要一起說出訊號次數；次數少於 5 次的勝率不具參考性，要明講。
8. 結尾加一句簡短提醒：這是根據數據整理的參考意見，不是投資建議。"""

ANALYST_FORMAT = """請用以下格式回答（Markdown，總長約 450–700 字）：

**💡 一句話結論**：（用一句話說目前適不適合買、處於什麼波浪階段、為什麼）

**🌊 波浪理論與黃金分割解讀**
- （用白話說明：目前走在第幾波或修正浪？前波起漲低點與高點在哪、目前已回吐幾 %？離 38.2%／50%／61.8% 黃金分割支撐或 127.2%／161.8% 延伸目標價有多近？有無頂底背離？並順帶提一下大盤氣象與歷史相似 K 線的勝率）

**🕯️ 有效K棒：短／中／長線買點雷達**
- （分別用一句話說明短線、中線、長線目前的有效K狀態與關鍵價位；如果有波浪位置標註，說明這根有效K可能代表第幾波起漲；若與「買進時機」同時偏多或互相矛盾，要點出來）

**🐂 多方觀點（有利訊號）**
- （最多 3 點，結合技術止跌、波浪位置、有效K、籌碼法人/融資、相對強弱 RS 或基本面）

**🐻 空方觀點（風險與反方挑刺）**
- （最多 3 點，指出目前最需要注意的扣分項、籌碼破口、波浪破壞風險、有效K失效或估值/消息面隱憂）

**🎯 實戰操作劇本（If-Then）**
- **進場與加碼**：（拉回買的人看哪些支撐或黃金分割價位；突破買的人看站上哪個 K高 才算確認）
- **停損與退場**：（引用 ATR 初始停損價、K低 或黃金分割防線，跌破哪裡代表結構轉弱應退場觀望？）
- **壓力與停利**：（上方最近壓力、Chandelier 移動停利價或波段延伸目標價在哪裡？）

**（提醒）**"""


def _client(api_key: str, timeout_ms: int = TIMEOUT_MS):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1)),
    )


def _generate(
    api_key: str,
    prompt: str,
    json_mode: bool = False,
    model: str = "",
    system: str | None = SYSTEM_PROMPT,
    temperature: float = 0.3,
    timeout_ms: int = TIMEOUT_MS,
) -> tuple[str, str]:
    """依序嘗試模型，成功就回傳（文字、實際使用的模型）。"""
    from google.genai import types

    client = _client(api_key, timeout_ms=timeout_ms)
    config = types.GenerateContentConfig(
        system_instruction=None if json_mode else system,
        response_mime_type="application/json" if json_mode else None,
        temperature=temperature,
    )
    models = [model or BEST_MODEL] + [m for m in FALLBACK_MODELS if m != model]
    candidates = [m for m in models if m not in _dead_models] or models
    last_err: Exception | None = None
    for m in candidates:
        try:
            r = client.models.generate_content(model=m, contents=prompt, config=config)
            if r.text:
                return r.text, m
        except Exception as e:  # 模型不存在、逾時、額度不足等，換下一個
            last_err = e
            msg = f"{type(e).__name__}: {e}".lower()
            if "api key" in msg or "permission_denied" in msg:
                break
            if any(k in msg for k in ("not_found", "deadline", "timed out", "timeout", "resource_exhausted", "429")):
                _dead_models.add(m)
    raise RuntimeError(f"Gemini 呼叫失敗：{last_err}")


# ---------------------------------------------------------------------------
# 新聞情緒
# ---------------------------------------------------------------------------
def classify_news(api_key: str, stock_name: str, news: list[dict], model: str = "") -> tuple[list[dict], float | None, str]:
    """回傳（加上 sentiment / reason 的新聞、整體情緒 −1~+1、狀態訊息或使用的模型）。"""
    if not news:
        return news, None, "近期沒有新聞"
    if not api_key:
        return news, None, "沒有設定 Gemini 金鑰，無法判讀新聞情緒"
    titles = "\n".join(f"{i}. {n['title']}" for i, n in enumerate(news))
    prompt = f"""以下是關於「{stock_name}」的近期新聞標題。請判斷每則新聞對這檔股票股價的影響：
1 = 偏正面、0 = 中性或無關、-1 = 偏負面。reason 用 15 字以內的繁體中文說明。
只回傳 JSON 陣列，格式：[{{"i": 0, "s": 1, "reason": "..."}}, ...]

{titles}"""
    try:
        text, used = _generate(api_key, prompt, json_mode=True, model=model, timeout_ms=15_000)
        text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
        result = {int(x["i"]): x for x in json.loads(text)}
    except Exception as e:
        return news, None, f"新聞情緒判讀失敗（{str(e)[:80]}）"
    out = []
    for i, n in enumerate(news):
        r = result.get(i, {})
        out.append({**n, "sentiment": int(r.get("s", 0)), "reason": r.get("reason", "")})
    vals = [n["sentiment"] for n in out]
    overall = sum(vals) / len(vals) if vals else None
    return out, overall, used


def news_to_score(overall: float | None) -> float | None:
    """整體情緒 −1~+1 → 0~10 分。"""
    if overall is None:
        return None
    return round(5 + 5 * max(-1.0, min(1.0, overall * 1.5)), 1)


# ---------------------------------------------------------------------------
# AI 分析師
# ---------------------------------------------------------------------------
def analyst_comment(api_key: str, context: dict, model: str = "") -> tuple[str, str]:
    """回傳（分析內容、使用的模型）。"""
    prompt = (
        "以下是這檔股票的完整分析數據（JSON）：\n\n"
        + json.dumps(context, ensure_ascii=False, indent=1, default=str)
        + "\n\n"
        + ANALYST_FORMAT
    )
    return _generate(api_key, prompt, model=model)


def rule_based_comment(context: dict) -> str:
    """沒有 AI 時的制式摘要。"""
    timing = context.get("買進時機分數（技術與波浪，0-100）", 0)
    wv = context.get("波浪理論與黃金分割", {})
    mkt = context.get("大盤環境與相對強弱", {})
    lines = [
        f"**💡 一句話結論**：買進時機 **{timing:.0f}** 分，建議「{context['買進建議']}」；目前處於「{wv.get('目前波浪階段', '-')}」。",
        "",
        "**🌊 波浪理論與黃金分割解讀**",
        f"- **波段轉折**：{wv.get('近期波段轉折序列', '-')}",
        f"- **黃金分割**：{wv.get('黃金分割分析', '-')}",
        f"- **背離訊號**：{wv.get('頂底背離訊號', '無明顯背離')}",
        f"- **大盤與強弱**：{mkt.get('大盤氣象', '')}（{mkt.get('個股相對強弱RS', '')}）",
        "",
        f"**📊 技術面情境**：{context['技術面']['情境']}",
    ]
    good, bad = [], []
    for k, v in context["各面向得分率"].items():
        if v is None:
            continue
        rate = float(v.rstrip("%")) / 100
        if rate >= 0.6:
            good.append(f"{k}（{v}）")
        elif rate < 0.4:
            bad.append(f"{k}（{v}）")
    if good:
        lines += ["", "**🐂 多方加分面向**：" + "、".join(good)]
    if bad:
        lines += ["", "**🐻 需留意面向**：" + "、".join(bad)]
    kb = next((v for k, v in context.items() if k.startswith("有效K棒")), None)
    if kb:
        lines += ["", "**🕯️ 有效K棒：短／中／長線買點雷達**"]
        for k, v in kb.items():
            if isinstance(v, dict):
                lines.append(f"- **{k.split('（')[0]}**：{v.get('狀態', '')}" + (f"　{v['說明']}" if v.get("說明") else ""))
            elif k.startswith("近10日有效黑K") and v != "無":
                lines.append(f"- **有效黑K**：{v}（短線風險提示，不是賣出訊號）")
    sr = context.get("壓力支撐與ATR風控", {})
    sup = sr.get("最近支撐", ["無明顯支撐"])[0] if sr.get("最近支撐") else "無明顯支撐"
    res = sr.get("最近壓力", ["無明顯壓力"])[0] if sr.get("最近壓力") else "無明顯壓力"
    stop = sr.get("建議初始停損價", "-")
    chand = sr.get("Chandelier移動停利價", "-")
    lines += [
        "",
        f"**🎯 實戰操作參考**：下方支撐 {sup}；上方壓力 {res}；ATR 初始停損價 **{stop}**；移動停利線 **{chand}**。",
        "",
        "_（設定 Gemini 金鑰後，這裡會改由 AI 用口語化方式解讀波浪位置、多空辯論與 If-Then 劇本）_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 盤中主動互動播報 & 輕量戰情聊天（極低 Token 設計）
# ---------------------------------------------------------------------------
def rule_based_pulse_comment(events: list[dict], profile_key: str, session_label: str) -> str:
    """沒有 AI 或額度暫滿時的規則式盤中播報（0 Token）。"""
    if not events:
        if profile_key == "enthusiastic":
            return f"👀 剛巡邏完一輪（{session_label}），目前熱門股跟名單都還在整理蓄力中，還沒看到明顯放量點火。等有風吹草動我馬上喊你！"
        if profile_key == "calm":
            return f"目前（{session_label}）無重大突破或風險警訊，維持紀律觀望即可。"
        return f"剛巡邏完最新盤面（{session_label}），暫時沒有符合門檻的量價異動，有訊號會再提醒你。"
    top = events[0]
    others = "、".join(f"{e['flag']}{e['name']}" for e in events[1:3])
    extra = f"；另外 {others} 也有異動" if others else ""
    if profile_key == "enthusiastic":
        return (
            f"🔥 發現亮點！**{top['flag']} {top['name']}**（{top['tag_str']}），{top['detail']}{extra}！"
            "要直接點下方按鈕切過去看 K 線圖，還是讓我幫你評估追高風險？"
        )
    if profile_key == "calm":
        return f"重點提醒：**{top['flag']} {top['name']}** 出現「{top['tag_str']}」（{top['detail']}）{extra}。建議先確認支撐與停損位置。"
    return f"盤面快報：**{top['flag']} {top['name']}** 出現「{top['tag_str']}」— {top['detail']}{extra}。點下方卡片可直接切換看圖。"


def pulse_comment(
    api_key: str,
    events: list[dict],
    profile_key: str,
    tone_desc: str,
    session_label: str,
    model: str = "",
) -> tuple[str, str]:
    """把 Python 篩出的 1–3 檔異動重點交給 AI 寫成 2–3 句有趣口語播報（約 500–700 tokens）。"""
    if not events:
        return rule_based_pulse_comment(events, profile_key, session_label), "0-token"
    if not api_key:
        return rule_based_pulse_comment(events, profile_key, session_label), "rule"

    brief = [
        {
            "股票": f"{e['flag']}{e['name']}",
            "在觀察名單": e["in_watch"],
            "訊號": e["tag_str"],
            "現價": e["close"],
            "漲跌": f"{e['ret_pct']:+.1f}%",
            "量比": f"{e['vol_ratio']:.1f}倍",
            "說明": e["detail"],
        }
        for e in events[:3]
    ]
    sys_msg = (
        "你是股票分析小幫手的盤中互動助理，用繁體中文口語跟使用者聊天播報。"
        f"說話風格：{tone_desc}\n"
        "嚴格規則：只能引用提供的數據；量能放大只能說是「量能異常放大／疑似大資金點火」，"
        "不可宣稱看到實際券商大戶明細；不可保證漲跌；總字數控制在 80–160 字（2–3 句話，不要列長清單）。"
    )
    prompt = f"目前市場狀態：{session_label}\n剛偵測到的異動：\n{json.dumps(brief, ensure_ascii=False)}\n請寫一段簡短自然的互動提示。"
    try:
        return _generate(api_key, prompt, model=model or "gemini-3.6-flash", system=sys_msg, temperature=0.55, timeout_ms=15_000)
    except Exception:
        return rule_based_pulse_comment(events, profile_key, session_label), "fallback"


def chat_reply(
    api_key: str,
    question: str,
    focus_ctx: dict,
    pulse_events: list[dict],
    history: list[dict],
    tone_desc: str,
    model: str = "",
) -> tuple[str, str]:
    """盤中互動助理對話回覆（精簡 Context，控制 Token 消耗）。"""
    if not api_key:
        return "目前沒有設定 Gemini API 金鑰，請先到右上角「⚙️ 設定」填入自己的金鑰，或設定 secrets.toml 後就能開始對話。", "none"

    compact_focus = {
        "目前查看股票": focus_ctx.get("股票"),
        "幣別": focus_ctx.get("幣別"),
        "收盤價": focus_ctx.get("收盤價"),
        "買進時機分數": focus_ctx.get("買進時機分數（技術與波浪，0-100）"),
        "買進建議": focus_ctx.get("買進建議"),
        "波浪": focus_ctx.get("波浪理論與黃金分割"),
        "有效K": focus_ctx.get("有效K棒起漲訊號（第二套獨立訊號，和買進時機分數分開）"),
        "壓力支撐與ATR風控": focus_ctx.get("壓力支撐與ATR風控"),
        "籌碼面": focus_ctx.get("籌碼面"),
        "大盤與RS": focus_ctx.get("大盤環境與相對強弱"),
    }
    compact_pulse = [
        {"股票": f"{e['flag']}{e['name']}", "訊號": e["tag_str"], "價": e["close"], "漲跌": f"{e['ret_pct']:+.1f}%", "量比": f"{e['vol_ratio']:.1f}x"}
        for e in (pulse_events or [])[:5]
    ]
    recent_hist = "\n".join(f"{'使用者' if m['role'] == 'user' else '助理'}：{m['content']}" for m in history[-6:])
    sys_msg = (
        "你是股票分析小幫手的盤中互動助理，用繁體中文回答使用者提問。"
        f"說話風格：{tone_desc}\n"
        "規則：只能依據提供的數據回答，不可編造數字或新聞；若問到清單以外的股票，請提醒使用者先在上方搜尋框加入觀察名單；"
        "回答精簡務實（150–280 字內），結尾提醒僅供參考、非投資建議。"
    )
    prompt = (
        f"【目前個股摘要】\n{json.dumps(compact_focus, ensure_ascii=False, default=str)}\n\n"
        f"【最新巡邏異動】\n{json.dumps(compact_pulse, ensure_ascii=False)}\n\n"
        + (f"【近期對話】\n{recent_hist}\n\n" if recent_hist else "")
        + f"【使用者提問】\n{question}"
    )
    return _generate(api_key, prompt, model=model or "gemini-3.6-flash", system=sys_msg, temperature=0.4, timeout_ms=25_000)
