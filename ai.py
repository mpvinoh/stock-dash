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
TIMEOUT_MS = 60_000
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


def _client(api_key: str):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)),
    )


def _generate(api_key: str, prompt: str, json_mode: bool = False, model: str = "") -> tuple[str, str]:
    """依序嘗試模型，成功就回傳（文字、實際使用的模型）。"""
    from google.genai import types

    client = _client(api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT if not json_mode else None,
        response_mime_type="application/json" if json_mode else None,
        temperature=0.3,
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
            msg = str(e)
            if "API key" in msg or "PERMISSION_DENIED" in msg:
                break
            if "NOT_FOUND" in msg or "DEADLINE" in msg or "timed out" in msg.lower():
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
        text, used = _generate(api_key, prompt, json_mode=True, model=model)
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
