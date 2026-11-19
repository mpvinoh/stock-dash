# AGENTS.md — Antigravity 跨機開發指引

當你在任何一台電腦使用 **Antigravity** 開啟本專案時，請先閱讀本檔案以快速掌握架構、設計規範與雲端同步流程。

## 1. 專案概覽
**股票分析小幫手（v3.4）** 是一個支援**台股（上市 `.TW`、上櫃 `.TWO`、ETF）與美股（S&P 500、Nasdaq 100、AI／半導體、ETF）**的 Streamlit 智慧股市分析應用程式。

### 核心模組分工
- `app.py`：Streamlit 主介面（頂部「⚡ 熱門快選」與「⚙️ 設定」＋搜尋框＋國旗觀察名單按鈕 → 狀態總覽卡與 K 線主圖 → 30 秒 3 欄結論 → 分層細節分頁（AI 分析放第一子頁）＋台美股起漲雷達＋名單總覽比較＋雲端密碼鎖）。
- `model.py`：第一套買點模型「**拉回買（買進時機分數）**」——長多趨勢 + 波浪黃金分割回檔（32%～68%）+ 近 10 日超賣序列記憶 + 止跌閘門 + ATR 動態停損停利回測。
- `keybar.py`：第二套獨立買點訊號「**有效 K 棒起漲確認（突破追強型）**」——放量長紅 K 的三條關鍵價位（K 高、實體 1/2、K 低）與短／中／長（週線）三週期狀態追蹤及回測。
- `waves.py`：ATR 動態 ZigZag 波浪轉折標示（`[0] 起漲 → ①②③④⑤ ／ ⓐⓑⓒ`）、費波那契黃金分割回撤／延伸價位、RSI 頂底背離、歷史前 10 次最相似 K 線比對。
- `factors.py`：台股籌碼（三大法人、投信連買／3 日轉買、融資 5 日斷頭清洗）、美股資金流替代指標（機構持股、空單比例、OBV 能量潮與紅 K 量佔比）、基本面與估值、支撐壓力與分價量表、五面向綜合評估。
- `scanner.py`：「🚀 起漲雷達」——支援 **🇹🇼 台股（證交所／櫃買成交值前 100～300 大）** 與 **🇺🇸 美股（熱門權值 ~105 檔／AI 科技精選 40 檔）**，共用同一套規則掃描剛啟動、準備啟動與拉回買點。
- `analysis.py`：整合所有模組、大盤氣象站（台股對比 `0050`、美股對比 `SPY`）與相對強弱 RS、最終建議降級濾網、組裝 AI 分析師 JSON Context。
- `sources.py` & `data.py`：Yahoo Finance（台美股價、美股季報時間序列、估值目標價、機構／空單與新聞）與 FinMind 資料抓取、台股最新日 K 自動補齊、台美股中英文股名解析、`data_cache/` 本機 Parquet/JSON 快取。
- `ai.py`：Gemini API 新聞情緒判讀與 AI 分析師（波浪解讀 × 有效 K × 多空辯論 × If-Then 操作劇本），自動區分台股（台幣／三大法人）與美股（美元／機構空單與 OBV），支援自動 fallback 與無金鑰規則摘要。

## 2. 開發與設計守則
1. **台股紅漲綠跌慣例**：上漲一律使用紅色 `#e5484d`，下跌一律使用綠色 `#30a46c`（`st.metric` 使用 `delta_color="inverse"`）。
2. **兩套買點訊號保持獨立**：`model.py`（拉回買便宜）與 `keybar.py`（放量長紅突破確認）邏輯相反，必須分開計算與顯示，切勿硬乘在一起。
3. **誠實回測、禁止前視偏誤**：所有指標與財報公布日對齊，不可偷看未來資料；回測結果據實呈現（包含優勢有限或樣本過少時的提醒）。
4. **金鑰安全**：永遠不要將 `.streamlit/secrets.toml` 提交進 Git（已在 `.gitignore` 排除）。新增環境變數時請同步更新 `.streamlit/secrets.toml.example`。

## 3. 在新電腦啟動與雲端同步流程
- **第一次在新電腦拉取專案**：
  ```bash
  git clone <你的_GitHub_倉庫網址> stock-dashboard
  cd stock-dashboard
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
  # 填入你的 GEMINI_API_KEY
  uv run streamlit run app.py
  ```
- **日常跨電腦開發與自動部署到 Cloud**：
  1. 修改前先拉取最新進度：`git pull`
  2. 修改並測試通過後推送：`git add -A && git commit -m "更新說明" && git push`
  3. 推送到 GitHub 後，已綁定的 **Streamlit Community Cloud**（或 **Railway**）會自動偵測並重新部署線上網站。
