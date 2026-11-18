#!/bin/zsh
# 雙擊這個檔案就能啟動「股票分析小幫手」，瀏覽器會自動打開。
# 關閉方式：關掉這個終端機視窗，或按 Ctrl + C。
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "📈 正在啟動股票分析小幫手…（第一次會比較久）"
uv run streamlit run app.py --server.headless false --browser.gatherUsageStats false
