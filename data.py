"""資料取得：股票代號解析、中文名稱對照、從 Yahoo Finance 下載歷史股價（含本機硬碟快取與台股當日最新收盤補齊）。"""

from __future__ import annotations

import pathlib
import time

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = pathlib.Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# 常見股票的中文名稱（顯示用，也可以直接輸入中文名稱查詢）
NAME_MAP: dict[str, str] = {
    # 台股 ETF
    "0050": "元大台灣50",
    "006208": "富邦台50",
    "0056": "元大高股息",
    "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息",
    "00981A": "主動統一台股增長",
    # 台股個股
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2382": "廣達",
    "2327": "國巨",
    "3231": "緯創",
    "6669": "緯穎",
    "2345": "智邦",
    "3017": "奇鋐",
    "2303": "聯電",
    "3711": "日月光投控",
    "2357": "華碩",
    "2412": "中華電",
    "2881": "富邦金",
    "2882": "國泰金",
    "2891": "中信金",
    "1301": "台塑",
    "2603": "長榮",
    # 美股
    "NVDA": "輝達",
    "AAPL": "蘋果",
    "MSFT": "微軟",
    "GOOGL": "Google",
    "AMZN": "亞馬遜",
    "META": "Meta",
    "AVGO": "博通",
    "AMD": "超微",
    "TSLA": "特斯拉",
    "TSM": "台積電 ADR",
    "SPY": "S&P 500 ETF",
    "QQQ": "那斯達克100 ETF",
}

# 中文名稱 → 代號（「台積電」對應台股 2330，不是 ADR）
ALIASES: dict[str, str] = {
    name: code for code, name in NAME_MAP.items() if code != "TSM"
}

MAX_YEARS = 15  # 最多保留 15 年資料（回測用）
TW_NAMES_FILE = CACHE_DIR / "tw_names.json"  # 由 scanner.fetch_quotes 從證交所／櫃買清單產生
_tw_cache: dict = {}


def tw_names() -> dict[str, list[str]]:
    """全部上市櫃股票：代號 → [名稱, 市場 TW/TWO]（沒有快取檔時回傳空字典）。"""
    if not _tw_cache and TW_NAMES_FILE.exists():
        try:
            import json

            _tw_cache.update(json.loads(TW_NAMES_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return _tw_cache


def parse_inputs(raw: str) -> list[str]:
    """把使用者輸入（逗號、空白、頓號分隔）拆成清單，去除重複。英文代號一律轉大寫。"""
    for sep in ["，", "、", ";", "；", " ", "\n"]:
        raw = raw.replace(sep, ",")
    items: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isascii():
            part = part.upper()
        if part and part not in items:
            items.append(part)
    return items


def _candidates(user_input: str) -> list[str]:
    """把使用者輸入轉成可能的 Yahoo Finance 代號。"""
    s = user_input.strip()
    s = ALIASES.get(s, s)
    if not s.isascii():  # 中文名稱：查全部上市櫃清單
        for code, (name, mkt) in tw_names().items():
            if name == s:
                return [f"{code}.{mkt}"]
        for code, (name, mkt) in tw_names().items():
            if s in name:
                return [f"{code}.{mkt}"]
    s = s.upper()
    if "." in s:  # 使用者已指定交易所後綴，例如 2330.TW
        return [s]
    if s[:1].isdigit():  # 台股：已知市場就直接用，否則先試上市 .TW，再試上櫃 .TWO
        known = tw_names().get(s)
        if known:
            return [f"{s}.{known[1]}"]
        return [f"{s}.TW", f"{s}.TWO"]
    return [s]  # 美股或其他


def display_name(ticker: str) -> str:
    code = ticker.split(".")[0]
    if code in NAME_MAP:
        return f"{NAME_MAP[code]}（{code}）"
    if code in tw_names():
        return f"{tw_names()[code][0]}（{code}）"
    try:
        info = yf.Ticker(ticker).get_info()
        short = info.get("shortName") or info.get("longName")
        if short:
            return f"{short}（{code}）"
    except Exception:
        pass
    return code


def _download(ticker: str, attempts: int = 3) -> pd.DataFrame | None:
    """Yahoo Finance 偶爾會暫時失敗，最多重試 3 次。"""
    for i in range(attempts):
        try:
            df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1.2 * (i + 1))
    return None


def _patch_latest_taiwan_bar(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo 台股日 K 常慢一天；若 FinMind 有更新的交易日且無除權息跳空，自動補上最新日 K。"""
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")) or df.empty:
        return df
    sid = ticker.split(".")[0]
    last_dt = df.index.max()
    start_dt = (last_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            FINMIND_URL,
            params={"dataset": "TaiwanStockPrice", "data_id": sid, "start_date": start_dt},
            timeout=15,
        )
        j = r.json()
        if j.get("status") != 200 or not j.get("data"):
            return df
        fm = pd.DataFrame(j["data"])
        fm["date"] = pd.to_datetime(fm["date"])
        fm = fm.sort_values("date").set_index("date")
        fm = fm[(fm["close"] > 0) & (fm["Trading_Volume"] > 0)]
        if fm.empty or fm.index.max() <= last_dt:
            return df
        # 比對重疊日收盤價，確認還原權息比例一致
        overlap = fm.index.intersection(df.index)
        ratio = 1.0
        if len(overlap) > 0:
            ref = overlap[-1]
            r_val = float(df.loc[ref, "Close"]) / float(fm.loc[ref, "close"])
            if 0.85 <= r_val <= 1.15:
                ratio = r_val
            else:
                return df
        new_rows = fm[fm.index > last_dt]
        patch = pd.DataFrame(
            {
                "Open": new_rows["open"].astype(float) * ratio,
                "High": new_rows["max"].astype(float) * ratio,
                "Low": new_rows["min"].astype(float) * ratio,
                "Close": new_rows["close"].astype(float) * ratio,
                "Volume": new_rows["Trading_Volume"].astype(float),
            },
            index=new_rows.index,
        )
        return pd.concat([df, patch]).sort_index()
    except Exception:
        return df


def fetch_history(user_input: str) -> tuple[str, pd.DataFrame]:
    """下載歷史股價（已還原除權息，並自動快取至本機硬碟）。找不到時拋出 ValueError。"""
    candidates = _candidates(user_input)
    for n, ticker in enumerate(candidates):
        cache_file = CACHE_DIR / f"yf_{ticker.replace('.', '_')}.parquet"
        df: pd.DataFrame | None = None
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 2.0:
                try:
                    df = pd.read_parquet(cache_file)
                except Exception:
                    df = None
        if df is None:
            raw = _download(ticker, attempts=3 if n == len(candidates) - 1 else 2)
            if raw is not None and not raw.empty:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df["Close"].notna() & (df["Close"] > 0)]
                df = _patch_latest_taiwan_bar(ticker, df)
                cutoff = df.index.max() - pd.DateOffset(years=MAX_YEARS)
                df = df[df.index >= cutoff]
                if len(df) >= 30:
                    try:
                        df.to_parquet(cache_file)
                    except Exception:
                        pass
            elif cache_file.exists():
                # 網路暫時異常時退回本機快取
                try:
                    df = pd.read_parquet(cache_file)
                except Exception:
                    df = None
        if df is not None and len(df) >= 30:
            return ticker, df
    raise ValueError(f"找不到「{user_input}」的股價資料，請確認代號是否正確。")
