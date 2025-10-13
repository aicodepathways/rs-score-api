from typing import Any, List, Dict, Tuple
import math
import os
import time
import requests
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Body, HTTPException

# --- NEW: imports for the first-15m endpoint ---
from datetime import time as dtime
import pytz
# ----------------------------------------------

app = FastAPI(title="RS5/RS10 API")

SPX = "^GSPC"  # S&P 500 index on Yahoo
WINDOWS = [5, 10]  # RS5, RS10

# ---- Finnhub token for /enrich_ext (set in Render env) ----
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "")
_ext_cache: dict[str, Tuple[float, Tuple[float | None, float | None]]] = {}
EXT_TTL = 60 * 30  # 30 minutes cache

# --- NEW: timezone for RTH filtering in first-15m endpoint ---
NY_TZ = pytz.timezone("America/New_York")
# ------------------------------------------------------------


def fetch(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise HTTPException(status_code=502, detail=f"No data for {symbol}")
    return df


def rs_rank(df_sym: pd.DataFrame, df_spx: pd.DataFrame, length: int) -> float | None:
    s1 = df_sym["Adj Close"].copy()
    s2 = df_spx["Adj Close"].copy()

    df = pd.concat([s1, s2], axis=1).dropna()
    df.columns = ["sym", "spx"]

    if len(df) < length:
        return None

    window = df.tail(length)
    rs = window["sym"] / window["spx"]
    lo, hi = rs.min(), rs.max()
    if math.isclose(hi - lo, 0.0):
        return 50.0
    latest = rs.iloc[-1]
    return round((latest - lo) / (hi - lo) * 100.0, 1)


def normalize(payload: Any) -> List[str]:
    # Accepts {"tickers":[...]}, or [{"Ticker":[...]}], or ["AAPL","MSFT"]
    if isinstance(payload, dict) and "tickers" in payload:
        seq = payload["tickers"]
    elif isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict) and "Ticker" in payload[0]:
        seq = payload[0]["Ticker"]
    elif isinstance(payload, list):
        seq = payload
    else:
        raise HTTPException(status_code=400, detail="Send {'tickers':[...]} or [{'Ticker':[...]}] or a list.")

    out = []
    for t in seq:
        if isinstance(t, str) and t.strip():
            tt = t.strip().upper()
            if tt not in out:
                out.append(tt)
    return out


@app.post("/rs")
def rs_endpoint(payload: Any = Body(...)) -> Dict[str, Any]:
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")
    df_spx = fetch(SPX)
    results = []
    for sym in tickers:
        try:
            df = fetch(sym)
            results.append({
                "ticker": sym,
                "RS5": rs_rank(df, df_spx, 5),
                "RS10": rs_rank(df, df_spx, 10),
            })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})
    return {"results": results}


@app.post("/enrich")
def enrich_endpoint(payload: Any = Body(...)) -> Dict[str, Any]:
    """
    Enrich each ticker with:
    - Gap % (today's open vs yesterday's close)
    - First 15m volume vs average first-15m volume over last 10 trading days
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    for sym in tickers:
        try:
            # ----- Gap: yesterday's close vs today's open (daily data)
            df_daily = fetch(sym, period="1mo", interval="1d")
            if len(df_daily) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")
            yesterday_close = float(df_daily["Close"].iloc[-2])
            today_open = float(df_daily["Open"].iloc[-1])
            gap_pct = round((today_open - yesterday_close) / yesterday_close * 100.0, 2)
            gap_up = today_open > yesterday_close

            # ----- First 15m volume today vs 10-day avg (15m intraday)
            df_intraday = fetch(sym, period="11d", interval="15m")
            if df_intraday.empty:
                raise HTTPException(status_code=502, detail=f"No intraday data for {sym}")

            # First 15m bar per session as scalars
            df_intraday["date"] = df_intraday.index.date
            daily_first = df_intraday.groupby("date")["Volume"].first()

            if len(daily_first) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough intraday history for {sym}")

            vol_today = int(daily_first.iloc[-1])
            # average of prior up to 10 sessions’ first-15m volume
            prior = daily_first.iloc[:-1].tail(10)
            vol_avg10 = int(round(float(prior.mean()))) if len(prior) > 0 else 0

            results.append({
                "ticker": sym,
                "gap_percent": gap_pct,
                "gap_up": gap_up,
                "vol_15m_today": vol_today,
                "vol_15m_avg10": vol_avg10,
            })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"results": results}


@app.post("/enrich_bear")
def enrich_bear_endpoint(payload: Any = Body(...)) -> Dict[str, Any]:
    """
    Bear-case enrichment:
    - Gap % (today's open vs yesterday's close), plus a boolean `gap_down`
    - First 15m volume vs average first-15m volume over last 10 trading days
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    for sym in tickers:
        try:
            # ----- Gap: yesterday's close vs today's open (daily data)
            df_daily = fetch(sym, period="1mo", interval="1d")
            if len(df_daily) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")
            yesterday_close = float(df_daily["Close"].iloc[-2])
            today_open = float(df_daily["Open"].iloc[-1])
            gap_pct = round((today_open - yesterday_close) / yesterday_close * 100.0, 2)
            gap_down = today_open < yesterday_close

            # ----- First 15m volume today vs 10-day avg (15m intraday)
            df_intraday = fetch(sym, period="11d", interval="15m")
            if df_intraday.empty:
                raise HTTPException(status_code=502, detail=f"No intraday data for {sym}")

            df_intraday["date"] = df_intraday.index.date
            daily_first = df_intraday.groupby("date")["Volume"].first()
            if len(daily_first) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough intraday history for {sym}")

            vol_today = int(daily_first.iloc[-1])
            prior = daily_first.iloc[:-1].tail(10)
            vol_avg10 = int(round(float(prior.mean()))) if len(prior) > 0 else 0

            results.append({
                "ticker": sym,
                "gap_percent": gap_pct,
                "gap_down": gap_down,
                "vol_15m_today": vol_today,
                "vol_15m_avg10": vol_avg10,
            })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"results": results}


# -------------------- NEW: /enrich_ext --------------------

def fetch_external(sym: str) -> Tuple[float | None, float | None]:
    """
    Returns (ExternalSellside, ExternalBuyside) in [0,1] using Finnhub analyst
    recommendations. Falls back to (None, None) if unavailable.
    """
    now = time.time()
    if sym in _ext_cache and now - _ext_cache[sym][0] < EXT_TTL:
        return _ext_cache[sym][1]

    if not FINNHUB_TOKEN:
        _ext_cache[sym] = (now, (None, None))
        return (None, None)

    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": sym, "token": FINNHUB_TOKEN},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            _ext_cache[sym] = (now, (None, None))
            return (None, None)

        snap = data[0]  # latest snapshot
        buy = int(snap.get("buy", 0))
        hold = int(snap.get("hold", 0))
        sell = int(snap.get("sell", 0))
        total = buy + hold + sell
        if total == 0:
            _ext_cache[sym] = (now, (None, None))
            return (None, None)

        buy_score = round(buy / total, 3)
        sell_score = round(sell / total, 3)

        _ext_cache[sym] = (now, (sell_score, buy_score))
        return (sell_score, buy_score)

    except requests.RequestException:
        _ext_cache[sym] = (now, (None, None))
        return (None, None)


@app.post("/enrich_ext")
def enrich_ext_endpoint(payload: Any = Body(...)) -> Dict[str, Any]:
    """
    Like /enrich, but also includes:
      - ExternalSellside / ExternalBuyside

    Behavior:
      1) If payload.externals[ticker] is provided, use those values.
      2) Otherwise, fetch from Finnhub analyst recommendations and normalize.

    Request examples:
      { "tickers": ["ANGO","CATX"] }
      { "tickers": ["AAPL"], "externals": {"AAPL": {"ExternalSellside": 0.2, "ExternalBuyside": 0.7}} }
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    externals: Dict[str, Dict[str, float]] = {}
    if isinstance(payload, dict) and isinstance(payload.get("externals"), dict):
        externals = payload["externals"]

    results = []
    for sym in tickers:
        try:
            # ----- Gap: yesterday's close vs today's open (daily data)
            df_daily = fetch(sym, period="1mo", interval="1d")
            if len(df_daily) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")
            yesterday_close = float(df_daily["Close"].iloc[-2])
            today_open = float(df_daily["Open"].iloc[-1])
            gap_pct = round((today_open - yesterday_close) / yesterday_close * 100.0, 2)
            gap_up = today_open > yesterday_close

            # ----- First 15m volume today vs 10-day avg (15m intraday)
            df_intraday = fetch(sym, period="11d", interval="15m")
            if df_intraday.empty:
                raise HTTPException(status_code=502, detail=f"No intraday data for {sym}")

            df_intraday["date"] = df_intraday.index.date
            daily_first = df_intraday.groupby("date")["Volume"].first()
            if len(daily_first) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough intraday history for {sym}")

            vol_today = int(daily_first.iloc[-1])
            prior = daily_first.iloc[:-1].tail(10)
            vol_avg10 = int(round(float(prior.mean()))) if len(prior) > 0 else 0

            # ----- External signals: prefer client-provided, else fetch
            ext = externals.get(sym, {})
            ext_sell = ext.get("ExternalSellside")
            ext_buy = ext.get("ExternalBuyside")
            if ext_sell is None or ext_buy is None:
                ext_sell, ext_buy = fetch_external(sym)

            results.append({
                "ticker": sym,
                "gap_percent": gap_pct,
                "gap_up": gap_up,
                "vol_15m_today": vol_today,
                "vol_15m_avg10": vol_avg10,
                "ExternalSellside": ext_sell,
                "ExternalBuyside": ext_buy,
            })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"results": results}


# -------------------- NEW: first 15-minute volume spikes --------------------

def first_15m_volumes(sym: str) -> Tuple[int, int, float]:
    """
    Returns (vol_today, vol_avg10, ratio).
    Uses 15m bars, RTH only (09:30–16:00 ET), first bar per session.
    """
    df = fetch(sym, period="11d", interval="15m")
    if df.empty:
        raise HTTPException(status_code=502, detail=f"No intraday data for {sym}")

    # ensure tz-aware -> NY time, then restrict to cash session
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert(NY_TZ)
    df_rth = df.between_time(dtime(9, 30), dtime(16, 0))

    # first 15m bar per RTH session
    df_rth = df_rth.copy()
    df_rth["session"] = df_rth.index.date
    daily_first = df_rth.groupby("session")["Volume"].first()
    if len(daily_first) < 2:
        raise HTTPException(status_code=502, detail=f"Not enough RTH sessions for {sym}")

    vol_today = int(daily_first.iloc[-1])
    prior = daily_first.iloc[:-1].tail(10)  # up to last 10 sessions
    vol_avg10 = int(round(float(prior.mean()))) if len(prior) > 0 else 0
    ratio = (vol_today / vol_avg10) if vol_avg10 > 0 else float("inf")
    return vol_today, vol_avg10, ratio


@app.post("/first15_spikes")
def first15_spikes_endpoint(
    payload: Any = Body(...),
    multiple: float = 2.0   # X times the 10-day avg (e.g., 2.0 = 200%)
) -> Dict[str, Any]:
    """
    Screen for tickers where today's first 15m volume >= multiple * avg(first 15m volume of last 10 sessions).
    Request body: {"tickers": ["AAPL","MSFT",...]}  (same normalize() as your other endpoints)
    Optional query param: ?multiple=2.5
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    spikes = []  # tickers that meet/exceed the multiple
    for sym in tickers:
        try:
            vol_today, vol_avg10, ratio = first_15m_volumes(sym)
            item = {
                "ticker": sym,
                "vol_15m_today": vol_today,
                "vol_15m_avg10": vol_avg10,
                "ratio": None if math.isinf(ratio) else round(ratio, 2),
                "meets_threshold": (ratio >= multiple) if not math.isinf(ratio) else (vol_avg10 == 0 and vol_today > 0)
            }
            results.append(item)
            if item["meets_threshold"]:
                spikes.append(item)
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"multiple": multiple, "spikes": spikes, "results": results}
