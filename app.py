from typing import Any, List, Dict
import math
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Body, HTTPException

app = FastAPI(title="RS5/RS10 API")

SPX = "^GSPC"  # S&P 500 index on Yahoo
WINDOWS = [5, 10]  # RS5, RS10


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

            # (Optional) force negative sign for down gaps:
            # if gap_down and gap_pct > 0:
            #     gap_pct = -gap_pct

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
