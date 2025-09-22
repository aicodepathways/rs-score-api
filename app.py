from typing import Any, List, Dict
import math
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Body, HTTPException

app = FastAPI(title="RS5/RS10 API")

SPX = "^GSPC"  # S&P 500 index on Yahoo
WINDOWS = [5, 10]  # RS5, RS10

def fetch(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise HTTPException(status_code=502, detail=f"No data for {symbol}")
    return df

def rs_rank(df_sym: pd.DataFrame, df_spx: pd.DataFrame, length: int) -> float | None:
    # Get Adj Close as Series
    s1 = df_sym["Adj Close"].copy()
    s2 = df_spx["Adj Close"].copy()

    # Align and FORCE column names
    df = pd.concat([s1, s2], axis=1).dropna()
    df.columns = ["sym", "spx"]  # ensure we have these names

    if len(df) < length:
        return None

    window = df.tail(length)
    rs = window["sym"] / window["spx"]
    lo, hi = rs.min(), rs.max()
    if math.isclose(hi - lo, 0.0):
        return 50.0  # neutral if flat
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
