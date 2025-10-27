from typing import Any, List, Dict, Tuple
import math
import os
import time
import requests
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Body, HTTPException

# --- NEW: imports for insider helpers & time handling ---
from datetime import datetime, timedelta
# -------------------------------------------------------

# --- NEW: imports for the first-15m/15m checks endpoints ---
from datetime import time as dtime
import pytz
# -----------------------------------------------------------

app = FastAPI(title="RS5/RS10 API")

SPX = "^GSPC"  # S&P 500 index on Yahoo
WINDOWS = [5, 10]  # RS5, RS10

# ---- Finnhub token for /enrich_ext (set in Render env) ----
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "")
_ext_cache: dict[str, Tuple[float, Tuple[float | None, float | None]]] = {}
EXT_TTL = 60 * 30  # 30 minutes cache

# --- Timezone for RTH filtering (define early so helpers can use it) ---
NY_TZ = pytz.timezone("America/New_York")
# ----------------------------------------------------------------------

# ---- Insider (Finnhub) helpers & cache ----
INSIDER_TTL = 60 * 15  # 15 minutes
_insider_cache: dict[str, Tuple[float, Any]] = {}

def _finnhub_get(path: str, params: dict) -> Any:
    if not FINNHUB_TOKEN:
        raise HTTPException(status_code=500, detail="FINNHUB_TOKEN not set")
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/{path}",
            params={**params, "token": FINNHUB_TOKEN},
            timeout=12,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Finnhub error: {e}")

def _cache_get(key: str):
    now = time.time()
    if key in _insider_cache and now - _insider_cache[key][0] < INSIDER_TTL:
        return _insider_cache[key][1]
    return None

def _cache_set(key: str, value: Any):
    _insider_cache[key] = (time.time(), value)

def fetch_insider_transactions(sym: str, days: int = 90) -> List[dict]:
    """
    Finnhub: /stock/insider-transactions
    Returns list of dicts (raw), limited to last `days`.
    """
    to_dt = datetime.now(NY_TZ).date()
    from_dt = to_dt - timedelta(days=days)
    cache_key = f"ins_txn:{sym}:{from_dt}:{to_dt}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    data = _finnhub_get(
        "stock/insider-transactions",
        {"symbol": sym, "from": str(from_dt), "to": str(to_dt)},
    )
    rows = data.get("data") or []
    _cache_set(cache_key, rows)
    return rows

def summarize_insider(rows: List[dict]) -> Dict[str, Any]:
    """
    Aggregate buys/sells, net shares & notional (if price present),
    recentness, and cluster (unique buyers).
    """
    buys_shares = 0
    sells_shares = 0
    buys_notional = 0.0
    sells_notional = 0.0
    buyers = set()
    sellers = set()
    most_recent = None

    for r in rows:
        code = str(r.get("transactionCode") or r.get("type") or "").upper()  # 'P' purchase / 'S' sale
        shares = float(r.get("change") or r.get("share") or r.get("shares") or 0)
        px = float(r.get("transactionPrice") or r.get("price") or 0.0)
        who = (r.get("name") or r.get("insiderName") or "").strip()

        dstr = r.get("transactionDate") or r.get("filingDate")
        if dstr:
            try:
                d = datetime.fromisoformat(dstr[:10])
                most_recent = d if (most_recent is None or d > most_recent) else most_recent
            except ValueError:
                pass

        if code == "P":
            buys_shares += max(shares, 0)
            if px > 0 and shares > 0:
                buys_notional += px * shares
            if who:
                buyers.add(who)
        elif code == "S":
            sells_shares += max(shares, 0)
            if px > 0 and shares > 0:
                sells_notional += px * shares
            if who:
                sellers.add(who)

    net_shares = buys_shares - sells_shares
    net_notional = round(buys_notional - sells_notional, 2)

    return {
        "total_buys_shares": int(buys_shares),
        "total_sells_shares": int(sells_shares),
        "net_shares": int(net_shares),
        "buy_notional": round(buys_notional, 2),
        "sell_notional": round(sells_notional, 2),
        "net_notional": net_notional,
        "unique_buyers": len(buyers),
        "unique_sellers": len(sellers),
        "most_recent_txn": most_recent.strftime("%Y-%m-%d") if most_recent else None,
    }
# ----------------------------------------------------------

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


# -------------------- /enrich_ext --------------------

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


# -------------------- Insider endpoints --------------------

@app.post("/insider_summary")
def insider_summary_endpoint(
    payload: Any = Body(...),
    days: int = 90
) -> Dict[str, Any]:
    """
    For each ticker: returns aggregate insider activity over the last `days`.
    Body: {"tickers": ["AAPL","MSFT", ...]}
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    for sym in tickers:
        try:
            rows = fetch_insider_transactions(sym, days=days)
            summary = summarize_insider(rows)
            results.append({"ticker": sym, "days": days, **summary})
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})
    return {"days": days, "results": results}

@app.post("/insider_cluster_buys")
def insider_cluster_buys_endpoint(
    payload: Any = Body(...),
    days: int = 60,
    min_unique_buyers: int = 3,
    min_net_shares: int = 10000,
    min_net_notional: float = 250000.0
) -> Dict[str, Any]:
    """
    Flags tickers with 'cluster buys' in the last `days`.
    Criteria are configurable via query params.
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    hits = []
    for sym in tickers:
        try:
            rows = fetch_insider_transactions(sym, days=days)
            s = summarize_insider(rows)

            meets = (
                (s["unique_buyers"] >= min_unique_buyers) and
                (s["net_shares"] >= min_net_shares) and
                (s["net_notional"] >= min_net_notional)
            )

            item = {"ticker": sym, "days": days, **s, "meets_threshold": meets}
            results.append(item)
            if meets:
                hits.append(item)
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {
        "params": {
            "days": days,
            "min_unique_buyers": min_unique_buyers,
            "min_net_shares": min_net_shares,
            "min_net_notional": min_net_notional
        },
        "hits": hits,
        "results": results
    }

def fetch_insider_sentiment(sym: str) -> Dict[str, Any]:
    cache_key = f"ins_sent:{sym}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    data = _finnhub_get("stock/insider-sentiment", {"symbol": sym})
    latest = (data.get("data") or [])[-1] if (data.get("data")) else None
    out = {"latest_month": latest}
    _cache_set(cache_key, out)
    return out

@app.post("/insider_score")
def insider_score_endpoint(
    payload: Any = Body(...),
    days: int = 90,
    w_cluster: float = 0.5,
    w_net_notional: float = 0.3,
    w_sentiment: float = 0.2
) -> Dict[str, Any]:
    """
    Heuristic 0–100 score combining cluster size, net notional, and Finnhub sentiment (mspr).
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    for sym in tickers:
        try:
            rows = fetch_insider_transactions(sym, days=days)
            s = summarize_insider(rows)
            sent = fetch_insider_sentiment(sym).get("latest_month") or {}

            cluster = min(s["unique_buyers"], 8) / 8.0
            net_notional = max(min(s["net_notional"] / 2_000_000.0, 1.0), 0.0)
            mspr = float(sent.get("mspr") or 0.0)
            mspr_norm = (max(min(mspr, 100.0), -100.0) + 100.0) / 200.0

            score01 = (w_cluster * cluster) + (w_net_notional * net_notional) + (w_sentiment * mspr_norm)
            score = round(score01 * 100.0, 1)

            results.append({
                "ticker": sym,
                "days": days,
                "unique_buyers": s["unique_buyers"],
                "net_notional": s["net_notional"],
                "mspr": sent.get("mspr"),
                "score": score
            })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})
    return {"results": results}


# -------------------- Helpers for 15m RTH analytics --------------------

def rth_15m_df(sym: str, days: int = 20) -> pd.DataFrame:
    """
    Fetch ~N trading days of 15m bars, convert to America/New_York, and filter to RTH (09:30–16:00).
    """
    df = fetch(sym, period=f"{days}d", interval="15m")
    if df.empty:
        raise HTTPException(status_code=502, detail=f"No intraday data for {sym}")

    # tz-aware and ET, cash session only
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert(NY_TZ)
    df = df.between_time(dtime(9, 30), dtime(16, 0)).copy()
    df["session"] = df.index.date
    return df


def prev_vs_26(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses the most recent completed RTH bar in df. Compares its volume vs 26 bars ago.
    """
    if len(df) < 27:
        raise HTTPException(status_code=502, detail="Not enough 15m RTH bars")
    prev_bar_time = df.index[-1]
    prev_bar_vol = int(df["Volume"].iloc[-1])
    vol_26ago = int(df["Volume"].iloc[-27])
    ratio_26 = (prev_bar_vol / vol_26ago) if vol_26ago > 0 else float("inf")
    return {
        "prev_bar_time_et": prev_bar_time.strftime("%Y-%m-%d %H:%M"),
        "prev_bar_vol": prev_bar_vol,
        "vol_26bars_ago": vol_26ago,
        "ratio_prev_vs_26ago": None if math.isinf(ratio_26) else round(ratio_26, 2),
    }


def first15_stats(df: pd.DataFrame, session_key) -> Dict[str, Any]:
    grp = df.groupby("session")["Volume"]
    daily_first = grp.first()
    first15_today = int(daily_first.loc[session_key])
    prev_days_first = daily_first.drop(index=session_key) if session_key in daily_first.index else daily_first
    first15_avg10 = int(round(float(prev_days_first.tail(10).mean()))) if len(prev_days_first) else 0
    first15_prevday = int(prev_days_first.iloc[-1]) if len(prev_days_first) >= 1 else 0
    ratio_avg10 = (first15_today / first15_avg10) if first15_avg10 > 0 else float("inf")
    ratio_prevday = (first15_today / first15_prevday) if first15_prevday > 0 else float("inf")
    return {
        "today": first15_today,
        "avg10": first15_avg10,
        "ratio_vs_avg10": None if math.isinf(ratio_avg10) else round(ratio_avg10, 2),
        "prevday": first15_prevday,
        "ratio_vs_prevday": None if math.isinf(ratio_prevday) else round(ratio_prevday, 2),
    }


def last15_stats(df: pd.DataFrame, session_key) -> Dict[str, Any]:
    grp = df.groupby("session")["Volume"]
    daily_last = grp.last()

    # determine if today's close bar exists (we keep the original >=16:00 check)
    rows_today = df.loc[df["session"] == session_key]
    has_today_close_bar = False
    today_val = None
    if len(rows_today) > 0 and rows_today.index[-1].time() >= dtime(16, 0):
        has_today_close_bar = True
        today_val = int(daily_last.loc[session_key])

    prev_days_last = daily_last.drop(index=session_key) if session_key in daily_last.index else daily_last
    avg10 = int(round(float(prev_days_last.tail(10).mean()))) if len(prev_days_last) else 0
    prevday = int(prev_days_last.iloc[-1]) if len(prev_days_last) >= 1 else 0

    ratio_avg10 = (today_val / avg10) if (has_today_close_bar and avg10 > 0) else float("inf")
    ratio_prevday = (today_val / prevday) if (has_today_close_bar and prevday > 0) else float("inf")

    return {
        "has_today_close_bar": has_today_close_bar,
        "today": today_val if has_today_close_bar else None,
        "avg10": avg10 if has_today_close_bar else None,
        "ratio_vs_avg10": None if (not has_today_close_bar or math.isinf(ratio_avg10)) else round(ratio_avg10, 2),
        "prevday": prevday if has_today_close_bar else None,
        "ratio_vs_prevday": None if (not has_today_close_bar or math.isinf(ratio_prevday)) else round(ratio_prevday, 2),
    }


# -------------------- Existing first-15 spike screen (9:47 ET) --------------------

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


# -------------------- Split 15m endpoints --------------------

@app.post("/bar15_open_checks")
def bar15_open_checks_endpoint(
    payload: Any = Body(...),
    greater_multiple: float = 2.0,   # >= 2.0x vs 26 bars ago
    less_multiple: float = 0.5       # <= 0.5x vs 26 bars ago
) -> Dict[str, Any]:
    """
    Run at 09:50 ET. Returns:
      - prev bar vs 26 bars ago ratio (+ threshold booleans)
      - first 15m stats: today's vs 10-day avg and vs previous day
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    hits = []
    for sym in tickers:
        try:
            df = rth_15m_df(sym, days=20)
            meta = prev_vs_26(df)

            # thresholds
            ratio = meta["ratio_prev_vs_26ago"]
            meets_gt = (ratio is not None) and (ratio >= greater_multiple)
            meets_lt = (ratio is not None) and (ratio <= less_multiple)

            # first 15m (use today's session key from latest bar)
            today_key = df.index[-1].date()
            first = first15_stats(df, today_key)

            out = {
                "ticker": sym,
                **meta,
                "first15": first,
                "meets_gt_multiple_vs_26": meets_gt,
                "meets_lt_multiple_vs_26": meets_lt,
            }
            results.append(out)
            if meets_gt or meets_lt:
                hits.append({"ticker": sym, "ratio_prev_vs_26ago": ratio, "gt": meets_gt, "lt": meets_lt})
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {
        "greater_multiple": greater_multiple,
        "less_multiple": less_multiple,
        "hits": hits,
        "results": results
    }


@app.post("/bar15_close_checks")
def bar15_close_checks_endpoint(
    payload: Any = Body(...),
    greater_multiple: float = 2.0,   # >= 2.0x vs 26 bars ago
    less_multiple: float = 0.5       # <= 0.5x vs 26 bars ago
) -> Dict[str, Any]:
    """
    Run at 16:05 ET. Returns:
      - prev bar vs 26 bars ago ratio (+ threshold booleans)
      - last 15m stats: today's (close bar) vs 10-day avg and vs previous day
        (last15 appears only if today's 15:45–16:00 bar exists; on early-close days this remains None)
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    hits = []
    for sym in tickers:
        try:
            df = rth_15m_df(sym, days=20)
            meta = prev_vs_26(df)

            # thresholds
            ratio = meta["ratio_prev_vs_26ago"]
            meets_gt = (ratio is not None) and (ratio >= greater_multiple)
            meets_lt = (ratio is not None) and (ratio <= less_multiple)

            today_key = df.index[-1].date()
            last = last15_stats(df, today_key)

            out = {
                "ticker": sym,
                **meta,
                "last15": last,
                "meets_gt_multiple_vs_26": meets_gt,
                "meets_lt_multiple_vs_26": meets_lt,
            }
            results.append(out)
            if meets_gt or meets_lt:
                hits.append({"ticker": sym, "ratio_prev_vs_26ago": ratio, "gt": meets_gt, "lt": meets_lt})
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {
        "greater_multiple": greater_multiple,
        "less_multiple": less_multiple,
        "hits": hits,
        "results": results
    }


# -------------------- NEW: After-close daily volume spikes (default 5×) --------------------

@app.post("/close_day_volume_spikes")
def close_day_volume_spikes_endpoint(
    payload: Any = Body(...),
    multiple: float = 5.0
) -> Dict[str, Any]:
    """
    After market close (e.g., 16:05 ET), flag tickers whose **today's daily volume**
    is >= `multiple` × **previous trading day's** volume.

    Query param:
      - multiple (float): default 5.0 for 5×.

    Body:
      { "tickers": ["AAPL","MSFT", ...] }

    Output:
      {
        "multiple": 5.0,
        "spikes": [ ...only tickers meeting/exceeding threshold... ],
        "results": [ ...all tickers with volumes/ratio... ]
      }
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    spikes = []
    results = []
    for sym in tickers:
        try:
            df_d = fetch(sym, period="2mo", interval="1d")
            if len(df_d) < 2:
                raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")

            vol_today = int(df_d["Volume"].iloc[-1])
            vol_prev = int(df_d["Volume"].iloc[-2])
            ratio = (vol_today / vol_prev) if vol_prev > 0 else float("inf")

            item = {
                "ticker": sym,
                "vol_today": vol_today,
                "vol_prevday": vol_prev,
                "ratio": None if math.isinf(ratio) else round(ratio, 2),
                "meets_threshold": (ratio >= multiple) if not math.isinf(ratio) else (vol_prev == 0 and vol_today > 0),
            }
            results.append(item)
            if item["meets_threshold"]:
                spikes.append(item)
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"multiple": multiple, "spikes": spikes, "results": results}


# -------------------- NEW: After-hours price move vs regular close --------------------

def afterhours_move(sym: str, threshold: float = 1.5) -> Dict[str, Any]:
    """
    Compare latest *after-hours* price (post RTH close, up to 20:00 ET) to today's regular-session close.
    Returns dict with after-hours price, close, pct change, and threshold flag.
    """
    # --- today's official close from daily bars ---
    df_daily = fetch(sym, period="1mo", interval="1d")
    if len(df_daily) < 1:
        raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")

    close_today = float(df_daily["Close"].iloc[-1])
    # trading day (naive date) corresponding to last daily row
    session_date = df_daily.index[-1].date()

    # --- find the RTH close timestamp for that session using intraday bars ---
    # Use 15m to determine the latest RTH bar time (handles early closes gracefully)
    df_15 = fetch(sym, period="5d", interval="15m")
    if df_15.index.tz is None:
        df_15 = df_15.tz_localize("UTC")
    df_15 = df_15.tz_convert(NY_TZ)

    # limit to that session_date
    df_15["session"] = df_15.index.date
    df_sess_15 = df_15.loc[df_15["session"] == session_date]
    if df_sess_15.empty:
        raise HTTPException(status_code=502, detail=f"No intraday session data for {sym} on {session_date}")

    # RTH window (approx) — take max timestamp <= 16:00 ET if present, else last timestamp of session
    rth_end_candidate = df_sess_15.index[df_sess_15.index.time <= dtime(16, 0)]
    close_ts = rth_end_candidate.max() if len(rth_end_candidate) else df_sess_15.index.max()

    # --- get latest after-hours price up to now (cap at 20:00 ET) ---
    df_5 = fetch(sym, period="5d", interval="5m")
    if df_5.index.tz is None:
        df_5 = df_5.tz_localize("UTC")
    df_5 = df_5.tz_convert(NY_TZ)
    df_5["session"] = df_5.index.date

    now_et = datetime.now(NY_TZ)
    window_end_time = min(now_et.time(), dtime(20, 0))

    # after-hours rows: strictly after close_ts and same session_date, up to 20:00 ET (or now, whichever is earlier)
    mask = (
        (df_5["session"] == session_date) &
        (df_5.index > close_ts) &
        (df_5.index.time <= window_end_time)
    )
    df_ah = df_5.loc[mask]
    if df_ah.empty:
        # No after-hours bars yet — report gracefully
        return {
            "after_hours_available": False,
            "asof_et": None,
            "after_hours_price": None,
            "regular_close": close_today,
            "pct_change": None,
            "abs_change": None,
            "threshold": threshold,
            "meets_threshold": False
        }

    last_row = df_ah.iloc[-1]
    last_ts = df_ah.index[-1]
    last_px = float(last_row["Close"])

    pct_change = round((last_px - close_today) / close_today * 100.0, 2)
    abs_change = abs(pct_change)
    meets = abs_change >= threshold

    return {
        "after_hours_available": True,
        "asof_et": last_ts.strftime("%Y-%m-%d %H:%M"),
        "after_hours_price": last_px,
        "regular_close": close_today,
        "pct_change": pct_change,
        "abs_change": abs_change,
        "threshold": threshold,
        "meets_threshold": meets
    }


@app.post("/afterhours_price_checks")
def afterhours_price_checks_endpoint(
    payload: Any = Body(...),
    threshold: float = 1.5  # absolute % move vs regular close
) -> Dict[str, Any]:
    """
    Call at 18:05 ET and 20:05 ET.
    Flags tickers whose *after-hours* price has moved by >= `threshold` percent
    relative to today's regular-session close.

    Body: {"tickers": ["AAPL","MSFT", ...]}
    Query param: ?threshold=1.5  (absolute percent)
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    hits = []
    for sym in tickers:
        try:
            out = afterhours_move(sym, threshold=threshold)
            row = {"ticker": sym, **out}
            results.append(row)
            if out.get("after_hours_available") and out.get("meets_threshold"):
                hits.append({
                    "ticker": sym,
                    "asof_et": out.get("asof_et"),
                    "pct_change": out.get("pct_change"),
                    "abs_change": out.get("abs_change"),
                })
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {
        "threshold": threshold,
        "hits": hits,
        "results": results
    }

# -------------------- Daily consolidation + breakout/breakdown --------------------

def _select_completed_daily(df_d: pd.DataFrame, include_incomplete: bool) -> pd.DataFrame:
    """
    Returns a daily DataFrame where the last row is a completed session bar, unless
    include_incomplete=True (in which case it may include today's partial bar).
    """
    if df_d.empty:
        return df_d
    now_et = datetime.now(NY_TZ)
    market_close = dtime(16, 0)
    last_date = df_d.index[-1].date()
    # If we’re on today's bar and before market close, drop it (unless include_incomplete=true)
    if (not include_incomplete) and (now_et.date() == last_date) and (now_et.time() < market_close):
        if len(df_d) >= 2:
            return df_d.iloc[:-1]
    return df_d


def analyze_consolidation_break(
    sym: str,
    lookback: int = 21,
    max_range_pct: float = 4.0,
    min_days: int = 15,
    include_incomplete: bool = False
) -> Dict[str, Any]:
    """
    Detects consolidation over the previous `lookback` daily bars (excluding latest bar),
    then checks if the latest bar takes out the consolidation HIGH or LOW (wick-based).
    """
    if lookback < max(min_days, 2):
        raise HTTPException(status_code=400, detail="lookback must be >= max(min_days, 2)")

    df_d = fetch(sym, period="6mo", interval="1d")
    if len(df_d) < (min_days + 1):
        raise HTTPException(status_code=502, detail=f"Not enough daily data for {sym}")

    df_d = _select_completed_daily(df_d, include_incomplete=include_incomplete)
    if len(df_d) < (min_days + 1):
        raise HTTPException(status_code=502, detail=f"Not enough completed daily bars for {sym}")

    window = df_d.tail(lookback + 1)
    cons_df = window.iloc[:-1]  # consolidation range
    latest = window.iloc[-1]    # latest bar = breakout test

    cons_high = float(cons_df["High"].max())
    cons_low = float(cons_df["Low"].min())
    ref_close = float(cons_df["Close"].iloc[-1])
    range_pct = round(((cons_high - cons_low) / ref_close) * 100.0, 2) if ref_close > 0 else None

    is_consolidating = (len(cons_df) >= min_days) and (range_pct is not None) and (range_pct <= max_range_pct)

    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    latest_date = window.index[-1].strftime("%Y-%m-%d")

    broke_up = is_consolidating and (latest_high > cons_high)
    broke_down = is_consolidating and (latest_low < cons_low)

    status = "no_consolidation"
    if is_consolidating:
        if broke_up and not broke_down:
            status = "break_up"
        elif broke_down and not broke_up:
            status = "break_down"
        elif broke_down and broke_up:
            status = "both_taken_out"
        else:
            status = "consolidating"

    return {
        "ticker": sym,
        "latest_session": latest_date,
        "lookback": lookback,
        "min_days": min_days,
        "max_range_pct": max_range_pct,
        "include_incomplete": include_incomplete,
        "is_consolidating": is_consolidating,
        "status": status,
        "consolidation_high": round(cons_high, 4),
        "consolidation_low": round(cons_low, 4),
        "consolidation_range_pct": range_pct,
        "latest_high": round(latest_high, 4),
        "latest_low": round(latest_low, 4),
        "broke_up_today": broke_up,
        "broke_down_today": broke_down,
    }

# -------------------- Market direction helper (optional inversion check) --------------------
def market_direction(sym: str, lookback: int = 21, min_trend_pct: float = 0.5) -> Dict[str, Any]:
    """
    Computes the benchmark's direction over `lookback` completed daily bars.
    Returns 'up' if pct_change >= min_trend_pct,
            'down' if pct_change <= -min_trend_pct,
            'flat' otherwise.
    """
    df = fetch(sym, period="6mo", interval="1d")
    df = _select_completed_daily(df, include_incomplete=False)
    if len(df) < lookback:
        raise HTTPException(status_code=502, detail=f"Not enough daily data for benchmark {sym}")
    win = df.tail(lookback)
    first_close = float(win["Close"].iloc[0])
    last_close = float(win["Close"].iloc[-1])
    if first_close <= 0:
        raise HTTPException(status_code=502, detail=f"Invalid benchmark close for {sym}")
    pct_change = round((last_close - first_close) / first_close * 100.0, 2)
    if pct_change >= min_trend_pct:
        direction = "up"
    elif pct_change <= -min_trend_pct:
        direction = "down"
    else:
        direction = "flat"
    return {"benchmark": sym, "lookback": lookback, "pct_change": pct_change, "direction": direction}


@app.post("/daily_consolidation_breaks")
def daily_consolidation_breaks_endpoint(
    payload: Any = Body(...),
    # consolidation params (same defaults)
    lookback: int = 21,
    max_range_pct: float = 4.0,
    min_days: int = 15,
    include_incomplete: bool = False,
    # OPTIONAL market inversion params (leave benchmark="" to disable)
    benchmark: str = "",            # e.g. "QQQ" or "^NDX"; empty string disables inversion check
    market_lookback: int = 21,
    min_trend_pct: float = 0.5,
    require_membership: bool = False   # if True, only flag inversion when ticker in provided list
) -> Dict[str, Any]:
    """
    Detects consolidation on daily candles and signals when highs/lows get taken out.
    Also (optionally) flags if the break is *inversing the market* relative to a benchmark.

    Inversion rules (only when `benchmark` is provided):
      - If market direction == 'up' AND stock breaks *down* -> inversing
      - If market direction == 'down' AND stock breaks *up* -> inversing

    Membership filter (optional):
      - If require_membership=true, pass 'ndx_members' or 'universe' list in the body.
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    # Optional membership/universe provided by caller (e.g., NDX/QQQ constituents)
    members = set()
    if isinstance(payload, dict):
        if isinstance(payload.get("ndx_members"), list):
            members = {str(x).upper() for x in payload["ndx_members"]}
        elif isinstance(payload.get("universe"), list):
            members = {str(x).upper() for x in payload["universe"]}

    # Compute market direction once if benchmark is provided
    bench = None
    if isinstance(benchmark, str) and benchmark.strip():
        bench = market_direction(sym=benchmark.strip(), lookback=market_lookback, min_trend_pct=min_trend_pct)

    results = []
    hits = []

    for sym in tickers:
        try:
            base = analyze_consolidation_break(
                sym,
                lookback=lookback,
                max_range_pct=max_range_pct,
                min_days=min_days,
                include_incomplete=include_incomplete
            )

            # Defaults (no inversion check)
            inversing_market = None
            inverse_reason = None
            is_member = True  # default allow unless require_membership gates it

            if bench is not None:
                # Apply membership gate if requested
                is_member = (not require_membership) or (sym.upper() in members)

                direction = bench["direction"]
                # Only count a break if the status indicates an actual break today
                broke_up = base["broke_up_today"] and base["status"] in ("break_up", "both_taken_out")
                broke_down = base["broke_down_today"] and base["status"] in ("break_down", "both_taken_out")

                if is_member:
                    if direction == "up" and broke_down:
                        inversing_market = True
                        inverse_reason = "market_up_stock_breaks_down"
                    elif direction == "down" and broke_up:
                        inversing_market = True
                        inverse_reason = "market_down_stock_breaks_up"
                    else:
                        inversing_market = False  # checked but not inversing

            row = {
                "ticker": sym,
                **base,
                "benchmark": bench if bench is not None else None,
                "require_membership": require_membership if bench is not None else None,
                "is_member": is_member if bench is not None else None,
                "inversing_market": inversing_market,  # None = not evaluated; True/False = evaluated
                "inverse_reason": inverse_reason,
            }
            results.append(row)

            if bench is not None and inversing_market:
                hits.append({
                    "ticker": sym,
                    "status": base["status"],
                    "inverse_reason": inverse_reason,
                    "consolidation_high": base["consolidation_high"],
                    "consolidation_low": base["consolidation_low"],
                })

        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {
        "params": {
            "lookback": lookback,
            "min_days": min_days,
            "max_range_pct": max_range_pct,
            "include_incomplete": include_incomplete,
            "benchmark": (benchmark if bench is not None else ""),
            "market_lookback": (market_lookback if bench is not None else None),
            "min_trend_pct": (min_trend_pct if bench is not None else None),
            "require_membership": (require_membership if bench is not None else None),
        },
        "hits": hits,
        "results": results
    }
