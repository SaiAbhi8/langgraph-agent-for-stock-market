# tools/events_window.py
# pip install yfinance pandas numpy

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf

def normalize_symbol(sym: str) -> str:
    """Add .NS for NSE equities; keep index tickers like ^NSEI as-is."""
    t = sym.strip().upper()
    return t if t.startswith("^") or t.endswith(".NS") else f"{t}.NS"

def _today_india() -> pd.Timestamp:
    """Date (midnight) in Asia/Kolkata."""
    return pd.Timestamp.today(tz="Asia/Kolkata").normalize()

def _parse_earnings_dates(tkr: yf.Ticker) -> Optional[pd.DatetimeIndex]:
    """
    Try to get a DatetimeIndex of earnings dates.
    yfinance.get_earnings_dates() returns a DF with index as dates on newer versions.
    Fallback tries to coerce any 'Earnings Date' column if present.
    """
    try:
        ed = tkr.get_earnings_dates(limit=12)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            if isinstance(ed.index, pd.DatetimeIndex):
                return ed.index.tz_localize(None).sort_values()
            if "Earnings Date" in ed.columns:
                return pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(None).sort_values()
            # last resort: first column
            return pd.to_datetime(ed.iloc[:, 0], errors="coerce").dropna().tz_localize(None).sort_values()
    except Exception:
        pass
    return None

def _estimate_next_ex_div(divs: pd.Series, now: pd.Timestamp) -> Dict[str, Optional[Any]]:
    """
    Estimate the next ex-div date from the median gap between past dividends.
    Returns:
      { 'last_div_date', 'last_div_amount', 'days_since_last_div',
        'est_next_ex_div', 'days_to_est_ex_div', 'in_exdiv_window_est',
        'ttm_dividend_sum' }
    """
    out = {
        "last_div_date": None,
        "last_div_amount": None,
        "days_since_last_div": None,
        "est_next_ex_div": None,
        "days_to_est_ex_div": None,
        "in_exdiv_window_est": False,
        "ttm_dividend_sum": 0.0,
    }
    if divs is None or divs.empty:
        return out

    divs = divs.sort_index()
    out["last_div_date"] = divs.index.max().date().isoformat()
    out["last_div_amount"] = float(divs.iloc[-1])
    out["days_since_last_div"] = int((now - divs.index.max().normalize()).days)

    # TTM dividend sum (calendar 365 days)
    cutoff_365 = now - pd.Timedelta(days=365)
    out["ttm_dividend_sum"] = float(divs.loc[divs.index >= cutoff_365].sum())

    # Estimate cycle from historical gaps (need ≥3 dividends to be meaningful)
    if len(divs) >= 3:
        gaps = divs.index.to_series().diff().dropna().dt.days
        if not gaps.empty:
            median_gap = int(np.median(gaps))
            est = divs.index.max() + pd.Timedelta(days=median_gap)
            if est >= now - pd.Timedelta(days=7):  # sanity: ignore ancient artifacts
                out["est_next_ex_div"] = est.date().isoformat()
                out["days_to_est_ex_div"] = int((est.normalize() - now).days)
    return out

def load_events_window(
    ticker: str,
    earnings_window_days: int = 21,
    exdiv_window_days: int = 30,
    lookback_years: int = 5,
    last_price: Optional[float] = None  # optional for ex-div tilt computation
) -> Dict[str, Any]:
    """
    Pull near-term catalysts from Yahoo:
      - next/last earnings (if available)
      - dividend history with estimated next ex-div date from median past gap
      - recent split
    Also suggests tiny widen/tilt heuristics for 1-month bands.
    """
    sym = normalize_symbol(ticker)
    now = _today_india()

    tkr = yf.Ticker(sym)

    # --- Earnings dates ---
    next_earnings_date = None
    last_earnings_date = None
    days_to_earnings = None
    in_earnings_window = False

    dates = _parse_earnings_dates(tkr)
    if isinstance(dates, pd.DatetimeIndex) and len(dates) > 0:
        future = [d for d in dates if d >= now.tz_localize(None)]
        past   = [d for d in dates if d <  now.tz_localize(None)]
        if future:
            next_earnings_date = future[0].date().isoformat()
            days_to_earnings = int((future[0].normalize() - now).days)
            in_earnings_window = (0 <= days_to_earnings <= earnings_window_days)
        if past:
            last_earnings_date = past[-1].date().isoformat()

    # --- Dividends (history) & estimated next ex-div ---
    divs = None
    try:
        divs = tkr.dividends
        if isinstance(divs, pd.Series) and not divs.empty and lookback_years and lookback_years > 0:
            cutoff = now - pd.DateOffset(years=lookback_years)
            divs = divs[divs.index >= cutoff]
    except Exception:
        pass

    div_info = _estimate_next_ex_div(divs, now) if isinstance(divs, pd.Series) else {
        "last_div_date": None, "last_div_amount": None, "days_since_last_div": None,
        "est_next_ex_div": None, "days_to_est_ex_div": None, "in_exdiv_window_est": False,
        "ttm_dividend_sum": 0.0
    }

    # Flag if estimated ex-div is within window (only if estimate exists and is in the future)
    in_exdiv_window_est = False
    if div_info["days_to_est_ex_div"] is not None:
        in_exdiv_window_est = (0 <= div_info["days_to_est_ex_div"] <= exdiv_window_days)
    div_info["in_exdiv_window_est"] = in_exdiv_window_est

    # --- Splits (recent info) ---
    last_split_date, last_split_ratio = None, None
    try:
        splits = tkr.splits
        if isinstance(splits, pd.Series) and not splits.empty:
            last_split_date = splits.index.max().date().isoformat()
            last_split_ratio = float(splits.iloc[-1])  # e.g., 2.0 for 2-for-1
    except Exception:
        pass

    # --- Heuristics: suggest widen/tilt (small, transparent) ---
    widen_bp = 0.0   # additive to band width (e.g., +0.015 = +1.5%)
    tilt = 0.0       # directional tilt to band center (e.g., -0.003 = -0.3%)
    reasons = []

    if in_earnings_window:
        widen_bp += 0.015  # widen ~150 bps if earnings within window
        reasons.append(f"Earnings in {days_to_earnings}d (≤{earnings_window_days}d).")

    # Ex-div estimation: small widening, small negative tilt proportional to dividend/price
    if in_exdiv_window_est and last_price is not None and div_info["last_div_amount"] is not None:
        drop_pct = float(div_info["last_div_amount"]) / float(last_price) if last_price else 0.0
        drop_pct = min(drop_pct, 0.02)  # cap to 2%
        if drop_pct > 0:
            widen_bp += min(0.008, drop_pct)     # widen up to 80 bps
            tilt     -= round(0.5 * drop_pct, 6) # half of expected drop spread across the month
            reasons.append(f"Estimated ex-div in {div_info['days_to_est_ex_div']}d; last dividend ≈ {div_info['last_div_amount']}.")

    # Recent split: tiny widen
    if last_split_date is not None:
        # If the split was within ~90 days, add a small widen
        try:
            split_dt = pd.to_datetime(last_split_date)
            if (now - split_dt).days <= 90:
                widen_bp += 0.003
                reasons.append("Recent split (≤90d).")
        except Exception:
            pass

    event_meta: Dict[str, Any] = {
        "as_of": now.date().isoformat(),
        "yahoo_symbol": sym,

        "earnings": {
            "next_earnings_date": next_earnings_date,
            "days_to_next_earnings": days_to_earnings,
            "in_earnings_window": in_earnings_window,
            "last_earnings_date": last_earnings_date,
            "window_days": earnings_window_days,
        },

        "dividends": {
            "last_div_date": div_info["last_div_date"],
            "last_div_amount": div_info["last_div_amount"],
            "days_since_last_div": div_info["days_since_last_div"],
            "ttm_dividend_sum": div_info["ttm_dividend_sum"],
            "est_next_ex_div": div_info["est_next_ex_div"],
            "days_to_est_ex_div": div_info["days_to_est_ex_div"],
            "in_exdiv_window_est": div_info["in_exdiv_window_est"],
            "window_days": exdiv_window_days,
            "estimated_from_history": True
        },

        "splits": {
            "last_split_date": last_split_date,
            "last_split_ratio": last_split_ratio
        },

        "suggestions": {
            "widen_bp": round(float(widen_bp), 6),  # e.g., 0.015 = +1.5%
            "tilt": round(float(tilt), 6),          # e.g., -0.003 = -0.3%
            "reasons": reasons
        }
    }
    return event_meta
