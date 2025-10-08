# tools/hist_bootstrap.py
# pip install pandas numpy

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def _ensure_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_index()
    if "Adj Close" not in d.columns:
        d["Adj Close"] = d["Close"]
    return d

def _forward_returns(adj: pd.Series, horizon_days: int) -> pd.Series:
    """
    Forward trading-day returns: R_t = AdjClose[t+h] / AdjClose[t] - 1
    """
    fr = adj.shift(-horizon_days) / adj - 1.0
    return fr.dropna()

def _winsorize(s: pd.Series, pct: float) -> pd.Series:
    """
    Clip tails at e.g. pct=0.01 => 1% & 99% quantiles.
    """
    if pct is None or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)

def estimate_hist_1m_bands(
    prices_df: pd.DataFrame,
    horizon_days: int = 21,
    lookback_years: int = 5,
    winsor_pct: float = 0.0,   # 0.01 = 1% winsorization
    extra_percentiles: Optional[list] = None
) -> Dict[str, Any]:
    """
    Compute empirical forward-return bands over a lookback window.

    Input:
      prices_df: DataFrame with at least ['Close','Adj Close'] columns, index=dates (daily).
      horizon_days: forward horizon in trading days (21 ~ 1 month).
      lookback_years: limit history to last N years (use 0 or None for all data).
      winsor_pct: optional tail clipping to reduce outliers.
      extra_percentiles: e.g., [5, 25, 75, 95] to include more quantiles.

    Output:
      hist_bands: dict with p10/p50/p90, optional extras, and price band (low/base/high).
    """
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df is empty.")

    df = _ensure_adj_close(prices_df)
    adj = df["Adj Close"].dropna()

    # Restrict to lookback window (calendar years)
    if lookback_years and lookback_years > 0:
        cutoff = adj.index.max() - pd.DateOffset(years=lookback_years)
        adj = adj.loc[adj.index >= cutoff]

    if len(adj) <= horizon_days + 5:
        raise ValueError(f"Not enough data in lookback to compute {horizon_days}-day forward returns.")

    # Build forward return series
    fwd = _forward_returns(adj, horizon_days=horizon_days)
    if fwd.empty:
        raise ValueError("Forward returns series is empty after processing.")

    # Optional winsorization (light outlier control)
    fwd_w = _winsorize(fwd, winsor_pct)

    # Percentiles
    p10 = float(np.percentile(fwd_w, 10))
    p50 = float(np.percentile(fwd_w, 50))
    p90 = float(np.percentile(fwd_w, 90))
    extras = {}
    if extra_percentiles:
        for p in extra_percentiles:
            try:
                extras[str(p)] = float(np.percentile(fwd_w, p))
            except Exception:
                extras[str(p)] = None

    # Convert to price bands
    last_price = float(adj.iloc[-1])
    low  = last_price * (1.0 + p10)
    base = last_price * (1.0 + p50)
    high = last_price * (1.0 + p90)

    hist_bands = {
        "as_of": adj.index[-1].strftime("%Y-%m-%d"),
        "last_price": last_price,
        "params": {
            "horizon_days": horizon_days,
            "lookback_years": lookback_years,
            "winsor_pct": winsor_pct
        },
        "sample_size": int(len(fwd_w)),
        "return_percentiles": {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            **({"extras": extras} if extras else {})
        },
        "price_bands": {
            "low": float(low),
            "base": float(base),
            "high": float(high)
        }
    }
    return hist_bands
