# tools/volatility_range.py
# pip install pandas numpy

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def _daily_log_returns(adj_close: pd.Series) -> pd.Series:
    r = np.log(adj_close / adj_close.shift(1))
    return r.dropna()

def _atr(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    """
    Expects columns: ['High','Low','Close'] (Adj Close is fine for price band math separately).
    True Range = max(H-L, |H-prev_close|, |L-prev_close|)
    """
    if df is None or df.empty or len(df) < window + 1:
        return None
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def _ewma_vol(returns: pd.Series, lam: float = 0.94) -> Optional[float]:
    """
    RiskMetrics-style EWMA volatility (daily).
    sigma^2_t = (1-lam) * sum_{i=0..n-1} lam^i * r_{t-i}^2
    """
    if returns is None or returns.empty:
        return None
    r = returns.dropna().values[::-1]  # newest first by reversing later math; we will just weight from latest
    # weight newest more (i=0 is most recent)
    w = (1 - lam) * lam ** np.arange(len(r))
    var = np.sum(w * (r ** 2))
    sigma = np.sqrt(var)
    return float(sigma)

def estimate_1m_range(
    prices_df: pd.DataFrame,
    z: float = 1.0,              # ~68% central band; try 1.28 (~80%) or 1.64 (~90%) if you want wider
    rv_window: int = 21,         # realized volatility lookback
    atr_window: int = 14,        # ATR lookback
    ewma_lambda: float = 0.94    # RiskMetrics lambda
) -> Dict[str, Any]:
    """
    Input:
      prices_df: pandas DataFrame with columns at least:
                 ['Open','High','Low','Close','Adj Close','Volume'], index = dates
    Output:
      vol_meta: dict with components and a blended [low, base, high]
    """
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df is empty.")

    df = prices_df.copy()
    df = df.sort_index()

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    last_price = float(df["Adj Close"].iloc[-1])

    # 1) Realized Volatility (RV)
    r = _daily_log_returns(df["Adj Close"])
    daily_vol_rv = float(r.tail(rv_window).std()) if len(r) >= 2 else None
    vol21_rv = float(daily_vol_rv * np.sqrt(21)) if daily_vol_rv is not None else None
    low_rv = float(last_price * np.exp(-z * vol21_rv)) if vol21_rv is not None else None
    high_rv = float(last_price * np.exp(z * vol21_rv)) if vol21_rv is not None else None

    # 2) EWMA Volatility
    daily_vol_ewma = _ewma_vol(r, ewma_lambda)
    vol21_ewma = float(daily_vol_ewma * np.sqrt(21)) if daily_vol_ewma is not None else None
    low_ewma = float(last_price * np.exp(-z * vol21_ewma)) if vol21_ewma is not None else None
    high_ewma = float(last_price * np.exp(z * vol21_ewma)) if vol21_ewma is not None else None

    # 3) ATR-based absolute band
    atr = _atr(df, window=atr_window)
    atr_21 = float(atr * np.sqrt(21 / atr_window)) if atr is not None else None
    low_atr = float(last_price - z * atr_21) if atr_21 is not None else None
    high_atr = float(last_price + z * atr_21) if atr_21 is not None else None

    # Blend available candidates (ignore Nones)
    lows  = [x for x in [low_rv, low_ewma, low_atr] if x is not None]
    highs = [x for x in [high_rv, high_ewma, high_atr] if x is not None]

    blended_low  = float(np.mean(lows))  if lows  else None
    blended_high = float(np.mean(highs)) if highs else None

    vol_meta = {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "last_price": last_price,
        "params": {
            "z": z,
            "rv_window": rv_window,
            "atr_window": atr_window,
            "ewma_lambda": ewma_lambda
        },

        # Components
        "rv": {
            "daily_vol": daily_vol_rv,
            "vol_21d": vol21_rv,
            "low": low_rv,
            "high": high_rv
        },
        "ewma": {
            "daily_vol": daily_vol_ewma,
            "vol_21d": vol21_ewma,
            "low": low_ewma,
            "high": high_ewma
        },
        "atr": {
            "atr": atr,
            "atr_projected_21d": atr_21,
            "low": low_atr,
            "high": high_atr
        },

        # Blended output
        "range_1m": {
            "low": blended_low,
            "base": last_price,
            "high": blended_high
        }
    }
    return vol_meta
