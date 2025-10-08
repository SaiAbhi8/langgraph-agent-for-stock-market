# tools/trend_gauge.py
# pip install pandas numpy

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _ensure_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_index()
    if "Adj Close" not in d.columns:
        d["Adj Close"] = d["Close"]
    return d

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi_ema(price: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using EMA for average gains/losses (classic 'Wilders' style via EMA).
    """
    delta = price.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _slope_linear(series: pd.Series, window: int) -> Optional[float]:
    """
    Linear regression slope over the last 'window' points.
    Returns slope per step in the series' units.
    """
    if series is None or len(series) < max(3, window):
        return None
    y = series.tail(window).values.astype(float)
    x = np.arange(len(y), dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return None
    num = np.sum((x - x_mean) * (y - y_mean))
    return float(num / denom)  # units of y per step

def estimate_trend_meta(
    prices_df: pd.DataFrame,
    ema_short: int = 20,
    ema_long: int = 50,
    slope_window: int = 20,
    rsi_period: int = 14
) -> Dict[str, Any]:
    """
    Input:
      prices_df: DataFrame with ['Open','High','Low','Close','Adj Close','Volume'].
    Output:
      trend_meta: dict with EMA slopes, price distances, RSI, and a trend score (-1..+1).
    """
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df is empty.")

    df = _ensure_adj_close(prices_df)

    close = df["Adj Close"]
    last_price = float(close.iloc[-1])

    ema_s = _ema(close, ema_short)
    ema_l = _ema(close, ema_long)

    ema_s_last = float(ema_s.iloc[-1])
    ema_l_last = float(ema_l.iloc[-1])

    # Slopes (per day). Also provide normalized (% of last price) for scale-free view.
    slope_s = _slope_linear(ema_s, slope_window)
    slope_l = _slope_linear(ema_l, slope_window)
    slope_s_pct = float(slope_s / last_price) if slope_s is not None and last_price != 0 else None
    slope_l_pct = float(slope_l / last_price) if slope_l is not None and last_price != 0 else None

    # Price distances from EMAs
    dist_s_pct = float((last_price - ema_s_last) / ema_s_last) if ema_s_last != 0 else None
    dist_l_pct = float((last_price - ema_l_last) / ema_l_last) if ema_l_last != 0 else None

    # RSI
    rsi_series = _rsi_ema(close, rsi_period)
    rsi_last = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

    # Heuristic score (-1..+1)
    score = 0.0
    # EMA slopes contribute
    if slope_s_pct is not None:
        score += 0.3 if slope_s_pct > 0 else -0.3
    if slope_l_pct is not None:
        score += 0.2 if slope_l_pct > 0 else -0.2
    # Price position vs EMAs
    if dist_s_pct is not None:
        score += 0.2 if dist_s_pct > 0 else -0.2
    if dist_l_pct is not None:
        score += 0.1 if dist_l_pct > 0 else -0.1
    # RSI band
    if rsi_last is not None:
        if rsi_last >= 55:
            score += 0.2
        elif rsi_last <= 45:
            score -= 0.2
        # else neutral

    # Clip score
    score = float(np.clip(score, -1.0, 1.0))

    trend_meta: Dict[str, Any] = {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "last_price": last_price,
        "params": {
            "ema_short": ema_short,
            "ema_long": ema_long,
            "slope_window": slope_window,
            "rsi_period": rsi_period
        },
        "ema": {
            "ema_short": ema_s_last,
            "ema_long": ema_l_last,
            "slope_short_per_day": slope_s,
            "slope_long_per_day": slope_l,
            "slope_short_pct_per_day": slope_s_pct,
            "slope_long_pct_per_day": slope_l_pct
        },
        "distance": {
            "price_vs_ema_short_pct": dist_s_pct,
            "price_vs_ema_long_pct": dist_l_pct
        },
        "rsi": rsi_last,
        "trend_score": score
    }
    return trend_meta
