# nodes/trend_node.py
from typing import Dict, Any

# Reuse your price loader if available
try:
    from tools.price_loader import load_prices
except Exception:
    load_prices = None

from tools.trend_gauge import estimate_trend_meta

def trend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
      - Optional params: ema_short (20), ema_long (50), slope_window (20), rsi_period (14)

    Output (state update):
      - 'trend_meta': dict with EMA slopes, distances, RSI, and trend_score (-1..+1)
    """
    prices_df = state.get("prices_df")

    if prices_df is None:
        if load_prices is None:
            raise RuntimeError("prices_df not in state and tools.price_loader.load_prices not available.")
        ticker   = state.get("ticker", "RELIANCE")
        period   = state.get("period", "1y")
        interval = state.get("interval", "1d")
        prices_df, _ = load_prices(ticker=ticker, period=period, interval=interval)

    trend_meta = estimate_trend_meta(
        prices_df=prices_df,
        ema_short=state.get("ema_short", 20),
        ema_long=state.get("ema_long", 50),
        slope_window=state.get("slope_window", 20),
        rsi_period=state.get("rsi_period", 14),
    )

    return {"trend_meta": trend_meta}
