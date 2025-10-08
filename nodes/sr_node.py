# nodes/sr_node.py
from typing import Dict, Any

# Reuse your price loader if available
try:
    from tools.price_loader import load_prices
except Exception:
    load_prices = None

from tools.sr_mapper import map_support_resistance

def sr_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
      - Optional params:
          swing_window (default 5)
          tolerance_pct (default 1.0)
          min_touches (default 2)
          max_levels_per_side (default 6)

    Output (state update):
      - 'sr_meta': dict with nearest support/resistance and level lists
    """
    prices_df = state.get("prices_df")

    if prices_df is None:
        if load_prices is None:
            raise RuntimeError("prices_df not in state and tools.price_loader.load_prices not available.")
        ticker   = state.get("ticker", "RELIANCE")
        period   = state.get("period", "1y")
        interval = state.get("interval", "1d")
        prices_df, _ = load_prices(ticker=ticker, period=period, interval=interval)

    sr_meta = map_support_resistance(
        prices_df=prices_df,
        swing_window=state.get("swing_window", 5),
        tolerance_pct=state.get("tolerance_pct", 1.0),
        min_touches=state.get("min_touches", 2),
        max_levels_per_side=state.get("max_levels_per_side", 6)
    )

    return {"sr_meta": sr_meta}
