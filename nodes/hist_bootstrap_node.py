# nodes/hist_bootstrap_node.py
from typing import Dict, Any

# Reuse your price loader if available
try:
    from tools.price_loader import load_prices
except Exception:
    load_prices = None

from tools.hist_bootstrap import estimate_hist_1m_bands

def hist_bootstrap_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
      - Optional params:
          horizon_days (default 21)
          lookback_years (default 5)
          winsor_pct (default 0.0)
          extra_percentiles (e.g., [5,25,75,95])

    Output (state update):
      - 'hist_bands': dict with empirical p10/p50/p90 and price_bands
    """
    prices_df = state.get("prices_df")

    if prices_df is None:
        if load_prices is None:
            raise RuntimeError("prices_df not in state and tools.price_loader.load_prices not available.")
        ticker   = state.get("ticker", "RELIANCE")
        period   = state.get("period", "5y")   # ensure enough lookback by default
        interval = state.get("interval", "1d")
        prices_df, _ = load_prices(ticker=ticker, period=period, interval=interval)

    hist_bands = estimate_hist_1m_bands(
        prices_df=prices_df,
        horizon_days=state.get("horizon_days", 21),
        lookback_years=state.get("lookback_years", 5),
        winsor_pct=state.get("winsor_pct", 0.0),
        extra_percentiles=state.get("extra_percentiles", None)
    )

    return {"hist_bands": hist_bands}
