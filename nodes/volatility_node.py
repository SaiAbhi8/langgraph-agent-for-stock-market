# nodes/volatility_node.py
# pip install pandas
from typing import Dict, Any

# Reuse your existing price loader if available
try:
    from tools.price_loader import load_prices
except Exception:
    load_prices = None

from tools.volatility_range import estimate_1m_range

def volatility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - Either provide 'prices_df' directly, or provide 'ticker' (plus optional period/interval)
      - Optional: 'z', 'rv_window', 'atr_window', 'ewma_lambda'
    Output (state update):
      - 'vol_meta': dict with component vols and blended 1M range
    """
    prices_df = state.get("prices_df")

    if prices_df is None:
        # Fallback: load prices if a ticker is provided
        if load_prices is None:
            raise RuntimeError("prices_df not in state and tools.price_loader.load_prices not available.")
        ticker   = state.get("ticker", "RELIANCE")
        period   = state.get("period", "1y")
        interval = state.get("interval", "1d")
        prices_df, _ = load_prices(ticker=ticker, period=period, interval=interval)

    vol_meta = estimate_1m_range(
        prices_df=prices_df,
        z=state.get("z", 1.0),
        rv_window=state.get("rv_window", 21),
        atr_window=state.get("atr_window", 14),
        ewma_lambda=state.get("ewma_lambda", 0.94),
    )

    return {"vol_meta": vol_meta}
