# nodes/events_node.py
from typing import Dict, Any

# Optional: use latest price from your price node if available
def _latest_price_from_state(state: Dict[str, Any]) -> float:
    try:
        pm = state.get("prices_meta") or state.get("company")  # your structure uses top-level 'company' and 'prices'
        # Your shared state shows latest price at state["prices"]["latest_close"]
        if "prices" in state and isinstance(state["prices"], dict):
            return float(state["prices"].get("latest_close"))
        # Fallback if someone stored meta differently
        if pm and isinstance(pm, dict):
            val = pm.get("latest_close")
            return float(val) if val is not None else None
    except Exception:
        pass
    return None

from tools.events_window import load_events_window

def events_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - ticker (e.g., "RELIANCE")
      - Optional:
          earnings_window_days (default 21)
          exdiv_window_days (default 30)
          lookback_years (default 5)
      - If available: prices.latest_close (for ex-div tilt)

    Output (state update):
      - 'event_meta': dict with earnings window, estimated ex-div, splits, and widen/tilt suggestions.
    """
    ticker = state.get("ticker", "RELIANCE")

    event_meta = load_events_window(
        ticker=ticker,
        earnings_window_days=state.get("earnings_window_days", 21),
        exdiv_window_days=state.get("exdiv_window_days", 30),
        lookback_years=state.get("lookback_years", 5),
        last_price=_latest_price_from_state(state),
    )
    return {"event_meta": event_meta}
