# pip install langgraph pandas
from typing import Dict, Any
from tools.price_loader import load_prices

def price_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal LangGraph-compatible node function.
    Expects: state has 'ticker' (and optionally 'period', 'interval')
    Returns: adds 'prices_df' and 'prices_meta' to state.
    """
    ticker = state.get("ticker", "RELIANCE")
    period = state.get("period", "1y")
    interval = state.get("interval", "1d")

    df, meta = load_prices(ticker=ticker, period=period, interval=interval)
    # return only the updates; the graph will merge into state
    return {"prices_df": df, "prices_meta": meta}
