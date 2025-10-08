# nodes/fundamentals_node.py
from typing import Dict, Any
from tools.fundamentals_momentum import load_fundamentals_momentum

def fundamentals_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      - ticker (e.g., "RELIANCE")
    Output:
      - 'fund_meta': TTM growth/margins/FCF + balance sheet risk + tiny tilt/widen suggestions
    """
    ticker = state.get("ticker", "RELIANCE")
    fund_meta = load_fundamentals_momentum(ticker)
    return {"fund_meta": fund_meta}
