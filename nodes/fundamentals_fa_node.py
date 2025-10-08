# nodes/fundamentals_fa_node.py
from typing import Dict, Any, Optional
from tools.fundamentals_fa import compute_fa

def fundamentals_fa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (in state):
      - 'ticker' (e.g., 'RELIANCE' or 'RELIANCE.NS')
      - 'fundamentals_xlsx' (optional path to Excel exported by yfin_fundamental_data_export.py)
      - 'use_llm' (bool, optional) -> uses tools/llm.py if OPENAI_API_KEY is set

    Output:
      - {'fa_meta': {...}}  # structured FA package with tilt/bandwidth + scenarios + narrative
    """
    ticker: str = state.get("ticker", "RELIANCE")
    xlsx_path: Optional[str] = state.get("fundamentals_xlsx")
    use_llm: bool = bool(state.get("use_llm", False))

    fa_meta = compute_fa(ticker, xlsx_path=xlsx_path, use_llm=use_llm)
    return {"fa_meta": fa_meta}
