# nodes/context_node.py
from typing import Dict, Any
from tools.context_loader import load_context

def context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ticker    = state.get("ticker", "INFY")
    period    = state.get("period", "1y")
    interval  = state.get("interval", "1d")
    benchmark = state.get("benchmark", "^NSEI")
    sector_in = state.get("sector", None)

    # sanitize sector
    sector = None if (sector_in is None or str(sector_in).strip().lower() in {"", "none", "null"}) else sector_in

    context_meta = load_context(
        ticker=ticker,
        period=period,
        interval=interval,
        benchmark=benchmark,
        sector=sector,
    )
    return {"context_meta": context_meta}
