# run_context_node.py
import sys
import json
from nodes.context_node import context_node

def main():
    """
    Usage:
      python run_context_node.py TICKER [BENCHMARK] [SECTOR] [PERIOD] [INTERVAL]
    Examples:
      python run_context_node.py RELIANCE
      python run_context_node.py TCS ^NSEI ^CNXIT 2y 1d
      python run_context_node.py ICICIBANK ^NSEI None 1y 1d
    """
    ticker    = sys.argv[1] if len(sys.argv) > 1 else "TCS"
    benchmark = sys.argv[2] if len(sys.argv) > 2 else "^NSEI"
    sector    = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "None" else "^CNXIT"
    period    = sys.argv[4] if len(sys.argv) > 4 else "2y"
    interval  = sys.argv[5] if len(sys.argv) > 5 else "1d"

    state = {
        "ticker": ticker,
        "benchmark": benchmark,
        "sector": sector,
        "period": period,
        "interval": interval,
    }

    out = context_node(state)
    print("=== context_meta ===")
    print(json.dumps(out["context_meta"], indent=2))

if __name__ == "__main__":
    main()
