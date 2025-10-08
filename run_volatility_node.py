# run_volatility_node.py
import sys
import json

from nodes.volatility_node import volatility_node

def main():
    """
    Usage:
      python run_volatility_node.py TICKER [PERIOD] [INTERVAL] [Z]
    Examples:
      python run_volatility_node.py RELIANCE
      python run_volatility_node.py TCS 2y 1d 1.28
    """
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    period   = sys.argv[2] if len(sys.argv) > 2 else "1y"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
    z        = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    state = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "z": z
    }

    out = volatility_node(state)
    print("=== vol_meta ===")
    print(json.dumps(out["vol_meta"], indent=2))

if __name__ == "__main__":
    main()
