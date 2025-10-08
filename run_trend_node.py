# run_trend_node.py
import sys
import json
from nodes.trend_node import trend_node

def main():
    """
    Usage:
      python run_trend_node.py TICKER [PERIOD] [INTERVAL]
    Examples:
      python run_trend_node.py RELIANCE
      python run_trend_node.py TCS 2y 1d
    """
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    period   = sys.argv[2] if len(sys.argv) > 2 else "1y"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"

    state = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        # knobs (optional):
        # "ema_short": 20,
        # "ema_long": 50,
        # "slope_window": 20,
        # "rsi_period": 14,
    }

    out = trend_node(state)
    print("=== trend_meta ===")
    print(json.dumps(out["trend_meta"], indent=2))

if __name__ == "__main__":
    main()
