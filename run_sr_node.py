# run_sr_node.py
import sys
import json
from nodes.sr_node import sr_node

def main():
    """
    Usage:
      python run_sr_node.py TICKER [PERIOD] [INTERVAL] [SWING_WINDOW] [TOL_PCT]
    Examples:
      python run_sr_node.py RELIANCE
      python run_sr_node.py TCS 2y 1d 7 1.0
    """
    ticker        = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    period        = sys.argv[2] if len(sys.argv) > 2 else "1y"
    interval      = sys.argv[3] if len(sys.argv) > 3 else "1d"
    swing_window  = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    tol_pct       = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    state = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "swing_window": swing_window,
        "tolerance_pct": tol_pct,
        # knobs:
        # "min_touches": 2,
        # "max_levels_per_side": 6,
    }

    out = sr_node(state)
    print("=== sr_meta ===")
    print(json.dumps(out["sr_meta"], indent=2))

if __name__ == "__main__":
    main()
