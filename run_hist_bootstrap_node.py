# run_hist_bootstrap_node.py
import sys
import json
from nodes.hist_bootstrap_node import hist_bootstrap_node

def main():
    """
    Usage:
      python run_hist_bootstrap_node.py TICKER [LOOKBACK_YEARS] [HORIZON_DAYS] [WINSOR_PCT]
    Examples:
      python run_hist_bootstrap_node.py RELIANCE
      python run_hist_bootstrap_node.py TCS 7 21 0.01
      python run_hist_bootstrap_node.py INFY 10 21 0.0
    """
    ticker         = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    lookback_years = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    horizon_days   = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    winsor_pct     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    state = {
        "ticker": ticker,
        "period": f"{max(lookback_years, 3)}y",  # ensure the loader pulls enough data
        "interval": "1d",
        "lookback_years": lookback_years,
        "horizon_days": horizon_days,
        "winsor_pct": winsor_pct,
        # "extra_percentiles": [5, 25, 75, 95],  # uncomment if you want more outputs
    }

    out = hist_bootstrap_node(state)
    print("=== hist_bands ===")
    print(json.dumps(out["hist_bands"], indent=2))

if __name__ == "__main__":
    main()
