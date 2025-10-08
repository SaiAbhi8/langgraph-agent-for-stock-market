# run_events_node.py
import sys
import json
from nodes.events_node import events_node

def main():
    """
    Usage:
      python run_events_node.py TICKER [EARN_WINDOW_DAYS] [EXDIV_WINDOW_DAYS] [LOOKBACK_YEARS] [LAST_PRICE]
    Examples:
      python run_events_node.py RELIANCE
      python run_events_node.py TCS 21 30 5 4200
    """
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    ewin     = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    dwin     = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    ly       = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    last_px  = float(sys.argv[5]) if len(sys.argv) > 5 else None

    state = {
        "ticker": ticker,
        "earnings_window_days": ewin,
        "exdiv_window_days": dwin,
        "lookback_years": ly,
        # If you already ran price_node, you don't need to pass last price here.
        "prices": {"latest_close": last_px} if last_px is not None else {}
    }

    out = events_node(state)
    print("=== event_meta ===")
    print(json.dumps(out["event_meta"], indent=2))

if __name__ == "__main__":
    main()
