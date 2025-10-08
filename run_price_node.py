import sys
from nodes.price_node import price_node
import json 
from pprint import pprint

def print_nested_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_nested_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

def main():
    # args: ticker [period] [interval]
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    period = sys.argv[2] if len(sys.argv) > 2 else "2y"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"

    state = {"ticker": ticker, "period": period, "interval": interval}
    out = price_node(state)
    # Show the metadata summary
    pprint(out, indent=1, width=160)
    print_nested_dict(out['prices_meta'])

    # Show a couple of rows to confirm data looks good
    df = out["prices_df"]
    print("\n=== sample rows ===")
    print(df.tail(3))

if __name__ == "__main__":
    main()
