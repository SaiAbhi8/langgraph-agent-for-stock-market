# run_fundamentals_node.py
import sys, json
from nodes.fundamentals_node import fundamentals_node
from pprint import pprint

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    out = fundamentals_node({"ticker": ticker})
    pprint(out, indent=2, width = 100)
    print("=== fund_meta ===")
    print(json.dumps(out["fund_meta"], indent=2))

if __name__ == "__main__":
    main()
