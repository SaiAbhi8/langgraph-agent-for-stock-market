# run_fundamentals_fa_node.py
import sys, json
from pprint import pprint
from nodes.fundamentals_fa_node import fundamentals_fa_node

def main():
    """
    Usage:
      python run_fundamentals_fa_node.py RELIANCE.NS
      python run_fundamentals_fa_node.py RELIANCE.NS path/to/RELIANCE.NS_fundamentals.xlsx
    """
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    xlsx = sys.argv[2] if len(sys.argv) > 2 else None

    state = {"ticker": ticker}
    if xlsx:
        state["fundamentals_xlsx"] = xlsx

    # set use_llm=True to enable narrative via tools/llm.py (needs OPENAI_API_KEY)
    state["use_llm"] = True

    out = fundamentals_fa_node(state)
    
    pprint(out, indent=2, width=100)

if __name__ == "__main__":
    main()
