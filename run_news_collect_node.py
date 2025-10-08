# run_news_collector_node.py
import argparse, json
from pathlib import Path
from nodes.news_collector_node import news_collector_node

DEFAULT_SITES = [
    "https://www.moneycontrol.com/news/business/stocks/",
    "https://economictimes.indiatimes.com/markets/stocks/news",
    "https://www.livemint.com/market/stock-market-news",
    "https://www.business-standard.com/markets/news",
    "https://www.financialexpress.com/market/"
]

def print_cards(items):
    for i, it in enumerate(items, 1):
        print(f"\n[{i}] {it.get('title') or '(no title)'}")
        print(f"    host: {it['host']}")
        print(f"    url : {it['url']}")
        print(f"    senti: {it['sentiment_label']} ({it['sentiment_score']})")
        print(f"    summary:\n      {it['summary']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="RELIANCE")
    ap.add_argument("--sites", nargs="*", default=None, help="Override sites list")
    ap.add_argument("--max_articles", type=int, default=10)
    ap.add_argument("--out", default=None, help="Save JSON to this path")
    args = ap.parse_args()

    state = {
        "ticker": args.ticker,
        "sites": args.sites or DEFAULT_SITES,
        "max_articles": args.max_articles
    } 
    out = news_collector_node(state)
    items = out.get("news_items", [])
    print(f"\nCollected {len(items)} item(s) for {args.ticker}")
    print_cards(items)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
