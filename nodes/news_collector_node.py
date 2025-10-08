# nodes/news_collector_node.py
# deps: pip install requests trafilatura beautifulsoup4
from typing import Dict, Any, List, Tuple
import re, time, json
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tools.web_fetch import extract_readable  # you added this earlier
from tools.llm import llm_json               # you added this earlier

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _get_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def _is_probable_article(href: str) -> bool:
    """Heuristics to keep article-like paths and drop noise (login, tags, etc.)."""
    if not href:
        return False
    href = href.lower()
    bad = ["login", "signup", "account", "privacy", "terms", "javascript:", "mailto:"]
    if any(b in href for b in bad):
        return False
    # Keep if looks like a news/article path
    good = ["news", "story", "article", "markets", "business", "companies", "stocks"]
    return any(g in href for g in good)

def _find_article_links(site_url: str, max_links: int = 30) -> List[Tuple[str, str]]:
    """
    Returns list of (absolute_url, anchor_text) discovered on the site homepage/section page.
    """
    links: List[Tuple[str, str]] = []
    try:
        html = _get_html(site_url)
    except Exception:
        return links
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = (a.get_text() or "").strip()
        if not _is_probable_article(href):
            continue
        abs_url = urljoin(site_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append((abs_url, text))
        if len(links) >= max_links:
            break
    return links

def _filter_by_query(links: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
    if not query:
        return links
    q = query.lower()
    out = []
    for url, text in links:
        if q in url.lower() or q in text.lower():
            out.append((url, text))
    # fallback: if nothing matched, return originals
    return out if out else links

def _clean_article(url: str) -> str:
    try:
        html = _get_html(url)
        text = extract_readable(html, url=url) or ""
        # keep it manageable for the LLM
        return text[:6000]
    except Exception:
        return ""

_SUMMARY_SCHEMA = """
Return STRICT JSON like:
{
  "title": "string",
  "summary_3_4_lines": "string (3-4 lines, crisp, factual)",
  "sentiment_label": "bullish|bearish|neutral",
  "sentiment_score": 0.0_to_1.0
}
- sentiment_score: map confidence in the label to [0,1].
- Be concise. Do not add extra keys.
"""

def _summarize_and_score(ticker: str, url: str, text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "title": None,
            "summary_3_4_lines": "Empty or unreadable article.",
            "sentiment_label": "neutral",
            "sentiment_score": 0.0
        }
    messages = [
        {"role": "system", "content": "You are an equity news assistant for Indian markets."},
        {"role": "user", "content":
            f"Ticker: {ticker}\nURL: {url}\n\n{_SUMMARY_SCHEMA}\n\nARTICLE TEXT:\n{text}"
        }
    ]
    js = llm_json(messages, model="gpt-4o-mini", temperature=0.2, response_format={"type": "json_object"})
    try:
        data = json.loads(js)
        # light validation
        for k in ["title", "summary_3_4_lines", "sentiment_label", "sentiment_score"]:
            data.setdefault(k, None)
        return data
    except Exception:
        return {
            "title": None,
            "summary_3_4_lines": "LLM could not produce valid JSON.",
            "sentiment_label": "neutral",
            "sentiment_score": 0.0
        }

def news_collector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (state):
      - ticker: str                       (e.g., 'RELIANCE')
      - sites: List[str]                  (homepage/section URLs to scan)
      - max_articles: int                 (default=10)
      - per_site_scan: int                (how many links to scan per site, default=30)
      - throttle_sec: float               (delay between requests, default=0.6)
    Output:
      - news_items: List[{
           "url","host","title","summary","sentiment_label","sentiment_score"
        }]
    """
    ticker = state.get("ticker", "").strip()
    sites: List[str] = state.get("sites", [])
    max_articles = int(state.get("max_articles", 10))
    per_site_scan = int(state.get("per_site_scan", 30))
    throttle = float(state.get("throttle_sec", 0.6))

    all_links: List[Tuple[str, str]] = []
    for site in sites:
        links = _find_article_links(site, max_links=per_site_scan)
        links = _filter_by_query(links, ticker)
        all_links.extend(links)
        time.sleep(throttle)

    # de-dupe by URL, keep order
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for u, t in all_links:
        if u in seen:
            continue
        seen.add(u)
        deduped.append((u, t))

    picked = deduped[:max_articles]

    results = []
    for url, _txt in picked:
        host = urlparse(url).netloc
        text = _clean_article(url)
        summary = _summarize_and_score(ticker, url, text)
        results.append({
            "url": url,
            "host": host,
            "title": summary.get("title"),
            "summary": summary.get("summary_3_4_lines"),
            "sentiment_label": summary.get("sentiment_label"),
            "sentiment_score": summary.get("sentiment_score"),
        })
        time.sleep(throttle)

    return {"news_items": results}
