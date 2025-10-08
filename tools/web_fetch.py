# tools/web_fetch.py
import requests, trafilatura, time
from urllib.parse import urlparse

def fetch_url(url, timeout=20):
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_readable(html, url=None):
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return downloaded or ""

def fetch_and_clean(urls, throttle_sec=0.8):
    out = []
    for u in urls:
        try:
            html = fetch_url(u)
            text = extract_readable(html, url=u)
            if text.strip():
                out.append({"url": u, "host": urlparse(u).netloc, "text": text})
        except Exception as e:
            out.append({"url": u, "host": None, "text": "", "error": str(e)})
        time.sleep(throttle_sec)
    return out
