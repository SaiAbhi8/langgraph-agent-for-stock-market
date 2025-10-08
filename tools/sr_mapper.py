# tools/sr_mapper.py
# pip install pandas numpy

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_index()
    for col in ["High", "Low", "Close"]:
        if col not in d.columns:
            raise ValueError(f"prices_df missing required column: {col}")
    if "Adj Close" not in d.columns:
        d["Adj Close"] = d["Close"]
    return d

def _find_swings(df: pd.DataFrame, window: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect swing highs/lows using a symmetric window.
    A swing high at i if High[i] == max(High[i-window : i+window])
    A swing low  at i if Low[i]  == min(Low[i-window : i+window])
    """
    if window < 1:
        window = 1

    highs = df["High"].values
    lows  = df["Low"].values
    idx   = df.index

    n = len(df)
    swings_hi: List[Dict[str, Any]] = []
    swings_lo: List[Dict[str, Any]] = []

    for i in range(window, n - window):
        h_slice = highs[i - window : i + window + 1]
        l_slice = lows[i - window  : i + window + 1]

        if highs[i] == np.max(h_slice):
            swings_hi.append({"date": idx[i], "price": float(highs[i])})
        if lows[i] == np.min(l_slice):
            swings_lo.append({"date": idx[i], "price": float(lows[i])})

    return {"highs": swings_hi, "lows": swings_lo}

def _cluster_levels(levels: List[Dict[str, Any]], tol_pct: float) -> List[Dict[str, Any]]:
    """
    Merge nearby levels within a relative tolerance (e.g., 1.0%).
    Uses simple one-pass clustering sorted by price; cluster price = median of member prices.
    """
    if not levels:
        return []

    # sort by price asc
    levels_sorted = sorted(levels, key=lambda x: x["price"])

    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / ((a + b) / 2.0)

    for lv in levels_sorted:
        if not current:
            current = [lv]
            continue

        # compare with current cluster center (median of prices so far)
        cur_prices = [c["price"] for c in current]
        center = float(np.median(cur_prices))
        if rel_diff(lv["price"], center) <= tol_pct / 100.0:
            current.append(lv)
        else:
            clusters.append(current)
            current = [lv]
    if current:
        clusters.append(current)

    # summarize clusters
    summarized: List[Dict[str, Any]] = []
    for cl in clusters:
        prices = np.array([c["price"] for c in cl], dtype=float)
        dates  = [c["date"] for c in cl]
        summarized.append({
            "price": float(np.median(prices)),
            "touches": int(len(cl)),                   # number of swing points merged
            "first_date": str(min(dates).date()),
            "last_date":  str(max(dates).date()),
        })
    return summarized

def map_support_resistance(
    prices_df: pd.DataFrame,
    swing_window: int = 5,
    tolerance_pct: float = 1.0,      # cluster width ~1%
    min_touches: int = 2,            # require at least 2 swing points per level
    max_levels_per_side: int = 6     # cap list size for readability
) -> Dict[str, Any]:
    """
    Find support/resistance levels:
      1) detect swing highs/lows
      2) cluster nearby swing levels
      3) filter by min_touches
      4) pick nearest support below and resistance above the last price

    Returns sr_meta with nearest levels and lists of levels.
    """
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df is empty.")

    df = _ensure_cols(prices_df)
    last_price = float(df["Adj Close"].iloc[-1])
    as_of = df.index[-1].strftime("%Y-%m-%d")

    swings = _find_swings(df, window=swing_window)

    # Cluster supports (from swing lows) and resistances (from swing highs) separately
    support_clusters = _cluster_levels(swings["lows"], tolerance_pct)
    resist_clusters  = _cluster_levels(swings["highs"], tolerance_pct)

    # Filter by touches
    supports = [s for s in support_clusters if s["touches"] >= min_touches]
    resists  = [r for r in resist_clusters  if r["touches"] >= min_touches]

    # Sort “strength” primarily by touches (desc), then recency (desc)
    supports.sort(key=lambda x: (x["touches"], x["last_date"]), reverse=True)
    resists.sort(key=lambda x: (x["touches"], x["last_date"]), reverse=True)

    # Compute distance from current price and split by side
    for s in supports:
        s["distance_pct"] = float((last_price - s["price"]) / last_price * 100.0)
    for r in resists:
        r["distance_pct"] = float((r["price"] - last_price) / last_price * 100.0)

    # Nearest on each side by absolute distance, but enforce side (below for support, above for resistance)
    supports_below = [s for s in supports if s["price"] <= last_price]
    resists_above  = [r for r in resists  if r["price"] >= last_price]

    nearest_support = min(supports_below, key=lambda x: abs(x["distance_pct"])) if supports_below else None
    nearest_resist  = min(resists_above,  key=lambda x: abs(x["distance_pct"])) if resists_above  else None

    # Truncate lists for readability; keep sorted by proximity to price
    supports_sorted = sorted(supports_below, key=lambda x: abs(x["distance_pct"]))[:max_levels_per_side]
    resists_sorted  = sorted(resists_above,  key=lambda x: abs(x["distance_pct"]))[:max_levels_per_side]

    sr_meta: Dict[str, Any] = {
        "as_of": as_of,
        "last_price": last_price,
        "params": {
            "swing_window": swing_window,
            "tolerance_pct": tolerance_pct,
            "min_touches": min_touches,
            "max_levels_per_side": max_levels_per_side
        },
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resist,
        "supports": supports_sorted,
        "resistances": resists_sorted
    }
    return sr_meta
