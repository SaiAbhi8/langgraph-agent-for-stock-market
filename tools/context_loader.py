# tools/context_loader.py
# pip install yfinance pandas numpy

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any, List

def normalize_symbol(sym: Optional[str]) -> Optional[str]:
    """Normalize NSE equities to end with .NS. Keep index tickers like ^NSEI as-is."""
    if sym is None:
        return None
    t = sym.strip().upper()
    if t.startswith("^"):
        return t
    return t if t.endswith(".NS") else f"{t}.NS"

def _pct_return(series: pd.Series, lookback_days: int) -> Optional[float]:
    if series is None or series.empty or len(series) <= lookback_days:
        return None
    try:
        return float(series.iloc[-1] / series.iloc[-1 - lookback_days] - 1.0)
    except Exception:
        return None

def _beta(stock_ret: pd.Series, bench_ret: pd.Series, window: int = 180) -> Optional[float]:
    if stock_ret is None or bench_ret is None:
        return None
    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    tail = df.tail(window)
    if len(tail) < 10:
        return None
    s = tail.iloc[:, 0]
    b = tail.iloc[:, 1]
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else None
    if not var_b:
        return None
    cov_sb = float(np.cov(s, b, ddof=1)[0, 1])
    return cov_sb / var_b

def _rolling_corr(stock_ret: pd.Series, bench_ret: pd.Series, window: int = 60) -> Optional[float]:
    if stock_ret is None or bench_ret is None:
        return None
    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(df) < window:
        return None
    return float(df.iloc[-window:, 0].corr(df.iloc[-window:, 1]))

def load_context(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    benchmark: str = "^NSEI",
    sector: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pulls Adj Close for stock, benchmark (and optional sector), then derives:
      - 1M/3M/6M returns (21/63/126 trading days)
      - out/under-performance vs benchmark
      - relative strength ratio (stock/bench) + 90d slope
      - 60d correlation and 180d beta vs benchmark

    Returns a single 'context_meta' dict.
    Gracefully handles missing benchmark/sector data (keeps stock-only stats).
    """
    stock = normalize_symbol(ticker)
    bench = normalize_symbol(benchmark) if benchmark else None
    sect  = normalize_symbol(sector) if sector else None

    symbols: List[str] = [stock]
    if bench: symbols.append(bench)
    if sect:  symbols.append(sect)

    # Download in one call. IMPORTANT: do NOT use group_by="ticker".
    data = yf.download(
        symbols, period=period, interval=interval, auto_adjust=False, progress=False
    )

    if data is None or data.empty:
        raise ValueError("Download returned no data. Check tickers or internet connectivity.")

    # Extract 'Adj Close' cleanly for both single- and multi-symbol cases.
    # Case A (multi-symbol): data['Adj Close'] is a DataFrame with columns = symbols.
    # Case B (single-symbol): data is OHLCV; data['Adj Close'] is a Series.
    adj = None
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-symbol case
        if ("Adj Close" in data.columns.get_level_values(0)):
            adj = data["Adj Close"].copy()
        else:
            # Some intervals/sources may lack 'Adj Close'; fallback to 'Close'
            adj = data["Close"].copy()
    else:
        # Single-symbol case
        if "Adj Close" in data.columns:
            s = data["Adj Close"].rename(stock)
        else:
            s = data["Close"].rename(stock)
        adj = s.to_frame()

    # Helper to pick a column safely
    def pick(sym: Optional[str]) -> Optional[pd.Series]:
        if sym is None:
            return None
        try:
            if isinstance(adj, pd.DataFrame):
                if sym in adj.columns:
                    return adj[sym].dropna()
                # Sometimes yfinance uses slightly different casing; try upper()
                up = sym.upper()
                if up in adj.columns:
                    return adj[up].dropna()
                return None
            else:
                # adj is a Series (single symbol only)
                return adj.dropna() if sym == stock else None
        except Exception:
            return None

    adj_stock = pick(stock)
    adj_bench = pick(bench) if bench else None
    adj_sect  = pick(sect)  if sect  else None

    if adj_stock is None or adj_stock.empty:
        raise ValueError(f"Could not extract Adj Close for stock: {stock}")

    # Align for joint calcs; keep stock unaligned for its own returns
    frames = [s for s in [adj_stock, adj_bench, adj_sect] if s is not None]
    adj_aligned = pd.concat(frames, axis=1).dropna(how="any") if len(frames) > 1 else adj_stock.to_frame(stock)

    # Daily returns for correlation/beta
    rets = adj_aligned.pct_change().dropna(how="any") if len(adj_aligned.columns) > 1 else None
    stock_ret = rets[stock] if rets is not None and stock in rets.columns else None
    bench_ret = rets[bench] if rets is not None and bench and bench in rets.columns else None

    # Trailing returns (trading days)
    r1m, r3m, r6m = _pct_return(adj_stock, 21), _pct_return(adj_stock, 63), _pct_return(adj_stock, 126)
    b1m = _pct_return(adj_bench, 21)  if adj_bench is not None else None
    b3m = _pct_return(adj_bench, 63)  if adj_bench is not None else None
    b6m = _pct_return(adj_bench, 126) if adj_bench is not None else None
    s1m = _pct_return(adj_sect, 21)   if adj_sect  is not None else None
    s3m = _pct_return(adj_sect, 63)   if adj_sect  is not None else None
    s6m = _pct_return(adj_sect, 126)  if adj_sect  is not None else None

    # Relative strength: stock / bench
    rs_ratio_last, rs_slope_90d = None, None
    if adj_bench is not None and not adj_bench.empty:
        ratio = (adj_stock.to_frame(stock).join(adj_bench.to_frame(bench), how="inner"))
        ratio = (ratio[stock] / ratio[bench]).dropna()
        if not ratio.empty:
            rs_ratio_last = float(ratio.iloc[-1])
            rs_window = ratio.tail(90) if len(ratio) >= 3 else ratio
            # Simple linear slope (per step)
            x = np.arange(len(rs_window), dtype=float)
            xm, ym = x.mean(), float(rs_window.mean())
            denom = np.sum((x - xm) ** 2)
            rs_slope_90d = float(np.sum((x - xm) * (rs_window.values - ym)) / denom) if denom != 0 else None

    # Correlation and Beta vs benchmark
    corr_60d  = _rolling_corr(stock_ret, bench_ret, window=60)  if bench_ret is not None else None
    beta_180d = _beta(stock_ret, bench_ret, window=180)         if bench_ret is not None else None

    meta: Dict[str, Any] = {
        "as_of": adj_stock.index.max().strftime("%Y-%m-%d"),
        "yahoo_symbol": stock,
        "benchmark": bench,
        "sector": sect,

        "returns": {
            "stock": {"1m": r1m, "3m": r3m, "6m": r6m},
            "bench": {"1m": b1m, "3m": b3m, "6m": b6m} if bench else None,
            "sector": {"1m": s1m, "3m": s3m, "6m": s6m} if sect else None,
        },

        "outperformance": (
            {
                "1m_vs_bench": (r1m is not None and b1m is not None and r1m > b1m),
                "3m_vs_bench": (r3m is not None and b3m is not None and r3m > b3m),
                "6m_vs_bench": (r6m is not None and b6m is not None and r6m > b6m),
            }
            if bench else None
        ),

        "relative_strength": (
            {"ratio_last": rs_ratio_last, "slope_90d": rs_slope_90d}
            if bench else None
        ),

        "correlation_60d": corr_60d,
        "beta_180d": beta_180d,
    }

    return meta
