# Docstring Index

## documentation_code.py
### Module: `documentation_code`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py`

```text
Extract all docstrings from a Python codebase without importing modules.

Usage:
  python extract_docstrings.py --root "C:/Users/lenovo/Agent" --out docstrings.md
  python extract_docstrings.py --root ./Agent --out docstrings.json --format json
  python extract_docstrings.py --root ./Agent --include-private
```

### Function: `should_skip_dir`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:23-25`

```text
_(no docstring)_
```

### Function: `read_text`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:27-37`

```text
_(no docstring)_
```

### Function: `node_docstring`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:39-41`

```text
_(no docstring)_
```

### Function: `format_location`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:43-50`

```text
_(no docstring)_
```

### Function: `collect_docstrings_from_file`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:52-148`

```text
_(no docstring)_
```

### Class: `collect_docstrings_from_file.StackVisitor`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:75-145`

```text
_(no docstring)_
```

### Method: `collect_docstrings_from_file.StackVisitor.visit_ClassDef`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:84-97`

```text
_(no docstring)_
```

### Method: `collect_docstrings_from_file.StackVisitor.visit_FunctionDef`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:99-112`

```text
_(no docstring)_
```

### Method: `collect_docstrings_from_file.StackVisitor.visit_AsyncFunctionDef`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:114-128`

```text
_(no docstring)_
```

### Method: `collect_docstrings_from_file.StackVisitor._parents.walk`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:134-143`

```text
_(no docstring)_
```

### Function: `walk_python_files`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:150-158`

```text
_(no docstring)_
```

### Function: `render_markdown`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:160-184`

```text
_(no docstring)_
```

### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\documentation_code.py:186-209`

```text
_(no docstring)_
```

## nodes\context_node.py
### Function: `context_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\context_node.py:5-22`

```text
_(no docstring)_
```

## nodes\events_node.py
### Function: `events_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\events_node.py:21-43`

```text
Inputs (state):
  - ticker (e.g., "RELIANCE")
  - Optional:
      earnings_window_days (default 21)
      exdiv_window_days (default 30)
      lookback_years (default 5)
  - If available: prices.latest_close (for ex-div tilt)

Output (state update):
  - 'event_meta': dict with earnings window, estimated ex-div, splits, and widen/tilt suggestions.
```

## nodes\fundamentals_node.py
### Function: `fundamentals_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\fundamentals_node.py:5-14`

```text
Inputs:
  - ticker (e.g., "RELIANCE")
Output:
  - 'fund_meta': TTM growth/margins/FCF + balance sheet risk + tiny tilt/widen suggestions
```

## nodes\hist_bootstrap_node.py
### Function: `hist_bootstrap_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\hist_bootstrap_node.py:12-43`

```text
Inputs (state):
  - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
  - Optional params:
      horizon_days (default 21)
      lookback_years (default 5)
      winsor_pct (default 0.0)
      extra_percentiles (e.g., [5,25,75,95])

Output (state update):
  - 'hist_bands': dict with empirical p10/p50/p90 and price_bands
```

## nodes\news_collector_node.py
### Function: `news_collector_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\news_collector_node.py:117-169`

```text
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
```

## nodes\price_node.py
### Function: `price_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\price_node.py:5-17`

```text
Minimal LangGraph-compatible node function.
Expects: state has 'ticker' (and optionally 'period', 'interval')
Returns: adds 'prices_df' and 'prices_meta' to state.
```

## nodes\run_events_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\nodes\run_events_node.py:6-31`

```text
Usage:
  python run_events_node.py TICKER [EARN_WINDOW_DAYS] [EXDIV_WINDOW_DAYS] [LOOKBACK_YEARS] [LAST_PRICE]
Examples:
  python run_events_node.py RELIANCE
  python run_events_node.py TCS 21 30 5 4200
```

## nodes\sr_node.py
### Function: `sr_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\sr_node.py:12-43`

```text
Inputs (state):
  - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
  - Optional params:
      swing_window (default 5)
      tolerance_pct (default 1.0)
      min_touches (default 2)
      max_levels_per_side (default 6)

Output (state update):
  - 'sr_meta': dict with nearest support/resistance and level lists
```

## nodes\trend_node.py
### Function: `trend_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\trend_node.py:12-39`

```text
Inputs (state):
  - Either provide 'prices_df', or provide 'ticker' (+ optional 'period','interval')
  - Optional params: ema_short (20), ema_long (50), slope_window (20), rsi_period (14)

Output (state update):
  - 'trend_meta': dict with EMA slopes, distances, RSI, and trend_score (-1..+1)
```

## nodes\volatility_node.py
### Function: `volatility_node`
- **Location:** `C:\Users\lenovo\Agent\nodes\volatility_node.py:13-40`

```text
Inputs (state):
  - Either provide 'prices_df' directly, or provide 'ticker' (plus optional period/interval)
  - Optional: 'z', 'rv_window', 'atr_window', 'ewma_lambda'
Output (state update):
  - 'vol_meta': dict with component vols and blended 1M range
```

## run_agent_1m.py
### Function: `extract_ticker`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:22-30`

```text
Very simple NSE ticker extractor. If input has .NS we keep it; else we append .NS later.
```

### Function: `normalize_nse_ticker`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:32-34`

```text
_(no docstring)_
```

### Class: `AgentState`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:37-53`

```text
_(no docstring)_
```

### Function: `reasoning_1m_node`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:56-122`

```text
Consumes *_meta dicts to produce a 1-month price range and a concise explanation.
Expected inputs present in state: prices_meta, trend_meta, vol_meta, context_meta, fund_meta, event_meta
```

### Function: `build_graph`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:125-155`

```text
_(no docstring)_
```

### Function: `run_1m_agent`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:158-199`

```text
user_text: free-form input like "What's the 1M range for Reliance?"
Returns: dict with ticker, latest, forecast_1m {low, high, confidence}, and reasoning text.
```

### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_agent_1m.py:201-217`

```text
_(no docstring)_
```

## run_context_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_context_node.py:6-31`

```text
Usage:
  python run_context_node.py TICKER [BENCHMARK] [SECTOR] [PERIOD] [INTERVAL]
Examples:
  python run_context_node.py RELIANCE
  python run_context_node.py TCS ^NSEI ^CNXIT 2y 1d
  python run_context_node.py ICICIBANK ^NSEI None 1y 1d
```

## run_fundamentals_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_fundamentals_node.py:5-9`

```text
_(no docstring)_
```

## run_hist_bootstrap_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_hist_bootstrap_node.py:6-32`

```text
Usage:
  python run_hist_bootstrap_node.py TICKER [LOOKBACK_YEARS] [HORIZON_DAYS] [WINSOR_PCT]
Examples:
  python run_hist_bootstrap_node.py RELIANCE
  python run_hist_bootstrap_node.py TCS 7 21 0.01
  python run_hist_bootstrap_node.py INFY 10 21 0.0
```

## run_news_collect_node.py
### Function: `print_cards`
- **Location:** `C:\Users\lenovo\Agent\run_news_collect_node.py:14-20`

```text
_(no docstring)_
```

### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_news_collect_node.py:22-44`

```text
_(no docstring)_
```

## run_price_node.py
### Function: `print_nested_dict`
- **Location:** `C:\Users\lenovo\Agent\run_price_node.py:4-10`

```text
_(no docstring)_
```

### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_price_node.py:12-26`

```text
_(no docstring)_
```

## run_sr_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_sr_node.py:6-33`

```text
Usage:
  python run_sr_node.py TICKER [PERIOD] [INTERVAL] [SWING_WINDOW] [TOL_PCT]
Examples:
  python run_sr_node.py RELIANCE
  python run_sr_node.py TCS 2y 1d 7 1.0
```

## run_sr_plot.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_sr_plot.py:54-131`

```text
_(no docstring)_
```

## run_trend_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_trend_node.py:6-31`

```text
Usage:
  python run_trend_node.py TICKER [PERIOD] [INTERVAL]
Examples:
  python run_trend_node.py RELIANCE
  python run_trend_node.py TCS 2y 1d
```

## run_volatility_node.py
### Function: `main`
- **Location:** `C:\Users\lenovo\Agent\run_volatility_node.py:7-29`

```text
Usage:
  python run_volatility_node.py TICKER [PERIOD] [INTERVAL] [Z]
Examples:
  python run_volatility_node.py RELIANCE
  python run_volatility_node.py TCS 2y 1d 1.28
```

## tools\context_loader.py
### Function: `normalize_symbol`
- **Location:** `C:\Users\lenovo\Agent\tools\context_loader.py:9-16`

```text
Normalize NSE equities to end with .NS. Keep index tickers like ^NSEI as-is.
```

### Function: `load_context`
- **Location:** `C:\Users\lenovo\Agent\tools\context_loader.py:49-193`

```text
Pulls Adj Close for stock, benchmark (and optional sector), then derives:
  - 1M/3M/6M returns (21/63/126 trading days)
  - out/under-performance vs benchmark
  - relative strength ratio (stock/bench) + 90d slope
  - 60d correlation and 180d beta vs benchmark

Returns a single 'context_meta' dict.
Gracefully handles missing benchmark/sector data (keeps stock-only stats).
```

### Function: `load_context.pick`
- **Location:** `C:\Users\lenovo\Agent\tools\context_loader.py:102-118`

```text
_(no docstring)_
```

## tools\events_window.py
### Function: `normalize_symbol`
- **Location:** `C:\Users\lenovo\Agent\tools\events_window.py:9-12`

```text
Add .NS for NSE equities; keep index tickers like ^NSEI as-is.
```

### Function: `load_events_window`
- **Location:** `C:\Users\lenovo\Agent\tools\events_window.py:77-209`

```text
Pull near-term catalysts from Yahoo:
  - next/last earnings (if available)
  - dividend history with estimated next ex-div date from median past gap
  - recent split
Also suggests tiny widen/tilt heuristics for 1-month bands.
```

## tools\fundamentals_momentum.py
### Function: `load_fundamentals_momentum`
- **Location:** `C:\Users\lenovo\Agent\tools\fundamentals_momentum.py:85-264`

```text
Builds TTM metrics from Yahoo statements with robust fallbacks:
  1) Try quarterly statements (preferred).
  2) Fallback to annual statements (approximate TTM with last annual).
  3) Fallback to get_info() keys for TTM revenue, margins, OCF/FCF, D/E, current ratio.

Returns 'fund_meta' with tilt/widen suggestions for 1-month usage.
```

## tools\hist_bootstrap.py
### Function: `estimate_hist_1m_bands`
- **Location:** `C:\Users\lenovo\Agent\tools\hist_bootstrap.py:30-111`

```text
Compute empirical forward-return bands over a lookback window.

Input:
  prices_df: DataFrame with at least ['Close','Adj Close'] columns, index=dates (daily).
  horizon_days: forward horizon in trading days (21 ~ 1 month).
  lookback_years: limit history to last N years (use 0 or None for all data).
  winsor_pct: optional tail clipping to reduce outliers.
  extra_percentiles: e.g., [5, 25, 75, 95] to include more quantiles.

Output:
  hist_bands: dict with p10/p50/p90, optional extras, and price band (low/base/high).
```

## tools\llm.py
### Function: `llm_json`
- **Location:** `C:\Users\lenovo\Agent\tools\llm.py:7-18`

```text
Ask ChatGPT for a STRICT JSON response. Raise if not JSON.
```

## tools\price_loader.py
### Function: `normalize_nse_ticker`
- **Location:** `C:\Users\lenovo\Agent\tools\price_loader.py:5-7`

```text
_(no docstring)_
```

### Function: `load_prices`
- **Location:** `C:\Users\lenovo\Agent\tools\price_loader.py:9-163`

```text
Fetch OHLCV data and return (df, meta).
meta is now a nested dictionary with category partitions for better organization.
Includes latest/previous close, 52w high/low, 30d avg volume,
and (best-effort) fundamentals from Yahoo, including valuation ratios and earnings/growth metrics.
```

### Function: `flatten_meta`
- **Location:** `C:\Users\lenovo\Agent\tools\price_loader.py:165-177`

```text
Helper to flatten nested meta back to a single-level dict for backward compatibility.
```

### Function: `preview`
- **Location:** `C:\Users\lenovo\Agent\tools\price_loader.py:179-182`

```text
_(no docstring)_
```

## tools\sr_mapper.py
### Function: `_cluster_levels.rel_diff`
- **Location:** `C:\Users\lenovo\Agent\tools\sr_mapper.py:59-60`

```text
_(no docstring)_
```

### Function: `map_support_resistance`
- **Location:** `C:\Users\lenovo\Agent\tools\sr_mapper.py:91-159`

```text
Find support/resistance levels:
  1) detect swing highs/lows
  2) cluster nearby swing levels
  3) filter by min_touches
  4) pick nearest support below and resistance above the last price

Returns sr_meta with nearest levels and lists of levels.
```

## tools\trend_gauge.py
### Function: `estimate_trend_meta`
- **Location:** `C:\Users\lenovo\Agent\tools\trend_gauge.py:46-134`

```text
Input:
  prices_df: DataFrame with ['Open','High','Low','Close','Adj Close','Volume'].
Output:
  trend_meta: dict with EMA slopes, price distances, RSI, and a trend score (-1..+1).
```

## tools\volatility_range.py
### Function: `estimate_1m_range`
- **Location:** `C:\Users\lenovo\Agent\tools\volatility_range.py:42-130`

```text
Input:
  prices_df: pandas DataFrame with columns at least:
             ['Open','High','Low','Close','Adj Close','Volume'], index = dates
Output:
  vol_meta: dict with components and a blended [low, base, high]
```

## tools\web_fetch.py
### Function: `fetch_url`
- **Location:** `C:\Users\lenovo\Agent\tools\web_fetch.py:5-8`

```text
_(no docstring)_
```

### Function: `extract_readable`
- **Location:** `C:\Users\lenovo\Agent\tools\web_fetch.py:10-12`

```text
_(no docstring)_
```

### Function: `fetch_and_clean`
- **Location:** `C:\Users\lenovo\Agent\tools\web_fetch.py:14-25`

```text
_(no docstring)_
```
