# run_sr_plot.py
# Usage:
#   python run_sr_plot.py TICKER [PERIOD] [INTERVAL] [SWING_WINDOW] [TOL_PCT] [LOOKBACK_DAYS] [SAVE_PATH]
# Examples:
#   python run_sr_plot.py RELIANCE
#   python run_sr_plot.py TCS 2y 1d 7 1.0 250 sr_tcs.png
#
# Requires:
#   pip install matplotlib pandas numpy yfinance
#
# Notes:
# - Uses your existing tools if present:
#     tools.price_loader.load_prices
#     tools.sr_mapper.map_support_resistance
# - Falls back to yfinance if needed.

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# Try to use your tools; fall back gracefully
try:
    from tools.price_loader import load_prices
except Exception:
    load_prices = None

try:
    from tools.sr_mapper import map_support_resistance
except Exception as e:
    print("ERROR: tools.sr_mapper.map_support_resistance not found. Make sure sr_mapper.py is in tools/ and importable.")
    raise

# Optional fallback if price_loader isn't available
def _fallback_load_prices(ticker: str, period: str = "1y", interval: str = "1d"):
    import yfinance as yf
    t = ticker.strip().upper()
    if not t.startswith("^") and not t.endswith(".NS"):
        t = f"{t}.NS"
    df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {t}")
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    meta = {
        "yahoo_symbol": t,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "latest_close": float(df["Adj Close"].iloc[-1]),
    }
    return df, meta

def main():
    ticker       = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    period       = sys.argv[2] if len(sys.argv) > 2 else "1y"
    interval     = sys.argv[3] if len(sys.argv) > 3 else "1d"
    swing_window = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    tol_pct      = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    lookback_days = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    save_path    = sys.argv[7] if len(sys.argv) > 7 else None

    # Load prices
    if load_prices is not None:
        df, meta = load_prices(ticker=ticker, period=period, interval=interval)
    else:
        df, meta = _fallback_load_prices(ticker=ticker, period=period, interval=interval)

    # Compute S/R
    sr_meta = map_support_resistance(
        prices_df=df,
        swing_window=swing_window,
        tolerance_pct=tol_pct,
        min_touches=2,
        max_levels_per_side=6
    )

    # Slice for plotting
    plot_df = df.copy()
    if lookback_days and len(plot_df) > lookback_days:
        plot_df = plot_df.tail(lookback_days)

    # Figure
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot Adj Close
    plot_df["Adj Close"].plot(ax=ax, label="Adj Close")

    # X-range for hlines
    x0, x1 = plot_df.index.min(), plot_df.index.max()

    # Draw nearest levels (if any)
    ns = sr_meta.get("nearest_support")
    nr = sr_meta.get("nearest_resistance")

    if ns:
        ax.hlines(y=ns["price"], xmin=x0, xmax=x1, linestyles="--", label=f"Nearest Support ({ns['price']:.2f})")
    if nr:
        ax.hlines(y=nr["price"], xmin=x0, xmax=x1, linestyles="--", label=f"Nearest Resistance ({nr['price']:.2f})")

    # Also draw a few nearby supports/resistances (already proximity-sorted)
    for i, s in enumerate(sr_meta.get("supports", [])):
        ax.hlines(y=s["price"], xmin=x0, xmax=x1, linestyles=":", label=None if i else "Supports (clustered)")
    for i, r in enumerate(sr_meta.get("resistances", [])):
        ax.hlines(y=r["price"], xmin=x0, xmax=x1, linestyles=":", label=None if i else "Resistances (clustered)")

    # Title & labels
    title_ticker = meta.get("yahoo_symbol", ticker)
    as_of = sr_meta.get("as_of")
    plt.title(f"{title_ticker} — Price with Support/Resistance (as of {as_of})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    # Save or show
    if save_path:
        # Make folder if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved plot to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    # Quick print of nearest levels
    print("\nNearest levels:")
    print("  Support:", ns)
    print("  Resistance:", nr)

if __name__ == "__main__":
    main()
