# pip install yfinance pandas
import yfinance as yf
import pandas as pd

def normalize_nse_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith(".NS") else f"{t}.NS"

def load_prices(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Fetch OHLCV data and return (df, meta).
    meta is now a nested dictionary with category partitions for better organization.
    Includes latest/previous close, 52w high/low, 30d avg volume,
    and (best-effort) fundamentals from Yahoo, including valuation ratios and earnings/growth metrics.
    """
    symbol = normalize_nse_ticker(ticker)
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
   
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol} (period={period}, interval={interval}).")
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    # --- Derived stats from OHLCV ---
    latest_close = float(df["Adj Close"].iloc[-1])
    prev_close = float(df["Adj Close"].iloc[-2]) if len(df) >= 2 else None
    # 52-week window (calendar 365 days)
    cutoff = df.index.max() - pd.Timedelta(days=365)
    window = df.loc[df.index >= cutoff]
    # 30 trading days average volume (works best with interval="1d")
    avg_volume_30d = int(df["Volume"].tail(30).mean()) if len(df) >= 1 else None
    latest_volume = int(df["Volume"].iloc[-1])
    # --- Fundamentals (best-effort via Yahoo) ---
    pe_ratio = None
    dividend_yield = None
    price_to_book = None
    peg_ratio = None
    enterprise_value = None
    profit_margins = None
    gross_margins = None
    operating_margins = None
    ebitda_margins = None
    return_on_assets = None
    return_on_equity = None
    debt_to_equity = None
    current_ratio = None
    beta = None
    trailing_eps = None
    forward_eps = None
    earnings_growth = None
    revenue_growth = None
    book_value = None
    price_to_sales = None
    try:
        info = yf.Ticker(symbol).get_info()  # may be slow / sometimes missing for some tickers
        # Existing fundamentals
        pe_ratio = info.get("trailingPE", None) or info.get("forwardPE", None)
        dividend_yield = info.get("dividendYield", None) or info.get("trailingAnnualDividendYield", None)
        company_name = info.get('longName', 'N/A')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        average_volume = info.get('averageVolume', 'N/A')
        exchange = info.get('exchange', 'N/A')
        quote_type = info.get('quoteType', 'N/A')
        high_52w = info.get('fiftyTwoWeekHigh', 'N/A')
        low_52w = info.get('fiftyTwoWeekLow', 'N/A')
        price_change_52w = info.get('52WeekChange', 'N/A')
        if price_change_52w != 'N/A':
            price_change_52w *= 100  # Convert to percentage
        # New valuation and financial ratios
        price_to_book = info.get('priceToBook', None)
        peg_ratio = info.get('pegRatio', None)
        enterprise_value = info.get('enterpriseValue', None)
        profit_margins = info.get('profitMargins', None)
        gross_margins = info.get('grossMargins', None)
        operating_margins = info.get('operatingMargins', None)
        ebitda_margins = info.get('ebitdaMargins', None)
        return_on_assets = info.get('returnOnAssets', None)
        return_on_equity = info.get('returnOnEquity', None)
        debt_to_equity = info.get('debtToEquity', None)
        current_ratio = info.get('currentRatio', None)
        beta = info.get('beta', None)
        # New earnings and growth metrics
        trailing_eps = info.get('trailingEps', None)
        forward_eps = info.get('forwardEps', None)
        earnings_growth = info.get('earningsGrowth', None)
        revenue_growth = info.get('revenueGrowth', None)
        book_value = info.get('bookValue', None)
        price_to_sales = info.get('priceToSalesTrailing12Months', None)
        # Coerce to floats where applicable
        for key in ['pe_ratio', 'dividend_yield', 'price_to_book', 'peg_ratio', 'enterprise_value',
                    'profit_margins', 'gross_margins', 'operating_margins', 'ebitda_margins',
                    'return_on_assets', 'return_on_equity', 'debt_to_equity', 'current_ratio',
                    'beta', 'trailing_eps', 'forward_eps', 'earnings_growth', 'revenue_growth',
                    'book_value', 'price_to_sales']:
            if locals()[key] is not None:
                locals()[key] = float(locals()[key])
    except Exception:
        # If unavailable, leave as None
        pass
    
    # Nested meta dictionary with categories
    meta = {
        "company": {
            "name": company_name,
            "input_ticker": ticker,
            "exchange": exchange,
            "quote_type": quote_type,
            "yahoo_symbol": symbol,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "as_of": df.index[-1].strftime("%Y-%m-%d")
        },
        "prices": {
            "latest_close": latest_close,
            "previous_close": prev_close,
            "52w_high": high_52w,
            "52w_low": low_52w,
            "price_change_52w": price_change_52w,
        },
        "volumes": {
            "latest_volume": latest_volume,
            "average_volume": average_volume,
            "avg_volume_30d": avg_volume_30d,
        },
        "fundamentals": {
            "valuation_ratios": {
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield,
                "price_to_book": price_to_book,
                "peg_ratio": peg_ratio,
                "enterprise_value": enterprise_value,
                "price_to_sales": price_to_sales,
            },
            "financial_ratios": {
                "profit_margins": profit_margins,
                "gross_margins": gross_margins,
                "operating_margins": operating_margins,
                "ebitda_margins": ebitda_margins,
                "return_on_assets": return_on_assets,
                "return_on_equity": return_on_equity,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "beta": beta,
            },
            "earnings_growth": {
                "trailing_eps": trailing_eps,
                "forward_eps": forward_eps,
                "earnings_growth": earnings_growth,
                "revenue_growth": revenue_growth,
                "book_value": book_value,
            },
        },
        "about_prices_df": {
            "prices_df_period": period,
            "prices_df_interval": interval,
            "no_of_rows_prices_df": int(len(df))
        },
    }
    return df, meta

def flatten_meta(meta: dict) -> dict:
    """
    Helper to flatten nested meta back to a single-level dict for backward compatibility.
    """
    flat = {}
    for category, subdict in meta.items():
        for key, value in subdict.items():
            if isinstance(subdict, dict):  # Handle deeper nesting in fundamentals
                for subkey, subvalue in subdict[key].items():
                    flat[f"{key}_{subkey}"] = subvalue
            else:
                flat[f"{category}_{key}"] = value
    return flat

def preview(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    return pd.concat([df.head(n), df.tail(n)])