# tools/fundamentals_momentum.py
# pip install yfinance pandas numpy

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- helpers ----------
def _norm(sym: str) -> str:
    s = sym.strip().upper()
    return s if s.startswith("^") or s.endswith(".NS") else f"{s}.NS"

def _get_df(tkr: yf.Ticker, candidates: List[str]) -> Optional[pd.DataFrame]:
    """
    Try a list of Ticker attributes or zero-arg methods and return the first non-empty DataFrame.
    Examples: ['quarterly_financials', 'financials'] or ['quarterly_cashflow', 'cashflow'].
    Safe even if attribute doesn't exist.
    """
    for name in candidates:
        try:
            obj = getattr(tkr, name, None)
            if obj is None:
                continue
            df = obj() if callable(obj) else obj
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    return None

def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Find a row by fuzzy label match (case-insensitive contains). Returns row (Series) or None."""
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(idx):
            if c == name or c in name:
                return df.iloc[i]
    return None

def _ttm_sum(row: pd.Series, k: int = 4) -> Optional[float]:
    """Sum last k columns of a statement row (columns are periods)."""
    if row is None or row.empty:
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    if vals.empty:
        return None
    return float(vals.tail(k).sum()) if len(vals) >= k else float(vals.sum())

def _ttm_sum_prev(row: pd.Series, k: int = 4) -> Optional[float]:
    """Sum previous k columns before the latest k (for YoY on TTM)."""
    if row is None or row.empty:
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    if len(vals) >= 2 * k:
        return float(vals.tail(2 * k).head(k).sum())
    return None

def _last_val(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    return float(vals.iloc[-1]) if not vals.empty else None

def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    try:
        if num is None or den in (None, 0):
            return None
        return float(num / den)
    except Exception:
        return None

def _latest_col_date(df: Optional[pd.DataFrame]) -> Optional[str]:
    try:
        if df is None or df.empty:
            return None
        return pd.to_datetime(df.columns.max()).date().isoformat()
    except Exception:
        return None

# ---------- core ----------
def load_fundamentals_momentum(ticker: str) -> Dict[str, Any]:
    """
    Builds TTM metrics from Yahoo statements with robust fallbacks:
      1) Try quarterly statements (preferred).
      2) Fallback to annual statements (approximate TTM with last annual).
      3) Fallback to get_info() keys for TTM revenue, margins, OCF/FCF, D/E, current ratio.

    Returns 'fund_meta' with tilt/widen suggestions for 1-month usage.
    """
    sym = _norm(ticker)
    tkr = yf.Ticker(sym)

    used = {"income": None, "cashflow": None, "balance_sheet": None, "info_fallbacks": []}

    # Pull statements (quarterly preferred; then annual)
    q_is = _get_df(tkr, ["quarterly_financials"])
    q_cf = _get_df(tkr, ["quarterly_cashflow"])
    q_bs = _get_df(tkr, ["quarterly_balance_sheet"])

    a_is = _get_df(tkr, ["financials"])            # annual IS
    a_cf = _get_df(tkr, ["cashflow"])              # annual CF
    a_bs = _get_df(tkr, ["balance_sheet"])         # annual BS

    # ----- Income Statement -----
    is_df = q_is if (q_is is not None and not q_is.empty) else a_is
    used["income"] = "quarterly" if is_df is q_is else ("annual" if is_df is a_is else None)

    row_rev = _pick_row(is_df, ["Total Revenue", "Revenue"])
    row_oi  = _pick_row(is_df, ["Operating Income", "Operating Income or Loss"])
    row_ni  = _pick_row(is_df, ["Net Income", "Net Income Applicable To Common Shares"])

    if is_df is q_is:
        rev_ttm  = _ttm_sum(row_rev, 4)
        rev_ttm_prev = _ttm_sum_prev(row_rev, 4)
        oi_ttm   = _ttm_sum(row_oi, 4)
        oi_ttm_prev = _ttm_sum_prev(row_oi, 4)
        ni_ttm   = _ttm_sum(row_ni, 4)
        ni_ttm_prev = _ttm_sum_prev(row_ni, 4)
    elif is_df is a_is:
        # Annual fallback: approximate TTM by last annual; YoY from last two annuals
        rev_ttm  = _last_val(row_rev)
        rev_ttm_prev = _ttm_sum_prev(row_rev, 1)   # previous annual
        oi_ttm   = _last_val(row_oi)
        oi_ttm_prev = _ttm_sum_prev(row_oi, 1)
        ni_ttm   = _last_val(row_ni)
        ni_ttm_prev = _ttm_sum_prev(row_ni, 1)
    else:
        rev_ttm = rev_ttm_prev = oi_ttm = oi_ttm_prev = ni_ttm = ni_ttm_prev = None

    op_margin_ttm = _safe_ratio(oi_ttm, rev_ttm)
    op_margin_prev = _safe_ratio(oi_ttm_prev, rev_ttm_prev)
    op_margin_delta_pp = (op_margin_ttm - op_margin_prev) * 100.0 if (op_margin_ttm is not None and op_margin_prev is not None) else None

    rev_yoy = (rev_ttm / rev_ttm_prev - 1.0) if (rev_ttm is not None and rev_ttm_prev not in (None, 0)) else None
    ni_yoy  = (ni_ttm / ni_ttm_prev - 1.0) if (ni_ttm is not None and ni_ttm_prev not in (None, 0)) else None

    # ----- Cashflow -----
    cf_df = q_cf if (q_cf is not None and not q_cf.empty) else a_cf
    used["cashflow"] = "quarterly" if cf_df is q_cf else ("annual" if cf_df is a_cf else None)

    row_ocf   = _pick_row(cf_df, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    row_capex = _pick_row(cf_df, ["Capital Expenditures", "Investment In Property, Plant, and Equipment"])

    if cf_df is q_cf:
        ocf_ttm   = _ttm_sum(row_ocf, 4)
        capex_ttm = _ttm_sum(row_capex, 4)
    elif cf_df is a_cf:
        ocf_ttm   = _last_val(row_ocf)
        capex_ttm = _last_val(row_capex)
    else:
        ocf_ttm = capex_ttm = None

    fcf_ttm = (ocf_ttm + capex_ttm) if (ocf_ttm is not None and capex_ttm is not None) else None  # CapEx often negative

    # ----- Balance Sheet (latest) -----
    bs_df = q_bs if (q_bs is not None and not q_bs.empty) else a_bs
    used["balance_sheet"] = "quarterly" if bs_df is q_bs else ("annual" if bs_df is a_bs else None)

    row_tot_debt = _pick_row(bs_df, ["Total Debt", "Short Long Term Debt", "Short Term Debt", "Long Term Debt"])
    row_equity   = _pick_row(bs_df, ["Total Stockholder Equity", "Total Shareholder Equity", "Total Equity"])
    row_ca       = _pick_row(bs_df, ["Total Current Assets"])
    row_cl       = _pick_row(bs_df, ["Total Current Liabilities"])

    total_debt = _last_val(row_tot_debt)
    equity     = _last_val(row_equity)
    cur_assets = _last_val(row_ca)
    cur_liab   = _last_val(row_cl)

    debt_to_equity = _safe_ratio(total_debt, equity)
    current_ratio  = _safe_ratio(cur_assets, cur_liab)

    # ----- Fallback to get_info() for commonly-available TTM items -----
    info = {}
    try:
        info = tkr.get_info() or {}
    except Exception:
        info = {}

    # Info-based fills (only if missing)
    if rev_ttm is None and info.get("totalRevenue") is not None:
        rev_ttm = float(info["totalRevenue"]); used["info_fallbacks"].append("totalRevenue→revenue")
    if op_margin_ttm is None and info.get("operatingMargins") is not None:
        op_margin_ttm = float(info["operatingMargins"]); used["info_fallbacks"].append("operatingMargins→op_margin")
    if ni_ttm is None and info.get("netIncomeToCommon") is not None:
        ni_ttm = float(info["netIncomeToCommon"]); used["info_fallbacks"].append("netIncomeToCommon→net_income")
    if ocf_ttm is None and info.get("operatingCashflow") is not None:
        ocf_ttm = float(info["operatingCashflow"]); used["info_fallbacks"].append("operatingCashflow→ocf")
    if fcf_ttm is None and info.get("freeCashflow") is not None:
        fcf_ttm = float(info["freeCashflow"]); used["info_fallbacks"].append("freeCashflow→fcf")
    if debt_to_equity is None and info.get("debtToEquity") is not None:
        debt_to_equity = float(info["debtToEquity"]); used["info_fallbacks"].append("debtToEquity")
    if current_ratio is None and info.get("currentRatio") is not None:
        current_ratio = float(info["currentRatio"]); used["info_fallbacks"].append("currentRatio")

    # as_of dates
    as_of_is = _latest_col_date(q_is if used["income"] == "quarterly" else (a_is if used["income"] == "annual" else None))
    as_of_cf = _latest_col_date(q_cf if used["cashflow"] == "quarterly" else (a_cf if used["cashflow"] == "annual" else None))
    as_of_bs = _latest_col_date(q_bs if used["balance_sheet"] == "quarterly" else (a_bs if used["balance_sheet"] == "annual" else None))

    # ----- Heuristics for 1-mo tilt/widen -----
    tilt = 0.0
    widen_bp = 0.0
    reasons = []

    # Growth & margins
    if rev_yoy is not None:
        if rev_yoy > 0.10: tilt += 0.01; reasons.append("Revenue TTM YoY strong (>10%).")
        elif rev_yoy < -0.05: tilt -= 0.01; reasons.append("Revenue TTM YoY weak (<-5%).")
    if ni_yoy is not None:
        if ni_yoy > 0.15: tilt += 0.01; reasons.append("Net income TTM YoY strong (>15%).")
        elif ni_yoy < -0.10: tilt -= 0.01; reasons.append("Net income TTM YoY weak (<-10%).")
    if op_margin_delta_pp is not None:
        if op_margin_delta_pp > 1.0: tilt += 0.005; reasons.append("Operating margin expanding (>+1 pp).")
        elif op_margin_delta_pp < -1.0: tilt -= 0.005; reasons.append("Operating margin contracting (<-1 pp).")

    # Balance sheet risk → widen a bit
    if debt_to_equity is not None and debt_to_equity > 1.5:
        widen_bp += 0.01; reasons.append("High leverage (D/E > 1.5) → widen.")
    if current_ratio is not None and current_ratio < 1.0:
        widen_bp += 0.005; reasons.append("Low liquidity (Current ratio < 1.0) → widen.")

    # Clip tilt
    tilt = float(np.clip(tilt, -0.02, 0.02))

    fund_meta: Dict[str, Any] = {
        "as_of": {
            "income_statement": as_of_is,
            "cashflow": as_of_cf,
            "balance_sheet": as_of_bs
        },
        "sources": used,  # shows what we used (quarterly/annual/info fallbacks)
        "metrics_ttm": {
            "revenue": rev_ttm,
            "operating_income": oi_ttm,
            "net_income": ni_ttm,
            "op_margin": op_margin_ttm,                 # 0..1
            "op_margin_delta_pp": op_margin_delta_pp,    # percentage points vs prior TTM
            "ocf": ocf_ttm,
            "capex": capex_ttm,
            "fcf": fcf_ttm
        },
        "yoy": {
            "revenue_yoy": rev_yoy,
            "net_income_yoy": ni_yoy
        },
        "balance_sheet": {
            "total_debt_latest": total_debt,
            "equity_latest": equity,
            "debt_to_equity": debt_to_equity,
            "current_assets_latest": cur_assets,
            "current_liabilities_latest": cur_liab,
            "current_ratio": current_ratio
        },
        "suggestions": {
            "tilt": tilt,             # −0.02..+0.02
            "widen_bp": round(widen_bp, 6),
            "reasons": reasons
        }
    }
    return fund_meta
