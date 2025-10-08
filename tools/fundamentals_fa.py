# tools/fundamentals_fa.py
# Core FA computation (numeric + optional LLM narrative)
# Requires: pandas, numpy, yfinance, openpyxl (for reading Excel), and your tools/llm.py (optional)

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os, math, json
import numpy as np
import pandas as pd
import yfinance as yf

# ---- Optional LLM helper (strict JSON) ----
_LLM_OK = False
try:
    from tools.llm import llm_json  # your helper returns a JSON string
    _LLM_OK = True if os.environ.get("OPENAI_API_KEY") else False
except Exception:
    _LLM_OK = False

# =========================
# Config: thresholds/weights
# =========================
FA_CONFIG: Dict[str, Any] = {
    # Normalization thresholds (you can tune)
    "growth":       {"rev_yoy_pos": 0.15, "rev_yoy_neg": -0.05, "ni_yoy_pos": 0.20, "ni_yoy_neg": -0.10,
                     "op_margin_delta_pos_pp": 1.0, "op_margin_delta_neg_pp": -1.0},
    "quality":      {"op_margin_good": 0.18, "op_margin_poor": 0.08, "fcf_margin_good": 0.08, "fcf_margin_poor": 0.0},
    "balance":      {"de_high": 2.0, "de_ok": 1.0, "current_ratio_low": 1.0, "current_ratio_good": 1.5},
    "cashflow":     {"fcf_stable_cv": 0.35},  # coefficient of variation threshold for stability
    "valuation":    {"pe_hi": 35.0, "pe_lo": 12.0, "pb_hi": 6.0, "pb_lo": 1.5, "ev_ebitda_hi": 20.0, "ev_ebitda_lo": 8.0},
    "dividend":     {"yield_defensive": 0.02, "payout_high": 0.8},
    "event":        {"earnings_horizon_days": 95},  # if earnings within ~3 months => raise bandwidth
    "risk":         {"beta_hi": 1.2, "beta_lo": 0.7},

    # Weights for composite signals
    "weights_tilt": {"G": 0.35, "Q": 0.20, "C": 0.10, "V": 0.15, "E": 0.15, "D": 0.05},
    "weights_bw":   {"B": 0.30, "V": 0.25, "E": 0.25, "R": 0.15, "C": 0.05},

    # Final clipping & scaling
    "tilt_clip": 0.03,        # max ±3% drift contribution from FA
    "bw_min": 0.02,           # at least 2% relative bandwidth adjust
    "bw_max": 0.25,           # cap at 25%
    "bw_scale": 0.12,         # scale composite to % (tunable)
}

# =========================
# Helpers (common)
# =========================
def _norm(sym: str) -> str:
    if not sym:
        return sym
    s = sym.strip().upper()
    return s if s.startswith("^") or s.endswith(".NS") else f"{s}.NS"

def _last_val(row: Optional[pd.Series]) -> Optional[float]:
    if row is None or not isinstance(row, pd.Series):
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    return float(vals.iloc[-1]) if len(vals) else None

def _ttm_sum(row: Optional[pd.Series], k: int = 4) -> Optional[float]:
    if row is None or not isinstance(row, pd.Series):
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    if not len(vals):
        return None
    return float(vals.tail(k).sum()) if len(vals) >= k else float(vals.sum())

def _ttm_prev(row: Optional[pd.Series], k: int = 4) -> Optional[float]:
    if row is None or not isinstance(row, pd.Series):
        return None
    vals = pd.to_numeric(row.dropna(), errors="coerce")
    if len(vals) >= 2 * k:
        return float(vals.tail(2 * k).head(k).sum())
    return None

def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)
    except Exception:
        return None

def _cv(x: List[float]) -> Optional[float]:
    arr = np.array([v for v in x if v is not None and (isinstance(v, (int, float)))], dtype=float)
    if len(arr) < 2:
        return None
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if mu == 0:
        return None
    return abs(float(sigma / mu))

def _pp(x: Optional[float]) -> Optional[float]:
    # percentage points to 0.xx
    if x is None:
        return None
    return float(x) / 100.0

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _latest_col_date(df: Optional[pd.DataFrame]) -> Optional[str]:
    try:
        if df is None or df.empty:
            return None
        # yfinance statements have dates as columns
        return pd.to_datetime(df.columns.max()).date().isoformat()
    except Exception:
        return None

# =========================
# Data Access: Yahoo or Excel
# =========================
def _fetch_from_yahoo(sym: str):
    tkr = yf.Ticker(sym)
    # Annual/Quarterly statements
    QIS = tkr.quarterly_financials
    QCF = tkr.quarterly_cashflow
    QBS = tkr.quarterly_balance_sheet
    AIS = tkr.financials
    ACF = tkr.cashflow
    ABS = tkr.balance_sheet

    # "Info" dict with ratios, market data, estimates (forward PE, beta, target price etc.)
    try:
        info = tkr.get_info() or {}
    except Exception:
        info = getattr(tkr, "info", {}) or {}

    # Recommendations, dividends (optional)
    recs = getattr(tkr, "recommendations", None)
    divs = getattr(tkr, "dividends", pd.Series(dtype=float))

    # Earnings calendar (next earnings date sometimes appears under 'earnings' / 'calendar')
    cal = getattr(tkr, "calendar", None)

    return {
        "QIS": QIS, "QCF": QCF, "QBS": QBS,
        "AIS": AIS, "ACF": ACF, "ABS": ABS,
        "INFO": info, "RECS": recs, "DIVS": divs, "CAL": cal
    }

def _fetch_from_excel(xlsx_path: str) -> Dict[str, Any]:
    xl = pd.ExcelFile(xlsx_path)
    def get_sheet(name: str) -> Optional[pd.DataFrame]:
        try:
            return xl.parse(name)
        except Exception:
            return None

    # Your export names (see reference script)
    QIS = get_sheet("Income_Quarterly")
    AIS = get_sheet("Income_Annual")
    QBS = get_sheet("Balance_Quarterly")
    ABS = get_sheet("Balance_Annual")
    QCF = get_sheet("Cashflow_Quarterly")
    ACF = get_sheet("Cashflow_Annual")
    INFO = get_sheet("Key_Statistics")
    RECS = get_sheet("Analyst_Recommendations")
    DIVS = get_sheet("Dividends")

    # Convert INFO sheet (Metric, Value) into a dict
    info_dict = {}
    if isinstance(INFO, pd.DataFrame) and not INFO.empty and "Metric" in INFO.columns and "Value" in INFO.columns:
        info_dict = pd.Series(INFO.Value.values, index=INFO.Metric).to_dict()

    # Make series for DIVS
    divs_series = pd.Series(dtype=float)
    if isinstance(DIVS, pd.DataFrame) and "Dividend" in DIVS.columns and DIVS.shape[1] == 1:
        # Index likely dates; leave as DataFrame—consumer does not rely heavily on it
        pass

    return {
        "QIS": _index_by_rows(QIS),
        "QCF": _index_by_rows(QCF),
        "QBS": _index_by_rows(QBS),
        "AIS": _index_by_rows(AIS),
        "ACF": _index_by_rows(ACF),
        "ABS": _index_by_rows(ABS),
        "INFO": info_dict,
        "RECS": RECS,
        "DIVS": DIVS,
        "CAL": None,  # not exported in the reference script
    }

def _index_by_rows(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # yfinance statements use rows as items (e.g., 'Total Revenue') and columns as periods
    if df is None or df.empty:
        return None
    if df.columns[0] != df.columns.name and df.columns[0] != 0 and "Unnamed" in str(df.columns[0]):
        # Try to fix messy headers—best effort
        pass
    if df.columns[0] != 0 and "Unnamed" not in str(df.columns[0]):
        # If first column is labels, set as index
        if df.columns[0] not in ("Breakdown", "Metric"):
            df = df.set_index(df.columns[0], drop=True)
    if not isinstance(df.index, pd.Index):
        return df
    # Ensure index is string for fuzzy matching
    df.index = df.index.map(lambda x: str(x))
    return df

# =========================
# Row pickers (fuzzy)
# =========================
def _pick_row(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(idx):
            if c == name or c in name:
                return df.iloc[i]
    return None

# =========================
# Core FA computation
# =========================
def compute_fa(sym: str, xlsx_path: Optional[str] = None, use_llm: bool = False) -> Dict[str, Any]:
    sym = _norm(sym)

    data = _fetch_from_excel(xlsx_path) if xlsx_path else _fetch_from_yahoo(sym)
    QIS, QCF, QBS = data["QIS"], data["QCF"], data["QBS"]
    AIS, ACF, ABS = data["AIS"], data["ACF"], data["ABS"]
    INFO, RECS, DIVS, CAL = data["INFO"], data["RECS"], data["DIVS"], data["CAL"]

    # Prefer quarterly statements; fall back to annual
    IS = QIS if (QIS is not None and not QIS.empty) else AIS
    CF = QCF if (QCF is not None and not QCF.empty) else ACF
    BS = QBS if (QBS is not None and not QBS.empty) else ABS

    used = {
        "income": "quarterly" if IS is QIS else ("annual" if IS is AIS else None),
        "cashflow": "quarterly" if CF is QCF else ("annual" if CF is ACF else None),
        "balance_sheet": "quarterly" if BS is QBS else ("annual" if BS is ABS else None),
        "source": "excel" if xlsx_path else "yahoo"
    }

    # ------ Extract metrics ------
    row_rev = _pick_row(IS, ["Total Revenue", "Revenue"])
    row_oi  = _pick_row(IS, ["Operating Income", "Operating Income or Loss"])
    row_ni  = _pick_row(IS, ["Net Income", "Net Income Applicable To Common Shares"])

    rev_ttm = _ttm_sum(row_rev, 4) if IS is QIS else _last_val(row_rev)
    rev_ttm_prev = _ttm_prev(row_rev, 4) if IS is QIS else _ttm_prev(row_rev, 1)

    oi_ttm = _ttm_sum(row_oi, 4) if IS is QIS else _last_val(row_oi)
    oi_ttm_prev = _ttm_prev(row_oi, 4) if IS is QIS else _ttm_prev(row_oi, 1)

    ni_ttm = _ttm_sum(row_ni, 4) if IS is QIS else _last_val(row_ni)
    ni_ttm_prev = _ttm_prev(row_ni, 4) if IS is QIS else _ttm_prev(row_ni, 1)

    op_margin_ttm = _safe_ratio(oi_ttm, rev_ttm)
    op_margin_prev = _safe_ratio(oi_ttm_prev, rev_ttm_prev)
    op_margin_delta_pp = ((op_margin_ttm - op_margin_prev) * 100.0
                          if (op_margin_ttm is not None and op_margin_prev is not None) else None)

    rev_yoy = (rev_ttm / rev_ttm_prev - 1.0) if (rev_ttm is not None and rev_ttm_prev not in (None, 0)) else None
    ni_yoy  = (ni_ttm / ni_ttm_prev - 1.0) if (ni_ttm is not None and ni_ttm_prev not in (None, 0)) else None

    # Cashflow: OCF, CapEx, FCF (+ basic stability)
    row_ocf   = _pick_row(CF, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    row_capex = _pick_row(CF, ["Capital Expenditures", "Investment In Property, Plant, and Equipment"])
    ocf_ttm   = _ttm_sum(row_ocf, 4) if CF is QCF else _last_val(row_ocf)
    capex_ttm = _ttm_sum(row_capex, 4) if CF is QCF else _last_val(row_capex)
    fcf_ttm   = (ocf_ttm + capex_ttm) if (ocf_ttm is not None and capex_ttm is not None) else None  # CapEx negative

    # FCF stability (quarterly series if available)
    fcf_quarters: List[float] = []
    if CF is not None and not CF.empty and CF is QCF:
        ocf_q = pd.to_numeric(row_ocf.dropna(), errors="coerce") if row_ocf is not None else pd.Series(dtype=float)
        capex_q = pd.to_numeric(row_capex.dropna(), errors="coerce") if row_capex is not None else pd.Series(dtype=float)
        if len(ocf_q) and len(capex_q):
            # align by index, sum last 4 individually
            aligned = pd.concat([ocf_q, capex_q], axis=1).tail(8)  # last few for cv
            aligned.columns = ["ocf", "capex"]
            fcf_qs = (aligned["ocf"] + aligned["capex"]).tail(4)
            fcf_quarters = fcf_qs.tolist()
    fcf_margin = _safe_ratio(fcf_ttm, rev_ttm)
    fcf_cv = _cv(fcf_quarters) if fcf_quarters else None

    # Balance sheet
    row_debt = _pick_row(BS, ["Total Debt", "Short Long Term Debt", "Short Term Debt", "Long Term Debt"])
    row_equity = _pick_row(BS, ["Total Stockholder Equity", "Total Shareholder Equity", "Total Equity"])
    row_ca = _pick_row(BS, ["Total Current Assets"])
    row_cl = _pick_row(BS, ["Total Current Liabilities"])

    total_debt = _last_val(row_debt)
    equity = _last_val(row_equity)
    cur_assets = _last_val(row_ca)
    cur_liab = _last_val(row_cl)

    de_ratio = _safe_ratio(total_debt, equity)
    current_ratio = _safe_ratio(cur_assets, cur_liab)

    # Info ratios / estimates
    # Note: keys differ slightly between get_info() and the INFO sheet; we handle common variants.
    def _get(info: Dict[str, Any], *keys, cast=float) -> Optional[float]:
        for k in keys:
            if k in info and info[k] is not None and info[k] == info[k]:
                try:
                    return cast(info[k])
                except Exception:
                    continue
        return None

    pe = _get(INFO, "trailingPE", "trailingPe", "trailing_pe")
    pb = _get(INFO, "priceToBook", "price_to_book")
    ev_ebitda = _get(INFO, "enterpriseToEbitda", "enterpriseToEBITDA", "evToEbitda", "ev_to_ebitda")
    beta = _get(INFO, "beta")
    dividend_yield = _get(INFO, "dividendYield", "trailingAnnualDividendYield")
    payout_ratio = _get(INFO, "payoutRatio", "payout_ratio")
    target_low = _get(INFO, "targetLowPrice")
    target_mean = _get(INFO, "targetMeanPrice")
    target_high = _get(INFO, "targetHighPrice")

    # Earnings event proximity (Excel export may not include)
    next_earnings_in_days = None
    try:
        if CAL is not None and not CAL.empty:
            # yfinance calendar uses rows as fields; try parse 'Earnings Date'
            if "Earnings Date" in CAL.index:
                dt = pd.to_datetime(CAL.loc["Earnings Date"].dropna().iloc[0])
                next_earnings_in_days = (dt.date() - pd.Timestamp.today().date()).days
    except Exception:
        pass

    # As-of dates
    as_of_is = _latest_col_date(QIS if used["income"] == "quarterly" else (AIS if used["income"] == "annual" else None))
    as_of_cf = _latest_col_date(QCF if used["cashflow"] == "quarterly" else (ACF if used["cashflow"] == "annual" else None))
    as_of_bs = _latest_col_date(QBS if used["balance_sheet"] == "quarterly" else (ABS if used["balance_sheet"] == "annual" else None))

    # ------ Scoring (−1..+1 per factor) ------
    cfg = FA_CONFIG
    reasons: List[str] = []

    # G: Growth
    G = 0.0
    if rev_yoy is not None:
        if rev_yoy >= cfg["growth"]["rev_yoy_pos"]: G += 0.5; reasons.append(f"Revenue YoY strong ({rev_yoy:.1%}).")
        elif rev_yoy <= cfg["growth"]["rev_yoy_neg"]: G -= 0.4; reasons.append(f"Revenue YoY weak ({rev_yoy:.1%}).")
    if ni_yoy is not None:
        if ni_yoy >= cfg["growth"]["ni_yoy_pos"]: G += 0.5; reasons.append(f"Net income YoY strong ({ni_yoy:.1%}).")
        elif ni_yoy <= cfg["growth"]["ni_yoy_neg"]: G -= 0.4; reasons.append(f"Net income YoY weak ({ni_yoy:.1%}).")
    if op_margin_delta_pp is not None:
        if op_margin_delta_pp >= cfg["growth"]["op_margin_delta_pos_pp"]: G += 0.2; reasons.append("Operating margin expanding.")
        elif op_margin_delta_pp <= cfg["growth"]["op_margin_delta_neg_pp"]: G -= 0.2; reasons.append("Operating margin contracting.")
    G = _clip(G, -1.0, 1.0)

    # Q: Quality
    Q = 0.0
    if op_margin_ttm is not None:
        if op_margin_ttm >= cfg["quality"]["op_margin_good"]: Q += 0.5
        elif op_margin_ttm <= cfg["quality"]["op_margin_poor"]: Q -= 0.4
    if fcf_margin is not None:
        if fcf_margin >= cfg["quality"]["fcf_margin_good"]: Q += 0.5
        elif fcf_margin <= cfg["quality"]["fcf_margin_poor"]: Q -= 0.3
    Q = _clip(Q, -1.0, 1.0)

    # B: Balance sheet
    B = 0.0
    if de_ratio is not None:
        if de_ratio >= cfg["balance"]["de_high"]: B += 0.6; reasons.append("High leverage (D/E elevated).")
        elif de_ratio <= cfg["balance"]["de_ok"]: B -= 0.2
    if current_ratio is not None:
        if current_ratio < cfg["balance"]["current_ratio_low"]: B += 0.3; reasons.append("Low current ratio.")
        elif current_ratio >= cfg["balance"]["current_ratio_good"]: B -= 0.1
    B = _clip(B, -1.0, 1.0)

    # C: Cashflow stability
    C = 0.0
    if fcf_cv is not None:
        if fcf_cv <= cfg["cashflow"]["fcf_stable_cv"]: C += 0.3
        else: C -= 0.2; reasons.append("FCF volatility elevated.")
    if (fcf_ttm is not None) and (fcf_ttm < 0):
        C -= 0.3; reasons.append("FCF negative.")
    C = _clip(C, -1.0, 1.0)

    # V: Valuation (penalize extremes; mild reward for reasonable)
    V = 0.0
    def _val_part(x, lo, hi):
        if x is None: return 0.0
        if x >= hi: return -0.6
        if x <= lo: return +0.3
        return 0.0
    V += _val_part(pe, cfg["valuation"]["pe_lo"], cfg["valuation"]["pe_hi"])
    V += _val_part(pb, cfg["valuation"]["pb_lo"], cfg["valuation"]["pb_hi"])
    V += _val_part(ev_ebitda, cfg["valuation"]["ev_ebitda_lo"], cfg["valuation"]["ev_ebitda_hi"])
    V = _clip(V, -1.0, 1.0)
    if V < -0.2: reasons.append("Valuation appears elevated vs simple thresholds.")
    elif V > 0.2: reasons.append("Valuation appears reasonable/low vs simple thresholds.")

    # D: Dividend support
    D = 0.0
    if dividend_yield is not None:
        if dividend_yield >= cfg["dividend"]["yield_defensive"]: D += 0.2
    if payout_ratio is not None:
        if payout_ratio > cfg["dividend"]["payout_high"]: D -= 0.2; reasons.append("Payout ratio high.")
    D = _clip(D, -1.0, 1.0)

    # E: Event/Estimates (if near earnings, we won’t tilt much but widen bandwidth; use targets loosely)
    E = 0.0
    if target_mean is not None and target_low is not None and target_high is not None:
        # If mean target is above current price we can't know here; leave narrative only
        pass
    if next_earnings_in_days is not None and next_earnings_in_days <= cfg["event"]["earnings_horizon_days"]:
        # Avoid directional bias; we’ll encode this mostly into bandwidth
        reasons.append(f"Earnings within {next_earnings_in_days} days.")
        # tiny nudge to neutrality:
        E += 0.0
    E = _clip(E, -1.0, 1.0)

    # R: Risk/Beta (for bandwidth only)
    R = 0.0
    if beta is not None:
        if beta >= cfg["risk"]["beta_hi"]: R += 0.6
        elif beta <= cfg["risk"]["beta_lo"]: R -= 0.2
    R = _clip(R, -1.0, 1.0)

    # ------ Compose Tilt & Bandwidth ------
    wT = cfg["weights_tilt"];   wB = cfg["weights_bw"]
    fa_tilt = (wT["G"]*G + wT["Q"]*Q + wT["C"]*C + wT["V"]*V + wT["E"]*E + wT["D"]*D)
    fa_tilt = float(_clip(fa_tilt, -cfg["tilt_clip"], cfg["tilt_clip"]))

    # For bandwidth we take positive magnitude; map to % range adjustment
    comp_bw = (wB["B"]*max(B, 0) + wB["V"]*abs(V) + wB["E"]*abs(E) + wB["R"]*max(R, 0) + wB["C"]*max(-C, 0))
    fa_bw = float(_clip(cfg["bw_scale"] * comp_bw, cfg["bw_min"], cfg["bw_max"]))

    # ------ Scenarios (for the range combiner to plug in) ------
    scenarios = {
        "bear": {"tilt": fa_tilt - 0.5*abs(fa_tilt), "bandwidth": min(cfg["bw_max"], fa_bw * 1.25)},
        "base": {"tilt": fa_tilt,                    "bandwidth": fa_bw},
        "bull": {"tilt": fa_tilt + 0.5*abs(fa_tilt), "bandwidth": max(cfg["bw_min"], fa_bw * 0.85)},
    }

    # ------ Narrative (optional LLM) ------
    explain = None
    if use_llm and _LLM_OK:
        prompt = [
            {"role":"system","content":"You are a careful equity analyst. Respond in STRICT JSON with keys: narrative, tags[]. Keep it concise and numeric-grounded."},
            {"role":"user","content": json.dumps({
                "symbol": sym,
                "as_of": {"IS": as_of_is, "CF": as_of_cf, "BS": as_of_bs},
                "metrics": {
                    "rev_ttm": rev_ttm, "rev_yoy": rev_yoy,
                    "ni_ttm": ni_ttm, "ni_yoy": ni_yoy,
                    "op_margin_ttm": op_margin_ttm, "op_margin_delta_pp": op_margin_delta_pp,
                    "ocf_ttm": ocf_ttm, "capex_ttm": capex_ttm, "fcf_ttm": fcf_ttm,
                    "fcf_margin": fcf_margin, "fcf_cv": fcf_cv,
                    "de_ratio": de_ratio, "current_ratio": current_ratio,
                    "pe": pe, "pb": pb, "ev_ebitda": ev_ebitda,
                    "beta": beta, "dividend_yield": dividend_yield, "payout_ratio": payout_ratio,
                    "next_earnings_in_days": next_earnings_in_days
                },
                "scores": {"G": G, "Q": Q, "B": B, "C": C, "V": V, "D": D, "E": E, "R": R},
                "fa_adjustments": {"tilt": fa_tilt, "bandwidth": fa_bw}
            }, default=str)}
        ]
        try:
            js = llm_json(prompt, model="gpt-4o-mini", temperature=0.2)
            parsed = json.loads(js)
            explain = {
                "narrative": parsed.get("narrative"),
                "tags": parsed.get("tags", []),
                "llm_model": "gpt-4o-mini"
            }
        except Exception:
            explain = None

    if explain is None:
        # Deterministic fallback
        bullets = []
        if rev_yoy is not None: bullets.append(f"Revenue YoY: {rev_yoy:.1%}")
        if ni_yoy is not None: bullets.append(f"Net Income YoY: {ni_yoy:.1%}")
        if op_margin_ttm is not None: bullets.append(f"Op Margin: {op_margin_ttm:.1%} ({'+' if (op_margin_delta_pp or 0)>=0 else ''}{(op_margin_delta_pp or 0):.2f} pp YoY)")
        if fcf_margin is not None: bullets.append(f"FCF Margin: {fcf_margin:.1%}")
        if de_ratio is not None: bullets.append(f"D/E: {de_ratio:.2f}")
        if current_ratio is not None: bullets.append(f"Current Ratio: {current_ratio:.2f}")
        if pe is not None: bullets.append(f"P/E: {pe:.1f}")
        if pb is not None: bullets.append(f"P/B: {pb:.1f}")
        if ev_ebitda is not None: bullets.append(f"EV/EBITDA: {ev_ebitda:.1f}")
        if beta is not None: bullets.append(f"Beta: {beta:.2f}")
        if dividend_yield is not None: bullets.append(f"Dividend Yield: {dividend_yield:.1%}")
        if payout_ratio is not None: bullets.append(f"Payout Ratio: {payout_ratio:.1%}")
        if next_earnings_in_days is not None: bullets.append(f"Earnings in ~{int(next_earnings_in_days)} days")

        explain = {
            "narrative": " | ".join(bullets) if bullets else "Fundamental snapshot computed. LLM summary disabled.",
            "tags": [],
            "llm_model": None
        }

    fa_meta: Dict[str, Any] = {
        "symbol": sym,
        "as_of": {"income_statement": as_of_is, "cashflow": as_of_cf, "balance_sheet": as_of_bs},
        "sources": used,
        "metrics_ttm": {
            "revenue": rev_ttm, "operating_income": oi_ttm, "net_income": ni_ttm,
            "op_margin": op_margin_ttm, "op_margin_delta_pp": op_margin_delta_pp,
            "ocf": ocf_ttm, "capex": capex_ttm, "fcf": fcf_ttm, "fcf_margin": fcf_margin, "fcf_cv": fcf_cv
        },
        "balance_sheet": {
            "total_debt_latest": total_debt, "equity_latest": equity,
            "debt_to_equity": de_ratio, "current_assets_latest": cur_assets,
            "current_liabilities_latest": cur_liab, "current_ratio": current_ratio
        },
        "info": {
            "pe": pe, "pb": pb, "ev_ebitda": ev_ebitda, "beta": beta,
            "dividend_yield": dividend_yield, "payout_ratio": payout_ratio,
            "target_low": target_low, "target_mean": target_mean, "target_high": target_high,
            "next_earnings_in_days": next_earnings_in_days
        },
        "scores": {"G": G, "Q": Q, "B": B, "C": C, "V": V, "D": D, "E": E, "R": R},
        "fa_adjustments": {"tilt": fa_tilt, "bandwidth": fa_bw, "rationales": reasons},
        "scenarios": scenarios,
        "explain": explain,
        "config": FA_CONFIG
    }
    return fa_meta
