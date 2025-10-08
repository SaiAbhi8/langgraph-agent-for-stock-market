# run_agent_1m.py
# deps: pip install langgraph openai pydantic
import os, re, json
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

# === import your existing nodes ===
from nodes.price_node import price_node            # adds prices_df, prices_meta  :contentReference[oaicite:6]{index=6}
from nodes.trend_node import trend_node            # adds trend_meta              :contentReference[oaicite:7]{index=7}
from nodes.volatility_node import volatility_node  # adds vol_meta                :contentReference[oaicite:8]{index=8}
from nodes.context_node import context_node        # adds context_meta            :contentReference[oaicite:9]{index=9}
from nodes.fundamentals_node import fundamentals_node  # adds fund_meta          :contentReference[oaicite:10]{index=10}
from nodes.events_node import events_node          # adds event_meta              :contentReference[oaicite:11]{index=11}

# --- minimal OpenAI chat wrapper (expects OPENAI_API_KEY) ---
from openai import OpenAI
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== Utilities ======
TICKER_REGEX = re.compile(r"\b([A-Z]{2,12})(?:\.NS)?\b")

def extract_ticker(user_text: str, fallback: str = "RELIANCE") -> str:
    """
    Very simple NSE ticker extractor. If input has .NS we keep it; else we append .NS later.
    """
    # prioritize uppercase tokens user likely typed (RELIANCE, TCS, INFY, etc.)
    m = TICKER_REGEX.search(user_text.upper())
    if m:
        return m.group(1)
    return fallback

def normalize_nse_ticker(t: str) -> str:
    t = t.strip().upper()
    return t if t.endswith(".NS") else f"{t}.NS"

# ====== Agent State ======
class AgentState(TypedDict, total=False):
    # input
    user_text: str
    ticker: str
    period: str
    interval: str
    # node outputs
    prices_df: Any
    prices_meta: Dict[str, Any]
    trend_meta: Dict[str, Any]
    vol_meta: Dict[str, Any]
    context_meta: Dict[str, Any]
    fund_meta: Dict[str, Any]
    event_meta: Dict[str, Any]
    # final
    forecast_1m: Dict[str, Any]
    reasoning_1m: str

# ====== Reasoning Node (LLM combines all signals) ======
def reasoning_1m_node(state: AgentState) -> Dict[str, Any]:
    """
    Consumes *_meta dicts to produce a 1-month price range and a concise explanation.
    Expected inputs present in state: prices_meta, trend_meta, vol_meta, context_meta, fund_meta, event_meta
    """
    pm   = state.get("prices_meta", {}) or {}
    tr   = state.get("trend_meta", {}) or {}
    vol  = state.get("vol_meta", {}) or {}
    ctx  = state.get("context_meta", {}) or {}
    fun  = state.get("fund_meta", {}) or {}
    ev   = state.get("event_meta", {}) or {}

    # pull a few helpful fields safely
    latest = pm.get("prices", {}).get("latest_close") if isinstance(pm.get("prices"), dict) else pm.get("latest_close")
    s52h   = (pm.get("prices", {}) or {}).get("52w_high")
    s52l   = (pm.get("prices", {}) or {}).get("52w_low")

    # hint for LLM: provide compact JSON context only
    context_for_llm = {
        "ticker": state.get("ticker"),
        "latest_close": latest,
        "meta_52w": {"high": s52h, "low": s52l},
        "trend_meta": tr,
        "vol_meta": vol,
        "context_meta": ctx,
        "fund_meta": fun,
        "event_meta": ev,
    }
    print(context_for_llm)
    system_prompt = (
        "You are a disciplined equity strategist for Indian markets. "
        "Using only the provided JSON facts, produce a 1-month price RANGE and a short explanation. "
        "Respect support/resistance if provided (if not, rely on volatility). "
        "Be measured; avoid sensational claims. Output STRICT JSON."
    )

    user_prompt = f"""
JSON_INPUT:
{json.dumps(context_for_llm, ensure_ascii=False)}

Return JSON with this schema:
{{
  "range": {{"low": number, "high": number, "confidence": 0.0_to_1.0}},
  "bias": "bullish|bearish|neutral",
  "rationale": "10-15 lines combining trend, volatility, fundamentals, context, and events. Avoid fluff."
}}
"""

    resp = _openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    js = resp.choices[0].message.content
    try:
        data = json.loads(js)
    except Exception:
        data = {
            "range": {"low": None, "high": None, "confidence": 0.0},
            "bias": "neutral",
            "rationale": "Model could not produce structured output."
        }
    return {"forecast_1m": data.get("range"), "reasoning_1m": data.get("rationale")}

# ====== Graph assembly ======
def build_graph():
    g = StateGraph(AgentState)

    # Nodes in sequence: price -> (trend, vol) ; plus side nodes (context, fundamentals, events)
    g.add_node("price", price_node)               # uses ticker/period/interval; sets prices_df, prices_meta
    g.add_node("trend", trend_node)               # reads prices_df if present
    g.add_node("vol", volatility_node)            # reads prices_df if present
    g.add_node("context", context_node)           # reads ticker/period/interval/benchmark/sector
    g.add_node("fundamentals", fundamentals_node) # reads ticker
    g.add_node("events", events_node)             # reads ticker (+ optionally prices_meta)
    g.add_node("reasoning_1m", reasoning_1m_node) # combines all meta dicts

    # Entry
    g.set_entry_point("price")
    # Fan-out after price
    g.add_edge("price", "trend")
    g.add_edge("price", "vol")
    # Parallel side nodes
    g.add_edge("price", "context")
    g.add_edge("price", "fundamentals")
    g.add_edge("price", "events")
    # Gather to reasoning
    g.add_edge("trend", "reasoning_1m")
    g.add_edge("vol", "reasoning_1m")
    g.add_edge("context", "reasoning_1m")
    g.add_edge("fundamentals", "reasoning_1m")
    g.add_edge("events", "reasoning_1m")
    # End
    g.add_edge("reasoning_1m", END)

    return g.compile()

# ====== Public API ======
def run_1m_agent(user_text: str,
                 period: str = "1y",
                 interval: str = "1d",
                 benchmark: str = "^NSEI",
                 sector: Optional[str] = None) -> Dict[str, Any]:
    """
    user_text: free-form input like "What's the 1M range for Reliance?"
    Returns: dict with ticker, latest, forecast_1m {low, high, confidence}, and reasoning text.
    """
    raw_ticker = extract_ticker(user_text, fallback="RELIANCE")
    base_ticker = raw_ticker  # without .NS
    ticker = normalize_nse_ticker(raw_ticker)     # ensure NSE symbol

    graph = build_graph()
    init: AgentState = {
        "user_text": user_text,
        "ticker": base_ticker,         # nodes that append .NS already handle this; price loader typically normalizes
        "period": period,
        "interval": interval,
        "benchmark": benchmark,
        "sector": sector or None,
    }
    final_state = graph.invoke(init)

    # Try to surface latest price cleanly
    pm = final_state.get("prices_meta", {}) or {}
    latest = (pm.get("prices", {}) or {}).get("latest_close") if isinstance(pm.get("prices"), dict) else pm.get("latest_close")

    return {
        "input": user_text,
        "ticker": normalize_nse_ticker(base_ticker),
        "latest_close": latest,
        "forecast_1m": final_state.get("forecast_1m"),
        "reasoning": final_state.get("reasoning_1m"),
        "debug": {
            "trend_meta": final_state.get("trend_meta"),
            "vol_meta": final_state.get("vol_meta"),
            "context_meta": final_state.get("context_meta"),
            "fund_meta": final_state.get("fund_meta"),
            "event_meta": final_state.get("event_meta"),
        }
    }

def main():
    # Ask the user for a ticker interactively
    user_text = input("Enter stock ticker or query (e.g., RELIANCE, TCS, INFY): ").strip()
    if not user_text:
        user_text = "RELIANCE"

    # Run the agent
    result = run_1m_agent(user_text=user_text)

    # Pretty print the output
    print("\n=== 1-Month Forecast Result ===")
    print(f"Ticker        : {result['ticker']}")
    print(f"Latest Close  : {result['latest_close']}")
    if result["forecast_1m"]:
        rng = result["forecast_1m"]
        print(f"Forecast Range: {rng.get('low')} – {rng.get('high')} (conf {rng.get('confidence')})")
    print("\nReasoning:\n", result["reasoning"])

    # Optional: dump the debug meta if you want
    # print("\n[DEBUG]", json.dumps(result["debug"], indent=2))

# If running directly in Spyder, just call main()
if __name__ == "__main__":
    main()