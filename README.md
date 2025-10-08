# 🧠 LangGraph Stock Market Analysis Agent

This project implements a **LangGraph-based multi-node agent** that performs various forms of **stock market analysis** and integrates them through an **OpenAI LLM** to estimate a stock’s **price range one month into the future**, accompanied by an interpretable reasoning summary.

---

## ⚙️ Overview

The project follows a modular, **graph-based architecture**.
Each node represents a specialized analysis unit (technical, sentiment, fundamental, etc.), and together they update a shared **state dictionary** that represents the evolving understanding of a stock’s behavior.

The **OpenAI reasoning node** then synthesizes all this information and generates:

* A **predicted price range** for one month ahead
* A **detailed explanation** summarizing insights from all analytical perspectives

---

## 🧩 Folder Structure and Workflow

The repository is organized into **two key components** — `tools/` and `nodes/` — with standalone testing scripts in the root directory.

```
C:\Users\lenovo\Agent\
├─ nodes\
│  ├─ price_node.py
│  ├─ context_node.py
│  ├─ volatility_node.py
│  ├─ hist_bootstrap_node.py
│  ├─ trend_node.py
│  ├─ sr_node.py
│  ├─ events_node.py
│  ├─ fundamentals_node.py
│  ├─ analyst_node.py
│  └─ range_combiner_node.py
├─ tools\
│  ├─ price_loader.py
│  ├─ context_loader.py
│  ├─ volatility_range.py
│  ├─ hist_bootstrap.py
│  ├─ trend_gauge.py
│  ├─ sr_mapper.py
│  ├─ events_window.py
│  ├─ fundamentals_momentum.py
│  └─ analyst_snapshot.py
├─ run_context_node.py
├─ run_fundamentals_node.py
├─ run_hist_bootstrap_node.py
├─ run_price_node.py
├─ run_sr_node.py
├─ run_sr_plot.py
├─ run_trend_node.py
└─ run_volatility_node.py
```


### 💡 How the Layers Interact

1. **`tools/` layer** → Handles all raw data fetching, numerical computations, and intermediate analytics.
2. **`nodes/` layer** → Wraps these functionalities inside LangGraph-compatible nodes that update the shared state.
3. **Root scripts** → Allow quick, independent testing of each node or the entire graph pipeline.

---

## 🧰 Tech Stack

* **Language:** Python
* **Framework:** [LangGraph](https://github.com/langchain-ai/langgraph)
* **LLM:** OpenAI GPT model (`gpt-4o` / `gpt-4-turbo`)
* **Libraries:** `yfinance`, `pandas`, `numpy`, `ta`, `transformers`, `requests`, `matplotlib`

---

## 🚀 Execution Flow

1. User provides a stock symbol (e.g., `TCS.NS`).
2. Each node (Technical, Sentiment, Fundamental, Correlation) runs sequentially or in parallel to analyze different aspects.
3. Each node updates the **shared state** with its computed insights.
4. The **LLM reasoning node** integrates the state into a final analysis, predicting a **price range** and providing a **reasoned explanation**.

---

## 🧪 Example Usage

```bash
python main.py --symbol INFY.NS
```

**Example Output:**

```
Predicted Range (1 month): ₹1,640 – ₹1,710

Reasoning:
Based on bullish RSI and MACD crossovers, stable earnings, and positive sentiment trends,
the stock is expected to maintain an upward bias with moderate volatility.
```

---

## 🧠 Future Work

* Add **new data nodes** (macro indicators, volatility indices, etc.)
* Introduce **reinforcement-based feedback** for adaptive reasoning.
* Integrate **Streamlit dashboards** for visual summaries.

---

⭐ *Developed as part of ongoing research on AI-driven financial reasoning and autonomous market intelligence systems.*



