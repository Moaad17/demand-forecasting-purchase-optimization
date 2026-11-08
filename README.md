# Supply Chain Optimizer

**Demand forecasting and ordering decisions under uncertainty.**

> How much of each product should we order next month when we don't know demand in advance?

Order too much and cash is tied up in stock. Order too little and sales are lost. This project turns a demand forecast, with its uncertainty, into an order quantity that minimizes expected cost under budget and storage constraints.

---

<!-- TODO: add a screenshot or GIF of the Streamlit app here: ![App demo](docs/demo.gif) -->

---

## How it works

```text
Sales history (CSV)
        │
        ▼
1. Forecasting      Prophet / XGBoost compared by backtest,
                    best model kept per product
        │           → central forecast + lower/upper bounds
        ▼
2. Optimization     Demand scenarios (5% / 50% / 95% quantiles)
                    → two-stage stochastic program
        │           minimizing expected total cost
        ▼
Order quantity per product + expected cost
```

The key difference from "forecast, then round": the optimizer does not use a single demand number. It reasons over several demand scenarios built from the forecast uncertainty and chooses the order that minimizes expected cost:

```text
Total cost = purchase cost + expected stockout cost + expected overstock cost
```

subject to a **budget** and a **storage capacity** constraint. The problem is solved with an **L-shaped (Benders) decomposition** (`app/optimization/lshaped.py`).

---

## Pipeline

### 1. Data (`app/data/`)
Input CSV columns: `date`, `product`, `quantity`, `unit_price` (optional). The data is validated (required columns, types, missing values, duplicates, minimum history), cleaned, aggregated by month, completed where months are missing, and enriched with time features.

### 2. Forecasting (`app/forecasting/`)
For each product, two models are trained and compared:

- **Prophet:** trend and seasonality.
- **XGBoost:** features from history (previous month, rolling means, trend, month of year, price when available).

Models are compared by backtesting on the last months of history, and the one with the lowest **MAE** is selected per product. The output is a central forecast plus a lower and an upper bound.

### 3. Optimization (`app/optimization/`)
One-month horizon. The 5%, 50% and 95% forecast quantiles are used as demand scenarios. The solver chooses order quantities that minimize expected cost under the budget and capacity constraints.

### 4. Interface (`frontend/app.py`)
A Streamlit app to run the pipeline interactively. You can change costs, budget, storage capacity and forecast horizon, and see forecasts, order quantities, demand scenarios, the strategy comparison, EVPI and VSS.

---

## Run it

```bash
pip install -r requirements.txt

# Interactive app
streamlit run frontend/app.py

# Command-line demo
cd app
python main.py
```

### Tests

```bash
pytest -v
```

The project has **44 tests** covering data loading, validation and cleaning, forecasting (including interval consistency and model-failure handling), optimization constraints and solution consistency, and a regression check of the L-shaped solver against the direct computation.

---

## Project structure

```text
Supply Chain Optimizer/
├── app/
│   ├── data/             # loading, validation, cleaning
│   ├── forecasting/      # Prophet, XGBoost, backtest selection
│   ├── optimization/     # stochastic model, lshaped.py
│   └── main.py
├── frontend/
│   └── app.py            # Streamlit interface
├── examples/
│   ├── sample_beverages.csv      # main demo: 5 products, 36 months
│   ├── sample_retail.csv         # multi-product test: 8 products, 24 months
│   └── sample_port_traffic.csv   # wrong schema on purpose, tests the validator
├── tests/
├── requirements.txt
└── README.md
```

---

## Limitations

- Only three demand scenarios per product (5% / 50% / 95% quantiles), which is a coarse approximation of the demand distribution.
- One-month planning horizon.
- Example datasets are for demonstration: [TODO: say whether they are synthetic or real, and if synthetic, say so here].
- Forecast intervals have not been calibrated against coverage yet [TODO: remove this line if you have checked it].

## Roadmap

- [ ] HTTP API (`app/API/`)
- [ ] `app/analysis/metrics.py`
- [ ] Dockerfile and `docker-compose.yaml`
- [ ] Deployment of the app
- [ ] More scenarios and multi-month horizon

---

## Tech stack

Python, Pandas, NumPy, Prophet, XGBoost, scikit-learn, Streamlit, pytest, Git/GitHub. Optimization: L-shaped (Benders) decomposition for two-stage stochastic programming [TODO: add OR-Tools here only if the code actually uses it].
