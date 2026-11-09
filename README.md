
# Supply Chain Optimizer
### Demand Forecasting & Stochastic Purchase Optimization

[![CI](https://github.com/Moaad17/demand-forecasting-purchase-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/Moaad17/demand-forecasting-purchase-optimization/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-supported-2496ED?logo=docker)
![Tests](https://img.shields.io/badge/Tests-47%20passing-success)

**Supply Chain Optimizer** is a decision-support application that combines demand forecasting and stochastic optimization to determine purchasing quantities under demand uncertainty, budget limitations, and storage capacity constraints.

The objective is to move beyond forecasting demand alone and translate predictive insights into actionable purchasing decisions.

---

## 1. Business Problem

How much of each product should a company purchase for the next planning period when future demand is uncertain?

Purchasing decisions involve a fundamental trade-off:

- **Overstocking:** Excess inventory increases holding costs and ties up working capital.
- **Understocking:** Insufficient inventory leads to stockouts, lost sales, and potential customer dissatisfaction.

Traditional purchasing approaches often rely on a single demand forecast or fixed safety-stock rules.

This project takes a different approach: it incorporates forecast uncertainty directly into the optimization model to identify purchasing decisions that minimize expected total cost.

## 2. Solution Overview

The application implements an end-to-end decision pipeline:

```text
                Historical Sales Data
                         |
                         v
                  Data Processing
                         |
                         v
                 Demand Forecasting
                   Prophet / XGBoost
                         |
                         v
                Model Selection
                  Backtesting / MAE
                         |
                         v
                Demand Scenarios
                  5% / 50% / 95%
                         |
                         v
             Stochastic Optimization
                L-shaped Decomposition
                         |
                         v
               Optimal Order Quantities
                         |
                         v
                 Decision Dashboard
```

The main idea is to connect predictive analytics with mathematical optimization.

Instead of optimizing against one predicted demand value, the model considers multiple possible demand scenarios and their associated costs.

## 3. Methodology

### 3.1 Data Processing

Historical sales data is processed before entering the forecasting pipeline.

The preprocessing stage includes:

- Data validation and type checking.
- Missing-value and duplicate handling.
- Monthly aggregation.
- Completion of missing monthly periods.
- Temporal feature engineering.
- Optional integration of product-level unit prices.

**Expected input columns:**

| Column | Description | Required |
|---|---|---|
| `date` | Sales transaction date | Yes |
| `product` | Product identifier | Yes |
| `quantity` | Quantity sold | Yes |
| `unit_price` | Unit purchasing price | No |

### 3.2 Demand Forecasting

Two forecasting approaches are implemented and compared independently for each product.

**Prophet**
- Captures trend and seasonal patterns.
- Provides forecast estimates and uncertainty intervals.

**XGBoost**
- Uses historical demand and engineered temporal features.
- Incorporates lagged demand, rolling statistics, trends, and month information.
- Uses unit price when available.

**Model selection**

Models are evaluated through time-based backtesting using the Mean Absolute Error (MAE).

For each product, the model with the lowest backtesting MAE is selected.

The forecasting stage produces:

- Central demand forecast.
- Lower prediction bound.
- Upper prediction bound.

### 3.3 Demand Uncertainty Modeling

Forecast uncertainty is translated into three demand scenarios:

| Scenario | Quantile | Interpretation |
|---|---:|---|
| Low demand | 5% | Lower-demand scenario |
| Median demand | 50% | Central scenario |
| High demand | 95% | Higher-demand scenario |

These scenarios allow the optimization model to account for demand variability rather than relying exclusively on the central forecast.

### 3.4 Stochastic Purchase Optimization

The purchasing problem is formulated as a two-stage stochastic optimization model.

The objective is to minimize expected total cost:

**Expected Total Cost = Purchase Cost + Expected Overstock Cost + Expected Stockout Cost**

The model considers:

- Product purchasing costs.
- Overstocking penalties.
- Stockout penalties.
- Available purchasing budget.
- Storage capacity.
- Integer purchasing quantities.

The optimization problem is solved using an **L-shaped decomposition (Benders decomposition)**, separating first-stage purchasing decisions from scenario-dependent recourse decisions.

The implementation also includes a direct deterministic-equivalent solver for smaller instances.

### 3.5 Decision Evaluation

The application includes comparisons between purchasing strategies and decision-quality indicators.

Implemented indicators include:

- Expected Value of Perfect Information (EVPI).
- Value of the Stochastic Solution (VSS).
- Expected cost comparison across purchasing strategies.

These indicators help evaluate the value of incorporating uncertainty into purchasing decisions.

---

## 4. Application

A Streamlit interface allows users to interact with the complete pipeline.

The user can:

- Upload or select historical sales data.
- Configure purchasing and inventory parameters.
- Set budget and storage capacity constraints.
- Generate demand forecasts.
- Inspect demand scenarios.
- Obtain recommended purchasing quantities.
- Compare purchasing strategies.
- Analyze EVPI and VSS.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Forecasting | Prophet, XGBoost |
| Machine Learning | Scikit-learn |
| Mathematical Optimization | PuLP, CBC |
| Optimization Method | Two-stage stochastic programming, L-shaped decomposition |
| Visualization & Interface | Streamlit |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| Continuous Integration | GitHub Actions |
| Version Control | Git, GitHub |

---

## 6. Installation & Execution

### Prerequisites

- Docker and Docker Compose
- Git

### Option A — Run with Docker (Recommended)

Clone the repository:

```bash
git clone https://github.com/Moaad17/demand-forecasting-purchase-optimization.git

cd demand-forecasting-purchase-optimization
```

Build the Docker image and start the application:

```bash
docker compose up --build
```

Open the application in your browser:

```text
http://localhost:8501
```

### Option B — Run Locally with Python

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run frontend/app.py
```

---

## 7. Testing & Continuous Integration

The project includes an automated test suite covering the main components of the pipeline.

**Current test coverage includes:**

- Data loading and validation.
- Data preprocessing.
- Forecasting outputs and prediction intervals.
- Forecasting model selection.
- Optimization constraints.
- Purchasing strategy evaluation.
- L-shaped decomposition consistency against the direct optimization approach.

Run the tests locally:

```bash
pytest -v tests/
```

**Continuous Integration**

GitHub Actions automatically:

1. Checks out the repository.
2. Builds the Docker image.
3. Runs the Pytest suite inside the Docker container.

The workflow is triggered by pushes to the `main` branch and pull requests targeting `main`.

---

## 8. Project Structure

```text
demand-forecasting-purchase-optimization/
│
├── app/
│   ├── data/                  # Data loading and preprocessing
│   ├── forecasting/           # Prophet, XGBoost and model selection
│   ├── optimization/          # Stochastic optimization
│   │   └── lshaped.py         # L-shaped decomposition
│   └── main.py                # Main pipeline
│
├── frontend/
│   └── app.py                 # Streamlit application
│
├── examples/                  # Example datasets
│
├── tests/                     # Automated test suite
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow
│
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 9. Current Scope & Limitations

The current implementation has the following scope:

- Single-month purchasing horizon.
- Three demand scenarios per product (5th, 50th and 95th percentiles).
- Forecasting based on historical sales data.
- Purchasing decisions subject to budget and storage constraints.

These assumptions provide a manageable framework for studying the interaction between demand uncertainty and purchasing decisions.

## 10. Future Improvements

Potential directions for further development:

- Multi-period inventory optimization.
- More demand scenarios and improved uncertainty calibration.
- Inventory replenishment and safety-stock policies.
- Supplier lead times and ordering constraints.
- Additional forecasting models and probabilistic forecasting methods.
- API integration and production deployment.
- Business performance evaluation using realistic operational datasets.

---

## 11. Project Objective

This project explores the intersection of **Data Science, Operations Research, and Business Decision-Making**.

Its central objective is to demonstrate how predictive models can be integrated with mathematical optimization to support operational decisions under uncertainty.

**Forecasting estimates what may happen. Optimization helps decide what to do about it.**
