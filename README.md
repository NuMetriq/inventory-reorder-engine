\# Inventory Reorder Recommendation Engine (M5 / WI)



Probabilistic demand forecasting + reorder recommendations at scale using the M5 dataset (Walmart), scoped to Wisconsin (WI) for fast iteration.



\## Goals

\- Global forecasting model across many SKUs/stores

\- Quantile forecasts (uncertainty)

\- Reorder policy (order-up-to S) under lead-time + service-level assumptions

\- Backtesting simulation to compare against baselines



\## Repo layout

\- `scripts/` runnable entry points (build, train, backtest)

\- `src/ire/` package code (features, model, policy, simulation)

\- `notebooks/` EDA only

\- `data/` local data (ignored by git)



\## Status

WIP (v1): weekly aggregation, P50 model, baseline backtest

