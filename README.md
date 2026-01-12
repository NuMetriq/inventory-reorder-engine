# Inventory Reorder Recommendation Engine  
**Probabilistic Forecasting + Inventory Policy Simulation (M5 / WI)**

This project implements a **decision-focused inventory reorder engine** that combines
probabilistic demand forecasting with an order-up-to inventory policy.  
Using the M5 Forecasting dataset (Walmart), the system forecasts demand uncertainty and
simulates reorder decisions under varying lead times.

> **Core idea:** forecasts are only valuable when they drive better decisions.  
> This project evaluates inventory performance, not just forecast accuracy.

---

## Decision Context

Retailers must decide **how much inventory to reorder** under:
- uncertain demand
- nonzero supplier lead times
- service-level expectations

Ordering too little causes stockouts and lost sales.  
Ordering too much ties up capital and increases holding costs.

This project answers:

> *How can probabilistic forecasts be translated into reorder policies that improve service levels while controlling inventory risk?*

---

## Data

- **Dataset:** M5 Forecasting (Walmart)
- **Scope:** Wisconsin (WI) only
- **Granularity:** Weekly demand
- **Scale:** ~9,100 SKU–store series, ~2.5M rows

Due to Kaggle licensing, raw data files are not included in the repository.
All results are fully reproducible using the provided scripts.

---

## Modeling Approach

### Forecasting
- **Global LightGBM model** trained across all SKUs and stores
- Quantile regression:
  - **P50** → expected demand
  - **P90** → demand uncertainty / safety stock
- Leakage-safe feature engineering:
  - demand lags and rolling statistics
  - intermittency indicators
  - calendar effects
  - price dynamics

### Baseline
- Trailing moving-average forecast
- Order-up-to policy derived from recent demand

---

## Inventory Policy

Weekly review, lost-sales assumption.

### ML Policy
Order-up-to level:
S = μ · (L + 1) + (P90 − P50) · √(L + 1)
Where:
- μ = P50 forecast
- L = supplier lead time (weeks)
- P90 − P50 approximates demand uncertainty

This formulation is transparent, interpretable, and tunable.

---

## Backtest Methodology

- Event-driven simulation over historical demand
- Compare **Baseline vs ML policy**
- Metrics tracked:
  - fill rate
  - lost sales
  - average inventory on hand
  - total units ordered
- Evaluated under multiple lead-time scenarios

---

## Results

### Lead Time = 2 Weeks (All Series)

| Metric | Baseline | ML Policy |
|------|---------:|----------:|
| Fill rate | 92.1% | **98.0%** |
| Lost sales | 50,850 | **12,873** |
| Avg inventory | 12.3 | 20.3 |
| Total units ordered | 613,580 | **608,189** |

**Key outcomes**
- ~**75% reduction in lost sales**
- **Higher service levels with slightly less total ordering**
- Intentional increase in on-hand inventory to achieve service targets

---

### Sensitivity to Lead Time

| Lead Time | Baseline Fill | ML Fill |
|---------:|--------------:|--------:|
| 1 week | 95.0% | **99.4%** |
| 2 weeks | 92.1% | **98.0%** |
| 4 weeks | 89.9% | **95.8%** |

The ML policy becomes **increasingly valuable as lead times grow**, where naive heuristics degrade rapidly.

---

## Repo Structure

inventory-reorder-engine/
├─ scripts/ # Build, train, backtest entry points
├─ src/ire/ # Feature, model, policy, simulation code
├─ data/ # Local data (ignored)
├─ outputs/ # Generated results (ignored)
└─ README.md

---

## Reproducibility

1. Download M5 data from Kaggle
2. Build dataset:
   ```bash
   python scripts/build_dataset.py --state WI
3. Train models:
   python scripts/train.py --alpha 0.5
   python scripts/train.py --alpha 0.9
4. Run inventory backtest:
   python scripts/backtest.py --lead-time 2

---

## Limitations

- Inventory levels and lead times are simulated (no public dataset provides full replenishment data)
- Lost-sales assumption (no backorders)
- Single-state scope for iteration speed

These assumptions are explicit and tested via sensitivity analysis.

---

## Future Work

- Service-level tuning (P80 / P90 / P95)
- Cash-constrained ordering across SKUs
- Hierarchical reconciliation
- Streamlit decision dashboard
- Online monitoring and retraining

---

## Takeaway

This project demonstrates how **probabilistic ML models can be operationalized into inventory decisions**, producing measurable improvements in service levels under real-world uncertainty.
