"""
backtest.py

Inventory simulation backtest comparing:
  - Baseline order-up-to policy from trailing mean
  - ML order-up-to policy using LightGBM P50 + safety stock from (P90 - P50)

Inputs:
  data/processed/m5_wi_weekly_panel.parquet
  models/lgbm_p50.txt
  models/lgbm_p90.txt
  models/features_p50.json
  models/features_p90.json

Outputs:
  outputs/
    backtest_summary.csv
    backtest_metrics.json

Run (PowerShell):
  python scripts/backtest.py --lead-time 2 --service-z 1.28 --sample-series 3000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "m5_wi_weekly_panel.parquet"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class BacktestConfig:
    data_path: Path
    models_dir: Path
    out_dir: Path
    lead_time: int
    sample_series: int
    seed: int
    min_hist_weeks: int
    baseline_window: int
    init_inventory_mode: str  # "S" or "zero"
    clip_pred_nonneg: bool


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inventory reorder backtest simulation (baseline vs ML).")
    ap.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--lead-time", type=int, default=2, help="Lead time in weeks (orders arrive after L weeks).")

    ap.add_argument(
        "--sample-series",
        type=int,
        default=3000,
        help="Number of (store,item) series to simulate (0 = all). Use 2000-5000 for speed.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hist-weeks", type=int, default=24, help="Skip first N weeks in each series (warm-up).")
    ap.add_argument("--baseline-window", type=int, default=8, help="Trailing window for baseline order-up-to.")
    ap.add_argument(
        "--init-inventory-mode",
        choices=["S", "zero"],
        default="S",
        help="Initialize starting inventory to first week's order-up-to S, or 0.",
    )
    ap.add_argument("--no-clip-nonneg", action="store_true", help="Do not clip predictions at 0.")
    return ap.parse_args()


# -------------------------
# Metrics
# -------------------------
def service_level_fill_rate(total_demand: float, total_lost: float) -> float:
    if total_demand <= 0:
        return float("nan")
    return float(1.0 - (total_lost / total_demand))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


# -------------------------
# Feature engineering
# (must match train.py logic)
# -------------------------
def add_features(df: pd.DataFrame, lags=(1, 2, 4, 8), rolls=(4, 8)) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "wm_yr_wk"]).copy()
    g = df.groupby(["store_id", "item_id"], sort=False)

    for k in lags:
        df[f"y_lag_{k}"] = g["demand"].shift(k).astype("float32")

    for w in rolls:
        s = g["demand"].shift(1)
        df[f"y_roll_mean_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).mean().astype("float32")
        df[f"y_roll_std_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).std().astype("float32")

    s8 = g["demand"].shift(1)
    df["pct_zero_8"] = (
        s8.rolling(8, min_periods=4)
        .apply(lambda x: float(np.mean(np.array(x) == 0.0)), raw=False)
        .astype("float32")
    )

    # weeks since sale (shifted by 1 week to avoid using current-week sale indicator)
    demand_now = df["demand"].astype("float32")
    df["weeks_since_sale"] = 0.0
    for (_, _), idx in g.indices.items():
        arr = demand_now.iloc[idx].to_numpy()
        out = np.zeros_like(arr, dtype=np.float32)
        last_sale = -1_000_000
        for i, v in enumerate(arr):
            if v > 0:
                last_sale = i
                out[i] = 0.0
            else:
                out[i] = float(i - last_sale) if last_sale >= 0 else float(i + 1)
        out = np.roll(out, 1)
        out[0] = np.nan
        df.loc[df.index[idx], "weeks_since_sale"] = out

    df["price_missing"] = df["sell_price"].isna().astype("int8")
    df["sell_price_ffill"] = g["sell_price"].ffill().astype("float32")
    df["price_chg_1"] = (df["sell_price_ffill"] / g["sell_price_ffill"].shift(1) - 1.0).astype("float32")

    p8 = g["sell_price_ffill"].shift(1)
    df["price_rel_8w"] = (df["sell_price_ffill"] / p8.rolling(8, min_periods=4).mean()).astype("float32")

    for c in ["price_chg_1", "price_rel_8w"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


def load_model_bundle(models_dir: Path, tag: str) -> Tuple[lgb.Booster, List[str]]:
    model_path = models_dir / f"lgbm_{tag}.txt"
    feats_path = models_dir / f"features_{tag}.json"

    # Backward compatibility if you kept older names (p50/p90)
    if not model_path.exists() and tag == "p50":
        model_path = models_dir / "lgbm_p50.txt"
    if not feats_path.exists() and tag == "p50":
        feats_path = models_dir / "features_p50.json"
    if not model_path.exists() and tag == "p90":
        model_path = models_dir / "lgbm_p90.txt"
    if not feats_path.exists() and tag == "p90":
        feats_path = models_dir / "features_p90.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing features list: {feats_path}")

    model = lgb.Booster(model_file=str(model_path))
    feat_cols = json.loads(feats_path.read_text())
    return model, feat_cols


def prepare_features_for_model(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Ensure categorical columns are category dtype (LightGBM expects same as training)
    for c in ["store_id", "item_id", "dept_id", "cat_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    X = df[feature_cols].copy()
    return X


# -------------------------
# Simulation core
# -------------------------
def simulate_series(
    series_df: pd.DataFrame,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
    cfg: BacktestConfig,
) -> Dict[str, float]:
    """
    Simulate one (store,item) series over the valid period.

    Lost sales model:
      sales_fulfilled = min(on_hand, demand)
      lost_sales = max(demand - on_hand, 0)

    Ordering:
      Each week, after observing demand and receiving arrivals, we compute inventory_position and place an order
      that arrives after lead_time weeks.

    Policies:
      Baseline: S = trailing mean(demand over last baseline_window weeks) * (lead_time + 1)
      ML:       S = p50*(lead_time+1) + max(0, (p90 - p50))*sqrt(lead_time+1)
                (simple aggregation of uncertainty; transparent and tunable)
    """
    n = len(series_df)
    if n == 0:
        return {}

    demand = series_df["demand"].to_numpy(dtype=np.float32)

    # warm-up: we assume series_df already contains only the weeks we simulate (e.g., valid weeks)
    # but we need trailing demand history for baseline; we will use series_df["demand_hist_*"] if present.
    # For simplicity, baseline will be based on y_roll_mean_{baseline_window} when possible, else fallback.
    # We'll compute a trailing mean directly from full series_df including warm-up if provided.

    # We'll operate on arrays per policy
    L = cfg.lead_time
    review = 1  # weekly review period
    horizon = L + review

    # Baseline S proxy uses trailing mean of past demand (from series_df features if available)
    # We'll reconstruct using y_roll_mean_{baseline_window} if present (already shifted).
    roll_col = f"y_roll_mean_{cfg.baseline_window}"
    if roll_col in series_df.columns:
        base_mu = series_df[roll_col].to_numpy(dtype=np.float32)
        base_mu = np.nan_to_num(base_mu, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # fallback: use p50 as baseline mean (not ideal but prevents crash)
        base_mu = pred_p50.astype(np.float32)

    # Compute order-up-to targets S per week
    # Baseline: expected demand over horizon
    S_base = base_mu * horizon

    # ML: expected over horizon + safety from spread
    mu = pred_p50.astype(np.float32)
    q90 = pred_p90.astype(np.float32)
    spread = np.maximum(q90 - mu, 0.0)

    # Simple uncertainty aggregation across horizon (transparent heuristic)
    # sqrt(horizon) scaling is a common approximation when aggregating independent-ish errors.
    S_ml = mu * horizon + spread * np.sqrt(horizon)

    # Avoid negative S
    S_base = np.clip(S_base, 0.0, None)
    S_ml = np.clip(S_ml, 0.0, None)

    # Initialize inventory
    on_hand_base = float(S_base[0]) if cfg.init_inventory_mode == "S" else 0.0
    on_hand_ml = float(S_ml[0]) if cfg.init_inventory_mode == "S" else 0.0

    # Pipeline of on-order arrivals (size L)
    pipe_base = np.zeros(L, dtype=np.float32)
    pipe_ml = np.zeros(L, dtype=np.float32)

    total_demand = 0.0
    lost_base = 0.0
    lost_ml = 0.0
    order_base_sum = 0.0
    order_ml_sum = 0.0
    inv_base_sum = 0.0
    inv_ml_sum = 0.0

    for t in range(n):
        # receive arrivals
        if L > 0:
            on_hand_base += float(pipe_base[0])
            on_hand_ml += float(pipe_ml[0])
            pipe_base[:-1] = pipe_base[1:]
            pipe_ml[:-1] = pipe_ml[1:]
            pipe_base[-1] = 0.0
            pipe_ml[-1] = 0.0

        # observe demand and fulfill
        d = float(demand[t])
        total_demand += d

        sold_base = min(on_hand_base, d)
        sold_ml = min(on_hand_ml, d)

        lost_base += max(d - sold_base, 0.0)
        lost_ml += max(d - sold_ml, 0.0)

        on_hand_base -= sold_base
        on_hand_ml -= sold_ml

        # inventory positions
        inv_pos_base = on_hand_base + float(pipe_base.sum())
        inv_pos_ml = on_hand_ml + float(pipe_ml.sum())

        # place orders to reach S
        order_qty_base = max(float(S_base[t]) - inv_pos_base, 0.0)
        order_qty_ml = max(float(S_ml[t]) - inv_pos_ml, 0.0)

        # push into pipeline
        if L > 0:
            pipe_base[-1] += order_qty_base
            pipe_ml[-1] += order_qty_ml
        else:
            # immediate arrival
            on_hand_base += order_qty_base
            on_hand_ml += order_qty_ml

        order_base_sum += order_qty_base
        order_ml_sum += order_qty_ml
        inv_base_sum += on_hand_base
        inv_ml_sum += on_hand_ml

    # Aggregate metrics
    fill_base = service_level_fill_rate(total_demand, lost_base)
    fill_ml = service_level_fill_rate(total_demand, lost_ml)

    return {
        "weeks": float(n),
        "total_demand": float(total_demand),
        "lost_sales_base": float(lost_base),
        "lost_sales_ml": float(lost_ml),
        "fill_rate_base": float(fill_base),
        "fill_rate_ml": float(fill_ml),
        "avg_on_hand_base": float(inv_base_sum / max(n, 1)),
        "avg_on_hand_ml": float(inv_ml_sum / max(n, 1)),
        "total_ordered_base": float(order_base_sum),
        "total_ordered_ml": float(order_ml_sum),
    }


def main() -> int:
    args = parse_args()
    cfg = BacktestConfig(
        data_path=Path(args.data_path),
        models_dir=Path(args.models_dir),
        out_dir=Path(args.out_dir),
        lead_time=int(args.lead_time),
        sample_series=int(args.sample_series),
        seed=int(args.seed),
        min_hist_weeks=int(args.min_hist_weeks),
        baseline_window=int(args.baseline_window),
        init_inventory_mode=str(args.init_inventory_mode),
        clip_pred_nonneg=not bool(args.no_clip_nonneg),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Missing data parquet: {cfg.data_path}")

    # Load models + features
    model_p50, feat_p50 = load_model_bundle(cfg.models_dir, "p50")
    model_p90, feat_p90 = load_model_bundle(cfg.models_dir, "p90")

    # Require same feature set for simplicity
    if feat_p50 != feat_p90:
        raise ValueError(
            "Feature lists differ between p50 and p90. For v1, keep them identical.\n"
            "Tip: train both with the same feature_cols list."
        )

    feat_cols = feat_p50

    print(f"[backtest] loading: {cfg.data_path}")
    df = pd.read_parquet(cfg.data_path)

    # Build features (same as training)
    print("[backtest] building features...")
    df = add_features(df)

    # We simulate ONLY on the validation period, but we need warm-up history to compute lags/rolls.
    # So we keep all rows, but mark which rows are sim weeks.
    # We'll simulate per-series over rows where split == 'valid' AND t_idx >= min_hist_weeks.
    df = df.sort_values(["store_id", "item_id", "wm_yr_wk"]).copy()
    df["t_idx"] = df.groupby(["store_id", "item_id"], sort=False).cumcount()
    sim_mask = (df["split"] == "valid") & (df["t_idx"] >= cfg.min_hist_weeks)

    # Predict for all rows (we'll only use valid+mask rows)
    print("[backtest] scoring p50/p90...")
    X_all = prepare_features_for_model(df, feat_cols)

    pred50 = model_p50.predict(X_all, num_iteration=model_p50.best_iteration)
    pred90 = model_p90.predict(X_all, num_iteration=model_p90.best_iteration)

    pred50 = pred50.astype(np.float32)
    pred90 = pred90.astype(np.float32)

    if cfg.clip_pred_nonneg:
        pred50 = np.clip(pred50, 0.0, None)
        pred90 = np.clip(pred90, 0.0, None)

    # WAPE on valid (for reference; interpret carefully for p90)
    valid_mask = sim_mask.to_numpy()
    y_valid = df.loc[sim_mask, "y_next_week"].to_numpy(dtype=np.float32)
    p50_valid = pred50[valid_mask]
    p90_valid = pred90[valid_mask]
    print(f"[backtest] valid WAPE p50 (reference): {wape(y_valid, p50_valid):.4f}")
    print(f"[backtest] valid WAPE p90 (reference): {wape(y_valid, p90_valid):.4f}  (WAPE not a quantile metric)")

    # Sample series
    series_keys = df.loc[sim_mask, ["store_id", "item_id"]].drop_duplicates()
    if cfg.sample_series > 0 and cfg.sample_series < len(series_keys):
        series_keys = series_keys.sample(cfg.sample_series, random_state=cfg.seed)

    print(f"[backtest] simulating series: {len(series_keys):,} (lead_time={cfg.lead_time}w)")

    # Index rows for fast access
    # Create a boolean mask per key for sim rows; we'll slice via groupby once
    results: List[Dict[str, float]] = []
    grouped = df.groupby(["store_id", "item_id"], sort=False)

    # For mapping predictions to the series rows, we'll rely on the fact df is in the same order as pred arrays.
    # We'll store integer positions.
    df_pos = np.arange(len(df), dtype=np.int64)

    for key_row in series_keys.itertuples(index=False):
        store_id = key_row.store_id
        item_id = key_row.item_id

        grp_idx = grouped.indices.get((store_id, item_id))
        if grp_idx is None:
            continue

        grp_pos = df_pos[grp_idx]
        # Select sim weeks within this group
        grp_sim_mask = sim_mask.iloc[grp_idx].to_numpy()
        if not grp_sim_mask.any():
            continue

        pos_sim = grp_pos[grp_sim_mask]
        series_df = df.iloc[pos_sim]

        # Align predictions with series_df order
        series_p50 = pred50[pos_sim]
        series_p90 = pred90[pos_sim]

        metrics = simulate_series(series_df, series_p50, series_p90, cfg)
        if not metrics:
            continue
        metrics["store_id"] = store_id
        metrics["item_id"] = item_id
        results.append(metrics)

    out_df = pd.DataFrame(results)
    if out_df.empty:
        raise RuntimeError("No series simulated. Try lowering --min-hist-weeks or confirm split=='valid' exists.")

    # Aggregate summary
    summary = {
        "lead_time_weeks": cfg.lead_time,
        "series_simulated": int(len(out_df)),
        "total_demand": float(out_df["total_demand"].sum()),
        "lost_sales_base": float(out_df["lost_sales_base"].sum()),
        "lost_sales_ml": float(out_df["lost_sales_ml"].sum()),
        "fill_rate_base": float(1.0 - out_df["lost_sales_base"].sum() / max(out_df["total_demand"].sum(), 1.0)),
        "fill_rate_ml": float(1.0 - out_df["lost_sales_ml"].sum() / max(out_df["total_demand"].sum(), 1.0)),
        "avg_on_hand_base": float(out_df["avg_on_hand_base"].mean()),
        "avg_on_hand_ml": float(out_df["avg_on_hand_ml"].mean()),
        "total_ordered_base": float(out_df["total_ordered_base"].sum()),
        "total_ordered_ml": float(out_df["total_ordered_ml"].sum()),
    }

    # Save outputs
    summary_csv = cfg.out_dir / "backtest_summary.csv"
    metrics_json = cfg.out_dir / "backtest_metrics.json"

    out_df.to_csv(summary_csv, index=False)
    metrics_json.write_text(json.dumps(summary, indent=2))

    print(f"[backtest] wrote: {summary_csv}")
    print(f"[backtest] wrote: {metrics_json}")
    print("[backtest] aggregate results:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())