"""
train.py

Train a baseline and a LightGBM quantile (P50) global model on the M5 WI weekly panel.

Input:
  data/processed/m5_wi_weekly_panel.parquet

Output:
  models/
    lgbm_p50.txt
    features_p50.json
    metrics.json

Run:
  python scripts/train.py
  python scripts/train.py --alpha 0.5 --lags 1 2 4 8 --rolls 4 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import lightgbm as lgb


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "m5_wi_weekly_panel.parquet"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    models_dir: Path
    alpha: float
    lags: tuple[int, ...]
    rolls: tuple[int, ...]
    min_train_weeks: int
    num_boost_round: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    seed: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train baseline + LightGBM P50 on M5 WI weekly panel.")
    ap.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--alpha", type=float, default=0.5, help="Quantile alpha (0.5 = median).")

    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8], help="Lag weeks.")
    ap.add_argument("--rolls", type=int, nargs="+", default=[4, 8], help="Rolling window sizes in weeks.")
    ap.add_argument(
        "--min-train-weeks",
        type=int,
        default=16,
        help="Drop rows with insufficient history (max lag/roll).",
    )

    # Reasonable CPU defaults for a first pass
    ap.add_argument("--num-boost-round", type=int, default=1500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=255)
    ap.add_argument("--min-data-in-leaf", type=int, default=200)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum(|e|)/sum(|y|)."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float(np.nan)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball loss for quantile alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def add_features(df: pd.DataFrame, lags: Iterable[int], rolls: Iterable[int]) -> pd.DataFrame:
    """
    Leakage-safe feature generation:
    - All features use demand and price from <= week t (predicting y_next_week).
    - We compute within each (store_id, item_id) series.
    """
    df = df.sort_values(["store_id", "item_id", "wm_yr_wk"]).copy()

    g = df.groupby(["store_id", "item_id"], sort=False)

    # Demand lags
    for k in lags:
        df[f"y_lag_{k}"] = g["demand"].shift(k).astype("float32")

    # Rolling stats (based on shifted demand so the window ends at t-1)
    for w in rolls:
        s = g["demand"].shift(1)
        df[f"y_roll_mean_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).mean().astype("float32")
        df[f"y_roll_std_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).std().astype("float32")

    # Intermittency features
    # pct_zero over last 8 weeks (shifted)
    s8 = g["demand"].shift(1)
    df["pct_zero_8"] = (
        s8.rolling(8, min_periods=4)
        .apply(lambda x: float(np.mean(np.array(x) == 0.0)), raw=False)
        .astype("float32")
    )

    # weeks since last sale (shifted view)
    # compute on the observed demand at time t (so for features predicting t+1, we use shift(0))
    # but we avoid peeking into t+1 by never using y_next_week.
    demand_now = df["demand"].astype("float32")
    # A robust way: per group, track last nonzero index.
    df["weeks_since_sale"] = 0.0
    for (store, item), idx in g.indices.items():
        arr = demand_now.iloc[idx].to_numpy()
        out = np.zeros_like(arr, dtype=np.float32)
        last_sale = -1_000_000
        for i, v in enumerate(arr):
            if v > 0:
                last_sale = i
                out[i] = 0.0
            else:
                out[i] = float(i - last_sale) if last_sale >= 0 else float(i + 1)
        # shift by 1 so we don't use current week's sale indicator to predict next week
        out = np.roll(out, 1)
        out[0] = np.nan
        df.loc[df.index[idx], "weeks_since_sale"] = out

    # Price features: forward-fill within series, add missing indicator
    df["price_missing"] = df["sell_price"].isna().astype("int8")
    df["sell_price_ffill"] = g["sell_price"].ffill().astype("float32")

    # price change vs last week
    df["price_chg_1"] = (df["sell_price_ffill"] / g["sell_price_ffill"].shift(1) - 1.0).astype("float32")

    # price relative to last 8w avg (shifted to avoid leakage)
    p8 = g["sell_price_ffill"].shift(1)
    df["price_rel_8w"] = (df["sell_price_ffill"] / p8.rolling(8, min_periods=4).mean()).astype("float32")

    # Calendar-like features already present from build_dataset.py:
    # week_of_year, woy_sin, woy_cos, event flags, snap_any, year, month

    # Clean up infs
    for c in ["price_chg_1", "price_rel_8w"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline: last 4-week mean of shifted demand (uses weeks t-1..t-4 to predict t+1).
    """
    g = df.groupby(["store_id", "item_id"], sort=False)
    s = g["demand"].shift(1)
    pred = s.rolling(4, min_periods=2).mean()
    return pred.to_numpy(dtype=np.float32)


def main() -> int:
    args = parse_args()
    cfg = TrainConfig(
        data_path=Path(args.data_path),
        models_dir=Path(args.models_dir),
        alpha=float(args.alpha),
        lags=tuple(int(x) for x in args.lags),
        rolls=tuple(int(x) for x in args.rolls),
        min_train_weeks=int(args.min_train_weeks),
        num_boost_round=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_data_in_leaf=int(args.min_data_in_leaf),
        feature_fraction=float(args.feature_fraction),
        bagging_fraction=float(args.bagging_fraction),
        bagging_freq=int(args.bagging_freq),
        seed=int(args.seed),
    )

    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Data not found: {cfg.data_path}")

    print(f"[train] loading: {cfg.data_path}")
    df = pd.read_parquet(cfg.data_path)

    required = {
        "state_id",
        "store_id",
        "item_id",
        "dept_id",
        "cat_id",
        "wm_yr_wk",
        "demand",
        "y_next_week",
        "split",
        "sell_price",
        "week_of_year",
        "woy_sin",
        "woy_cos",
        "event_1_any",
        "event_2_any",
        "snap_any",
        "year",
        "month",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    print("[train] building features...")
    df = add_features(df, lags=cfg.lags, rolls=cfg.rolls)

    # Build baseline predictions (before dropping)
    df["baseline_pred"] = baseline_predict(df)

    # Drop rows with insufficient history / missing features or target
    # We require target exists (should already) and at least max(lag) available.
    max_lag = max(cfg.lags) if cfg.lags else 1
    min_hist = max(max_lag + 1, cfg.min_train_weeks)

    # Use per-series week index to ensure enough history
    df["t_idx"] = df.groupby(["store_id", "item_id"], sort=False).cumcount()
    before = len(df)
    df = df.loc[df["t_idx"] >= min_hist].copy()
    after = len(df)
    print(f"[train] dropped for history: {before - after:,} rows (kept {after:,})")

    target = "y_next_week"

    # Feature columns
    feature_cols = [
        # demand-derived
        *[f"y_lag_{k}" for k in cfg.lags],
        *[f"y_roll_mean_{w}" for w in cfg.rolls],
        *[f"y_roll_std_{w}" for w in cfg.rolls],
        "pct_zero_8",
        "weeks_since_sale",
        # price-derived
        "price_missing",
        "sell_price_ffill",
        "price_chg_1",
        "price_rel_8w",
        # calendar-ish
        "week_of_year",
        "woy_sin",
        "woy_cos",
        "event_1_any",
        "event_2_any",
        "snap_any",
        "year",
        "month",
        # categorical IDs encoded as category (LightGBM handles categories nicely)
        "store_id",
        "item_id",
        "dept_id",
        "cat_id",
    ]

    # Ensure categories
    for c in ["store_id", "item_id", "dept_id", "cat_id"]:
        df[c] = df[c].astype("category")

    # Train/valid split
    train_df = df.loc[df["split"] == "train"].copy()
    valid_df = df.loc[df["split"] == "valid"].copy()

    print(f"[train] train rows: {len(train_df):,} | valid rows: {len(valid_df):,}")

    # Prepare datasets
    X_train = train_df[feature_cols]
    y_train = train_df[target].astype("float32").to_numpy()

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target].astype("float32").to_numpy()

    # Baseline eval
    base_pred_valid = valid_df["baseline_pred"].to_numpy(dtype=np.float32)
    # Fill any missing baseline preds with 0 (very early history)
    base_pred_valid = np.nan_to_num(base_pred_valid, nan=0.0, posinf=0.0, neginf=0.0)
    base_wape = wape(y_valid, base_pred_valid)

    print(f"[train] baseline WAPE: {base_wape:.4f}")

    # LightGBM config
    params = {
        "objective": "quantile",
        "alpha": cfg.alpha,
        "metric": "quantile",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbosity": -1,
        "seed": cfg.seed,
        "force_row_wise": True,  # good default for many rows / wide-ish data
    }

    print("[train] training LightGBM quantile model...")
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=["store_id", "item_id", "dept_id", "cat_id"])
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, categorical_feature=["store_id", "item_id", "dept_id", "cat_id"])

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=cfg.num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    # Predict and evaluate
    pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    pred_valid = np.clip(pred_valid, 0.0, None)  # demand can't be negative
    lgb_wape = wape(y_valid, pred_valid)
    lgb_pinball = pinball_loss(y_valid, pred_valid, alpha=cfg.alpha)

    print(f"[train] LGBM WAPE: {lgb_wape:.4f}")
    print(f"[train] LGBM pinball(alpha={cfg.alpha}): {lgb_pinball:.4f}")

    # Save artifacts
    tag = f"p{int(cfg.alpha*100):02d}"
    model_path = cfg.models_dir / f"lgbm_{tag}.txt"
    feats_path = cfg.models_dir / f"features_{tag}.json"
    metrics_path = cfg.models_dir / f"metrics_{tag}.json"

    model.save_model(str(model_path))
    feats_path.write_text(json.dumps(feature_cols, indent=2))
    metrics = {
        "state": "WI",
        "alpha": cfg.alpha,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "baseline_wape": base_wape,
        "lgbm_wape": lgb_wape,
        "lgbm_pinball": lgb_pinball,
        "best_iteration": int(model.best_iteration),
        "params": params,
        "feature_cols": feature_cols,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train] saved model: {model_path}")
    print(f"[train] saved features: {feats_path}")
    print(f"[train] saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())