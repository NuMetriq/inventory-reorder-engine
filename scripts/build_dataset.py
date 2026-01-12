"""
build_dataset.py

Build a WI-only weekly modeling dataset from the M5 Forecasting dataset.

Expected raw files (NOT committed to git):
  data/m5_raw/calendar.csv
  data/m5_raw/sales_train_validation.csv
  data/m5_raw/sell_prices.csv

Outputs:
  data/processed/m5_wi_weekly_panel.parquet

Run:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --state WI --valid-weeks 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# If you added settings.py as recommended, this will work:
# from ire.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
# Otherwise, we infer repo root from this script location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "m5_raw"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    out_dir: Path

    @property
    def calendar_csv(self) -> Path:
        return self.raw_dir / "calendar.csv"

    @property
    def sales_csv(self) -> Path:
        return self.raw_dir / "sales_train_validation.csv"

    @property
    def prices_csv(self) -> Path:
        return self.raw_dir / "sell_prices.csv"


def _check_paths(p: Paths) -> None:
    missing = [str(x) for x in [p.calendar_csv, p.sales_csv, p.prices_csv] if not x.exists()]
    if missing:
        msg = (
            "Missing required M5 raw files:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nExpected layout:\n"
            "  data/m5_raw/calendar.csv\n"
            "  data/m5_raw/sales_train_validation.csv\n"
            "  data/m5_raw/sell_prices.csv\n"
        )
        raise FileNotFoundError(msg)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build WI-only weekly panel from M5.")
    ap.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Directory with M5 raw CSVs.")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Directory for processed outputs.")
    ap.add_argument("--state", type=str, default="WI", help="State to filter to (M5 supports CA, TX, WI).")
    ap.add_argument(
        "--valid-weeks",
        type=int,
        default=8,
        help="How many final weeks to label as validation (time-based split flag).",
    )
    ap.add_argument(
        "--keep-zero-demand",
        action="store_true",
        help="Keep weeks with zero demand (default keeps them). This flag is here for clarity.",
    )
    return ap.parse_args()


def build_weekly_panel(paths: Paths, state_id: str) -> pd.DataFrame:
    # 1) Calendar: maps d_# -> date and wm_yr_wk
    cal = pd.read_csv(paths.calendar_csv, usecols=["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1",
                                                   "event_type_1", "event_name_2", "event_type_2", "snap_CA",
                                                   "snap_TX", "snap_WI"])
    cal["date"] = pd.to_datetime(cal["date"])

    # Precompute week start/end per wm_yr_wk for nice time features
    week_bounds = (
        cal.groupby("wm_yr_wk", as_index=False)
        .agg(week_start=("date", "min"), week_end=("date", "max"), year=("year", "max"), month=("month", "max"))
    )

    # For weekly event flags: if any day in week has an event, mark it
    cal["has_event_1"] = cal["event_name_1"].notna().astype(np.int8)
    cal["has_event_2"] = cal["event_name_2"].notna().astype(np.int8)

    week_events = (
        cal.groupby("wm_yr_wk", as_index=False)
        .agg(event_1_any=("has_event_1", "max"), event_2_any=("has_event_2", "max"))
    )

    # SNAP: take max within week for the chosen state (if any day is SNAP, week flag = 1)
    snap_col = f"snap_{state_id}"
    if snap_col not in cal.columns:
        raise ValueError(f"State '{state_id}' not found in calendar SNAP columns. Expected one of CA, TX, WI.")

    week_snap = cal.groupby("wm_yr_wk", as_index=False).agg(snap_any=(snap_col, "max"))

    # 2) Sales: filter to state, then melt d_ columns to long
    sales = pd.read_csv(paths.sales_csv)
    if "state_id" not in sales.columns:
        raise ValueError("sales_train_validation.csv missing 'state_id' column (unexpected format).")

    sales = sales.loc[sales["state_id"] == state_id].copy()
    if sales.empty:
        raise ValueError(f"No rows found for state_id='{state_id}'. Check the --state value (CA, TX, WI).")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_# columns found in sales_train_validation.csv (unexpected format).")

    long = sales.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="demand")
    # demand should be int-ish; keep as int32
    long["demand"] = long["demand"].astype(np.int32)

    # Merge calendar to get wm_yr_wk
    long = long.merge(cal[["d", "wm_yr_wk"]], on="d", how="left")
    if long["wm_yr_wk"].isna().any():
        raise ValueError("Found unmapped 'd' values after merging calendar. Dataset files may be inconsistent.")

    # Weekly aggregate demand
    weekly = (
        long.groupby(["state_id", "store_id", "item_id", "dept_id", "cat_id", "wm_yr_wk"], as_index=False)
        .agg(demand=("demand", "sum"))
    )

    # 3) Prices: merge sell_price (store_id, item_id, wm_yr_wk)
    prices = pd.read_csv(paths.prices_csv, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    # In rare cases, there can be duplicates; aggregate to mean price in week
    prices = prices.groupby(["store_id", "item_id", "wm_yr_wk"], as_index=False).agg(sell_price=("sell_price", "mean"))

    weekly = weekly.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # 4) Add week time features + event/snap flags
    weekly = weekly.merge(week_bounds, on="wm_yr_wk", how="left")
    weekly = weekly.merge(week_events, on="wm_yr_wk", how="left")
    weekly = weekly.merge(week_snap, on="wm_yr_wk", how="left")

    # Some weeks might not have price (sell_price NaN). Keep it; handle in feature engineering later.
    # Add cyclical week-of-year features
    # week_start is a date; use ISO week number
    iso_week = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["week_of_year"] = iso_week.astype(np.int16)
    weekly["woy_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52.0).astype(np.float32)
    weekly["woy_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52.0).astype(np.float32)

    # 5) Create leakage-safe target: next week's demand per (store, item)
    weekly = weekly.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)
    weekly["y_next_week"] = weekly.groupby(["store_id", "item_id"])["demand"].shift(-1).astype("float32")

    return weekly


def add_time_split_flags(df: pd.DataFrame, valid_weeks: int) -> pd.DataFrame:
    # Use global time ordering (same wm_yr_wk across all series)
    all_weeks = np.sort(df["wm_yr_wk"].unique())
    if len(all_weeks) <= valid_weeks + 1:
        raise ValueError(
            f"Not enough weeks ({len(all_weeks)}) to hold out valid_weeks={valid_weeks}. Reduce --valid-weeks."
        )

    cutoff_weeks = set(all_weeks[-valid_weeks:])
    df = df.copy()
    df["split"] = np.where(df["wm_yr_wk"].isin(cutoff_weeks), "valid", "train")
    return df


def main() -> int:
    args = _parse_args()
    paths = Paths(raw_dir=Path(args.raw_dir), out_dir=Path(args.out_dir))
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    _check_paths(paths)

    print(f"[build_dataset] raw_dir={paths.raw_dir}")
    print(f"[build_dataset] out_dir={paths.out_dir}")
    print(f"[build_dataset] state={args.state}")

    df = build_weekly_panel(paths=paths, state_id=args.state)
    df = add_time_split_flags(df, valid_weeks=args.valid_weeks)

    # Drop rows where target is missing (last available week per series)
    before = len(df)
    df = df.loc[~df["y_next_week"].isna()].copy()
    after = len(df)

    out_path = paths.out_dir / f"m5_{args.state.lower()}_weekly_panel.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[build_dataset] wrote: {out_path}")
    print(f"[build_dataset] rows (before drop y_next_week NaN): {before:,}")
    print(f"[build_dataset] rows (after): {after:,}")
    print(f"[build_dataset] columns: {len(df.columns)}")
    print("[build_dataset] split counts:")
    print(df["split"].value_counts())

    # Quick sanity checks
    if (df["wm_yr_wk"].isna().any()) or (df["week_start"].isna().any()):
        print("[build_dataset] WARNING: missing week metadata after merges.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())