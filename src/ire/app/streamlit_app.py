import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json

st.set_page_config(page_title="Inventory Reorder Engine", layout="wide")

# -------------------------------------------------
# Paths
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "processed" / "m5_wi_weekly_panel.parquet"
MODELS_DIR = REPO_ROOT / "models"

# -------------------------------------------------
# Load assets
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_resource
def load_model(tag: str):
    model = lgb.Booster(model_file=str(MODELS_DIR / f"lgbm_{tag}.txt"))
    features = json.loads((MODELS_DIR / f"features_{tag}.json").read_text())
    return model, features

df = load_data()
model_p50, feat_cols = load_model("p50")
model_p90, _ = load_model("p90")

def pct(x: float) -> str:
    return f"{100*x:.2f}%"

@st.cache_data
def load_backtest_metrics(path: Path):
    return json.loads(path.read_text())

@st.cache_data
def load_all_backtest_metrics(repo_root: Path):
    out_dir = repo_root / "outputs"
    rows = []
    for lt in [1, 2, 3, 4]:
        p = out_dir / f"backtest_metrics_lt{lt}.json"
        if p.exists():
            m = json.loads(p.read_text())
            rows.append({
                "lead_time": lt,
                "fill_base": m["fill_rate_base"],
                "fill_ml": m["fill_rate_ml"],
                "inv_base": m["avg_on_hand_base"],
                "inv_ml": m["avg_on_hand_ml"],
                "lost_base": m["lost_sales_base"],
                "lost_ml": m["lost_sales_ml"],
                "series": m["series_simulated"],
            })
    return pd.DataFrame(rows).sort_values("lead_time")
# -------------------------------------------------
# Feature engineering (single-SKU slice)
# -------------------------------------------------
def add_features(df: pd.DataFrame):
    df = df.sort_values("wm_yr_wk").copy()

    for k in [1, 2, 4, 8]:
        df[f"y_lag_{k}"] = df["demand"].shift(k)

    for w in [4, 8]:
        s = df["demand"].shift(1)
        df[f"y_roll_mean_{w}"] = s.rolling(w, min_periods=2).mean()
        df[f"y_roll_std_{w}"] = s.rolling(w, min_periods=2).std()

    s8 = df["demand"].shift(1)
    df["pct_zero_8"] = (
        s8.rolling(8, min_periods=4)
        .apply(lambda x: np.mean(np.array(x) == 0), raw=False)
    )

    # weeks since last sale
    weeks_since = []
    last_sale = None
    for i, v in enumerate(df["demand"].values):
        if v > 0:
            last_sale = i
            weeks_since.append(0)
        else:
            weeks_since.append(i - last_sale if last_sale is not None else np.nan)
    df["weeks_since_sale"] = pd.Series(weeks_since).shift(1)

    df["price_missing"] = df["sell_price"].isna().astype(int)
    df["sell_price_ffill"] = df["sell_price"].ffill()
    df["price_chg_1"] = df["sell_price_ffill"] / df["sell_price_ffill"].shift(1) - 1

    p8 = df["sell_price_ffill"].shift(1)
    df["price_rel_8w"] = df["sell_price_ffill"] / p8.rolling(8, min_periods=4).mean()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📦 Inventory Reorder Recommendation Engine")

st.sidebar.header("Selection")

store_id = st.sidebar.selectbox(
    "Store",
    sorted(df["store_id"].unique())
)

item_id = st.sidebar.selectbox(
    "Item",
    sorted(df.loc[df["store_id"] == store_id, "item_id"].unique())
)

lead_time = st.sidebar.slider("Lead time (weeks)", 1, 4, 2)

service_level = st.sidebar.selectbox(
    "Target service level",
    ["P80 (Lean)", "P90 (Balanced)", "P95 (Conservative)"],
    index=1
)

SERVICE_K = {
    "P80 (Lean)": 0.5,
    "P90 (Balanced)": 1.0,
    "P95 (Conservative)": 1.5,
}

k = SERVICE_K[service_level]

sku_df = (
    df[(df["store_id"] == store_id) & (df["item_id"] == item_id)]
    .sort_values("wm_yr_wk")
    .copy()
)

sku_df = add_features(sku_df)

# Ensure categorical dtype
for c in ["store_id", "item_id", "dept_id", "cat_id"]:
    sku_df[c] = sku_df[c].astype("category")

# Use last available row for forecast
X_last = sku_df.iloc[-1:][feat_cols]

p50 = float(model_p50.predict(X_last)[0])
p90 = float(model_p90.predict(X_last)[0])

p50 = max(p50, 0.0)
p90 = max(p90, p50)

# Inventory policy
horizon = lead_time + 1
uncertainty = max(p90 - p50, 0.0)
S_ml = p50 * horizon + k * uncertainty * np.sqrt(horizon)

# Mock current inventory (for demo)
current_on_hand = st.sidebar.number_input(
    "Current on-hand inventory (units)",
    min_value=0,
    value=int(round(p50 * lead_time)),
)

order_qty = max(S_ml - current_on_hand, 0.0)

# -------------------------------------------------
# Display
# -------------------------------------------------
st.subheader(f"Store: {store_id} | Item: {item_id}")

left, right = st.columns([2, 1])

with left:
    chart_df = sku_df[["week_start", "demand"]].set_index("week_start")
    st.line_chart(chart_df, height=300)

with right:
    st.metric("P50 Forecast (next week)", f"{p50:.1f}")
    st.metric("P90 Forecast (next week)", f"{p90:.1f}")
    st.metric("Order-up-to level (S)", f"{S_ml:.1f}")
    st.metric("Recommended order qty", f"{order_qty:.1f}")

st.caption(
    f"Order-up-to policy: S = P50·(L+1) + k·(P90 − P50)·√(L+1), "
    f"where k = {k:.1f} reflects the chosen service level. "
    "Higher service levels increase safety stock to reduce stockouts."
)
st.divider()

st.subheader("📊 Backtest Snapshot (Precomputed)")

metrics_path = REPO_ROOT / "outputs" / f"backtest_metrics_lt{lead_time}.json"

if metrics_path.exists():
    m = load_backtest_metrics(metrics_path)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fill rate (Baseline)", pct(m["fill_rate_base"]))
    with col2:
        st.metric("Fill rate (ML)", pct(m["fill_rate_ml"]))
    with col3:
        lost_reduction = 1.0 - (m["lost_sales_ml"] / max(m["lost_sales_base"], 1e-9))
        st.metric("Lost sales ↓", pct(lost_reduction))
    with col4:
        inv_delta = m["avg_on_hand_ml"] - m["avg_on_hand_base"]
        st.metric("Avg inventory Δ", f"{inv_delta:.2f}")

    st.caption(
        f"Backtest across {m['series_simulated']} series (lead time = {m['lead_time_weeks']} weeks). "
        "These metrics are computed offline via scripts/backtest.py and loaded instantly in the app."
    )
else:
    st.warning(
        f"No precomputed backtest metrics found for lead time = {lead_time}. "
        "Generate them by running:\n\n"
        f"python scripts/backtest.py --lead-time {lead_time} --sample-series 0"
    )

st.subheader("📈 Tradeoff Across Lead Times")

trade = load_all_backtest_metrics(REPO_ROOT)

if not trade.empty:
    # Show a small table
    show = trade.copy()
    show["fill_base"] = (show["fill_base"] * 100).round(2)
    show["fill_ml"] = (show["fill_ml"] * 100).round(2)
    show["inv_base"] = show["inv_base"].round(2)
    show["inv_ml"] = show["inv_ml"].round(2)
    st.dataframe(
        show.rename(columns={
            "lead_time": "Lead time (wks)",
            "fill_base": "Fill % (Baseline)",
            "fill_ml": "Fill % (ML)",
            "inv_base": "Avg Inv (Baseline)",
            "inv_ml": "Avg Inv (ML)",
            "series": "Series simulated"
        })[["Lead time (wks)", "Fill % (Baseline)", "Fill % (ML)", "Avg Inv (Baseline)", "Avg Inv (ML)", "Series simulated"]],
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.line_chart(trade.set_index("lead_time")[["fill_base", "fill_ml"]], height=260)
        st.caption("Fill rate vs lead time (higher is better).")

    with c2:
        st.line_chart(trade.set_index("lead_time")[["inv_base", "inv_ml"]], height=260)
        st.caption("Average on-hand inventory vs lead time (lower is leaner).")
else:
    st.info("No backtest metric files found in outputs/. Run scripts/backtest.py for lead times 1–4.")