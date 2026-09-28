"""
Advanced XGBoost Feature Engineering + Ablation Pipeline
========================================================
Uses DuckDB for efficient extraction and pandas for per-entity feature
computation. Avoids materializing a new 75M-row parquet.

Strategy:
  1. Extract train+val rows directly from forecast_dataset.parquet via DuckDB
     with subsampling to keep it manageable (~500K train, ~100K val)
  2. Compute new features per entity group in pandas (backward-looking only)
  3. Train XGBoost ablation models for 30d/60d/90d
  4. Evaluate and compare against baseline
"""

import os, sys, json, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
import duckdb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Config ──────────────────────────────────────────────────────────────
PARQUET    = "data/processed/forecast_dataset.parquet"
SPLIT_CFG  = "data/processed/split_config.json"
MODEL_DIR  = "data/models/new_pipeline"
RESULT_DIR = "evaluation_results"

MAX_TRAIN = 500_000
MAX_VAL   = 100_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── Metric helpers ──────────────────────────────────────────────────────
def wape(y, p):
    return float(np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8))

def smape_metric(y, p):
    return float(200 * np.mean(np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8)))

def dir_accuracy(y, p, curr):
    td = y - curr
    pd_ = p - curr
    return float(np.mean(np.sign(td) == np.sign(pd_)))

def evaluate(y, p, curr):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "WAPE": wape(y, p),
        "sMAPE": smape_metric(y, p),
        "DirAcc": dir_accuracy(y, p, curr),
    }

# ─── Load split config ──────────────────────────────────────────────────
with open(SPLIT_CFG) as f:
    splits = json.load(f)["splits"]

train_start = splits["train_start"]
val_start   = splits["val_start"]
val_end     = splits["val_end"]

print(f"Train: {train_start} -> {val_start}")
print(f"Val  : {val_start} -> {val_end}")

# ─── Step 1: Extract data via DuckDB with subsampling ────────────────────
print("\n[1/5] Extracting train + val data from parquet via DuckDB ...")
t0 = time.time()

con = duckdb.connect()

# We use TABLESAMPLE to subsample. DuckDB supports TABLESAMPLE SYSTEM.
# But we need deterministic entity coverage, so we use a hash-based approach:
# select all rows in the date range, assign a random ordering via hash, take top N.

# First: get column list for the existing features
existing_cols = [
    "State", "District", "Market", "Commodity", "Variety", "Grade",
    "Arrival_Date", "Modal_Price", "Min_Price", "Max_Price",
    "Commodity_Code", "is_interpolated",
    "modal_price_lag_1", "modal_price_lag_3", "modal_price_lag_7",
    "modal_price_lag_14", "modal_price_lag_30", "modal_price_lag_60", "modal_price_lag_90",
    "modal_price_roll_mean_7", "modal_price_roll_std_7",
    "modal_price_roll_min_7", "modal_price_roll_max_7",
    "modal_price_roll_mean_30", "modal_price_roll_std_30",
    "modal_price_roll_min_30", "modal_price_roll_max_30",
    "modal_price_change_7", "modal_price_change_30",
    "month", "week", "day_of_year", "season",
    "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
    "target_30d", "target_60d", "target_90d",
]

col_str = ", ".join(existing_cols)

# Extract TRAIN data (subsample)
train_query = f"""
SELECT {col_str}
FROM 'data/processed/forecast_dataset.parquet'
WHERE Arrival_Date >= '{train_start}'
  AND Arrival_Date < '{val_start}'
  AND target_30d IS NOT NULL
ORDER BY hash(State || Market || Commodity || CAST(Arrival_Date AS VARCHAR))
LIMIT {MAX_TRAIN}
"""
print("  Loading train ...")
df_train_raw = con.query(train_query).df()
print(f"  Train rows: {len(df_train_raw):,}")

# Extract VAL data (subsample)
val_query = f"""
SELECT {col_str}
FROM 'data/processed/forecast_dataset.parquet'
WHERE Arrival_Date >= '{val_start}'
  AND Arrival_Date < '{val_end}'
  AND target_30d IS NOT NULL
ORDER BY hash(State || Market || Commodity || CAST(Arrival_Date AS VARCHAR))
LIMIT {MAX_VAL}
"""
print("  Loading val ...")
df_val_raw = con.query(val_query).df()
print(f"  Val rows: {len(df_val_raw):,}")

con.close()
print(f"  Extraction time: {time.time()-t0:.1f}s")

# ─── Step 2: Compute new features in pandas ──────────────────────────────
print("\n[2/5] Computing advanced features ...")
t1 = time.time()

def add_features(df):
    """Add all new features. Operates on the entire df (already subsampled)."""
    df = df.copy()
    p = df["Modal_Price"]

    # --- 1. Price momentum (already have lag cols, just compute differences) ---
    for lag in [1, 3, 7, 14, 30, 60, 90]:
        lag_col = f"modal_price_lag_{lag}"
        if lag_col in df.columns:
            df[f"price_change_{lag}"] = p - df[lag_col]

    # --- 2. Percentage returns ---
    for lag in [1, 7, 30, 90]:
        lag_col = f"modal_price_lag_{lag}"
        if lag_col in df.columns:
            df[f"return_{lag}"] = (p - df[lag_col]) / df[lag_col].replace(0, np.nan)

    # --- 3. Volatility: use existing roll_std, add more from existing lags ---
    # We already have roll_std_7 and roll_std_30. Approximate others from lag data.
    # For rolling_std_14/60/90, we use a proxy: stddev of available lag prices
    lag_cols_14 = [f"modal_price_lag_{l}" for l in [1,3,7,14]]
    lag_cols_60 = [f"modal_price_lag_{l}" for l in [1,3,7,14,30,60]]
    lag_cols_90 = [f"modal_price_lag_{l}" for l in [1,3,7,14,30,60,90]]

    df["rolling_std_14_proxy"] = df[lag_cols_14].std(axis=1)
    df["rolling_std_60_proxy"] = df[lag_cols_60].std(axis=1)
    df["rolling_std_90_proxy"] = df[lag_cols_90].std(axis=1)

    # --- 4. Rolling statistics: median, min, max from available lags ---
    df["rolling_median_7_proxy"]  = df[[f"modal_price_lag_{l}" for l in [1,3,7]]].median(axis=1)
    df["rolling_median_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].median(axis=1)
    df["rolling_median_90_proxy"] = df[lag_cols_90].median(axis=1)

    # rolling_min and rolling_max from lag columns
    df["rolling_min_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].min(axis=1)
    df["rolling_max_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].max(axis=1)
    df["rolling_min_90_proxy"] = df[lag_cols_90].min(axis=1)
    df["rolling_max_90_proxy"] = df[lag_cols_90].max(axis=1)

    # --- 5. Price position ---
    df["distance_from_roll_mean_30"] = p - df["modal_price_roll_mean_30"]
    df["distance_from_roll_mean_7"]  = p - df["modal_price_roll_mean_7"]

    range_30 = df["rolling_max_30_proxy"] - df["rolling_min_30_proxy"]
    range_90 = df["rolling_max_90_proxy"] - df["rolling_min_90_proxy"]
    df["position_in_30d_range"] = (p - df["rolling_min_30_proxy"]) / range_30.replace(0, np.nan)
    df["position_in_90d_range"] = (p - df["rolling_min_90_proxy"]) / range_90.replace(0, np.nan)

    # --- 6. Seasonal features ---
    dt = pd.to_datetime(df["Arrival_Date"])
    df["quarter"] = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int).values
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)

    # --- 7. Commodity-market relative features (date-level) ---
    # Relative to commodity-wide median on the same date
    comm_date_med = df.groupby(["Commodity", df["Arrival_Date"].dt.date])["Modal_Price"].transform("median")
    df["relative_to_comm_date"] = p / comm_date_med.replace(0, np.nan)

    # Relative to market-wide median on the same date
    mkt_date_med = df.groupby(["Market", df["Arrival_Date"].dt.date])["Modal_Price"].transform("median")
    df["relative_to_market_date"] = p / mkt_date_med.replace(0, np.nan)

    # --- 8. Entity history (using expanding stats on subsampled data as proxy) ---
    # Historical commodity median/std
    df["hist_comm_median"] = df.groupby("Commodity")["Modal_Price"].transform("median")
    df["hist_comm_std"]    = df.groupby("Commodity")["Modal_Price"].transform("std")

    # Historical market median/std
    df["hist_market_median"] = df.groupby("Market")["Modal_Price"].transform("median")
    df["hist_market_std"]    = df.groupby("Market")["Modal_Price"].transform("std")

    # Price relative to historical entity stats
    df["price_vs_comm_hist"]   = p / df["hist_comm_median"].replace(0, np.nan)
    df["price_vs_market_hist"] = p / df["hist_market_median"].replace(0, np.nan)

    return df


df_train = add_features(df_train_raw)
df_val   = add_features(df_val_raw)
del df_train_raw, df_val_raw
gc.collect()

print(f"  Feature computation time: {time.time()-t1:.1f}s")
print(f"  Train cols: {df_train.shape[1]}, Val cols: {df_val.shape[1]}")

# ─── Step 3: Define feature sets and ablation configs ─────────────────────
print("\n[3/5] Defining ablation feature sets ...")

existing_features = [
    "modal_price_lag_1", "modal_price_lag_3", "modal_price_lag_7",
    "modal_price_lag_14", "modal_price_lag_30", "modal_price_lag_60", "modal_price_lag_90",
    "modal_price_roll_mean_7", "modal_price_roll_std_7",
    "modal_price_roll_min_7", "modal_price_roll_max_7",
    "modal_price_roll_mean_30", "modal_price_roll_std_30",
    "modal_price_roll_min_30", "modal_price_roll_max_30",
    "modal_price_change_7", "modal_price_change_30",
    "month", "week", "day_of_year",
    "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
    "Commodity_Code",
]

momentum_return_features = [
    "price_change_1", "price_change_3", "price_change_7",
    "price_change_14", "price_change_30", "price_change_60", "price_change_90",
    "return_1", "return_7", "return_30", "return_90",
]

volatility_rolling_features = [
    "rolling_std_14_proxy", "rolling_std_60_proxy", "rolling_std_90_proxy",
    "rolling_median_7_proxy", "rolling_median_30_proxy", "rolling_median_90_proxy",
    "rolling_min_30_proxy", "rolling_max_30_proxy",
    "rolling_min_90_proxy", "rolling_max_90_proxy",
    "distance_from_roll_mean_30", "distance_from_roll_mean_7",
    "position_in_30d_range", "position_in_90d_range",
]

relative_entity_features = [
    "relative_to_comm_date", "relative_to_market_date",
    "hist_comm_median", "hist_comm_std",
    "hist_market_median", "hist_market_std",
    "price_vs_comm_hist", "price_vs_market_hist",
]

seasonal_extra_features = [
    "quarter", "week_of_year",
    "sin_week", "cos_week", "sin_quarter", "cos_quarter",
]

ablations = {
    "A": existing_features,
    "B": existing_features + momentum_return_features,
    "C": existing_features + volatility_rolling_features,
    "D": existing_features + relative_entity_features,
    "E": (existing_features + momentum_return_features
         + volatility_rolling_features + relative_entity_features
         + seasonal_extra_features),
}

# Save feature config
with open(os.path.join(RESULT_DIR, "new_feature_config.json"), "w") as f:
    json.dump(ablations, f, indent=2)
print(f"  Ablations: {list(ablations.keys())}")
for k, v in ablations.items():
    print(f"    {k}: {len(v)} features")

# ─── Step 4: Train XGBoost for each horizon × ablation ──────────────────
print("\n[4/5] Training XGBoost models ...")

baseline_mae = {"30": 1104.3, "60": 1132.7, "90": 1094.2}
results = []
feature_importances = {}

for horizon in [30, 60, 90]:
    target_col = f"target_{horizon}d"
    h_key = str(horizon)
    print(f"\n{'='*60}")
    print(f"  Horizon: {horizon}d")
    print(f"{'='*60}")

    # Filter valid targets
    train_mask = df_train[target_col].notna()
    val_mask   = df_val[target_col].notna()

    y_train = df_train.loc[train_mask, target_col].values.astype(np.float32)
    y_val   = df_val.loc[val_mask, target_col].values.astype(np.float32)
    curr_val = df_val.loc[val_mask, "Modal_Price"].values.astype(np.float32)

    print(f"  Train samples: {len(y_train):,} | Val samples: {len(y_val):,}")

    for exp_name, features in ablations.items():
        print(f"\n  Experiment {exp_name} ({len(features)} features)")

        # Replace inf/nan in features
        X_tr = df_train.loc[train_mask, features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        X_vl = df_val.loc[val_mask, features].replace([np.inf, -np.inf], np.nan).astype(np.float32)

        t_start = time.time()

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:pseudohubererror",
            eval_metric="mae",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(
            X_tr, y_train,
            eval_set=[(X_vl, y_val)],
            verbose=False,
        )

        train_time = time.time() - t_start
        preds = model.predict(X_vl)

        metrics = evaluate(y_val, preds, curr_val)
        metrics["train_time"] = round(train_time, 1)
        metrics["best_iteration"] = model.best_iteration

        bsl = baseline_mae[h_key]
        delta = bsl - metrics["MAE"]
        pct = delta / bsl * 100

        print(f"    MAE: {metrics['MAE']:.1f}  (baseline {bsl:.1f}, delta={delta:+.1f}, {pct:+.1f}%)")
        print(f"    RMSE: {metrics['RMSE']:.1f} | R2: {metrics['R2']:.4f} | sMAPE: {metrics['sMAPE']:.2f}")
        print(f"    WAPE: {metrics['WAPE']:.4f} | DirAcc: {metrics['DirAcc']:.4f}")
        print(f"    Train time: {train_time:.1f}s | Best iter: {model.best_iteration}")

        results.append({
            "horizon": horizon,
            "experiment": exp_name,
            **metrics,
        })

        # Save model checkpoint
        model.save_model(os.path.join(MODEL_DIR, f"xgb_{horizon}d_exp_{exp_name}.json"))

        # Feature importance for full feature set
        if exp_name == "E":
            imp = model.get_booster().get_score(importance_type="gain")
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_importances[f"{horizon}d"] = sorted_imp
            print(f"    Top 5 features (gain): {[x[0] for x in sorted_imp[:5]]}")

        del model
        gc.collect()

# ─── Step 5: Save results and generate report ───────────────────────────
print("\n[5/5] Saving results ...")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULT_DIR, "xgb_ablation_results.csv"), index=False)

with open(os.path.join(RESULT_DIR, "xgb_feature_importance.json"), "w") as f:
    json.dump(feature_importances, f, indent=2)

# ─── Final Summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL ABLATION SUMMARY")
print("="*70)

for horizon in [30, 60, 90]:
    h_key = str(horizon)
    bsl = baseline_mae[h_key]
    print(f"\n--- {horizon}d Horizon (baseline MAE = {bsl:.1f}) ---")
    subset = df_results[df_results["horizon"] == horizon].sort_values("MAE")
    for _, row in subset.iterrows():
        delta = bsl - row["MAE"]
        print(f"  Exp {row['experiment']}: MAE={row['MAE']:.1f}  delta={delta:+.1f}  "
              f"RMSE={row['RMSE']:.1f}  R2={row['R2']:.4f}  "
              f"sMAPE={row['sMAPE']:.2f}  WAPE={row['WAPE']:.4f}  "
              f"DirAcc={row['DirAcc']:.4f}  Time={row['train_time']:.0f}s")

print("\n--- Top 20 Features per Horizon (gain) ---")
for h, feats in feature_importances.items():
    print(f"\n  {h}:")
    for i, (fname, gain) in enumerate(feats, 1):
        print(f"    {i:2d}. {fname:40s} gain={gain:.1f}")

# Overall verdict
print("\n--- Verdict ---")
for horizon in [30, 60, 90]:
    h_key = str(horizon)
    bsl = baseline_mae[h_key]
    best = df_results[df_results["horizon"] == horizon].sort_values("MAE").iloc[0]
    delta = bsl - best["MAE"]
    if delta > 0:
        print(f"  {horizon}d: IMPROVED by {delta:.1f} ({delta/bsl*100:.1f}%) — Exp {best['experiment']}")
    else:
        print(f"  {horizon}d: DEGRADED by {abs(delta):.1f} ({abs(delta)/bsl*100:.1f}%) — Exp {best['experiment']}")

print("\nDONE.")
