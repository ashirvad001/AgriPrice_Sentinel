"""
Advanced XGBoost Diagnostics and Tuning Pipeline
"""

import os, sys, json, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
import duckdb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler

# --- Config ---
PARQUET = "data/processed/forecast_dataset.parquet"
SPLIT_CFG = "data/processed/split_config.json"
MODEL_DIR = "data/models/new_pipeline_v2"
RESULT_DIR = "evaluation_results/v2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Metric Helpers ---
def wape(y, p): return float(np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8))
def smape_metric(y, p): return float(200 * np.mean(np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8)))
def dir_accuracy(y, p, curr): return float(np.mean(np.sign(y - curr) == np.sign(p - curr)))
def evaluate(y, p, curr):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "WAPE": wape(y, p),
        "sMAPE": smape_metric(y, p),
        "DirAcc": dir_accuracy(y, p, curr),
    }

with open(SPLIT_CFG) as f: splits = json.load(f)["splits"]
train_start = splits["train_start"]
val_start = splits["val_start"]
val_end = splits["val_end"]

# --- Feature Defs ---
existing_features = ["modal_price_lag_1", "modal_price_lag_3", "modal_price_lag_7", "modal_price_lag_14", "modal_price_lag_30", "modal_price_lag_60", "modal_price_lag_90", "modal_price_roll_mean_7", "modal_price_roll_std_7", "modal_price_roll_min_7", "modal_price_roll_max_7", "modal_price_roll_mean_30", "modal_price_roll_std_30", "modal_price_roll_min_30", "modal_price_roll_max_30", "modal_price_change_7", "modal_price_change_30", "month", "week", "day_of_year", "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos", "Commodity_Code"]
momentum_return_features = ["price_change_1", "price_change_3", "price_change_7", "price_change_14", "price_change_30", "price_change_60", "price_change_90", "return_1", "return_7", "return_30", "return_90"]
volatility_rolling_features = ["rolling_std_14_proxy", "rolling_std_60_proxy", "rolling_std_90_proxy", "rolling_median_7_proxy", "rolling_median_30_proxy", "rolling_median_90_proxy", "rolling_min_30_proxy", "rolling_max_30_proxy", "rolling_min_90_proxy", "rolling_max_90_proxy", "distance_from_roll_mean_30", "distance_from_roll_mean_7", "position_in_30d_range", "position_in_90d_range"]
relative_entity_features = ["relative_to_comm_date", "relative_to_market_date", "hist_comm_median", "hist_comm_std", "hist_market_median", "hist_market_std", "price_vs_comm_hist", "price_vs_market_hist"]
seasonal_extra_features = ["quarter", "week_of_year", "sin_week", "cos_week", "sin_quarter", "cos_quarter"]

feature_sets = {
    "C": existing_features + volatility_rolling_features,
    "E": existing_features + momentum_return_features + volatility_rolling_features + relative_entity_features + seasonal_extra_features
}

all_cols = ["State", "District", "Market", "Commodity", "Arrival_Date", "Modal_Price", "target_30d", "target_60d", "target_90d"] + existing_features
col_str = ", ".join(all_cols)

def load_and_featurize(train_limit, val_limit, target_col):
    con = duckdb.connect()
    
    train_query = f"SELECT {col_str} FROM 'data/processed/forecast_dataset.parquet' WHERE Arrival_Date >= '{train_start}' AND Arrival_Date < '{val_start}' AND {target_col} IS NOT NULL ORDER BY hash(State || Market || Commodity || CAST(Arrival_Date AS VARCHAR)) LIMIT {train_limit}"
    val_query = f"SELECT {col_str} FROM 'data/processed/forecast_dataset.parquet' WHERE Arrival_Date >= '{val_start}' AND Arrival_Date < '{val_end}' AND {target_col} IS NOT NULL ORDER BY hash(State || Market || Commodity || CAST(Arrival_Date AS VARCHAR)) LIMIT {val_limit}"
    
    df_train = con.query(train_query).df()
    df_val = con.query(val_query).df()
    con.close()
    
    def add_features(df):
        df = df.copy()
        p = df["Modal_Price"]
        for lag in [1, 3, 7, 14, 30, 60, 90]:
            if f"modal_price_lag_{lag}" in df.columns:
                df[f"price_change_{lag}"] = p - df[f"modal_price_lag_{lag}"]
        for lag in [1, 7, 30, 90]:
            if f"modal_price_lag_{lag}" in df.columns:
                df[f"return_{lag}"] = (p - df[f"modal_price_lag_{lag}"]) / df[f"modal_price_lag_{lag}"].replace(0, np.nan)
        lag_cols_14 = [f"modal_price_lag_{l}" for l in [1,3,7,14]]
        lag_cols_60 = [f"modal_price_lag_{l}" for l in [1,3,7,14,30,60]]
        lag_cols_90 = [f"modal_price_lag_{l}" for l in [1,3,7,14,30,60,90]]
        df["rolling_std_14_proxy"] = df[lag_cols_14].std(axis=1)
        df["rolling_std_60_proxy"] = df[lag_cols_60].std(axis=1)
        df["rolling_std_90_proxy"] = df[lag_cols_90].std(axis=1)
        df["rolling_median_7_proxy"]  = df[[f"modal_price_lag_{l}" for l in [1,3,7]]].median(axis=1)
        df["rolling_median_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].median(axis=1)
        df["rolling_median_90_proxy"] = df[lag_cols_90].median(axis=1)
        df["rolling_min_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].min(axis=1)
        df["rolling_max_30_proxy"] = df[[f"modal_price_lag_{l}" for l in [1,3,7,14,30]]].max(axis=1)
        df["rolling_min_90_proxy"] = df[lag_cols_90].min(axis=1)
        df["rolling_max_90_proxy"] = df[lag_cols_90].max(axis=1)
        df["distance_from_roll_mean_30"] = p - df["modal_price_roll_mean_30"]
        df["distance_from_roll_mean_7"]  = p - df["modal_price_roll_mean_7"]
        range_30 = df["rolling_max_30_proxy"] - df["rolling_min_30_proxy"]
        range_90 = df["rolling_max_90_proxy"] - df["rolling_min_90_proxy"]
        df["position_in_30d_range"] = (p - df["rolling_min_30_proxy"]) / range_30.replace(0, np.nan)
        df["position_in_90d_range"] = (p - df["rolling_min_90_proxy"]) / range_90.replace(0, np.nan)
        dt = pd.to_datetime(df["Arrival_Date"])
        df["quarter"] = dt.dt.quarter
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int).values
        df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)
        comm_date_med = df.groupby(["Commodity", df["Arrival_Date"].dt.date])["Modal_Price"].transform("median")
        df["relative_to_comm_date"] = p / comm_date_med.replace(0, np.nan)
        mkt_date_med = df.groupby(["Market", df["Arrival_Date"].dt.date])["Modal_Price"].transform("median")
        df["relative_to_market_date"] = p / mkt_date_med.replace(0, np.nan)
        df["hist_comm_median"] = df.groupby("Commodity")["Modal_Price"].transform("median")
        df["hist_comm_std"]    = df.groupby("Commodity")["Modal_Price"].transform("std")
        df["hist_market_median"] = df.groupby("Market")["Modal_Price"].transform("median")
        df["hist_market_std"]    = df.groupby("Market")["Modal_Price"].transform("std")
        df["price_vs_comm_hist"]   = p / df["hist_comm_median"].replace(0, np.nan)
        df["price_vs_market_hist"] = p / df["hist_market_median"].replace(0, np.nan)
        return df

    return add_features(df_train), add_features(df_val)

def train_eval_xgb(X_tr, y_tr, X_vl, y_vl, curr_vl, params):
    t_start = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    train_time = time.time() - t_start
    preds = model.predict(X_vl)
    metrics = evaluate(y_vl, preds, curr_vl)
    metrics["train_time"] = round(train_time, 1)
    metrics["best_iteration"] = model.best_iteration
    return metrics, model

results = []

# --- Task 1: Diagnose 30d instability ---
print("--- TASK 1: Diagnose 30d instability ---")
df_t, df_v = load_and_featurize(500000, 100000, "target_30d")
feats_E = feature_sets["E"]
X_tr = df_t[feats_E].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_vl = df_v[feats_E].replace([np.inf, -np.inf], np.nan).astype(np.float32)
y_tr = df_t["target_30d"].values.astype(np.float32)
y_vl = df_v["target_30d"].values.astype(np.float32)
curr_vl = df_v["Modal_Price"].values.astype(np.float32)

best_obj = None
best_obj_mae = float('inf')

for obj in ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]:
    params = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective=obj, eval_metric="mae", early_stopping_rounds=50, n_jobs=-1, random_state=42)
    m, _ = train_eval_xgb(X_tr, y_tr, X_vl, y_vl, curr_vl, params)
    print(f"Obj: {obj} | MAE: {m['MAE']:.1f} | Best Iter: {m['best_iteration']}")
    results.append({"task": "1_obj", "obj": obj, **m})
    if m["MAE"] < best_obj_mae:
        best_obj_mae = m["MAE"]
        best_obj = obj

del df_t, df_v, X_tr, X_vl, y_tr, y_vl, curr_vl
gc.collect()

# --- Task 2: Remove sampling as confounder ---
print(f"\n--- TASK 2: Sample Size Impact (Best Obj: {best_obj}) ---")
sizes = [(500000, 100000), (1000000, 200000), (3000000, 600000)] # 3M max practical for fast mem

for t_size, v_size in sizes:
    print(f"Evaluating sample size: Train={t_size}, Val={v_size}")
    df_t, df_v = load_and_featurize(t_size, v_size, "target_30d")
    X_tr = df_t[feats_E].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X_vl = df_v[feats_E].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y_tr = df_t["target_30d"].values.astype(np.float32)
    y_vl = df_v["target_30d"].values.astype(np.float32)
    curr_vl = df_v["Modal_Price"].values.astype(np.float32)
    
    params = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective=best_obj, eval_metric="mae", early_stopping_rounds=50, n_jobs=-1, random_state=42)
    m, _ = train_eval_xgb(X_tr, y_tr, X_vl, y_vl, curr_vl, params)
    print(f"Size: {t_size} | MAE: {m['MAE']:.1f}")
    results.append({"task": "2_size", "t_size": t_size, **m})
    
    del df_t, df_v, X_tr, X_vl, y_tr, y_vl, curr_vl
    gc.collect()

# --- Task 3 & 4: Full data + Tuning ---
print("\n--- TASK 3 & 4: Full Data Validation & Tuning ---")
tune_space = {
    "max_depth": [5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 1, 5]
}
sampler = list(ParameterSampler(tune_space, n_iter=5, random_state=42))

tasks = [("target_30d", "C"), ("target_60d", "E"), ("target_90d", "E")]
best_tuned = {}

for tgt, fset in tasks:
    print(f"\nTarget: {tgt}, Features: {fset}")
    df_t, df_v = load_and_featurize(3000000, 600000, tgt)
    feats = feature_sets[fset]
    X_tr = df_t[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X_vl = df_v[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y_tr = df_t[tgt].values.astype(np.float32)
    y_vl = df_v[tgt].values.astype(np.float32)
    curr_vl = df_v["Modal_Price"].values.astype(np.float32)
    
    # Baseline with best_obj
    params = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective=best_obj, eval_metric="mae", early_stopping_rounds=50, n_jobs=-1, random_state=42)
    m_base, _ = train_eval_xgb(X_tr, y_tr, X_vl, y_vl, curr_vl, params)
    print(f"Baseline MAE: {m_base['MAE']:.1f}")
    results.append({"task": f"3_full_{tgt}", "type": "baseline", **m_base})
    
    # Tuning
    best_tune_mae = float('inf')
    best_model = None
    for i, p in enumerate(sampler):
        p_full = dict(**p, n_estimators=1000, tree_method="hist", objective=best_obj, eval_metric="mae", early_stopping_rounds=50, n_jobs=-1, random_state=42)
        m, mod = train_eval_xgb(X_tr, y_tr, X_vl, y_vl, curr_vl, p_full)
        if m["MAE"] < best_tune_mae:
            best_tune_mae = m["MAE"]
            best_model = mod
            best_tuned[tgt] = p
            
    print(f"Tuned MAE: {best_tune_mae:.1f}")
    best_tuned_metrics = evaluate(y_vl, best_model.predict(X_vl), curr_vl)
    results.append({"task": f"4_tune_{tgt}", "type": "tuned", **best_tuned_metrics})
    best_model.save_model(os.path.join(MODEL_DIR, f"best_tuned_{tgt}.json"))
    
    del df_t, df_v, X_tr, X_vl, y_tr, y_vl, curr_vl
    gc.collect()

pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, "validation_metrics.csv"), index=False)
with open(os.path.join(RESULT_DIR, "tuning_results.json"), "w") as f:
    json.dump(best_tuned, f, indent=2)

print("\nDONE")
