"""
Train baseline XGBoost models for 30d, 60d, and 90d forecasting.
Uses tabular features from the LAST timestep of the 30-day sequence.
"""

import sys
import os
import json
import time

import torch
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ml.dataset import create_dataloader

PARQUET   = "data/processed/forecast_dataset.parquet"
PREP_PATH = "data/processed/preprocessor.json"
CONF_PATH = "data/processed/split_config.json"
MODEL_DIR = "data/models"

MAX_TRAIN_SAMPLES = 100_000
MAX_VAL_SAMPLES   = 20_000


def extract_tabular_data(dataloader, max_samples: int):
    """
    Extracts (X, Y, Y_mask) where X is [batch, features + 2 (cat ids)]
    from the last timestep of the sequence.
    """
    X_list, Y_list, Mask_list = [], [], []
    count = 0
    
    for x, y, y_mask, cat in dataloader:
        # x is [batch, seq_len, features]
        # We want the LAST timestep: [batch, features]
        x_last = x[:, -1, :].numpy()
        
        # Append categorical IDs to the end
        # cat is [batch, 2]
        c = cat.numpy()
        x_tab = np.concatenate([x_last, c], axis=1)
        
        X_list.append(x_tab)
        Y_list.append(y.numpy())
        Mask_list.append(y_mask.numpy())
        
        count += len(x)
        if count >= max_samples:
            break
            
    X = np.concatenate(X_list, axis=0)[:max_samples]
    Y = np.concatenate(Y_list, axis=0)[:max_samples]
    Mask = np.concatenate(Mask_list, axis=0)[:max_samples]
    
    return X, Y, Mask


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.perf_counter()
    
    print("[1/3] Loading data subsets for XGBoost...")
    train_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="train", batch_size=2048)
    val_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="val", batch_size=2048)
    
    X_train, Y_train, M_train = extract_tabular_data(train_dl, MAX_TRAIN_SAMPLES)
    X_val, Y_val, M_val = extract_tabular_data(val_dl, MAX_VAL_SAMPLES)
    
    print(f"  Train shape: {X_train.shape}, Targets: {Y_train.shape}")
    print(f"  Val shape  : {X_val.shape}, Targets: {Y_val.shape}")
    
    target_names = ["30d", "60d", "90d"]
    
    metrics = {}
    
    print("\n[2/3] Training XGBoost models (one per horizon)...")
    for idx, horizon in enumerate(target_names):
        print(f"  -> Horizon: {horizon}")
        
        # Filter where target is valid
        t_mask = M_train[:, idx] > 0
        v_mask = M_val[:, idx] > 0
        
        xt = X_train[t_mask]
        yt = Y_train[t_mask, idx]
        
        xv = X_val[v_mask]
        yv = Y_val[v_mask, idx]
        
        if len(yt) == 0:
            print(f"     No valid targets found for {horizon}. Skipping.")
            continue
            
        print(f"     Valid Train Samples: {len(yt):,} | Valid Val Samples: {len(yv):,}")
        
        # Note: We use reg:pseudohubererror to be robust to extreme outliers (prices can vary wildly)
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective="reg:pseudohubererror",
            tree_method="hist",
            early_stopping_rounds=10,
            eval_metric="mae"
        )
        
        model.fit(
            xt, yt,
            eval_set=[(xv, yv)],
            verbose=False
        )
        
        best_iter = model.best_iteration
        best_score = model.best_score
        print(f"     Best Iteration: {best_iter} | Val MAE: {best_score:.2f}")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"xgboost_{horizon}.json")
        model.save_model(model_path)
        
    print(f"\n[3/3] Done. Time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
