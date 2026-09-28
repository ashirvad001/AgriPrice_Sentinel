"""
Evaluate baseline models on the VALIDATION set and generate a comparison report.
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
from app.ml.models import MandiLSTM, MandiBiLSTM
from app.ml.metrics import calculate_metrics

PARQUET   = "data/processed/forecast_dataset.parquet"
PREP_PATH = "data/processed/preprocessor.json"
CONF_PATH = "data/processed/split_config.json"
MODEL_DIR = "data/models"

MAX_EVAL_SAMPLES = 20_000

def load_data(dataloader, max_samples: int):
    X_seq_list, X_tab_list, Y_list, Mask_list = [], [], [], []
    count = 0
    
    for x, y, y_mask, cat in dataloader:
        # Sequential X
        X_seq_list.append(x.numpy())
        
        # Tabular X
        x_last = x[:, -1, :].numpy()
        c = cat.numpy()
        x_tab = np.concatenate([x_last, c], axis=1)
        X_tab_list.append(x_tab)
        
        Y_list.append(y.numpy())
        Mask_list.append(y_mask.numpy())
        
        count += len(x)
        if count >= max_samples:
            break
            
    X_seq = np.concatenate(X_seq_list, axis=0)[:max_samples]
    X_tab = np.concatenate(X_tab_list, axis=0)[:max_samples]
    Y = np.concatenate(Y_list, axis=0)[:max_samples]
    Mask = np.concatenate(Mask_list, axis=0)[:max_samples]
    
    return X_seq, X_tab, Y, Mask


def main():
    t0 = time.perf_counter()
    print("=" * 60)
    print("  MODEL EVALUATION (VALIDATION SET)")
    print("=" * 60)
    
    print("[1/3] Loading Validation Data...")
    val_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="val", batch_size=2048)
    X_seq, X_tab, Y, Mask = load_data(val_dl, MAX_EVAL_SAMPLES)
    
    print(f"  Samples: {len(Y):,}")
    
    report = {}
    
    # ── 1. Persistence (Naive Baseline) ─────────────────────────────
    print("\n[2/3] Evaluating Models...")
    print("  -> Persistence (Naive Baseline)")
    # Persistence: predict that the future price equals the current price.
    # Current price is the first feature (index 0) in the last timestep.
    # Note: features are robust-scaled. To predict correctly, we just use the scaled current price.
    # Wait, the targets in Y are unscaled prices!
    # Let's check `dataset.py`. Ah, `Y` is NOT scaled. It's the raw target price!
    # But `X` IS scaled. So the current price in X is scaled.
    # To do persistence, we need the unscaled current price. 
    # Let's read preprocessor to unscale `Modal_Price` (col 0).
    with open(PREP_PATH) as f:
        prep = json.load(f)
    stats = prep["numeric_stats"]["Modal_Price"]
    median = stats["median"]
    iqr = stats["iqr"]
    
    current_price_scaled = X_tab[:, 0]
    current_price_unscaled = (current_price_scaled * iqr) + median
    
    # Persistence predicts current price for all horizons
    Y_pred_naive = np.column_stack([current_price_unscaled]*3)
    naive_metrics = calculate_metrics(Y, Y_pred_naive, Mask)
    report["persistence"] = {"30d": naive_metrics[0], "60d": naive_metrics[1], "90d": naive_metrics[2]}
    
    
    # ── 2. XGBoost ──────────────────────────────────────────────────
    print("  -> XGBoost")
    xgb_preds = np.zeros_like(Y)
    xgb_models_exist = True
    for idx, h in enumerate(["30d", "60d", "90d"]):
        path = os.path.join(MODEL_DIR, f"xgboost_{h}.json")
        if os.path.exists(path):
            model = xgb.XGBRegressor()
            model.load_model(path)
            xgb_preds[:, idx] = model.predict(X_tab)
        else:
            xgb_models_exist = False
            
    if xgb_models_exist:
        xgb_metrics = calculate_metrics(Y, xgb_preds, Mask)
        report["xgboost"] = {"30d": xgb_metrics[0], "60d": xgb_metrics[1], "90d": xgb_metrics[2]}
    else:
        print("     WARNING: XGBoost models not found.")
        

    # ── 3. LSTM & BiLSTM ─────────────────────────────────────────────
    input_dim = 27
    
    def eval_nn(name, model_class):
        print(f"  -> {name}")
        path = os.path.join(MODEL_DIR, f"{name.lower()}.pt")
        if not os.path.exists(path):
            print(f"     WARNING: {name} model not found.")
            return None
            
        model = model_class(input_dim=input_dim, hidden_dim=64, num_layers=2)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            x_t = torch.from_numpy(X_seq)
            preds = model(x_t).numpy()
            
        metrics = calculate_metrics(Y, preds, Mask)
        return {"30d": metrics[0], "60d": metrics[1], "90d": metrics[2]}

    report["lstm"] = eval_nn("LSTM", MandiLSTM)
    report["bilstm"] = eval_nn("BiLSTM", MandiBiLSTM)
    
    
    # ── 4. Save and Print Report ──────────────────────────────────────
    print("\n[3/3] Generating Report...")
    report_path = os.path.join(MODEL_DIR, "model_comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    for h in ["30d", "60d", "90d"]:
        print(f"\n--- Horizon: {h} ---")
        best_mae = float('inf')
        best_model = None
        for m_name, m_res in report.items():
            if m_res and m_res[h] and m_res[h]["MAE"] is not None:
                mae = m_res[h]["MAE"]
                print(f"  {m_name:12s} | MAE: {mae:8.1f} | sMAPE: {m_res[h]['sMAPE']:5.1f}% | R2: {m_res[h]['R2']:5.2f}")
                if mae < best_mae:
                    best_mae = mae
                    best_model = m_name
        print(f"  Best Model: {best_model} (MAE: {best_mae:.1f})")

    print(f"\nReport saved to {report_path}")
    print(f"Total Evaluation Time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
