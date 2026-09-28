"""
Evaluate BiLSTM v2 Ablation models against baselines.
"""

import sys
import os
import json
import time
import gc

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ml.dataset import create_dataloader
from app.ml.models_v2 import BiLSTM_v2
from app.ml.metrics import calculate_metrics

PARQUET   = "data/processed/forecast_dataset.parquet"
PREP_PATH = "data/processed/preprocessor.json"
CONF_PATH = "data/processed/split_config.json"
MODEL_DIR = "data/models"

MAX_EVAL_SAMPLES = 20_000


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(dataloader, max_samples: int):
    X_seq_list, Y_list, Mask_list, Cat_list = [], [], [], []
    count = 0
    
    for x, y, y_mask, cat in dataloader:
        X_seq_list.append(x.numpy())
        Y_list.append(y.numpy())
        Mask_list.append(y_mask.numpy())
        Cat_list.append(cat.numpy())
        
        count += len(x)
        if count >= max_samples:
            break
            
    X_seq = np.concatenate(X_seq_list, axis=0)[:max_samples]
    Y = np.concatenate(Y_list, axis=0)[:max_samples]
    Mask = np.concatenate(Mask_list, axis=0)[:max_samples]
    Cat = np.concatenate(Cat_list, axis=0)[:max_samples]
    
    return X_seq, Y, Mask, Cat


def main():
    t0 = time.perf_counter()
    device = get_device()
    
    print("=" * 60)
    print("  ABLATION MODEL EVALUATION (VALIDATION SET)")
    print("=" * 60)
    
    print("[1/3] Loading Validation Data...")
    val_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="val", batch_size=2048)
    X_seq, Y, Mask, Cat = load_data(val_dl, MAX_EVAL_SAMPLES)
    
    print(f"  Samples: {len(Y):,}")
    
    with open(PREP_PATH) as f:
        prep = json.load(f)
    n_comm = prep['n_commodities']
    n_mkt = prep['n_markets']
    
    models_to_test = {
        "ablation_A": BiLSTM_v2(use_commodity_emb=False, use_market_emb=False, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_B": BiLSTM_v2(use_commodity_emb=True, use_market_emb=False, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_C": BiLSTM_v2(use_commodity_emb=True, use_market_emb=True, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_D": BiLSTM_v2(use_commodity_emb=True, use_market_emb=True, use_attention=True, n_commodities=n_comm, n_markets=n_mkt),
    }
    
    report_path = os.path.join(MODEL_DIR, "model_comparison_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {}

    print("\n[2/3] Evaluating Ablation Models...")
    
    x_t = torch.from_numpy(X_seq).to(device)
    cat_t = torch.from_numpy(Cat).to(device)
    
    for name, model_class in models_to_test.items():
        print(f"  -> {name}")
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        
        if not os.path.exists(path):
            print(f"     WARNING: {name} model not found.")
            continue
            
        model = model_class.to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            preds = model(x_t, cat_t).cpu().numpy()
            
        metrics = calculate_metrics(Y, preds, Mask)
        report[name] = {"30d": metrics[0], "60d": metrics[1], "90d": metrics[2]}
        
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    print("\n[3/3] Generating Report...")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    for h in ["30d", "60d", "90d"]:
        print(f"\n--- Horizon: {h} ---")
        best_mae = float('inf')
        best_model = None
        for m_name, m_res in report.items():
            if m_res and m_res.get(h) and m_res[h]["MAE"] is not None:
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
