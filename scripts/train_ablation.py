"""
Train BiLSTM v2 Ablation Models.
"""

import sys
import os
import json
import time
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ml.dataset import create_dataloader
from app.ml.models_v2 import BiLSTM_v2

PARQUET   = "data/processed/forecast_dataset.parquet"
PREP_PATH = "data/processed/preprocessor.json"
CONF_PATH = "data/processed/split_config.json"
MODEL_DIR = "data/models"

BATCH_SIZE = 512
EPOCHS = 5
PATIENCE = 3
# To make it finish in a reasonable time, limit batches per epoch
BATCHES_PER_EPOCH_TRAIN = 50
BATCHES_PER_EPOCH_VAL   = 10


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    
    for i, (x, y, y_mask, cat) in enumerate(dataloader):
        if i >= BATCHES_PER_EPOCH_TRAIN:
            break
            
        x, y, y_mask, cat = x.to(device), y.to(device), y_mask.to(device), cat.to(device)
        
        optimizer.zero_grad()
        preds = model(x, cat)
        
        loss = criterion(preds, y)
        loss = (loss * y_mask).sum() / (y_mask.sum() + 1e-8)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / max(count, 1)


@torch.no_grad()
def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    
    for i, (x, y, y_mask, cat) in enumerate(dataloader):
        if i >= BATCHES_PER_EPOCH_VAL:
            break
            
        x, y, y_mask, cat = x.to(device), y.to(device), y_mask.to(device), cat.to(device)
            
        preds = model(x, cat)
        
        loss = criterion(preds, y)
        loss = (loss * y_mask).sum() / (y_mask.sum() + 1e-8)
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / max(count, 1)


def train_model(model_name: str, model: nn.Module, train_dl, val_dl, device):
    print(f"\n{'='*40}")
    print(f" Training: {model_name}")
    print(f"{'='*40}")
    
    model = model.to(device)
    
    # Huber Loss
    criterion = nn.SmoothL1Loss(reduction='none')
    # AdamW Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # LR Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss = val_epoch(model, val_dl, criterion, device)
        
        scheduler.step(val_loss)
        
        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            path = os.path.join(MODEL_DIR, f"{model_name}.pt")
            torch.save(model.state_dict(), path)
            print(f"    -> Saved new best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping triggered after {epoch+1} epochs.")
                break
                
    print(f"Finished {model_name}. Best Val Loss: {best_val_loss:.2f}")
    
    # Cleanup memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.perf_counter()
    device = get_device()
    print(f"Using device: {device}")
    
    print("[1/2] Initializing DataLoaders...")
    train_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="train", batch_size=BATCH_SIZE)
    val_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="val", batch_size=BATCH_SIZE)
    
    with open(PREP_PATH) as f:
        prep = json.load(f)
    n_comm = prep['n_commodities']
    n_mkt = prep['n_markets']
    
    models_to_train = {
        "ablation_A": BiLSTM_v2(use_commodity_emb=False, use_market_emb=False, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_B": BiLSTM_v2(use_commodity_emb=True, use_market_emb=False, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_C": BiLSTM_v2(use_commodity_emb=True, use_market_emb=True, use_attention=False, n_commodities=n_comm, n_markets=n_mkt),
        "ablation_D": BiLSTM_v2(use_commodity_emb=True, use_market_emb=True, use_attention=True, n_commodities=n_comm, n_markets=n_mkt),
    }
    
    print("\n[2/2] Training Ablation Models...")
    for name, model in models_to_train.items():
        train_model(name, model, train_dl, val_dl, device)
    
    print(f"\nDone. Total Time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
