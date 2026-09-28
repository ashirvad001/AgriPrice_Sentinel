"""
Train baseline Neural Network models (LSTM and BiLSTM) for forecasting.
"""

import sys
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ml.dataset import create_dataloader
from app.ml.models import MandiLSTM, MandiBiLSTM

PARQUET   = "data/processed/forecast_dataset.parquet"
PREP_PATH = "data/processed/preprocessor.json"
CONF_PATH = "data/processed/split_config.json"
MODEL_DIR = "data/models"

# To keep training fast for the baseline, we will limit the number of batches per epoch
BATCHES_PER_EPOCH_TRAIN = 100
BATCHES_PER_EPOCH_VAL   = 20
EPOCHS = 5
BATCH_SIZE = 512

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    count = 0
    
    for i, (x, y, y_mask, cat) in enumerate(dataloader):
        if i >= BATCHES_PER_EPOCH_TRAIN:
            break
            
        optimizer.zero_grad()
        preds = model(x)
        
        # Huber loss weighted by target availability mask
        loss = criterion(preds, y)
        loss = (loss * y_mask).sum() / (y_mask.sum() + 1e-8)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / max(count, 1)


@torch.no_grad()
def val_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    
    for i, (x, y, y_mask, cat) in enumerate(dataloader):
        if i >= BATCHES_PER_EPOCH_VAL:
            break
            
        preds = model(x)
        loss = criterion(preds, y)
        loss = (loss * y_mask).sum() / (y_mask.sum() + 1e-8)
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / max(count, 1)


def train_model(model_name: str, model: nn.Module, train_dl, val_dl):
    print(f"\n--- Training {model_name} ---")
    
    # Huber Loss (Smooth L1) is robust to outliers
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_loss = val_epoch(model, val_dl, criterion)
        
        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(MODEL_DIR, f"{model_name}.pt")
            torch.save(model.state_dict(), path)
            print(f"    -> Saved new best model")
            
    print(f"Finished {model_name}. Best Val Loss: {best_val_loss:.2f}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.perf_counter()
    
    print("[1/2] Initializing DataLoaders...")
    train_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="train", batch_size=BATCH_SIZE)
    val_dl = create_dataloader(PARQUET, PREP_PATH, CONF_PATH, split="val", batch_size=BATCH_SIZE)
    
    # The dataset has 27 features. We don't embed the categorical IDs in the LSTM sequence directly
    # for this baseline to keep it simple, just using the 27 numerical/time features.
    input_dim = 27
    
    # Train Vanilla LSTM
    lstm = MandiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    train_model("lstm", lstm, train_dl, val_dl)
    
    # Train BiLSTM
    bilstm = MandiBiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    train_model("bilstm", bilstm, train_dl, val_dl)
    
    print(f"\n[2/2] Done. Total Time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
