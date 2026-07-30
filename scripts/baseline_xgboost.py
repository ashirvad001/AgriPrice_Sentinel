"""
scripts/baseline_xgboost.py
───────────────────────────
Trains an XGBoost model as a fast baseline to compare against the
existing BiLSTM model. This helps us quantify if the extra latency of the
BiLSTM (plus MC Dropout) is worth the accuracy gain over a fast tree-based model.
"""

import os
import time
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.feature_engineering import engineer_features
from app.tasks.retrain import _load_crop_data, _create_sequences
from app.database import SyncSession

def train_xgboost_baseline(crop: str, mandi: str):
    print(f"Training XGBoost Baseline for {crop} at {mandi}...")
    
    session = SyncSession()
    if session is None:
        print("Database not available.")
        return

    try:
        # Load 3 years of data
        df = _load_crop_data(session, crop, mandi, years=3)
        if len(df) < 100:
            print(f"Insufficient data for {crop}/{mandi}.")
            return
            
        features_df = engineer_features(df)
        if len(features_df) < 100:
            print("Insufficient data after feature engineering.")
            return

        values = features_df.values.astype(np.float32)
        
        # Target is modal_price (column 0)
        target_col = 0
        targets = values[:, target_col]
        
        # For XGBoost, we want to predict the next 30 days.
        # XGBoost doesn't natively do multi-output time-series easily like an LSTM,
        # so we will train it to predict just t+1, or we can use MultiOutputRegressor.
        # Let's use it for t+30 direct forecasting for simplicity.
        HORIZON = 30
        
        X, y = [], []
        # We don't need sequences for XGBoost, just the current row's features
        # to predict the price `HORIZON` days from now.
        for i in range(len(values) - HORIZON):
            X.append(values[i])
            y.append(targets[i + HORIZON])
            
        X = np.array(X)
        y = np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        t0 = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_test)
        inference_time = time.time() - t1
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        print(f"\n--- XGBoost Results ({HORIZON} day horizon) ---")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Training time:  {training_time:.2f} seconds")
        print(f"Inference time: {inference_time*1000:.2f} ms for {len(X_test)} samples")
        print(f"Inference/sample: {(inference_time/len(X_test))*1000:.3f} ms")
        
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=str, default="Wheat")
    parser.add_argument("--mandi", type=str, default="Indore Mandi")
    args = parser.parse_args()
    
    train_xgboost_baseline(args.crop, args.mandi)
