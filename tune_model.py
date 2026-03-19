import os
import asyncio
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit
from database import AsyncSessionLocal, ModelConfig, init_db
from forecast_model import build_hypermodel
from datetime import datetime

# Disable GPU/OneDNN info logs to keep output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def prepare_windows(data, targets, seq_len):
    """
    Numpy-based sliding window generation for speed.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(targets[i+seq_len-1]) # Forecast starts after the sequence
    return np.array(X), np.array(y)

class TSCrossValidationTuner(kt.RandomSearch):
    """
    Custom Keras Tuner that implements Time-Series Cross-Validation.
    Uses numpy data for maximum speed during search.
    """
    def __init__(self, data_x, data_y, n_splits=3, **kwargs):
        super().__init__(**kwargs)
        self.data_x = data_x
        self.data_y = data_y
        self.n_splits = n_splits
        
    def run_trial(self, trial, *args, **kwargs):
        """Overrides run_trial to perform 3-fold TS CV."""
        hp = trial.hyperparameters
        seq_len = hp.Choice('sequence_length', [30, 60, 90])
        batch_size = hp.Choice('batch_size', [16, 32, 64])
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        val_rmses = []
        
        for train_idx, val_idx in tscv.split(self.data_x):
            # Split original data
            X_train_raw, X_val_raw = self.data_x[train_idx], self.data_x[val_idx]
            y_train_raw, y_val_raw = self.data_y[train_idx], self.data_y[val_idx]
            
            # Skip if not enough data for windows
            if len(X_train_raw) <= seq_len or len(X_val_raw) <= seq_len:
                continue
                
            # Prepare windows for this specific split
            X_train, y_train = prepare_windows(X_train_raw, y_train_raw, seq_len)
            X_val, y_val = prepare_windows(X_val_raw, y_val_raw, seq_len)
            
            # Step-specific model build
            model = self.hypermodel.build(hp)
            
            # Fit with minimal epochs for search
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=1, # Single-epoch search for extreme speed in demonstration.
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=1)]
            )
            
            if 'val_rmse' in history.history:
                val_rmses.append(min(history.history['val_rmse']))
                
        if val_rmses:
            avg_rmse = np.mean(val_rmses)
            print(f"Trial {trial.trial_id} | Avg Val RMSE: {avg_rmse:.4f}")
            self.oracle.update_trial(trial.trial_id, {'val_rmse': avg_rmse})
            # No need for save_trial, update_trial handles it.
        else:
            self.oracle.update_trial(trial.trial_id, {'val_rmse': float('inf')})
        
        # Mark trial as finished
        trial.status = "COMPLETED"


async def save_best_config(crop_name, best_hp, best_rmse):
    """Saves the best tuning configuration to PostgreSQL"""
    async with AsyncSessionLocal() as session:
        config = ModelConfig(
            crop=crop_name,
            lstm_units=best_hp.get('lstm_units'),
            dropout=best_hp.get('dropout'),
            learning_rate=best_hp.get('learning_rate'),
            sequence_length=best_hp.get('sequence_length'),
            batch_size=best_hp.get('batch_size'),
            rmse=float(best_rmse),
            optimized_at=datetime.utcnow()
        )
        session.add(config)
        await session.commit()
        print(f"✅ Saved best config for {crop_name} to DB (RMSE: {best_rmse:.4f}).")

async def main():
    try:
        await init_db()
    except Exception as e:
        print(f"DB Offline: {e}")
        
    num_samples, num_features = 400, 48
    X_data = np.random.randn(num_samples, num_features).astype(np.float32)
    Y_data = np.random.randn(num_samples, 30).astype(np.float32)
    
    tuner = TSCrossValidationTuner(
        data_x=X_data, data_y=Y_data, n_splits=2,
        hypermodel=build_hypermodel,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials=10,
        directory='kt_tuning', project_name='crop_price_optimization',
        overwrite=True
    )
    
    print("🚀 Starting 50-trial RandomSearch (Accelerated with NumPy)...")
    tuner.search()
    
    best_trials = tuner.oracle.get_best_trials(num_trials=5)
    print("\n🏆 Top 5 Configurations:")
    for i, t in enumerate(best_trials):
        hp = t.hyperparameters
        print(f"{i+1}. Trial {t.trial_id} | RMSE: {t.score:.4f} | Units: {hp.get('lstm_units')} | DO: {hp.get('dropout')} | Seq: {hp.get('sequence_length')}")
        
    if best_trials:
        try:
            await save_best_config("Wheat", best_trials[0].hyperparameters, best_trials[0].score)
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
