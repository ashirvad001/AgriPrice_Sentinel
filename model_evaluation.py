"""
model_evaluation.py
━━━━━━━━━━━━━━━━━━━
Comprehensive model evaluation script for AgriPrice Sentinel MSP Forecaster.

Compares 5 models × 16 crops × 3 mandis × 3 horizons = 720 experiment runs.
Models : ARIMA(5,1,2), SARIMA, Facebook Prophet, vanilla LSTM, BiLSTM+Attention.
Metrics: RMSE, MAE, MAPE, Directional Accuracy, Prediction Interval Coverage (95% CI).
Outputs: LaTeX table, RMSE heatmap, baseline improvement chart, PostgreSQL persistence.

Usage:
    python model_evaluation.py                     # Run all experiments
    python model_evaluation.py --crops Wheat Rice  # Run for specific crops
    python model_evaluation.py --dry-run           # Preview experiment grid
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from itertools import product
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_eval")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 16 crops (MSP values for 2025-26 in ₹/quintal)
CROPS = {
    "Wheat": 2275, "Rice": 2320, "Maize": 2090, "Bajra": 2625,
    "Jowar": 3371, "Ragi": 3846, "Barley": 1850, "Gram": 5440,
    "Tur": 7000, "Moong": 8558, "Urad": 6950, "Groundnut": 6377,
    "Soybean": 4600, "Mustard": 5650, "Cotton": 7020, "Sugarcane": 315,
}

# 3 representative mandis across different states
MANDIS = ["Lucknow Mandi", "Indore Mandi", "Amritsar Mandi"]

# 3 forecast horizons
HORIZONS = [30, 60, 90]

# 5 models + 1 challenger (TFT)
# Note: TFT requires GPU for full 720 runs; add --models TFT flag for selective testing
MODEL_NAMES = [
    "ARIMA(5,1,2)",
    "SARIMA",
    "Prophet",
    "Vanilla LSTM",
    "BiLSTM+Attention",
    "TFT",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentResult:
    """Single experiment run result."""
    crop: str
    mandi: str
    horizon: int
    model_name: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    pi_coverage_95: float
    training_time_s: float
    n_train_samples: int
    n_test_samples: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading & Preparation
# ─────────────────────────────────────────────────────────────────────────────
def load_price_data(crop: str, mandi: str) -> pd.DataFrame:
    """
    Load historical price data from the PostgreSQL database.
    Falls back to synthetic data if the database is unavailable.
    """
    try:
        import asyncio
        from sqlalchemy import select, text
        from database import AsyncSessionLocal, RawPrice

        async def _fetch():
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    text("""
                        SELECT fetch_date AS date,
                               (raw_data->>'modal_price')::float AS modal_price,
                               (raw_data->>'min_price')::float   AS min_price,
                               (raw_data->>'max_price')::float   AS max_price,
                               (raw_data->>'arrivals_tonnes')::float AS arrivals_tonnes
                        FROM raw_prices
                        WHERE crop = :crop
                          AND raw_data->>'mandi' = :mandi
                        ORDER BY fetch_date
                    """),
                    {"crop": crop, "mandi": mandi}
                )
                rows = result.fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=["date", "modal_price", "min_price",
                                                        "max_price", "arrivals_tonnes"])
                return None

        df = asyncio.run(_fetch())
        if df is not None and len(df) >= 120:
            logger.info(f"  Loaded {len(df)} records from database for {crop}@{mandi}")
            return df
    except Exception as e:
        logger.debug(f"  DB unavailable ({e}), using synthetic data")

    # Fallback: generate synthetic price data with realistic patterns
    return _generate_synthetic_data(crop, mandi)


def _generate_synthetic_data(crop: str, mandi: str) -> pd.DataFrame:
    """Generate 2 years of synthetic crop price data with seasonal patterns."""
    np.random.seed(hash(f"{crop}_{mandi}") % 2**31)
    n_days = 730  # 2 years

    msp = CROPS.get(crop, 2000)
    base_price = msp * np.random.uniform(0.85, 1.15)
    dates = pd.date_range(end=datetime.now().date(), periods=n_days, freq="D")

    # Seasonal component (yearly cycle)
    seasonal = np.sin(2 * np.pi * np.arange(n_days) / 365) * base_price * 0.08
    # Trend component
    trend = np.linspace(0, base_price * 0.05, n_days)
    # Random walk component
    noise = np.cumsum(np.random.normal(0, base_price * 0.005, n_days))

    prices = base_price + seasonal + trend + noise
    prices = np.maximum(prices, base_price * 0.5)  # floor

    spread = np.abs(np.random.normal(0, base_price * 0.03, n_days))

    return pd.DataFrame({
        "date": dates,
        "modal_price": np.round(prices, 2),
        "min_price": np.round(prices - spread, 2),
        "max_price": np.round(prices + spread, 2),
        "arrivals_tonnes": np.round(np.abs(np.random.normal(200, 80, n_days)), 1),
    })


def prepare_train_test(
    df: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Split data into train/test, scaled for neural networks."""
    prices = df["modal_price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices).flatten()

    test_size = max(horizon, int(len(scaled) * 0.15))
    train, test = scaled[:-test_size], scaled[-test_size:]

    return train, test, prices[:-test_size].flatten(), prices[-test_size:].flatten(), scaler


# ─────────────────────────────────────────────────────────────────────────────
#  Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all 5 evaluation metrics."""
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    # RMSE
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    # MAE
    mae = float(mean_absolute_error(actual, predicted))

    # MAPE (avoid division by zero)
    nonzero = actual != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
    else:
        mape = 0.0

    # Directional Accuracy (correct sign of day-over-day change)
    if n > 1:
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = 50.0

    # Prediction Interval Coverage (95% CI)
    if lower is not None and upper is not None:
        lower, upper = lower[:n], upper[:n]
        coverage = float(np.mean((actual >= lower) & (actual <= upper)) * 100)
    else:
        coverage = 0.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(dir_acc, 2),
        "pi_coverage_95": round(coverage, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model Implementations
# ─────────────────────────────────────────────────────────────────────────────
def run_arima(train_raw: np.ndarray, test_raw: np.ndarray, horizon: int) -> dict:
    """ARIMA(5,1,2) model."""
    from statsmodels.tsa.arima.model import ARIMA

    predictions, lowers, uppers = [], [], []
    history = list(train_raw)

    for i in range(min(horizon, len(test_raw))):
        try:
            model = ARIMA(history, order=(5, 1, 2))
            fit = model.fit()
            fc = fit.get_forecast(steps=1)
            yhat = fc.predicted_mean[0]
            ci = fc.conf_int(alpha=0.05)
            predictions.append(yhat)
            lowers.append(ci[0, 0])
            uppers.append(ci[0, 1])
        except Exception:
            predictions.append(history[-1])
            lowers.append(history[-1] * 0.95)
            uppers.append(history[-1] * 1.05)
        history.append(test_raw[i])

    return {
        "predicted": np.array(predictions),
        "lower": np.array(lowers),
        "upper": np.array(uppers),
    }


def run_sarima(train_raw: np.ndarray, test_raw: np.ndarray, horizon: int) -> dict:
    """SARIMA model with seasonal order (1,1,1,12)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    predictions, lowers, uppers = [], [], []
    history = list(train_raw)

    for i in range(min(horizon, len(test_raw))):
        try:
            model = SARIMAX(history, order=(5, 1, 2), seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=50)
            fc = fit.get_forecast(steps=1)
            yhat = fc.predicted_mean[0]
            ci = fc.conf_int(alpha=0.05)
            predictions.append(yhat)
            lowers.append(ci.iloc[0, 0])
            uppers.append(ci.iloc[0, 1])
        except Exception:
            predictions.append(history[-1])
            lowers.append(history[-1] * 0.95)
            uppers.append(history[-1] * 1.05)
        history.append(test_raw[i])

    return {
        "predicted": np.array(predictions),
        "lower": np.array(lowers),
        "upper": np.array(uppers),
    }


def run_prophet(train_raw: np.ndarray, test_raw: np.ndarray, horizon: int,
                dates: pd.DatetimeIndex) -> dict:
    """Facebook Prophet model."""
    from prophet import Prophet

    # Prepare dataframe for Prophet
    train_dates = dates[:len(train_raw)]
    prophet_df = pd.DataFrame({"ds": train_dates, "y": train_raw})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    # Forecast
    n_predict = min(horizon, len(test_raw))
    future = model.make_future_dataframe(periods=n_predict)
    forecast = model.predict(future)

    # Extract predictions for test period
    preds = forecast.tail(n_predict)

    return {
        "predicted": preds["yhat"].values,
        "lower": preds["yhat_lower"].values,
        "upper": preds["yhat_upper"].values,
    }


def run_vanilla_lstm(
    train_scaled: np.ndarray,
    test_raw: np.ndarray,
    horizon: int,
    scaler: MinMaxScaler,
    seq_length: int = 60,
) -> dict:
    """Vanilla LSTM model (single-layer, no attention)."""
    import tensorflow as tf

    # Build sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train = X_train.reshape(-1, seq_length, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(seq_length, 1), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    # Rolling prediction
    n_predict = min(horizon, len(test_raw))
    combined = np.concatenate([train_scaled, scaler.transform(test_raw.reshape(-1, 1)).flatten()])
    predictions = []

    for i in range(n_predict):
        offset = len(train_scaled) + i
        seq = combined[offset - seq_length:offset].reshape(1, seq_length, 1)
        pred_scaled = model.predict(seq, verbose=0)[0, 0]
        predictions.append(pred_scaled)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Simple CI using rolling std
    std_estimate = np.std(test_raw[:n_predict] - predictions[:n_predict]) if n_predict > 1 else predictions.mean() * 0.05
    lower = predictions - 1.96 * std_estimate
    upper = predictions + 1.96 * std_estimate

    return {"predicted": predictions, "lower": lower, "upper": upper}


def run_bilstm_attention(
    train_scaled: np.ndarray,
    test_raw: np.ndarray,
    horizon: int,
    scaler: MinMaxScaler,
    seq_length: int = 60,
) -> dict:
    """BiLSTM + Bahdanau Attention with MC Dropout (project's primary model)."""
    from forecast_model import MCDropout, BahdanauAttention
    import tensorflow as tf

    # Build sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train = X_train.reshape(-1, seq_length, 1)

    # Simplified BiLSTM+Attention
    inputs = tf.keras.layers.Input(shape=(seq_length, 1))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inputs)
    x = MCDropout(0.2)(x)

    lstm_out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)
    )(x)
    lstm_out = MCDropout(0.2)(lstm_out)

    state_h = tf.keras.layers.Concatenate()([fh, bh])
    context = BahdanauAttention(32)(lstm_out, state_h)
    combined = tf.keras.layers.Concatenate()([context, state_h])

    x = tf.keras.layers.Dense(64, activation="relu")(combined)
    x = MCDropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(delta=1.0))
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    # MC Dropout predictions (30 passes for speed)
    n_predict = min(horizon, len(test_raw))
    combined_data = np.concatenate([train_scaled, scaler.transform(test_raw.reshape(-1, 1)).flatten()])

    mc_preds_all = []
    for _ in range(30):
        preds = []
        for i in range(n_predict):
            offset = len(train_scaled) + i
            seq = combined_data[offset - seq_length:offset].reshape(1, seq_length, 1)
            pred = model(seq, training=True).numpy()[0, 0]
            preds.append(pred)
        mc_preds_all.append(preds)

    mc_preds = np.array(mc_preds_all)
    mean_pred = np.mean(mc_preds, axis=0)
    std_pred = np.std(mc_preds, axis=0)

    predictions = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    lower = scaler.inverse_transform((mean_pred - 1.96 * std_pred).reshape(-1, 1)).flatten()
    upper = scaler.inverse_transform((mean_pred + 1.96 * std_pred).reshape(-1, 1)).flatten()

    return {"predicted": predictions, "lower": lower, "upper": upper}


def run_tft(train_raw: np.ndarray, test_raw: np.ndarray, horizon: int, dates: pd.DatetimeIndex) -> dict:
    """Temporal Fusion Transformer via pytorch-forecasting."""
    import warnings
    warnings.filterwarnings("ignore")
    
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    
    n_predict = min(horizon, len(test_raw))
    
    df_train = pd.DataFrame({
        "time_idx": np.arange(len(train_raw)),
        "modal_price": train_raw,
        "group": "single"
    })
    
    max_encoder_length = 60
    max_prediction_length = horizon
    
    # Must have enough samples
    if len(train_raw) < max_encoder_length + max_prediction_length:
        raise ValueError("Not enough data for TFT")
        
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="modal_price",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["modal_price"],
        time_varying_known_reals=[],
    )
    
    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = training.to_dataloader(train=False, batch_size=32, num_workers=0)
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    encoder_data = df_train.iloc[-max_encoder_length:].copy()
    decoder_data = pd.DataFrame({
        "time_idx": np.arange(len(train_raw), len(train_raw) + max_prediction_length),
        "modal_price": encoder_data["modal_price"].iloc[-1],
        "group": "single"
    })
    
    predict_df = pd.concat([encoder_data, decoder_data], ignore_index=True)
    predict_dataloader = TimeSeriesDataSet.from_dataset(training, predict_df, predict=True, stop_randomization=True).to_dataloader(train=False, batch_size=1, num_workers=0)
    
    preds = trainer.predict(tft, dataloaders=predict_dataloader)
    quantiles = preds[0][0]
    
    pred_p50 = quantiles[:, 1].numpy()[:n_predict]
    pred_p10 = quantiles[:, 0].numpy()[:n_predict]
    pred_p90 = quantiles[:, 2].numpy()[:n_predict]
    
    return {
        "predicted": pred_p50,
        "lower": pred_p10,
        "upper": pred_p90
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    crop: str, mandi: str, horizon: int, model_name: str
) -> ExperimentResult:
    """Run a single model evaluation experiment."""
    logger.info(f"  ▶ {model_name} | {crop} @ {mandi} | {horizon}d")

    df = load_price_data(crop, mandi)
    train_scaled, test_scaled, train_raw, test_raw, scaler = prepare_train_test(df, horizon)
    n_predict = min(horizon, len(test_raw))

    t0 = time.time()

    try:
        if model_name == "ARIMA(5,1,2)":
            result = run_arima(train_raw, test_raw, horizon)
        elif model_name == "SARIMA":
            result = run_sarima(train_raw, test_raw, horizon)
        elif model_name == "Prophet":
            result = run_prophet(train_raw, test_raw, horizon, df["date"])
        elif model_name == "Vanilla LSTM":
            result = run_vanilla_lstm(train_scaled, test_raw, horizon, scaler)
        elif model_name == "BiLSTM+Attention":
            result = run_bilstm_attention(train_scaled, test_raw, horizon, scaler)
        elif model_name == "TFT":
            result = run_tft(train_raw, test_raw, horizon, df["date"])
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"    ✗ {model_name} failed: {e}")
        return ExperimentResult(
            crop=crop, mandi=mandi, horizon=horizon, model_name=model_name,
            rmse=999.0, mae=999.0, mape=999.0, directional_accuracy=0.0,
            pi_coverage_95=0.0, training_time_s=time.time() - t0,
            n_train_samples=len(train_raw), n_test_samples=len(test_raw),
        )

    elapsed = time.time() - t0
    actual = test_raw[:n_predict]
    metrics = compute_metrics(
        actual, result["predicted"][:n_predict],
        result.get("lower"), result.get("upper"),
    )

    logger.info(f"    ✓ RMSE={metrics['rmse']:.1f} MAE={metrics['mae']:.1f} "
                f"MAPE={metrics['mape']:.1f}% DirAcc={metrics['directional_accuracy']:.1f}% "
                f"CI95={metrics['pi_coverage_95']:.1f}% ({elapsed:.1f}s)")

    return ExperimentResult(
        crop=crop, mandi=mandi, horizon=horizon, model_name=model_name,
        training_time_s=round(elapsed, 2),
        n_train_samples=len(train_raw), n_test_samples=len(test_raw),
        **metrics,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Output: LaTeX Table
# ─────────────────────────────────────────────────────────────────────────────
def generate_latex_table(results_df: pd.DataFrame) -> str:
    """Generate a LaTeX-formatted results table."""
    # Aggregate by model across all crops/mandis/horizons
    agg = results_df.groupby("model_name").agg({
        "rmse": "mean", "mae": "mean", "mape": "mean",
        "directional_accuracy": "mean", "pi_coverage_95": "mean",
        "training_time_s": "mean",
    }).round(2)

    agg = agg.reindex(MODEL_NAMES)

    # Find best (min for error metrics, max for accuracy)
    best_rmse = agg["rmse"].idxmin()
    best_mae = agg["mae"].idxmin()
    best_mape = agg["mape"].idxmin()
    best_dir = agg["directional_accuracy"].idxmax()
    best_ci = agg["pi_coverage_95"].idxmax()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model Comparison — Averaged Across 16 Crops $\times$ 3 Mandis $\times$ 3 Horizons}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RMSE (₹)} & \textbf{MAE (₹)} & \textbf{MAPE (\%)} "
        r"& \textbf{Dir. Acc. (\%)} & \textbf{95\% CI Cov. (\%)} \\",
        r"\midrule",
    ]

    for model_name in MODEL_NAMES:
        row = agg.loc[model_name]
        rmse_str = f"\\textbf{{{row['rmse']:.2f}}}" if model_name == best_rmse else f"{row['rmse']:.2f}"
        mae_str = f"\\textbf{{{row['mae']:.2f}}}" if model_name == best_mae else f"{row['mae']:.2f}"
        mape_str = f"\\textbf{{{row['mape']:.2f}}}" if model_name == best_mape else f"{row['mape']:.2f}"
        dir_str = f"\\textbf{{{row['directional_accuracy']:.1f}}}" if model_name == best_dir else f"{row['directional_accuracy']:.1f}"
        ci_str = f"\\textbf{{{row['pi_coverage_95']:.1f}}}" if model_name == best_ci else f"{row['pi_coverage_95']:.1f}"

        lines.append(f"  {model_name} & {rmse_str} & {mae_str} & {mape_str} & {dir_str} & {ci_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)

    # Save to file
    path = os.path.join(RESULTS_DIR, "model_comparison.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info(f"LaTeX table saved → {path}")

    return latex


# ─────────────────────────────────────────────────────────────────────────────
#  Output: RMSE Heatmap (Crop × Model)
# ─────────────────────────────────────────────────────────────────────────────
def generate_rmse_heatmap(results_df: pd.DataFrame) -> str:
    """Generate heatmap of average RMSE by crop × model."""
    pivot = results_df.pivot_table(
        values="rmse", index="crop", columns="model_name", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "RMSE (₹)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("RMSE by Crop × Model (Averaged Across Mandis & Horizons)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Crop", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "rmse_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"RMSE heatmap saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Output: Improvement Over ARIMA Baseline (Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────
def generate_improvement_chart(results_df: pd.DataFrame) -> str:
    """Generate bar chart showing RMSE improvement over ARIMA baseline."""
    # Average RMSE per model
    avg_rmse = results_df.groupby("model_name")["rmse"].mean()
    arima_rmse = avg_rmse.get("ARIMA(5,1,2)", avg_rmse.mean())

    other_models = [m for m in MODEL_NAMES if m != "ARIMA(5,1,2)"]
    improvements = [(arima_rmse - avg_rmse.get(m, 0)) / arima_rmse * 100 for m in other_models]

    colors = ["#ef4444" if imp < 0 else "#10b981" for imp in improvements]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(other_models, improvements, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        ypos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + (0.5 if ypos >= 0 else -1.5),
                f"{imp:+.1f}%", ha="center", va="bottom" if ypos >= 0 else "top",
                fontweight="bold", fontsize=12)

    ax.axhline(y=0, color="#64748b", linewidth=1, linestyle="--")
    ax.set_title("RMSE Improvement Over ARIMA(5,1,2) Baseline",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Improvement (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "improvement_over_arima.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Improvement chart saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Output: Additional Charts
# ─────────────────────────────────────────────────────────────────────────────
def generate_horizon_comparison(results_df: pd.DataFrame) -> str:
    """Generate grouped bar chart: RMSE by horizon for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = results_df.pivot_table(values="rmse", index="model_name", columns="horizon", aggfunc="mean")
    pivot = pivot.reindex(MODEL_NAMES)

    x = np.arange(len(MODEL_NAMES))
    width = 0.25
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    for i, h in enumerate(HORIZONS):
        if h in pivot.columns:
            ax.bar(x + i * width, pivot[h], width, label=f"{h}-day", color=colors[i], edgecolor="white")

    ax.set_title("RMSE by Model × Forecast Horizon", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("RMSE (₹)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODEL_NAMES, rotation=15, ha="right")
    ax.legend(title="Horizon")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "rmse_by_horizon.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Horizon comparison saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  PostgreSQL Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_to_postgres(results: list[ExperimentResult]) -> None:
    """Save all results to the 'model_evaluations' table."""
    import asyncio
    from sqlalchemy import text
    from database import engine

    async def _save():
        async with engine.begin() as conn:
            # Create table if not exists
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id SERIAL PRIMARY KEY,
                    crop VARCHAR(100) NOT NULL,
                    mandi VARCHAR(200) NOT NULL,
                    horizon INTEGER NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    rmse FLOAT NOT NULL,
                    mae FLOAT NOT NULL,
                    mape FLOAT NOT NULL,
                    directional_accuracy FLOAT NOT NULL,
                    pi_coverage_95 FLOAT NOT NULL,
                    training_time_s FLOAT,
                    n_train_samples INTEGER,
                    n_test_samples INTEGER,
                    evaluated_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Insert all results
            for r in results:
                await conn.execute(text("""
                    INSERT INTO model_evaluations
                        (crop, mandi, horizon, model_name, rmse, mae, mape,
                         directional_accuracy, pi_coverage_95, training_time_s,
                         n_train_samples, n_test_samples)
                    VALUES (:crop, :mandi, :horizon, :model_name, :rmse, :mae, :mape,
                            :directional_accuracy, :pi_coverage_95, :training_time_s,
                            :n_train_samples, :n_test_samples)
                """), {
                    "crop": r.crop, "mandi": r.mandi, "horizon": r.horizon,
                    "model_name": r.model_name, "rmse": r.rmse, "mae": r.mae,
                    "mape": r.mape, "directional_accuracy": r.directional_accuracy,
                    "pi_coverage_95": r.pi_coverage_95, "training_time_s": r.training_time_s,
                    "n_train_samples": r.n_train_samples, "n_test_samples": r.n_test_samples,
                })

        logger.info(f"✓ Saved {len(results)} results to PostgreSQL 'model_evaluations' table")

    try:
        asyncio.run(_save())
    except Exception as e:
        logger.warning(f"DB save failed ({e}), results still available in CSV")


# ─────────────────────────────────────────────────────────────────────────────
#  Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AgriPrice Sentinel — Model Evaluation")
    parser.add_argument("--crops", nargs="+", default=None, help="Specific crops to evaluate")
    parser.add_argument("--mandis", nargs="+", default=None, help="Specific mandis to evaluate")
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="Specific horizons")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to run")
    parser.add_argument("--dry-run", action="store_true", help="Preview experiment grid only")
    parser.add_argument("--no-db", action="store_true", help="Skip PostgreSQL persistence")
    args = parser.parse_args()

    crops = args.crops or list(CROPS.keys())
    mandis = args.mandis or MANDIS
    horizons = args.horizons or HORIZONS
    models = args.models or MODEL_NAMES

    experiments = list(product(crops, mandis, horizons, models))
    total = len(experiments)

    logger.info("=" * 70)
    logger.info(f"AgriPrice Sentinel — Model Evaluation Suite")
    logger.info(f"  Crops    : {len(crops)}")
    logger.info(f"  Mandis   : {len(mandis)}")
    logger.info(f"  Horizons : {horizons}")
    logger.info(f"  Models   : {len(models)}")
    logger.info(f"  Total    : {total} experiment runs")
    logger.info("=" * 70)

    if args.dry_run:
        for i, (c, m, h, mod) in enumerate(experiments[:20], 1):
            print(f"  [{i:3d}/{total}] {mod:20s} │ {c:12s} │ {m:16s} │ {h:2d}d")
        if total > 20:
            print(f"  ... and {total - 20} more")
        return

    # Run all experiments
    results: list[ExperimentResult] = []
    t_start = time.time()

    for i, (crop, mandi, horizon, model_name) in enumerate(experiments, 1):
        logger.info(f"[{i:3d}/{total}] ─────────────────────────────────────────")
        result = run_experiment(crop, mandi, horizon, model_name)
        results.append(result)

    elapsed_total = time.time() - t_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"All {total} experiments complete in {elapsed_total / 60:.1f} minutes")
    logger.info(f"{'=' * 70}")

    # Build results DataFrame
    results_df = pd.DataFrame([asdict(r) for r in results])

    # Save raw CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Raw results CSV → {csv_path}")

    # Generate outputs
    logger.info("\n📊 Generating outputs...")
    latex = generate_latex_table(results_df)
    print("\n" + latex + "\n")

    generate_rmse_heatmap(results_df)
    generate_improvement_chart(results_df)
    generate_horizon_comparison(results_df)

    # Save to PostgreSQL
    if not args.no_db:
        save_to_postgres(results)

    # Print summary
    logger.info("\n📋 Summary of outputs:")
    logger.info(f"  • CSV           → {csv_path}")
    logger.info(f"  • LaTeX table   → {RESULTS_DIR}/model_comparison.tex")
    logger.info(f"  • RMSE heatmap  → {RESULTS_DIR}/rmse_heatmap.png")
    logger.info(f"  • ARIMA baseline→ {RESULTS_DIR}/improvement_over_arima.png")
    logger.info(f"  • Horizon chart → {RESULTS_DIR}/rmse_by_horizon.png")
    logger.info(f"  • PostgreSQL    → model_evaluations table")


if __name__ == "__main__":
    main()
