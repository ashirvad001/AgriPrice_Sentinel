"""
tasks/retrain.py
────────────────
Celery task that retrains LSTM crop models every Sunday at 2 AM.

Pipeline per crop/mandi:
    1. Load 3 years of raw price data from PostgreSQL
    2. Run feature engineering (48 features)
    3. Build model with best hyperparameters from model_configs
    4. Train and evaluate on held-out test set
    5. Champion-challenger: promote only if RMSE improves > 2%
    6. Log retraining metadata to retraining_logs table
    7. Send Slack webhook notification
"""

import os
import time
import json
import traceback
import logging
from typing import Any
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
import joblib

from celery_app import app as celery_app
from feature_engineering import engineer_features
from forecast_model import build_hypermodel

# ── Sync SQLAlchemy for Celery workers (Celery is synchronous) ───────────────
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from dotenv import load_dotenv

load_dotenv()

_SYNC_DB_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
).replace("+asyncpg", "+psycopg2").replace("asyncpg://", "psycopg2://")

try:
    sync_engine = create_engine(_SYNC_DB_URL, echo=False)
    SyncSession = sessionmaker(bind=sync_engine)
except Exception:
    sync_engine = None
    SyncSession = None

logger = logging.getLogger("retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_CROPS = [
    "Wheat", "Rice", "Maize", "Bajra", "Jowar", "Ragi", "Barley", "Gram",
    "Tur", "Moong", "Urad", "Groundnut", "Soybean", "Mustard", "Cotton", "Sugarcane",
]
MANDIS = ["Lucknow Mandi", "Indore Mandi", "Amritsar Mandi"]

TARGET_CROP_MANDIS = [(crop, mandi) for crop in TARGET_CROPS for mandi in MANDIS]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
PROMOTION_THRESHOLD = 0.02  # 2% improvement required
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
LOOKBACK_YEARS = 3
SEQUENCE_LENGTH_DEFAULT = 60
OUTPUT_STEPS = 30


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_crop_data(session: Session, crop: str, mandi: str, years: int = 3) -> pd.DataFrame:
    """
    Load raw price data from PostgreSQL for the past `years` years.
    Reconstructs a DataFrame from the raw_data JSON column.
    """
    cutoff = date.today() - timedelta(days=years * 365)
    query = text("""
        SELECT fetch_date, raw_data
        FROM raw_prices
        WHERE LOWER(crop) = LOWER(:crop)
          AND raw_data->>'market_name' ILIKE :mandi
          AND fetch_date >= :cutoff
        ORDER BY fetch_date ASC
    """)
    rows = session.execute(query, {"crop": crop, "mandi": mandi, "cutoff": cutoff}).fetchall()

    if not rows:
        logger.warning(f"No data found for {crop} at {mandi} — generating synthetic data for demo")
        return _generate_synthetic_data(crop)

    records = []
    for fetch_date, raw_data in rows:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        record = {
            "date": fetch_date,
            "modal_price": raw_data.get("modal_price", np.nan),
            "msp": raw_data.get("msp", 2000.0),
            "min_price": raw_data.get("min_price", np.nan),
            "max_price": raw_data.get("max_price", np.nan),
            "arrivals_tonnes": raw_data.get("arrivals_tonnes", np.nan),
            "rainfall_mm": raw_data.get("rainfall_mm", 0.0),
            "max_temp": raw_data.get("max_temp", 35.0),
            "min_temp": raw_data.get("min_temp", 20.0),
            "freight_index": raw_data.get("freight_index", 100.0),
            "futures_price": raw_data.get("futures_price", np.nan),
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records for {crop} at {mandi} (from {cutoff})")
    return df


def _generate_synthetic_data(crop: str, n: int = 1100) -> pd.DataFrame:
    """Generate synthetic data for demo/testing when no real data exists."""
    np.random.seed(hash(crop) % 2**31)
    dates = pd.date_range(end=date.today(), periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "modal_price": np.random.uniform(1500, 3500, n).cumsum() / np.arange(1, n + 1) + 2000,
        "msp": np.full(n, 2275.0),
        "min_price": np.random.uniform(1400, 2500, n),
        "max_price": np.random.uniform(2500, 3500, n),
        "arrivals_tonnes": np.random.uniform(50, 500, n),
        "rainfall_mm": np.random.uniform(0, 15, n),
        "max_temp": np.random.uniform(28, 45, n),
        "min_temp": np.random.uniform(12, 28, n),
        "freight_index": np.random.uniform(90, 120, n),
        "futures_price": np.random.uniform(1400, 2800, n),
    })


def _load_best_hyperparams(session: Session, crop: str) -> dict:
    """Load the best hyperparameters from model_configs table."""
    query = text("""
        SELECT lstm_units, dropout, learning_rate, sequence_length, batch_size
        FROM model_configs
        WHERE LOWER(crop) = LOWER(:crop)
        ORDER BY rmse ASC LIMIT 1
    """)
    row = session.execute(query, {"crop": crop}).fetchone()

    if row:
        return {
            "lstm_units": row[0], "dropout": row[1], "learning_rate": row[2],
            "sequence_length": row[3], "batch_size": row[4],
        }
    # Defaults if no tuned config exists
    return {
        "lstm_units": 128, "dropout": 0.2, "learning_rate": 1e-3,
        "sequence_length": SEQUENCE_LENGTH_DEFAULT, "batch_size": 32,
    }


def _create_sequences(data: np.ndarray, targets: np.ndarray, seq_len: int):
    """Sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_len - OUTPUT_STEPS + 1):
        X.append(data[i : i + seq_len])
        y.append(targets[i + seq_len : i + seq_len + OUTPUT_STEPS])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _get_deployed_rmse(crop: str, mandi: str) -> float | None:
    """Load the deployed model and return its RMSE, or None if no model exists."""
    rmse_path = os.path.join(MODELS_DIR, f"{crop.lower()}_{mandi.lower().replace(' ', '_')}_rmse.txt")
    if os.path.exists(rmse_path):
        with open(rmse_path, "r") as f:
            return float(f.read().strip())
    return None


def _save_model(model, scaler, crop: str, mandi: str, rmse: float):
    """Save a trained model, its scaler, and its RMSE to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    base_name = f"{crop.lower()}_{mandi.lower().replace(' ', '_')}"
    model_path = os.path.join(MODELS_DIR, f"{base_name}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{base_name}_scaler.pkl")
    rmse_path = os.path.join(MODELS_DIR, f"{base_name}_rmse.txt")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(rmse_path, "w") as f:
        f.write(str(rmse))
    logger.info(f"Model and scaler saved → {model_path} (RMSE: {rmse:.4f})")


def _compute_feature_importance_delta(old_path: str, new_model, X_sample: np.ndarray) -> dict:
    """
    Compute the change in feature importance between old and new models.
    Uses gradient-based attribution (simplified).
    """
    from shap_explainer import FEATURE_NAMES, CropShapExplainer

    try:
        new_explainer = CropShapExplainer(new_model, X_sample[:50])
        new_attr = new_explainer.explain(X_sample[:10], m_steps=10)
        new_imp = np.mean(np.abs(new_attr), axis=(0, 1))  # (n_features,)

        if os.path.exists(old_path):
            old_model = tf.keras.models.load_model(old_path, compile=False)
            old_explainer = CropShapExplainer(old_model, X_sample[:50])
            old_attr = old_explainer.explain(X_sample[:10], m_steps=10)
            old_imp = np.mean(np.abs(old_attr), axis=(0, 1))
            delta = (new_imp - old_imp).tolist()
        else:
            delta = new_imp.tolist()

        n = min(len(FEATURE_NAMES), len(delta))
        return {FEATURE_NAMES[i]: round(delta[i], 6) for i in range(n)}
    except Exception as e:
        logger.warning(f"Feature importance delta skipped: {e}")
        return {}


def _log_retraining(session: Session, log_data: dict):
    """Insert a retraining log record into PostgreSQL."""
    query = text("""
        INSERT INTO retraining_logs
            (crop, mandi, started_at, finished_at, duration_seconds,
             rmse_before, rmse_after, improvement_pct,
             model_promoted, feature_importance_delta, status, error_message)
        VALUES
            (:crop, :mandi, :started_at, :finished_at, :duration_seconds,
             :rmse_before, :rmse_after, :improvement_pct,
             :model_promoted, :feature_importance_delta, :status, :error_message)
    """)
    session.execute(query, {
        **log_data,
        "feature_importance_delta": json.dumps(log_data.get("feature_importance_delta", {})),
    })
    session.commit()


def _send_slack_notification(message: str):
    """Send a Slack webhook notification."""
    if not SLACK_WEBHOOK_URL:
        logger.info(f"Slack (no webhook configured): {message}")
        return

    try:
        import urllib.request
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Slack notification sent")
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE RETRAINING FUNCTION (per crop and mandi)
# ═══════════════════════════════════════════════════════════════════════════════

def retrain_single_crop(crop: str, mandi: str, session: Session) -> dict:
    """
    Full retraining pipeline for a single crop and mandi.

    Returns a dict with retraining metadata.
    """
    start_time = datetime.utcnow()
    t0 = time.time()
    logger.info(f"{'='*50}")
    logger.info(f"RETRAINING: {crop} at {mandi}")
    logger.info(f"{'='*50}")

    log_data: dict[str, Any] = {
        "crop": crop,
        "mandi": mandi,
        "started_at": start_time,
        "finished_at": None,
        "duration_seconds": None,
        "rmse_before": None,
        "rmse_after": None,
        "improvement_pct": None,
        "model_promoted": False,
        "feature_importance_delta": {},
        "status": "running",
        "error_message": None,
    }

    try:
        # ── 1. Load data ─────────────────────────────────────────────────
        df = _load_crop_data(session, crop, mandi, years=LOOKBACK_YEARS)
        logger.info(f"  Data loaded: {len(df)} rows")

        # ── 2. Feature engineering ───────────────────────────────────────
        features_df = engineer_features(df)
        logger.info(f"  Features engineered: {features_df.shape}")

        if len(features_df) < 200:
            raise ValueError(f"Insufficient data after feature engineering: {len(features_df)} rows")

        # ── 3. Load best hyperparameters ─────────────────────────────────
        best_hp = _load_best_hyperparams(session, crop)
        seq_len = best_hp["sequence_length"]
        batch_size = best_hp["batch_size"]
        logger.info(f"  Hyperparams: units={best_hp['lstm_units']}, "
                     f"dropout={best_hp['dropout']}, lr={best_hp['learning_rate']}, "
                     f"seq={seq_len}, bs={batch_size}")

        # ── 4. Prepare train/test split ──────────────────────────────────
        values = features_df.values.astype(np.float32)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        # Target = modal_price (column 0)
        target_col = 0
        targets = scaled[:, target_col]
        # Expand targets to multi-step
        multi_targets = np.column_stack([
            np.roll(targets, -i) for i in range(OUTPUT_STEPS)
        ])[: -(OUTPUT_STEPS - 1)]
        scaled_trimmed = scaled[: -(OUTPUT_STEPS - 1)]

        X, y = _create_sequences(scaled_trimmed, multi_targets, seq_len)
        logger.info(f"  Sequences: X={X.shape}, y={y.shape}")

        # 80/20 temporal split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ── 5. Build model ───────────────────────────────────────────────
        hp = kt.HyperParameters()
        hp.Choice("lstm_units", [best_hp["lstm_units"]])
        hp.Choice("dropout", [best_hp["dropout"]])
        hp.Choice("learning_rate", [best_hp["learning_rate"]])

        model = build_hypermodel(hp, input_shape=(seq_len, features_df.shape[1]))
        logger.info(f"  Model built ({model.count_params():,} params)")

        # ── 6. Train ─────────────────────────────────────────────────────
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_rmse", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_rmse", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        logger.info("  Training complete")

        # ── 7. Evaluate ──────────────────────────────────────────────────
        eval_result = model.evaluate(X_test, y_test, verbose=0)
        new_rmse = eval_result[1] if len(eval_result) > 1 else eval_result[0]
        log_data["rmse_after"] = float(new_rmse)  # type: ignore[arg-type]
        logger.info(f"  New RMSE: {new_rmse:.4f}")

        # ── 8. Champion-Challenger ───────────────────────────────────────
        deployed_rmse = _get_deployed_rmse(crop, mandi)
        log_data["rmse_before"] = deployed_rmse

        if deployed_rmse is not None:
            improvement = (deployed_rmse - new_rmse) / deployed_rmse
            log_data["improvement_pct"] = round(float(improvement * 100), 2)
            logger.info(f"  Deployed RMSE: {deployed_rmse:.4f}, Improvement: {improvement*100:.1f}%")

            if improvement > PROMOTION_THRESHOLD:
                # ── Compute feature importance delta before overwriting ───
                old_path = os.path.join(MODELS_DIR, f"{crop.lower()}_{mandi.lower().replace(' ', '_')}_model.keras")
                fi_delta = _compute_feature_importance_delta(old_path, model, X_test)
                log_data["feature_importance_delta"] = fi_delta

                _save_model(model, scaler, crop, mandi, new_rmse)
                log_data["model_promoted"] = True
                logger.info(f"  ✅ MODEL PROMOTED (>{PROMOTION_THRESHOLD*100}% improvement)")
            else:
                logger.info(f"  ⏸  Not promoted (improvement {improvement*100:.1f}% <= {PROMOTION_THRESHOLD*100}%)")
        else:
            # No existing model → always save
            fi_delta = _compute_feature_importance_delta("", model, X_test)
            log_data["feature_importance_delta"] = fi_delta
            _save_model(model, scaler, crop, mandi, new_rmse)
            log_data["model_promoted"] = True
            log_data["improvement_pct"] = 100.0
            logger.info("  ✅ First model saved (no prior deployment)")

        log_data["status"] = "success"

    except Exception as e:
        log_data["status"] = "failed"
        log_data["error_message"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"  ❌ FAILED: {e}")

    finally:
        elapsed = time.time() - t0
        log_data["finished_at"] = datetime.utcnow()
        log_data["duration_seconds"] = round(float(elapsed), 1)
        logger.info(f"  Duration: {elapsed:.1f}s")

    return log_data


# ═══════════════════════════════════════════════════════════════════════════════
#  CELERY TASK — RETRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(name="tasks.retrain.retrain_all_models", bind=True, max_retries=1)
def retrain_all_models(self):
    """
    Celery task: retrain all LSTM crop+mandi models.

    Scheduled via celery-beat for every Sunday at 2:00 AM IST.
    Can also be triggered manually:
        from tasks.retrain import retrain_all_models
        retrain_all_models.delay()
    """
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info(f"  WEEKLY LSTM RETRAINING — {len(TARGET_CROP_MANDIS)} Crop-Mandi Combinations")
    logger.info(f"  Started at: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    results = {"promoted": [], "skipped": [], "failed": []}

    # Use sync session for Celery workers
    if SyncSession is None:
        logger.error("Database unavailable — running in demo mode with synthetic data")
        session = None
    else:
        session = SyncSession()

    try:
        for crop, mandi in TARGET_CROP_MANDIS:
            name = f"{crop} at {mandi}"
            try:
                log_data = retrain_single_crop(crop, mandi, session)

                # Log to PostgreSQL
                if session:
                    try:
                        _log_retraining(session, log_data)
                    except Exception as db_err:
                        logger.warning(f"DB log failed for {name}: {db_err}")

                # Categorize result
                if log_data["status"] == "failed":
                    results["failed"].append(name)
                elif log_data.get("model_promoted"):
                    results["promoted"].append(name)
                else:
                    results["skipped"].append(name)

            except Exception as e:
                logger.error(f"Unexpected error for {name}: {e}")
                results["failed"].append(name)

    finally:
        if session:
            session.close()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = round(float(time.time() - overall_start), 1)
    summary = (
        f"🌾 *Weekly LSTM Retraining Complete*\n"
        f"⏱ Duration: {elapsed_total}s\n"
        f"✅ Promoted ({len(results['promoted'])}): {', '.join(results['promoted']) or 'None'}\n"
        f"⏸ Skipped ({len(results['skipped'])}): {', '.join(results['skipped']) or 'None'}\n"
        f"❌ Failed ({len(results['failed'])}): {', '.join(results['failed']) or 'None'}"
    )
    logger.info("\n" + summary)
    _send_slack_notification(summary)

    return {
        "duration_seconds": elapsed_total,
        "promoted": results["promoted"],
        "skipped": results["skipped"],
        "failed": results["failed"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECT EXECUTION (for testing without Celery)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running retrain_all_models directly (no Celery)…")
    result = retrain_all_models()
    print(json.dumps(result, indent=2))
