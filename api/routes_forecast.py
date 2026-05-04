"""
api/routes_forecast.py
──────────────────────
Forecast endpoint with Redis caching and Sell/Hold recommendation.

Inference priority:
  1. Trained Keras model (MLflow registry → local saved_models/)
     with MC Dropout confidence intervals.
  2. Statistical baseline (linear trend + seasonal decomposition)
     when no model is available.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path

from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from api.schemas import ForecastResponse, ForecastDay
from database import RawPrice
from api.deps import cache_get, cache_set, get_db

import mlflow
import mlflow.keras

logger = logging.getLogger(__name__)

# ── MLflow tracking URI ──────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# ── Path helpers ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"


def _model_stem(crop: str, mandi: str) -> str:
    """Canonical file stem for a crop–mandi pair."""
    return f"{crop.lower()}_{mandi.lower().replace(' ', '_')}"


def load_crop_model(crop: str, mandi: str):
    """
    Tries MLflow model registry first, then falls back to local
    saved_models/{stem}_model.keras.  Returns None if nothing found.
    """
    # 1. MLflow registry
    try:
        model_uri = f"models:/CropPrice_{crop}_{mandi.replace(' ', '_')}/Production"
        return mlflow.keras.load_model(model_uri)
    except Exception:
        logger.info(f"MLflow load failed for {crop}/{mandi}, trying local fallback…")

    # 2. Local .keras file
    try:
        import tensorflow as tf
        local_path = SAVED_MODELS_DIR / f"{_model_stem(crop, mandi)}_model.keras"
        if local_path.exists():
            return tf.keras.models.load_model(str(local_path), compile=False)
    except Exception as exc:
        logger.warning(f"Local model load failed: {exc}")

    return None


def _load_scaler(crop: str, mandi: str):
    """Load the joblib scaler for a crop–mandi pair.  Returns None on failure."""
    import joblib
    path = SAVED_MODELS_DIR / f"{_model_stem(crop, mandi)}_scaler.pkl"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning(f"Scaler load failed ({path}): {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Fetch historical price rows from DB
# ─────────────────────────────────────────────────────────────────────────────
async def _fetch_price_rows(
    db: AsyncSession, crop: str, mandi: str, days: int = 365,
) -> list[dict]:
    """Return a chronological list of {date, modal_price, min_price, max_price, …}
    dicts from the raw_prices table."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(RawPrice)
        .where(and_(RawPrice.crop.ilike(crop), RawPrice.fetch_date >= cutoff))
        .order_by(desc(RawPrice.fetch_date))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    records: list[dict] = []
    for row in rows:
        raw = row.raw_data or {}
        row_mandi = raw.get("market_name", raw.get("mandi", raw.get("market", "")))
        if mandi.lower() not in row_mandi.lower() and row_mandi:
            continue
        records.append({
            "date": row.fetch_date,
            "modal_price": raw.get("modal_price"),
            "min_price": raw.get("min_price"),
            "max_price": raw.get("max_price"),
            "arrivals_tonnes": raw.get("arrivals_tonnes", 0),
            "rainfall_mm": raw.get("rainfall_mm", 0),
            "max_temp": raw.get("max_temp"),
            "min_temp": raw.get("min_temp"),
            "freight_index": raw.get("freight_index", 100),
            "futures_price": raw.get("futures_price"),
            "msp": raw.get("msp", 0),
        })

    # Return in chronological order
    records.sort(key=lambda r: r["date"])
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Model-based inference path
# ─────────────────────────────────────────────────────────────────────────────
def _run_model_inference(
    model,
    scaler,
    records: list[dict],
    horizon: int,
) -> tuple[list[float], list[float], list[float]] | None:
    """Build input sequence from DB records, scale, run MC Dropout.

    Returns (means, lowers, uppers) — each a list of length `horizon`,
    or None if there is insufficient data.
    """
    from forecast_model import get_mc_dropout_predictions
    from feature_engineering import engineer_features

    # Build a DataFrame the feature pipeline expects
    df = pd.DataFrame(records)
    # Fill missing columns with sensible defaults so engineer_features won't crash
    for col in ["msp", "arrivals_tonnes", "rainfall_mm", "max_temp",
                 "min_temp", "freight_index", "futures_price"]:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0.0)

    features_df = engineer_features(df)
    if features_df.empty or len(features_df) < 30:
        logger.warning("Not enough rows after feature engineering to build input sequence.")
        return None

    # Determine sequence_length from model input shape
    try:
        seq_len = model.input_shape[1]  # (batch, seq_len, features)
        if seq_len is None:
            seq_len = 60  # default if model accepts variable length
    except Exception:
        seq_len = 60

    if len(features_df) < seq_len:
        logger.warning(f"Only {len(features_df)} rows available but model needs {seq_len}.")
        return None

    # Take last `seq_len` rows, scale, reshape
    X_raw = features_df.iloc[-seq_len:].values.astype(np.float32)

    if scaler is not None:
        X_raw = scaler.transform(X_raw)

    X = X_raw.reshape(1, seq_len, -1)  # (1, seq_len, n_features)

    # MC Dropout inference (50 stochastic passes)
    mean_pred, lower_bound, upper_bound = get_mc_dropout_predictions(
        model, X, n_iter=50,
    )

    # mean_pred shape: (1, output_steps) — model may output >horizon steps
    means = mean_pred[0].tolist()
    lowers = lower_bound[0].tolist()
    uppers = upper_bound[0].tolist()

    # If scaler was used, the predictions are in scaled space for the
    # "modal_price" target column.  In typical training the target is
    # the raw price (not scaled), so we leave predictions as-is.
    # If your training pipeline also scales the target, add inverse_transform here.

    # Trim or pad to exactly `horizon` days
    if len(means) >= horizon:
        return means[:horizon], lowers[:horizon], uppers[:horizon]

    # Pad by repeating the last predicted value with widening CI
    while len(means) < horizon:
        last_mean = means[-1]
        last_spread = (uppers[-1] - lowers[-1]) / 2
        new_spread = last_spread * 1.05  # CI widens 5% per extra day
        means.append(last_mean)
        lowers.append(last_mean - new_spread)
        uppers.append(last_mean + new_spread)

    return means[:horizon], lowers[:horizon], uppers[:horizon]


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical-baseline fallback
# ─────────────────────────────────────────────────────────────────────────────
def _statistical_baseline(
    records: list[dict], horizon: int,
) -> tuple[list[float], list[float], list[float]]:
    """Linear trend + seasonal decomposition baseline.

    Uses the last 90 days of modal_price.  Returns (means, lowers, uppers).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    prices = [
        float(r["modal_price"])
        for r in records
        if r.get("modal_price") is not None
    ]
    # Keep at most the last 90 days
    prices = prices[-90:] if len(prices) > 90 else prices

    if len(prices) < 14:
        # Not enough data even for a simple projection — use last known price
        last = prices[-1] if prices else 2000.0
        means = [last] * horizon
        lowers = [last * 0.95] * horizon
        uppers = [last * 1.05] * horizon
        return means, lowers, uppers

    series = pd.Series(prices, dtype=float)

    # Linear trend via polyfit
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, 1)  # slope, intercept
    slope, intercept = coeffs

    # Seasonal decomposition (additive)
    # Period = 7 (weekly seasonality) — sensible for daily mandi prices
    period = min(7, len(series) // 3)
    if period < 2:
        period = 2
    try:
        decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
        seasonal_component = decomp.seasonal.values  # one full cycle
    except Exception:
        seasonal_component = np.zeros(period)

    # Residual std for confidence intervals
    residual_std = float(series.diff().dropna().std())
    if np.isnan(residual_std) or residual_std == 0:
        residual_std = float(series.mean()) * 0.03  # 3% fallback

    # Project forward
    means, lowers, uppers = [], [], []
    for i in range(1, horizon + 1):
        trend_val = slope * (len(series) + i) + intercept
        season_val = float(seasonal_component[(len(series) + i) % len(seasonal_component)])
        pred = trend_val + season_val

        # CI widens with sqrt of time
        ci = 1.96 * residual_std * np.sqrt(i)
        means.append(round(float(pred), 2))
        lowers.append(round(float(pred - ci), 2))
        uppers.append(round(float(pred + ci), 2))

    return means, lowers, uppers


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═════════════════════════════════════════════════════════════════════════════
router = APIRouter(prefix="/api/v1", tags=["Forecast"])

# ── MSP lookup (₹ per quintal, 2025-26 Rabi & Kharif) ───────────────────────
MSP_TABLE: dict[str, float] = {
    "wheat": 2275.0, "rice": 2320.0, "maize": 2090.0, "bajra": 2625.0,
    "jowar": 3371.0, "ragi": 3846.0, "barley": 1850.0, "gram": 5440.0,
    "tur": 7000.0, "moong": 8558.0, "urad": 6950.0, "groundnut": 6377.0,
    "soybean": 4600.0, "mustard": 5650.0, "cotton": 7020.0, "sugarcane": 315.0,
}


@router.get(
    "/forecast/{crop}/{mandi}",
    response_model=ForecastResponse,
    summary="Get multi-step crop price forecast",
    description=(
        "Returns a **30/60/90-day price forecast** for the specified crop and mandi, "
        "including 95% confidence intervals from Monte Carlo Dropout, MSP comparison, "
        "and a **Sell / Hold** recommendation. Results are Redis-cached for 1 hour."
    ),
)
async def get_forecast(
    crop: str,
    mandi: str,
    horizon: int = Query(30, ge=1, le=90, description="Forecast horizon in days (30, 60, or 90)"),
    db: AsyncSession = Depends(get_db),
):
    """
    **Forecast** crop prices for a given mandi.

    - **crop**: Crop name (e.g. `wheat`, `rice`, `maize`)
    - **mandi**: Market name (e.g. `Azadpur`, `Lasalgaon`)
    - **horizon**: Number of days to forecast (default: 30)

    The response includes:
    - Daily predicted prices with 95% confidence bounds
    - MSP comparison and a **SELL** / **HOLD** recommendation
    - Recommendation logic: SELL if avg predicted > MSP, else HOLD

    **Caching**: Results are cached in Redis with a 1-hour TTL.
    """
    cache_key = f"forecast:v2:{crop.lower()}:{mandi.lower()}:{horizon}"

    # ── Check Redis cache ────────────────────────────────────────────────
    cached = await cache_get(cache_key)
    if cached:
        return ForecastResponse(**cached)

    # ── Fetch historical price rows from DB ──────────────────────────────
    records = await _fetch_price_rows(db, crop, mandi, days=365)

    # Determine base_price (latest known modal_price)
    msp_value = MSP_TABLE.get(crop.lower())
    base_price: float | None = None
    for r in reversed(records):
        if r.get("modal_price") is not None:
            base_price = float(r["modal_price"])
            break
    if base_price is None:
        base_price = msp_value if msp_value else 2000.0

    # ── Attempt model inference ──────────────────────────────────────────
    forecast_source = "model"
    means: list[float] | None = None

    model = load_crop_model(crop, mandi)
    if model is not None:
        scaler = _load_scaler(crop, mandi)
        result = _run_model_inference(model, scaler, records, horizon)
        if result is not None:
            means, lowers, uppers = result
            logger.info(f"Forecast via trained model for {crop}/{mandi}")

    # ── Fallback: statistical baseline ───────────────────────────────────
    if means is None:
        forecast_source = "statistical-baseline"
        if records:
            means, lowers, uppers = _statistical_baseline(records, horizon)
            logger.info(f"Forecast via statistical baseline for {crop}/{mandi}")
        else:
            # No data at all — flat projection from base_price
            means = [base_price] * horizon
            spread = base_price * 0.05
            lowers = [base_price - spread] * horizon
            uppers = [base_price + spread] * horizon
            logger.info(f"Forecast via flat fallback for {crop}/{mandi}")

    # ── Build forecast days ──────────────────────────────────────────────
    forecast_days: list[ForecastDay] = []
    for i in range(horizon):
        forecast_days.append(ForecastDay(
            date=date.today() + timedelta(days=i + 1),
            predicted_price=round(float(means[i]), 2),
            lower_bound=round(float(lowers[i]), 2),
            upper_bound=round(float(uppers[i]), 2),
        ))

    avg_price = round(float(np.mean(means)), 2)

    # ── Recommendation logic (unchanged) ─────────────────────────────────
    if msp_value and avg_price > msp_value:
        pct = round(float((avg_price - msp_value) / msp_value * 100), 1)
        recommendation = "SELL"
        reason = f"Predicted avg price ₹{avg_price:,.0f} is {pct}% above MSP ₹{msp_value:,.0f}"
    elif msp_value:
        pct = round(float((msp_value - avg_price) / msp_value * 100), 1)
        recommendation = "HOLD"
        reason = f"Predicted avg price ₹{avg_price:,.0f} is {pct}% below MSP ₹{msp_value:,.0f} — consider holding"
    else:
        recommendation = "HOLD"
        reason = f"No MSP data available for {crop}. Average predicted price: ₹{avg_price:,.0f}"

    response = ForecastResponse(
        crop=crop,
        mandi=mandi,
        horizon_days=horizon,
        current_price=round(float(base_price), 2),
        msp=msp_value,
        avg_predicted_price=avg_price,
        recommendation=recommendation,
        recommendation_reason=reason,
        forecast=forecast_days,
    )

    # ── Cache in Redis (TTL 1 hour) ──────────────────────────────────────
    await cache_set(cache_key, response.model_dump(), ttl=3600)

    # ── Return with source header when using fallback ────────────────────
    if forecast_source != "model":
        resp = JSONResponse(content=response.model_dump(mode="json"))
        resp.headers["X-Forecast-Source"] = forecast_source
        return resp

    return response
