"""
app/services/forecast_service.py
────────────────────────────────
Business logic layer for generating crop price forecasts.
Orchestrates data fetching, caching, ML inference, and statistical fallback.

Fallback policy (Option B — degraded + transparent):
  • Model available    → real MC Dropout inference, source="model"
  • Model unavailable  → statistical baseline if data exists, source="statistical-baseline"
  • No data at all     → HTTP 503 (never returns fake flat values)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

import mlflow.keras

from app.repositories.forecast_repository import ForecastRepository
from app.api.schemas import ForecastResponse, ForecastDay
from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# ── MSP lookup (₹ per quintal, 2025-26 Rabi & Kharif) ───────────────────────
MSP_TABLE: dict[str, float] = {
    "wheat": 2275.0, "rice": 2320.0, "maize": 2090.0, "bajra": 2625.0,
    "jowar": 3371.0, "ragi": 3846.0, "barley": 1850.0, "gram": 5440.0,
    "tur": 7000.0, "moong": 8558.0, "urad": 6950.0, "groundnut": 6377.0,
    "soybean": 4600.0, "mustard": 5650.0, "cotton": 7020.0, "sugarcane": 315.0,
}


class ForecastService:
    def __init__(self, db: AsyncSession):
        self.repo = ForecastRepository(db)

    @staticmethod
    def _model_stem(crop: str, mandi: str) -> str:
        """Canonical file stem for a crop–mandi pair."""
        return f"{crop.lower()}_{mandi.lower().replace(' ', '_')}"

    @classmethod
    def load_crop_model(cls, crop: str, mandi: str):
        """Tries MLflow model registry first, then falls back to local file.

        Returns the loaded Keras model or None.  All failures are logged at
        WARNING level so operators can see *why* inference fell back.
        """
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        try:
            model_uri = f"models:/CropPrice_{crop}_{mandi.replace(' ', '_')}/Production"
            model = mlflow.keras.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry for {crop}/{mandi}")
            return model
        except Exception as exc:
            logger.warning(
                f"MLflow model load failed for {crop}/{mandi}: {type(exc).__name__}: {exc}"
            )

        try:
            import tensorflow as tf
            local_path = SAVED_MODELS_DIR / f"{cls._model_stem(crop, mandi)}_model.keras"
            if local_path.exists():
                model = tf.keras.models.load_model(str(local_path), compile=False)
                logger.info(f"Loaded model from local file: {local_path}")
                return model
            else:
                logger.warning(f"Local model file not found: {local_path}")
        except Exception as exc:
            logger.warning(f"Local model load failed: {type(exc).__name__}: {exc}")

        return None

    @classmethod
    def load_scaler(cls, crop: str, mandi: str):
        import joblib
        path = SAVED_MODELS_DIR / f"{cls._model_stem(crop, mandi)}_scaler.pkl"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.warning(f"Scaler load failed ({path}): {exc}")
        return None

    def _run_model_inference(
        self, model, scaler, records: list[dict], horizon: int
    ) -> tuple[list[float], list[float], list[float]] | None:
        """Run MC Dropout inference on the historical data sequence."""
        from app.forecast_model import get_mc_dropout_predictions
        from app.feature_engineering import engineer_features

        df = pd.DataFrame(records)
        for col in ["msp", "arrivals_tonnes", "rainfall_mm", "max_temp",
                     "min_temp", "freight_index", "futures_price"]:
            if col not in df.columns:
                df[col] = 0.0
        df = df.fillna(0.0)

        features_df = engineer_features(df)
        if features_df.empty or len(features_df) < 30:
            logger.warning("Not enough rows after feature engineering.")
            return None

        seq_len = settings.SEQUENCE_LENGTH
        try:
            shape_len = model.input_shape[1]
            if shape_len is not None:
                seq_len = shape_len
        except Exception:
            pass

        if len(features_df) < seq_len:
            logger.warning(f"Only {len(features_df)} rows available but model needs {seq_len}.")
            return None

        X_raw = features_df.iloc[-seq_len:].values.astype(np.float32)
        if scaler is not None:
            X_raw = scaler.transform(X_raw)

        X = X_raw.reshape(1, seq_len, -1)
        mean_pred, lower_bound, upper_bound = get_mc_dropout_predictions(
            model, X, n_iter=settings.MC_DROPOUT_ITERATIONS,
        )

        means = mean_pred[0].tolist()
        lowers = lower_bound[0].tolist()
        uppers = upper_bound[0].tolist()

        if len(means) >= horizon:
            return means[:horizon], lowers[:horizon], uppers[:horizon]

        while len(means) < horizon:
            last_mean = means[-1]
            last_spread = (uppers[-1] - lowers[-1]) / 2
            new_spread = last_spread * 1.05
            means.append(last_mean)
            lowers.append(last_mean - new_spread)
            uppers.append(last_mean + new_spread)

        return means[:horizon], lowers[:horizon], uppers[:horizon]

    def _statistical_baseline(
        self, records: list[dict], horizon: int
    ) -> tuple[list[float], list[float], list[float]]:
        """Linear trend + seasonal decomposition baseline."""
        from statsmodels.tsa.seasonal import seasonal_decompose

        prices = [float(r["modal_price"]) for r in records if r.get("modal_price") is not None]
        prices = prices[-90:] if len(prices) > 90 else prices

        if len(prices) < 14:
            last = prices[-1] if prices else 2000.0
            return [last] * horizon, [last * 0.95] * horizon, [last * 1.05] * horizon

        series = pd.Series(prices, dtype=float)
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values, 1)

        period = min(7, len(series) // 3)
        if period < 2:
            period = 2
        try:
            decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
            seasonal_component = decomp.seasonal.values
        except Exception:
            seasonal_component = np.zeros(period)

        residual_std = float(series.diff().dropna().std())
        if np.isnan(residual_std) or residual_std == 0:
            residual_std = float(series.mean()) * 0.03

        means, lowers, uppers = [], [], []
        for i in range(1, horizon + 1):
            trend_val = slope * (len(series) + i) + intercept
            season_val = float(seasonal_component[(len(series) + i) % len(seasonal_component)])
            pred = trend_val + season_val

            ci = 1.96 * residual_std * np.sqrt(i)
            means.append(round(float(pred), 2))
            lowers.append(round(float(pred - ci), 2))
            uppers.append(round(float(pred + ci), 2))

        return means, lowers, uppers

    async def get_forecast(self, crop: str, mandi: str, horizon: int, force_refresh: bool = False) -> tuple[ForecastResponse, str]:
        """
        Orchestrates forecast fetching. 
        Returns (ForecastResponse, forecast_source).
        """
        cache_key = f"forecast:v2:{crop.lower()}:{mandi.lower()}:{horizon}"
        
        if not force_refresh:
            cached = await self.repo.get_cached_forecast(cache_key)
            if cached:
                return ForecastResponse(**cached), "cache"

        records = await self.repo.fetch_historical_prices(crop, mandi, days=365)
        
        msp_value = MSP_TABLE.get(crop.lower())
        base_price: float | None = None
        for r in reversed(records):
            if r.get("modal_price") is not None:
                base_price = float(r["modal_price"])
                break
        if base_price is None:
            base_price = msp_value if msp_value else 2000.0

        forecast_source = "model"
        means: list[float] | None = None

        model = self.load_crop_model(crop, mandi)
        if model is not None:
            scaler = self.load_scaler(crop, mandi)
            result = self._run_model_inference(model, scaler, records, horizon)
            if result is not None:
                means, lowers, uppers = result
                logger.info(f"Forecast via trained model for {crop}/{mandi}")
            else:
                logger.warning(
                    f"Model loaded but inference failed for {crop}/{mandi} — "
                    "falling back to statistical baseline"
                )

        if means is None:
            forecast_source = "statistical-baseline"
            if records:
                means, lowers, uppers = self._statistical_baseline(records, horizon)
                logger.warning(
                    f"No ML model available for {crop}/{mandi}. "
                    f"Returning statistical baseline forecast (source='{forecast_source}')."
                )
            else:
                # ── NO fake data: fail loudly ────────────────────────────
                logger.error(
                    f"Cannot produce forecast for {crop}/{mandi}: "
                    "no trained model AND no historical price data available."
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"Forecast unavailable for {crop}/{mandi}: "
                        "no trained model and no historical data. "
                        "Please ensure a model is trained or price data is ingested."
                    ),
                )

        forecast_days = [
            ForecastDay(
                date=date.today() + timedelta(days=i + 1),
                predicted_price=round(float(means[i]), 2),
                lower_bound=round(float(lowers[i]), 2),
                upper_bound=round(float(uppers[i]), 2),
            )
            for i in range(horizon)
        ]

        avg_price = round(float(np.mean(means)), 2)

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
            forecast_source=forecast_source,
            forecast=forecast_days,
        )

        await self.repo.set_cached_forecast(cache_key, response.model_dump(), ttl=3600)

        return response, forecast_source
