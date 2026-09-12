"""
app/services/forecast_service.py
--------------------------------
Business logic layer for generating crop price forecasts.
Orchestrates data fetching, caching, and ML inference.

Strict forecast policy:
  - Forecast responses must come from the trained model.
  - Cached non-model payloads are ignored.
  - If the model cannot be loaded or cannot produce a real forecast,
    the API returns a 5xx error instead of synthetic data.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ForecastDay, ForecastResponse
from app.config import get_settings
from app.logger import get_logger
from app.repositories.forecast_repository import ForecastRepository

logger = get_logger(__name__)
settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
FORECAST_CACHE_PREFIX = "forecast:v3"
CANARY_CROP = "wheat"
CANARY_MANDI = "Azadpur"

MSP_TABLE: dict[str, float] = {
    "wheat": 2275.0,
    "rice": 2320.0,
    "maize": 2090.0,
    "bajra": 2625.0,
    "jowar": 3371.0,
    "ragi": 3846.0,
    "barley": 1850.0,
    "gram": 5440.0,
    "tur": 7000.0,
    "moong": 8558.0,
    "urad": 6950.0,
    "groundnut": 6377.0,
    "soybean": 4600.0,
    "mustard": 5650.0,
    "cotton": 7020.0,
    "sugarcane": 315.0,
}


class ForecastInferenceError(RuntimeError):
    """Raised when the trained model cannot produce a usable forecast."""


class ForecastService:
    def __init__(self, db: AsyncSession):
        self.repo = ForecastRepository(db)

    @staticmethod
    def _model_stem(crop: str, mandi: str) -> str:
        """Canonical file stem for a crop-mandi pair."""
        return f"{crop.lower()}_{mandi.lower().replace(' ', '_')}"

    @staticmethod
    def _resolve_sequence_length(model) -> int:
        seq_len = settings.SEQUENCE_LENGTH
        try:
            shape_len = model.input_shape[1]
            if shape_len is not None:
                seq_len = int(shape_len)
        except Exception:
            pass
        return seq_len

    @staticmethod
    def _resolve_feature_count(model) -> int | None:
        try:
            feature_count = model.input_shape[-1]
            if feature_count is not None:
                return int(feature_count)
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_output_steps(model) -> int | None:
        try:
            output_shape = model.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            output_steps = output_shape[-1]
            if output_steps is not None:
                return int(output_steps)
        except Exception:
            pass
        return None

    @classmethod
    def load_crop_model(cls, crop: str, mandi: str):
        """Try MLflow first, then a local saved Keras model."""
        try:
            import mlflow
            import mlflow.keras

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
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
            logger.warning(f"Local model file not found: {local_path}")
        except Exception as exc:
            logger.warning(f"Local model load failed for {crop}/{mandi}: {type(exc).__name__}: {exc}")

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

    @classmethod
    def probe_model_readiness(cls, crop: str = CANARY_CROP, mandi: str = CANARY_MANDI) -> tuple[bool, str]:
        """Verify a canary model can be loaded and run a sample prediction."""
        model = cls.load_crop_model(crop, mandi)
        if model is None:
            return False, f"Cannot load canary model for {crop}/{mandi}"

        seq_len = cls._resolve_sequence_length(model)
        num_features = cls._resolve_feature_count(model) or settings.NUM_FEATURES
        dummy_input = np.zeros((1, seq_len, num_features), dtype=np.float32)

        try:
            prediction = model(dummy_input, training=False)
            prediction_array = prediction.numpy() if hasattr(prediction, "numpy") else np.asarray(prediction)
        except Exception as exc:
            return False, f"Canary model prediction failed: {type(exc).__name__}"

        if prediction_array.size == 0:
            return False, "Canary model returned an empty prediction"

        if not np.all(np.isfinite(prediction_array)):
            return False, "Canary model returned non-finite values"

        return True, f"{crop}/{mandi}"

    def _run_model_inference(
        self,
        model,
        scaler,
        records: list[dict],
        horizon: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """Run real MC Dropout inference on historical data."""
        from app.feature_engineering import engineer_features
        from app.forecast_model import get_mc_dropout_predictions

        if not records:
            raise ForecastInferenceError("no historical price data is available for inference")

        model_output_steps = self._resolve_output_steps(model)
        if model_output_steps is not None and horizon > model_output_steps:
            raise ForecastInferenceError(
                f"requested {horizon} forecast days but the trained model only outputs {model_output_steps}"
            )

        df = pd.DataFrame(records)
        for column in [
            "msp",
            "arrivals_tonnes",
            "rainfall_mm",
            "max_temp",
            "min_temp",
            "freight_index",
            "futures_price",
        ]:
            if column not in df.columns:
                df[column] = 0.0
        df = df.fillna(0.0)

        features_df = engineer_features(df)
        if features_df.empty:
            raise ForecastInferenceError("feature engineering produced no rows")

        seq_len = self._resolve_sequence_length(model)
        if len(features_df) < seq_len:
            raise ForecastInferenceError(
                f"only {len(features_df)} engineered rows are available but the model requires {seq_len}"
            )

        X_raw = features_df.iloc[-seq_len:].values.astype(np.float32)
        expected_features = self._resolve_feature_count(model)
        if expected_features is not None and X_raw.shape[1] != expected_features:
            raise ForecastInferenceError(
                f"engineered feature width {X_raw.shape[1]} does not match model expectation {expected_features}"
            )

        if scaler is not None:
            try:
                X_raw = scaler.transform(X_raw)
            except Exception as exc:
                raise ForecastInferenceError(f"scaler transform failed: {type(exc).__name__}") from exc

        X = X_raw.reshape(1, seq_len, -1)

        try:
            mean_pred, lower_bound, upper_bound = get_mc_dropout_predictions(
                model,
                X,
                n_iter=settings.MC_DROPOUT_ITERATIONS,
            )
        except Exception as exc:
            raise ForecastInferenceError(f"MC Dropout prediction failed: {type(exc).__name__}") from exc

        means = np.asarray(mean_pred[0], dtype=np.float32)
        lowers = np.asarray(lower_bound[0], dtype=np.float32)
        uppers = np.asarray(upper_bound[0], dtype=np.float32)

        if means.size == 0:
            raise ForecastInferenceError("model returned an empty forecast")
        if means.size < horizon:
            raise ForecastInferenceError(
                f"model returned {means.size} forecast steps but {horizon} were requested"
            )
        if not (np.all(np.isfinite(means[:horizon])) and np.all(np.isfinite(lowers[:horizon])) and np.all(np.isfinite(uppers[:horizon]))):
            raise ForecastInferenceError("model returned non-finite forecast values")

        return means[:horizon].tolist(), lowers[:horizon].tolist(), uppers[:horizon].tolist()

    async def get_forecast(
        self,
        crop: str,
        mandi: str,
        horizon: int,
        force_refresh: bool = False,
    ) -> tuple[ForecastResponse, str]:
        """Generate a forecast using the trained model or fail with a 5xx error."""
        cache_key = f"{FORECAST_CACHE_PREFIX}:{crop.lower()}:{mandi.lower()}:{horizon}"

        if not force_refresh:
            cached = await self.repo.get_cached_forecast(cache_key)
            if cached:
                cached_response = ForecastResponse(**cached)
                if cached_response.forecast_source == "model":
                    return cached_response, "model"
                logger.warning(
                    f"Ignoring cached non-model forecast for {crop}/{mandi} "
                    f"(source={cached_response.forecast_source})"
                )

        records = await self.repo.fetch_historical_prices(crop, mandi, days=365)

        msp_value = MSP_TABLE.get(crop.lower())
        base_price: float | None = None
        for record in reversed(records):
            if record.get("modal_price") is not None:
                base_price = float(record["modal_price"])
                break
        if base_price is None:
            base_price = msp_value if msp_value is not None else 2000.0

        model = self.load_crop_model(crop, mandi)
        if model is None:
            detail = (
                f"Forecast model unavailable for {crop}/{mandi}: "
                "trained model could not be loaded."
            )
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

        scaler = self.load_scaler(crop, mandi)
        try:
            means, lowers, uppers = self._run_model_inference(model, scaler, records, horizon)
        except ForecastInferenceError as exc:
            detail = f"Forecast inference unavailable for {crop}/{mandi}: {exc}."
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from None
        except Exception:
            logger.exception(f"Unexpected forecast inference failure for {crop}/{mandi}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Forecast inference failed for {crop}/{mandi}. See server logs for details.",
            ) from None

        logger.info(f"Forecast via trained model for {crop}/{mandi}")

        forecast_days = [
            ForecastDay(
                date=date.today() + timedelta(days=index + 1),
                predicted_price=round(float(means[index]), 2),
                lower_bound=round(float(lowers[index]), 2),
                upper_bound=round(float(uppers[index]), 2),
            )
            for index in range(horizon)
        ]

        avg_price = round(float(np.mean(means)), 2)

        if msp_value and avg_price > msp_value:
            pct = round(float((avg_price - msp_value) / msp_value * 100), 1)
            recommendation = "SELL"
            reason = f"Predicted avg price Rs.{avg_price:,.0f} is {pct}% above MSP Rs.{msp_value:,.0f}"
        elif msp_value:
            pct = round(float((msp_value - avg_price) / msp_value * 100), 1)
            recommendation = "HOLD"
            reason = (
                f"Predicted avg price Rs.{avg_price:,.0f} is {pct}% below MSP Rs.{msp_value:,.0f}; "
                "consider holding"
            )
        else:
            recommendation = "HOLD"
            reason = f"No MSP data available for {crop}. Average predicted price: Rs.{avg_price:,.0f}"

        response = ForecastResponse(
            crop=crop,
            mandi=mandi,
            horizon_days=horizon,
            current_price=round(float(base_price), 2),
            msp=msp_value,
            avg_predicted_price=avg_price,
            recommendation=recommendation,
            recommendation_reason=reason,
            forecast_source="model",
            forecast=forecast_days,
        )

        await self.repo.set_cached_forecast(cache_key, response.model_dump(), ttl=3600)

        return response, "model"
