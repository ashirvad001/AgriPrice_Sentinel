import importlib
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


_BASE_ENV = {
    "SECRET_KEY": "a" * 64,
    "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
}

_FORECAST_MODULES = [
    "app.api.routes_forecast",
    "app.api.deps",
    "app.api.schemas",
    "app.database",
    "app.logger",
    "app.config",
    "app.services.forecast_service",
    "app.repositories.forecast_repository",
]


class DummyModel:
    input_shape = (None, 60, 53)
    output_shape = (None, 30)


def _clear_forecast_modules() -> None:
    for module_name in _FORECAST_MODULES:
        sys.modules.pop(module_name, None)


def _build_forecast_test_app() -> FastAPI:
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        _clear_forecast_modules()
        deps = importlib.import_module("app.api.deps")
        routes = importlib.import_module("app.api.routes_forecast")

        app = FastAPI()
        app.include_router(routes.router)

        async def _override_db():
            yield object()

        app.dependency_overrides[deps.get_db] = _override_db
        return app


def _make_records(base_price: float) -> list[dict]:
    start = date(2026, 1, 1)
    records = []
    for offset in range(120):
        price = base_price + offset * 3
        records.append(
            {
                "date": start + timedelta(days=offset),
                "modal_price": price,
                "min_price": price - 20,
                "max_price": price + 20,
                "arrivals_tonnes": 100 + offset,
                "rainfall_mm": float(offset % 7),
                "max_temp": 30.0 + (offset % 5),
                "min_temp": 18.0 + (offset % 3),
                "freight_index": 100.0 + offset * 0.1,
                "futures_price": price + 10,
                "msp": 2275.0,
            }
        )
    return records


async def _fake_get_cached_forecast(self, cache_key: str):
    return None


async def _fake_set_cached_forecast(self, cache_key: str, data: dict, ttl: int = 3600):
    return None


async def _fake_fetch_historical_prices(self, crop: str, mandi: str, days: int = 365):
    key = (crop.lower(), mandi.lower())
    if key == ("wheat", "azadpur"):
        return _make_records(1800.0)
    if key == ("rice", "karnal"):
        return _make_records(2600.0)
    raise AssertionError(f"Unexpected crop/mandi pair: {crop}/{mandi}")


def _fake_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    last_price = float(df["modal_price"].iloc[-1])
    rows = 70
    base_column = np.linspace(last_price, last_price + rows - 1, rows, dtype=np.float32).reshape(-1, 1)
    feature_matrix = np.repeat(base_column, 53, axis=1)
    return pd.DataFrame(feature_matrix)


def test_forecast_endpoint_invokes_real_inference_and_varies_by_input():
    app = _build_forecast_test_app()

    def _inference_side_effect(model, X, n_iter=100, confidence_level=0.95):
        base = float(X.mean())
        steps = np.arange(30, dtype=np.float32)
        mean = (base + steps).reshape(1, 30)
        lower = mean - 2.5
        upper = mean + 2.5
        return mean, lower, upper

    with patch(
        "app.repositories.forecast_repository.ForecastRepository.get_cached_forecast",
        _fake_get_cached_forecast,
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.set_cached_forecast",
        _fake_set_cached_forecast,
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.fetch_historical_prices",
        _fake_fetch_historical_prices,
    ), patch(
        "app.services.forecast_service.ForecastService.load_crop_model",
        return_value=DummyModel(),
    ), patch(
        "app.services.forecast_service.ForecastService.load_scaler",
        return_value=None,
    ), patch(
        "app.feature_engineering.engineer_features",
        side_effect=_fake_engineer_features,
    ), patch(
        "app.forecast_model.get_mc_dropout_predictions",
        side_effect=_inference_side_effect,
    ) as mock_inference:
        with TestClient(app) as client:
            wheat_response = client.get("/api/v1/forecast/wheat/Azadpur?horizon=30")
            rice_response = client.get("/api/v1/forecast/rice/Karnal?horizon=30")

    assert wheat_response.status_code == 200
    assert rice_response.status_code == 200
    assert wheat_response.headers["X-Forecast-Source"] == "model"
    assert rice_response.headers["X-Forecast-Source"] == "model"
    assert mock_inference.call_count == 2

    wheat_payload = wheat_response.json()
    rice_payload = rice_response.json()

    assert wheat_payload["forecast_source"] == "model"
    assert rice_payload["forecast_source"] == "model"
    assert wheat_payload["avg_predicted_price"] != rice_payload["avg_predicted_price"]
    assert wheat_payload["forecast"][0]["predicted_price"] != rice_payload["forecast"][0]["predicted_price"]
    assert wheat_payload["forecast"][0]["predicted_price"] != wheat_payload["forecast"][-1]["predicted_price"]
    assert rice_payload["forecast"][0]["predicted_price"] != rice_payload["forecast"][-1]["predicted_price"]
