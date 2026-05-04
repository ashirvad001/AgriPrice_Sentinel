import pytest
from unittest.mock import patch, MagicMock
from datetime import date
from fastapi.responses import JSONResponse
import json
import pandas as pd
import numpy as np

from api.routes_forecast import get_forecast
from api.schemas import ForecastResponse

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def sample_records():
    # Provide enough records to avoid "Not enough rows" baseline shortcut
    today = date.today()
    return [{"date": today, "modal_price": 2500.0, "msp": 2000.0}] * 100

@patch("api.routes_forecast._fetch_price_rows")
@patch("api.routes_forecast.load_crop_model")
@patch("api.routes_forecast.cache_get")
@patch("api.routes_forecast.cache_set")
async def test_fallback_path(mock_cache_set, mock_cache_get, mock_load, mock_fetch, mock_db, sample_records):
    """
    Test that when no model is available, the endpoint returns a valid fallback forecast
    via the statistical baseline, returned as a JSONResponse.
    """
    mock_cache_get.return_value = None
    mock_load.return_value = None
    mock_fetch.return_value = sample_records
    
    response = await get_forecast("wheat", "New Delhi", horizon=30, db=mock_db)
    
    assert isinstance(response, JSONResponse)
    assert response.headers["X-Forecast-Source"] == "statistical-baseline"
    
    data = json.loads(response.body.decode())
    assert "forecast" in data
    assert len(data["forecast"]) == 30
    assert data["crop"] == "wheat"
    assert data["mandi"] == "New Delhi"
    assert "recommendation" in data

@patch("api.routes_forecast._fetch_price_rows")
@patch("api.routes_forecast.load_crop_model")
@patch("api.routes_forecast._load_scaler")
@patch("forecast_model.get_mc_dropout_predictions")
@patch("feature_engineering.engineer_features")
@patch("api.routes_forecast.cache_get")
@patch("api.routes_forecast.cache_set")
async def test_inference_path(
    mock_cache_set, mock_cache_get, mock_engineer, mock_mc_dropout,
    mock_scaler, mock_load, mock_fetch, mock_db, sample_records
):
    """
    Test that when a model is available, the endpoint correctly delegates to it,
    calling get_mc_dropout_predictions and returning the true model values.
    """
    mock_cache_get.return_value = None
    
    # Mock the Keras model
    mock_keras_model = MagicMock()
    mock_keras_model.input_shape = (None, 60, 53)
    mock_load.return_value = mock_keras_model
    
    mock_scaler.return_value = MagicMock()
    mock_fetch.return_value = sample_records
    
    # Mock feature engineering so it returns a non-empty DF >= seq_len (60)
    mock_engineer.return_value = pd.DataFrame(np.random.randn(70, 53))
    
    # Mock MC Dropout to return deterministic arrays
    # get_mc_dropout_predictions returns (mean_pred, lower_bound, upper_bound)
    # Each is a 2D array: (batch_size=1, output_steps)
    means = np.full((1, 30), 3200.0)
    lowers = np.full((1, 30), 3100.0)
    uppers = np.full((1, 30), 3300.0)
    mock_mc_dropout.return_value = (means, lowers, uppers)
    
    response = await get_forecast("wheat", "New Delhi", horizon=30, db=mock_db)
    
    # When source is "model", it returns the ForecastResponse directly
    assert isinstance(response, ForecastResponse)
    assert len(response.forecast) == 30
    assert response.forecast[0].predicted_price == 3200.0
    assert response.forecast[0].lower_bound == 3100.0
    assert response.forecast[0].upper_bound == 3300.0
