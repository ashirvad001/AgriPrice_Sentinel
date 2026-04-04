"""
test_stationarity.py
────────────────────
Pytest suite for the ADF stationarity preprocessing module.

Patches ``database`` in ``sys.modules`` before import so tests run
without asyncpg / a live PostgreSQL instance.
Patches ``_log_to_db`` at the call site so the real DB layer is never
reached during unit tests.
"""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

# ── mock out database module BEFORE importing stationarity ───────────────────
from sqlalchemy.orm import declarative_base  # type: ignore
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime  # type: ignore

_Base = declarative_base()

class _DummyModelDiagnostic(_Base):
    __tablename__ = "dummy_model_diagnostics"
    id = Column(Integer, primary_key=True)
    crop = Column(String)
    mandi = Column(String)
    test_name = Column(String)
    stage = Column(String)
    adf_statistic = Column(Float)
    p_value = Column(Float)
    critical_1pct = Column(Float)
    critical_5pct = Column(Float)
    critical_10pct = Column(Float)
    is_stationary = Column(Boolean)
    differencing_applied = Column(Boolean)
    tested_at = Column(DateTime)

_mock_database = ModuleType("database")
_mock_database.AsyncSessionLocal = MagicMock()           # type: ignore[attr-defined]
_mock_database.ModelDiagnostic = _DummyModelDiagnostic   # type: ignore[attr-defined]
sys.modules.setdefault("database", _mock_database)

from stationarity import check_and_make_stationary, _run_adf  # type: ignore  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _stationary_series(n: int = 200, seed: int = 0) -> pd.Series:
    """White noise — should be stationary."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n))


def _nonstationary_series(n: int = 200, seed: int = 0) -> pd.Series:
    """Random walk — should be non-stationary."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


# ── unit tests: _run_adf ─────────────────────────────────────────────────────

class TestRunAdf:
    def test_returns_required_keys(self):
        result = _run_adf(_stationary_series())
        expected_keys = {
            "adf_statistic", "p_value",
            "critical_1pct", "critical_5pct", "critical_10pct",
        }
        assert expected_keys == set(result.keys())

    def test_stationary_series_low_pvalue(self):
        result = _run_adf(_stationary_series())
        assert result["p_value"] < 0.05, "White noise should be stationary"

    def test_nonstationary_series_high_pvalue(self):
        result = _run_adf(_nonstationary_series())
        assert result["p_value"] > 0.05, "Random walk should be non-stationary"


# ── integration tests: check_and_make_stationary ─────────────────────────────

class TestCheckAndMakeStationary:
    """Tests patch _log_to_db so the real DB layer is never reached."""

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_stationary_series_not_differenced(self, mock_log):
        series = _stationary_series(300)
        result, diffed = await check_and_make_stationary(
            series, crop="wheat", mandi="Karnal",
        )
        assert diffed is False, "Stationary series should not be differenced"
        assert len(result) == len(series)

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_nonstationary_series_is_differenced(self, mock_log):
        series = _nonstationary_series(300)
        result, diffed = await check_and_make_stationary(
            series, crop="rice", mandi="Ludhiana",
        )
        assert diffed is True, "Random walk should trigger differencing"
        # First-order diff drops one element
        assert len(result) == len(series) - 1

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_output_has_no_nan(self, mock_log):
        series = _nonstationary_series(300)
        result, _ = await check_and_make_stationary(
            series, crop="maize", mandi="Indore",
        )
        assert result.isna().sum() == 0, "Output must be NaN-free"

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_nonstationary_logs_with_diff_result(self, mock_log):
        """Non-stationary series → _log_to_db called with diff_result not None."""
        series = _nonstationary_series(300)
        await check_and_make_stationary(
            series, crop="soybean", mandi="Ujjain",
        )
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["crop"] == "soybean"
        assert call_kwargs["mandi"] == "Ujjain"
        assert call_kwargs["differencing_applied"] is True
        assert call_kwargs["diff_result"] is not None

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_stationary_logs_without_diff_result(self, mock_log):
        """Stationary series → _log_to_db called with diff_result=None."""
        series = _stationary_series(300)
        await check_and_make_stationary(
            series, crop="chana", mandi="Bikaner",
        )
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["crop"] == "chana"
        assert call_kwargs["mandi"] == "Bikaner"
        assert call_kwargs["differencing_applied"] is False
        assert call_kwargs["diff_result"] is None

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_custom_significance(self, mock_log):
        """With a very high threshold (0.99) even random walk original
        might pass, so differencing might not be applied."""
        series = _stationary_series(300)
        result, diffed = await check_and_make_stationary(
            series, crop="tur", mandi="Latur",
            significance=0.99,
        )
        # Stationary series at 0.99 → certainly not differenced
        assert diffed is False

    @pytest.mark.asyncio
    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_original_series_unchanged(self, mock_log):
        """Input Series must not be mutated."""
        series = _nonstationary_series(300)
        original_values = series.copy()
        await check_and_make_stationary(
            series, crop="urad", mandi="Agra",
        )
        pd.testing.assert_series_equal(series, original_values)
