"""
test_stationarity.py
────────────────────
Pytest suite for the ADF stationarity preprocessing module.

Shared fixtures (stationary_series, nonstationary_series, mock_db_session)
are defined in conftest.py.  The database module mock is also set up there
so tests run without asyncpg / a live PostgreSQL instance.

asyncio_mode = "auto" is configured in pyproject.toml, so no
@pytest.mark.asyncio decorators are needed.
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from stationarity import check_and_make_stationary, _run_adf  # noqa: E402


# ── helpers (ad-hoc generators for _run_adf unit tests) ──────────────────────

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

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_stationary_series_not_differenced(self, mock_log, stationary_series):
        result, diffed = await check_and_make_stationary(
            stationary_series, crop="wheat", mandi="Karnal",
        )
        assert diffed is False, "Stationary series should not be differenced"
        assert len(result) == len(stationary_series)

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_nonstationary_series_is_differenced(self, mock_log, nonstationary_series):
        result, diffed = await check_and_make_stationary(
            nonstationary_series, crop="rice", mandi="Ludhiana",
        )
        assert diffed is True, "Random walk should trigger differencing"
        # First-order diff drops one element
        assert len(result) == len(nonstationary_series) - 1

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_output_has_no_nan(self, mock_log, nonstationary_series):
        result, _ = await check_and_make_stationary(
            nonstationary_series, crop="maize", mandi="Indore",
        )
        assert result.isna().sum() == 0, "Output must be NaN-free"

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_nonstationary_logs_with_diff_result(self, mock_log, nonstationary_series):
        """Non-stationary series → _log_to_db called with diff_result not None."""
        await check_and_make_stationary(
            nonstationary_series, crop="soybean", mandi="Ujjain",
        )
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["crop"] == "soybean"
        assert call_kwargs["mandi"] == "Ujjain"
        assert call_kwargs["differencing_applied"] is True
        assert call_kwargs["diff_result"] is not None

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_stationary_logs_without_diff_result(self, mock_log, stationary_series):
        """Stationary series → _log_to_db called with diff_result=None."""
        await check_and_make_stationary(
            stationary_series, crop="chana", mandi="Bikaner",
        )
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["crop"] == "chana"
        assert call_kwargs["mandi"] == "Bikaner"
        assert call_kwargs["differencing_applied"] is False
        assert call_kwargs["diff_result"] is None

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_custom_significance(self, mock_log, stationary_series):
        """With a very high threshold (0.99) even random walk original
        might pass, so differencing might not be applied."""
        result, diffed = await check_and_make_stationary(
            stationary_series, crop="tur", mandi="Latur",
            significance=0.99,
        )
        # Stationary series at 0.99 → certainly not differenced
        assert diffed is False

    @patch("stationarity._log_to_db", new_callable=AsyncMock)
    async def test_original_series_unchanged(self, mock_log, nonstationary_series):
        """Input Series must not be mutated."""
        original_values = nonstationary_series.copy()
        await check_and_make_stationary(
            nonstationary_series, crop="urad", mandi="Agra",
        )
        pd.testing.assert_series_equal(nonstationary_series, original_values)
