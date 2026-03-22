"""
test_feature_engineering.py
───────────────────────────
Pytest suite for the 48-feature crop price engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from feature_engineering import engineer_features


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_sample_df(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic merged DataFrame spanning *n* days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "modal_price": rng.uniform(1500, 3000, n),
            "msp": np.full(n, 2000.0),
            "min_price": rng.uniform(1400, 2500, n),
            "max_price": rng.uniform(2500, 3500, n),
            "arrivals_tonnes": rng.uniform(50, 500, n),
            "rainfall_mm": rng.uniform(0, 15, n),
            "max_temp": rng.uniform(30, 45, n),
            "min_temp": rng.uniform(15, 28, n),
            "freight_index": rng.uniform(90, 120, n),
            "futures_price": rng.uniform(1400, 2800, n),
        }
    )


# ── tests ────────────────────────────────────────────────────────────────────

class TestEngineerFeatures:
    """Core contract tests for engineer_features()."""

    df: pd.DataFrame
    result: pd.DataFrame

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.df = _make_sample_df(180)
        self.result = engineer_features(self.df)

    def test_output_has_48_columns(self):
        assert self.result.shape[1] == 48, (
            f"Expected 48 columns, got {self.result.shape[1]}: {list(self.result.columns)}"
        )

    def test_no_nan_values(self):
        nan_count = self.result.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in output"

    def test_date_column_absent(self):
        assert "date" not in self.result.columns

    def test_output_has_rows(self):
        # 180 days minus 90 (max lag) → at least ~89 rows expected
        assert len(self.result) > 0, "Output DataFrame is empty"

    # ── feature value spot-checks ────────────────────────────────────────

    def test_msp_distance_values(self):
        """msp_distance must equal modal_price − msp for every row."""
        expected = self.result["modal_price"] - self.result["msp"]
        pd.testing.assert_series_equal(
            self.result["msp_distance"], expected, check_names=False
        )

    def test_msp_pct_values(self):
        expected = (self.result["modal_price"] - self.result["msp"]) / self.result["msp"] * 100
        pd.testing.assert_series_equal(
            self.result["msp_pct"], expected, check_names=False
        )

    def test_price_spread_values(self):
        expected = self.result["max_price"] - self.result["min_price"]
        pd.testing.assert_series_equal(
            self.result["price_spread"], expected, check_names=False
        )

    def test_futures_basis_values(self):
        expected = self.result["modal_price"] - self.result["futures_price"]
        pd.testing.assert_series_equal(
            self.result["futures_basis"], expected, check_names=False
        )

    def test_temp_range_values(self):
        expected = self.result["max_temp"] - self.result["min_temp"]
        pd.testing.assert_series_equal(
            self.result["temp_range"], expected, check_names=False
        )

    def test_drought_flag_binary(self):
        unique_vals = set(self.result["drought_flag"].unique())
        assert unique_vals.issubset({0, 1}), f"drought_flag has non-binary values: {unique_vals}"

    # ── seasonality checks ───────────────────────────────────────────────

    def test_rabi_kharif_mutually_exclusive_or_neither(self):
        """A row should never be both rabi AND kharif at the same time."""
        both = (self.result["is_rabi"] == 1) & (self.result["is_kharif"] == 1)
        assert not bool(both.any()), "Some rows are flagged as both rabi and kharif"

    def test_is_rabi_flag(self):
        """is_rabi must be 1 for months 10-12 and 1-3 (Oct–Mar)."""
        # Re-derive from original df after alignment
        df_with_month = self.df.copy()
        df_with_month["date"] = pd.to_datetime(df_with_month["date"])
        df_with_month = df_with_month.sort_values("date").reset_index(drop=True)
        # The output has been dropna'd; take the tail that survived
        n_out = len(self.result)
        tail_months = df_with_month["date"].dt.month.iloc[-n_out:].values
        expected_rabi = np.isin(tail_months, [10, 11, 12, 1, 2, 3]).astype(int)
        np.testing.assert_array_equal(self.result["is_rabi"].values, expected_rabi)

    def test_month_sin_cos_range(self):
        assert self.result["month_sin"].between(-1, 1).all()
        assert self.result["month_cos"].between(-1, 1).all()

    def test_week_sin_cos_range(self):
        assert self.result["week_sin"].between(-1, 1).all()
        assert self.result["week_cos"].between(-1, 1).all()

    # ── lag sanity ───────────────────────────────────────────────────────

    def test_lag_columns_exist(self):
        for lag in [7, 14, 21, 30, 60, 90]:
            assert f"price_lag_{lag}" in self.result.columns

    def test_arrivals_lag_columns_exist(self):
        for lag in [7, 14, 30]:
            assert f"arrivals_lag_{lag}" in self.result.columns

    # ── rolling columns ──────────────────────────────────────────────────

    def test_rolling_columns_exist(self):
        for w in [7, 14, 30]:
            assert f"price_rmean_{w}" in self.result.columns
            assert f"price_rstd_{w}" in self.result.columns
            assert f"arrivals_rmean_{w}" in self.result.columns
            assert f"arrivals_rstd_{w}" in self.result.columns


class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_unsorted_input_still_works(self):
        """Pipeline should sort by date internally even if input is shuffled."""
        df = _make_sample_df(180)
        shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
        result = engineer_features(shuffled)
        assert result.shape[1] == 48
        assert result.isna().sum().sum() == 0

    def test_string_dates_accepted(self):
        """date column as ISO strings should be parsed correctly."""
        df = _make_sample_df(180)
        df["date"] = df["date"].astype(str)
        result = engineer_features(df)
        assert result.shape[1] == 48

    def test_original_df_not_mutated(self):
        """engineer_features must not modify the caller's DataFrame."""
        df = _make_sample_df(180)
        original_cols = list(df.columns)
        _ = engineer_features(df)
        assert list(df.columns) == original_cols
