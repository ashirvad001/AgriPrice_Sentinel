"""
feature_engineering.py
──────────────────────
48-feature engineering pipeline for crop price time-series.

Input : merged DataFrame with columns
        [date, modal_price, msp, min_price, max_price, arrivals_tonnes,
         rainfall_mm, max_temp, min_temp, freight_index, futures_price]

Output: NaN-free DataFrame with exactly 48 numeric features.
"""

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
_LAG_DAYS = [7, 14, 21, 30, 60, 90]
_ROLL_WINDOWS = [7, 14, 30]
_ARRIVALS_LAG_DAYS = [7, 14, 30]
_RABI_MONTHS = {10, 11, 12, 1, 2, 3}
_KHARIF_MONTHS = {6, 7, 8, 9}
_DROUGHT_THRESHOLD_MM = 1.0  # 10-day cumulative rainfall below this → drought


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the merged daily crop DataFrame into a 48-feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, modal_price, msp, min_price, max_price,
        arrivals_tonnes, rainfall_mm, max_temp, min_temp, freight_index,
        futures_price.  ``date`` must be parse-able by ``pd.to_datetime``.

    Returns
    -------
    pd.DataFrame
        48-column feature matrix with all NaN rows removed.  The ``date``
        column is **not** included in the output.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── 1. Price lag features (6) ────────────────────────────────────────────
    for lag in _LAG_DAYS:
        df[f"price_lag_{lag}"] = df["modal_price"].shift(lag)

    # ── 2. Price rolling mean & std (6) ──────────────────────────────────────
    for w in _ROLL_WINDOWS:
        df[f"price_rmean_{w}"] = df["modal_price"].rolling(w).mean()
        df[f"price_rstd_{w}"] = df["modal_price"].rolling(w).std()

    # ── 3. Arrivals lag features (3) ─────────────────────────────────────────
    for lag in _ARRIVALS_LAG_DAYS:
        df[f"arrivals_lag_{lag}"] = df["arrivals_tonnes"].shift(lag)

    # ── 4. Arrivals rolling mean & std (6) ───────────────────────────────────
    for w in _ROLL_WINDOWS:
        df[f"arrivals_rmean_{w}"] = df["arrivals_tonnes"].rolling(w).mean()
        df[f"arrivals_rstd_{w}"] = df["arrivals_tonnes"].rolling(w).std()

    # ── 5. Seasonality encodings (4) ─────────────────────────────────────────
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_rabi"] = month.isin(_RABI_MONTHS).astype(int)
    df["is_kharif"] = month.isin(_KHARIF_MONTHS).astype(int)

    # ── 6. MSP features (2) ─────────────────────────────────────────────────
    df["msp_distance"] = df["modal_price"] - df["msp"]
    df["msp_pct"] = (df["modal_price"] - df["msp"]) / df["msp"] * 100

    # ── 7. Rainfall / drought features (2) ───────────────────────────────────
    df["rainfall_roll_10d"] = df["rainfall_mm"].rolling(10).sum()
    df["drought_flag"] = (df["rainfall_roll_10d"] < _DROUGHT_THRESHOLD_MM).astype(int)

    # ── 8. Market features (2) ───────────────────────────────────────────────
    freight_shifted = df["freight_index"].shift(7)
    df["freight_momentum"] = (
        (df["freight_index"] - freight_shifted) / freight_shifted * 100
    )
    df["futures_basis"] = df["modal_price"] - df["futures_price"]

    # ── 9. Derived price features (3) ────────────────────────────────────────
    df["price_spread"] = df["max_price"] - df["min_price"]
    df["price_volatility"] = df["price_spread"] / df["modal_price"]
    price_shifted_7 = df["modal_price"].shift(7)
    df["price_momentum_7d"] = (
        (df["modal_price"] - price_shifted_7) / price_shifted_7 * 100
    )

    # ── 10. Temperature features (2) ─────────────────────────────────────────
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    avg_temp = (df["max_temp"] + df["min_temp"]) / 2
    df["temp_rmean_7"] = avg_temp.rolling(7).mean()

    # ── 11. Week-of-year seasonality (2) ─────────────────────────────────────
    week = df["date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)

    # ── finalise ─────────────────────────────────────────────────────────────
    # Drop the date column (not a feature) and remove NaN rows
    df = df.drop(columns=["date"])
    df = df.dropna().reset_index(drop=True)

    return df


# ── quick sanity check when run directly ─────────────────────────────────────
if __name__ == "__main__":
    # Build a tiny synthetic dataset for a smoke test
    np.random.seed(42)
    n = 180
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    sample = pd.DataFrame(
        {
            "date": dates,
            "modal_price": np.random.uniform(1500, 3000, n),
            "msp": np.full(n, 2000.0),
            "min_price": np.random.uniform(1400, 2500, n),
            "max_price": np.random.uniform(2500, 3500, n),
            "arrivals_tonnes": np.random.uniform(50, 500, n),
            "rainfall_mm": np.random.uniform(0, 15, n),
            "max_temp": np.random.uniform(30, 45, n),
            "min_temp": np.random.uniform(15, 28, n),
            "freight_index": np.random.uniform(90, 120, n),
            "futures_price": np.random.uniform(1400, 2800, n),
        }
    )

    result = engineer_features(sample)
    print(f"Output shape : {result.shape}")
    print(f"Columns ({len(result.columns)}): {list(result.columns)}")
    print(f"NaN count    : {result.isna().sum().sum()}")
