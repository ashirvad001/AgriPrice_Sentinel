"""
app/data_loader.py
──────────────────
Reusable dataset loader for historical Indian mandi price CSV files.

Reads CSV data containing real Agmarknet-format columns:
    date, state, district, mandi, commodity, variety,
    min_price, max_price, modal_price, arrival_quantity

The dataset path is configurable via the MANDI_DATASET_PATH environment
variable (defaults to ``data/raw/mandi_prices.csv`` relative to the project
root).

Usage
-----
    from app.data_loader import load_mandi_dataset, load_crop_from_csv

    # Load entire CSV (lazy-cached)
    df = load_mandi_dataset()

    # Load a filtered + feature-ready DataFrame for a specific crop/mandi
    df = load_crop_from_csv("Wheat", mandi="Indore")
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column name normalisation map ────────────────────────────────────────────
# Real-world Agmarknet CSVs use inconsistent headers.  This map handles the
# most common variants and normalises them to snake_case canonical names.
_COLUMN_ALIASES: dict[str, str] = {
    # date variants
    "date": "date",
    "arrival_date": "date",
    "price_date": "date",
    "reported_date": "date",
    # location
    "state": "state",
    "state_name": "state",
    "district": "district",
    "district_name": "district",
    "mandi": "mandi",
    "market": "mandi",
    "market_name": "mandi",
    "market_center": "mandi",
    # commodity
    "commodity": "commodity",
    "commodity_name": "commodity",
    "crop": "commodity",
    "variety": "variety",
    "grade": "variety",
    # prices
    "min_price": "min_price",
    "minimum_price": "min_price",
    "max_price": "max_price",
    "maximum_price": "max_price",
    "modal_price": "modal_price",
    "model_price": "modal_price",
    # arrivals
    "arrival_quantity": "arrival_quantity",
    "arrivals": "arrival_quantity",
    "arrivals_tonnes": "arrival_quantity",
    "quantity": "arrival_quantity",
}

# Columns required for downstream feature engineering (after normalisation)
_REQUIRED_COLUMNS = {"date", "commodity", "modal_price"}

# Default relative path from project root
_DEFAULT_DATASET_REL = os.path.join("data", "raw", "mandi_prices.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_path() -> str:
    """Return the resolved absolute path to the mandi price CSV.

    Priority:
        1. ``MANDI_DATASET_PATH`` environment variable (absolute or relative)
        2. Default: ``<project_root>/data/raw/mandi_prices.csv``
    """
    env_path = os.environ.get("MANDI_DATASET_PATH", "").strip()
    if env_path:
        if os.path.isabs(env_path):
            return env_path
        # Relative paths resolved from the project root
        project_root = Path(__file__).resolve().parent.parent
        return str(project_root / env_path)

    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / _DEFAULT_DATASET_REL)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level CSV reader (cached)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical snake_case names using the alias map."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        canonical = _COLUMN_ALIASES.get(key)
        if canonical and canonical not in rename_map.values():
            rename_map[col] = canonical
    return df.rename(columns=rename_map)


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Robustly parse date strings found in Agmarknet data."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to float, coercing non-numeric values to NaN."""
    return pd.to_numeric(series, errors="coerce")


@lru_cache(maxsize=4)
def load_mandi_dataset(path: str | None = None) -> pd.DataFrame:
    """Load and normalise a mandi price CSV into a clean DataFrame.

    Parameters
    ----------
    path : str, optional
        Explicit CSV path.  Falls back to ``get_dataset_path()``.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with parsed dates and numeric price columns.

    Raises
    ------
    FileNotFoundError
        If the resolved CSV path does not exist.
    ValueError
        If required columns are missing after normalisation.
    """
    csv_path = path or get_dataset_path()
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Mandi dataset not found at: {csv_path}\n"
            f"Set the MANDI_DATASET_PATH env var or place a CSV at {_DEFAULT_DATASET_REL}"
        )

    file_size = os.path.getsize(csv_path)
    if file_size == 0:
        raise ValueError(f"Mandi dataset is empty (0 bytes): {csv_path}")

    logger.info("Loading mandi dataset from %s (%.1f MB)", csv_path, file_size / 1e6)

    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names
    df = _normalise_columns(df)

    # Validate required columns
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns after normalisation: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse dates
    df["date"] = _parse_date_column(df["date"])
    df = df.dropna(subset=["date"])

    # Coerce price / quantity columns to numeric
    for col in ("min_price", "max_price", "modal_price", "arrival_quantity"):
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Strip whitespace from string columns
    for col in ("state", "district", "mandi", "commodity", "variety"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop rows where modal_price is missing (unusable for training)
    df = df.dropna(subset=["modal_price"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Loaded %d records (%d commodities, date range %s → %s)",
        len(df),
        df["commodity"].nunique(),
        df["date"].min().date() if len(df) else "N/A",
        df["date"].max().date() if len(df) else "N/A",
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  High-level: load filtered data for a single crop, ready for feature eng.
# ─────────────────────────────────────────────────────────────────────────────

def load_crop_from_csv(
    commodity: str,
    *,
    mandi: str | None = None,
    state: str | None = None,
    path: str | None = None,
    min_records: int = 90,
) -> pd.DataFrame | None:
    """Load and filter CSV data for one commodity, returning a DataFrame
    compatible with :func:`app.feature_engineering.engineer_features`.

    The returned DataFrame has columns:
        date, modal_price, min_price, max_price, arrivals_tonnes,
        msp, rainfall_mm, max_temp, min_temp, freight_index, futures_price

    Missing auxiliary columns (weather, freight, futures) are filled with
    sensible defaults so the 53-feature pipeline never fails.

    Parameters
    ----------
    commodity : str
        Commodity / crop name (case-insensitive partial match).
    mandi : str, optional
        Market name filter (case-insensitive substring match).
    state : str, optional
        State name filter (case-insensitive substring match).
    path : str, optional
        Explicit CSV path override.
    min_records : int
        Minimum number of records required.  Returns ``None`` if fewer.

    Returns
    -------
    pd.DataFrame or None
        Feature-engineering-ready DataFrame, or ``None`` if data is
        insufficient.
    """
    try:
        full_df = load_mandi_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Cannot load mandi dataset: %s", exc)
        return None

    # Case-insensitive commodity filter
    mask = full_df["commodity"].str.lower() == commodity.lower()

    if mandi and "mandi" in full_df.columns:
        mask &= full_df["mandi"].str.lower().str.contains(mandi.lower(), na=False)

    if state and "state" in full_df.columns:
        mask &= full_df["state"].str.lower().str.contains(state.lower(), na=False)

    filtered = full_df.loc[mask].copy()

    if len(filtered) < min_records:
        logger.warning(
            "Only %d records for %s (mandi=%s, state=%s); need %d",
            len(filtered), commodity, mandi, state, min_records,
        )
        return None

    # Aggregate to daily: if multiple entries per date, take mean prices
    daily = (
        filtered
        .groupby("date")
        .agg({
            "modal_price": "mean",
            "min_price": "mean",
            "max_price": "mean",
            **({"arrival_quantity": "sum"} if "arrival_quantity" in filtered.columns else {}),
        })
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rename arrival_quantity → arrivals_tonnes (what feature_engineering expects)
    if "arrival_quantity" in daily.columns:
        daily = daily.rename(columns={"arrival_quantity": "arrivals_tonnes"})
    else:
        daily["arrivals_tonnes"] = 0.0

    # Fill min/max if missing
    if daily["min_price"].isna().all():
        daily["min_price"] = daily["modal_price"] * 0.95
    if daily["max_price"].isna().all():
        daily["max_price"] = daily["modal_price"] * 1.05

    # Add placeholder columns required by feature_engineering that aren't in
    # the mandi CSV.  These can be enriched later from weather/market APIs.
    daily["msp"] = _estimate_msp(commodity, daily["modal_price"].median())
    daily["rainfall_mm"] = 0.0
    daily["max_temp"] = 35.0
    daily["min_temp"] = 20.0
    daily["freight_index"] = 100.0
    daily["futures_price"] = daily["modal_price"] * 1.01  # small contango proxy

    # Forward-fill any remaining NaNs in price columns
    daily[["modal_price", "min_price", "max_price"]] = (
        daily[["modal_price", "min_price", "max_price"]].ffill().bfill()
    )

    logger.info(
        "Prepared %d daily records for %s (mandi=%s), date range %s → %s",
        len(daily), commodity, mandi,
        daily["date"].min().date(), daily["date"].max().date(),
    )
    return daily


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

# Approximate MSP values (₹/quintal) for major crops — used as fallback when
# no MSP data is available in the CSV.  These are based on recent Government
# of India announcements and are intentionally conservative.
_MSP_LOOKUP: dict[str, float] = {
    "wheat": 2275.0,
    "rice": 2183.0,
    "paddy": 2183.0,
    "maize": 2090.0,
    "bajra": 2500.0,
    "jowar": 3180.0,
    "ragi": 3846.0,
    "barley": 1735.0,
    "gram": 5440.0,
    "tur": 7000.0,
    "moong": 8558.0,
    "urad": 6950.0,
    "groundnut": 6377.0,
    "soybean": 4600.0,
    "mustard": 5650.0,
    "cotton": 6620.0,
    "sugarcane": 315.0,
    "onion": 1200.0,
    "potato": 800.0,
}


def _estimate_msp(commodity: str, median_price: float) -> float:
    """Return the MSP for a commodity, falling back to 80% of median price."""
    return _MSP_LOOKUP.get(commodity.lower(), median_price * 0.80)


def list_available_commodities(path: str | None = None) -> list[str]:
    """Return a sorted list of unique commodity names in the dataset."""
    try:
        df = load_mandi_dataset(path)
        return sorted(df["commodity"].unique().tolist())
    except (FileNotFoundError, ValueError):
        return []


def dataset_summary(path: str | None = None) -> dict:
    """Return a summary dict describing the loaded dataset."""
    try:
        df = load_mandi_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}

    return {
        "path": path or get_dataset_path(),
        "total_records": len(df),
        "commodities": sorted(df["commodity"].unique().tolist()),
        "states": sorted(df["state"].unique().tolist()) if "state" in df.columns else [],
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "columns": list(df.columns),
    }


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Dataset path:", get_dataset_path())
    print()

    summary = dataset_summary()
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
    else:
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nCommodities ({len(summary['commodities'])}):")
        for c in summary["commodities"]:
            print(f"  • {c}")
