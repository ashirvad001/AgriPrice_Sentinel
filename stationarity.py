"""
stationarity.py
───────────────
Augmented Dickey-Fuller stationarity check with automatic first-order
differencing.  Logs every test result to the ``model_diagnostics``
PostgreSQL table.

Usage
-----
    import asyncio
    from stationarity import check_and_make_stationary

    series, was_diffed = asyncio.run(
        check_and_make_stationary(price_series, crop="wheat", mandi="Karnal")
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

_SIGNIFICANCE = 0.05  # p-value threshold


# ── public API ───────────────────────────────────────────────────────────────

async def check_and_make_stationary(
    series: pd.Series,
    crop: str,
    mandi: str,
    *,
    significance: float = _SIGNIFICANCE,
    db_session=None,
) -> Tuple[pd.Series, bool]:
    """Run ADF test; difference once if needed; log results to Postgres.

    Parameters
    ----------
    series : pd.Series
        The raw price time-series (must be numeric, no NaN).
    crop : str
        Crop name (e.g. ``"wheat"``).
    mandi : str
        Mandi / market name (e.g. ``"Karnal"``).
    significance : float
        P-value threshold for stationarity (default 0.05).
    db_session : AsyncSession | None
        Optional pre-existing async DB session.  If ``None`` a new one
        is created and committed automatically.

    Returns
    -------
    (transformed_series, differencing_applied)
        ``transformed_series`` is either the original or its first
        difference (with the leading NaN dropped).
        ``differencing_applied`` is ``True`` when differencing was used.
    """
    series = series.copy().reset_index(drop=True).astype(float)

    # ── step 1: ADF on original series ───────────────────────────────────
    orig_result = _run_adf(series)
    is_stationary = orig_result["p_value"] <= significance
    differencing_applied = False

    logger.info(
        "ADF [%s / %s] original  → stat=%.4f  p=%.6f  stationary=%s",
        crop, mandi, orig_result["adf_statistic"],
        orig_result["p_value"], is_stationary,
    )

    diff_result = None

    # ── step 2: difference if non-stationary ─────────────────────────────
    if not is_stationary:
        diffed = series.diff().dropna().reset_index(drop=True)
        diff_result = _run_adf(diffed)
        is_stationary_after = diff_result["p_value"] <= significance
        differencing_applied = True

        logger.info(
            "ADF [%s / %s] differenced → stat=%.4f  p=%.6f  stationary=%s",
            crop, mandi, diff_result["adf_statistic"],
            diff_result["p_value"], is_stationary_after,
        )

        series = diffed

    # ── step 3: persist to model_diagnostics ─────────────────────────────
    await _log_to_db(
        crop=crop,
        mandi=mandi,
        orig_result=orig_result,
        diff_result=diff_result,
        differencing_applied=differencing_applied,
        db_session=db_session,
    )

    return series, differencing_applied


# ── internals ────────────────────────────────────────────────────────────────

def _run_adf(series: pd.Series) -> dict:
    """Run ADF test and return a tidy dict of results."""
    stat, pval, _usedlag, _nobs, crit, _icbest = adfuller(series, autolag="AIC")
    return {
        "adf_statistic": float(stat),
        "p_value": float(pval),
        "critical_1pct": float(crit.get("1%", np.nan)),
        "critical_5pct": float(crit.get("5%", np.nan)),
        "critical_10pct": float(crit.get("10%", np.nan)),
    }


async def _log_to_db(
    *,
    crop: str,
    mandi: str,
    orig_result: dict,
    diff_result: dict | None,
    differencing_applied: bool,
    db_session=None,
) -> None:
    """Upsert ADF diagnostics into the ``model_diagnostics`` table."""
    from sqlalchemy import delete as sa_delete
    from database import AsyncSessionLocal, ModelDiagnostic

    own_session = db_session is None
    session = db_session or AsyncSessionLocal()

    try:
        # Build rows to upsert
        rows = [
            _build_row(crop, mandi, "original", orig_result, differencing_applied),
        ]
        if diff_result is not None:
            rows.append(
                _build_row(crop, mandi, "differenced", diff_result, differencing_applied),
            )

        for row in rows:
            # Delete existing row for this (crop, mandi, test_name, stage) combo
            await session.execute(
                sa_delete(ModelDiagnostic).where(
                    ModelDiagnostic.crop == row.crop,
                    ModelDiagnostic.mandi == row.mandi,
                    ModelDiagnostic.test_name == row.test_name,
                    ModelDiagnostic.stage == row.stage,
                )
            )
            session.add(row)

        if own_session:
            await session.commit()
    except Exception:
        if own_session:
            await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()


def _build_row(
    crop: str,
    mandi: str,
    stage: str,
    result: dict,
    differencing_applied: bool,
):
    from database import ModelDiagnostic
    return ModelDiagnostic(
        crop=crop,
        mandi=mandi,
        test_name="ADF",
        stage=stage,
        adf_statistic=result["adf_statistic"],
        p_value=result["p_value"],
        critical_1pct=result["critical_1pct"],
        critical_5pct=result["critical_5pct"],
        critical_10pct=result["critical_10pct"],
        is_stationary=result["p_value"] <= _SIGNIFICANCE,
        differencing_applied=differencing_applied,
        tested_at=datetime.utcnow(),
    )
