"""
tasks/drift_detection.py
────────────────────────
Weekly data drift detection using the Kolmogorov-Smirnov test.

Compares the distribution of modal_price from the last 90 days (live)
against the 90 days before that (training window) for each crop.
If significant drift is detected (p < 0.05), triggers selective retraining.
"""

import os
import logging
from datetime import date, timedelta, datetime, timezone

from celery_app import app as celery_app
from scipy.stats import ks_2samp
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Sync DB setup (same pattern as tasks/retrain.py) ─────────────────────────
from tasks.retrain import _build_sync_url

_raw_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres",
)

try:
    _SYNC_DB_URL = _build_sync_url(_raw_db_url)
    _drift_engine = create_engine(_SYNC_DB_URL, echo=False, pool_pre_ping=True)
    _DriftSession = sessionmaker(bind=_drift_engine)
except Exception as e:
    logger.error(f"Drift detection DB engine failed: {e}")
    _drift_engine = None
    _DriftSession = None

# ── Crops to monitor ────────────────────────────────────────────────────────
MONITORED_CROPS = [
    "Wheat", "Rice", "Maize", "Bajra", "Jowar", "Ragi", "Barley", "Gram",
    "Tur", "Moong", "Urad", "Groundnut", "Soybean", "Mustard", "Cotton", "Sugarcane",
]

DRIFT_THRESHOLD = 0.05  # KS test p-value threshold


def _fetch_modal_prices(session, crop: str, start_date: date, end_date: date) -> list[float]:
    """Fetch modal_price values from raw_prices for a crop within a date range."""
    query = text("""
        SELECT (raw_data->>'modal_price')::float AS modal_price
        FROM raw_prices
        WHERE LOWER(crop) = LOWER(:crop)
          AND fetch_date >= :start_date
          AND fetch_date < :end_date
          AND raw_data->>'modal_price' IS NOT NULL
        ORDER BY fetch_date ASC
    """)
    rows = session.execute(query, {
        "crop": crop,
        "start_date": start_date,
        "end_date": end_date,
    }).fetchall()
    return [float(row[0]) for row in rows if row[0] is not None]


@celery_app.task(name="tasks.drift_detection.detect_drift_weekly")
def detect_drift_weekly():
    """
    Weekly drift detection job using Kolmogorov-Smirnov test.

    Compares last 90 days of modal_price (live window) against the 90 days
    before that (training window) for each monitored crop.
    Triggers selective retraining for crops where p < 0.05.
    """
    logger.info("=" * 60)
    logger.info("  WEEKLY DRIFT DETECTION — KS Test")
    logger.info(f"  Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    if _DriftSession is None:
        logger.error("Database unavailable — cannot run drift detection")
        return {"error": "no_database"}

    session = _DriftSession()
    drift_detected_crops = []

    try:
        today = date.today()
        live_start = today - timedelta(days=90)
        train_start = today - timedelta(days=180)
        train_end = live_start

        for crop in MONITORED_CROPS:
            try:
                training_prices = _fetch_modal_prices(session, crop, train_start, train_end)
                recent_prices = _fetch_modal_prices(session, crop, live_start, today)

                if len(training_prices) < 10 or len(recent_prices) < 10:
                    logger.warning(
                        f"[{crop}] Insufficient data for KS test "
                        f"(train={len(training_prices)}, live={len(recent_prices)})"
                    )
                    continue

                # Perform Two-Sample Kolmogorov-Smirnov Test
                statistic, p_value = ks_2samp(training_prices, recent_prices)
                logger.info(
                    f"[{crop}] KS test: statistic={statistic:.4f}, "
                    f"p_value={p_value:.4f}, n_train={len(training_prices)}, "
                    f"n_live={len(recent_prices)}"
                )

                if p_value < DRIFT_THRESHOLD:
                    logger.warning(f"[{crop}] ⚠ Data drift detected (p={p_value:.4f} < {DRIFT_THRESHOLD})")
                    drift_detected_crops.append(crop)

            except Exception as e:
                logger.error(f"[{crop}] Error during drift detection: {e}")

    finally:
        session.close()

    # ── Trigger selective retraining for drifting crops ───────────────────
    if drift_detected_crops:
        logger.info(f"Triggering retraining for drifting crops: {drift_detected_crops}")
        from tasks.retrain import retrain_all_models
        retrain_all_models.delay(crops=drift_detected_crops)
    else:
        logger.info("No drift detected across all monitored crops ✅")

    return {
        "checked": len(MONITORED_CROPS),
        "drift_detected": drift_detected_crops,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
