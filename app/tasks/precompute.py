"""
app/tasks/precompute.py
-----------------------
Nightly batch job to pre-compute forecast arrays for all active crop-mandi
pairs and cache them in Redis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from app.celery_app import app as celery_app
from app.services.forecast_service import ForecastService
from app.tasks.retrain import SyncSession, TARGET_CROP_MANDIS

logger = logging.getLogger("precompute")


@celery_app.task(name="app.tasks.precompute.precompute_forecasts")
def precompute_forecasts():
    """Run fresh model inference for monitored crop/mandi pairs and cache the result."""
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("  NIGHTLY FORECAST PRE-COMPUTE")
    logger.info(f"  Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    if SyncSession is None:
        logger.error("Database unavailable - cannot pre-compute forecasts")
        return {"error": "no_database"}

    from app.database import AsyncSessionLocal

    async def _run_all():
        results = {"success": 0, "failed": 0, "failed_pairs": []}
        async with AsyncSessionLocal() as db:
            service = ForecastService(db)
            for crop, mandi in TARGET_CROP_MANDIS:
                for horizon in [30, 60, 90]:
                    try:
                        await service.get_forecast(crop, mandi, horizon, force_refresh=True)
                        results["success"] += 1
                        logger.info(f"Precomputed {crop} @ {mandi} ({horizon}d)")
                    except Exception as exc:
                        logger.error(f"Failed to precompute {crop} @ {mandi} ({horizon}d): {exc}")
                        results["failed"] += 1
                        results["failed_pairs"].append(f"{crop}/{mandi}/{horizon}d")
        return results

    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(_run_all())

    elapsed_total = round(float(time.time() - overall_start), 1)
    logger.info("=" * 60)
    logger.info(f"  PRE-COMPUTE COMPLETE in {elapsed_total}s")
    logger.info(f"  Success: {results['success']} | Failed: {results['failed']}")
    logger.info("=" * 60)

    return results
