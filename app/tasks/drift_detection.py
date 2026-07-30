"""
app/tasks/drift_detection.py
────────────────────────────
Weekly Celery job to detect data drift in commodity prices using EvidentlyAI.
Logs drift metrics to MLflow and triggers retraining if drift is severe.
"""

import os
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from app.celery_app import app as celery_app
from app.tasks.retrain import TARGET_CROP_MANDIS, retrain_all_models
from app.config import get_settings
from app.logger import get_logger

logger = get_logger("drift_detection")
settings = get_settings()

def _get_prices_for_window(engine, crop: str, mandi: str, start: date, end: date) -> pd.DataFrame:
    """Fetch prices for a specific date window."""
    query = f"""
        SELECT fetch_date, CAST(raw_data->>'modal_price' AS FLOAT) as modal_price
        FROM raw_prices
        WHERE crop = '{crop}' 
          AND raw_data->>'mandi' = '{mandi}'
          AND fetch_date BETWEEN '{start}' AND '{end}'
          AND raw_data->>'modal_price' IS NOT NULL
        ORDER BY fetch_date ASC
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error fetching data for drift detection: {e}")
        return pd.DataFrame()


@celery_app.task(name="app.tasks.drift_detection.detect_drift_weekly")
def detect_drift_weekly():
    """
    Weekly drift detection job using EvidentlyAI.
    Compares the last 7 days of prices (Current) against the 30 days prior (Reference).
    """
    logger.info("Starting weekly EvidentlyAI data drift detection...")
    
    # ── Database connection ──────────────────────────────────────────────────
    # Create sync engine for pandas read_sql
    driver_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(driver_url)
    
    today = date.today()
    current_start = today - timedelta(days=7)
    reference_start = current_start - timedelta(days=30)
    
    # Track which models need retraining due to drift
    drifted_pairs = []
    
    for crop, mandi in TARGET_CROP_MANDIS:
        # 1. Fetch Reference Data (T-37 to T-7)
        ref_df = _get_prices_for_window(engine, crop, mandi, reference_start, current_start - timedelta(days=1))
        
        # 2. Fetch Current Data (T-7 to T)
        cur_df = _get_prices_for_window(engine, crop, mandi, current_start, today)
        
        if len(ref_df) < 10 or len(cur_df) < 5:
            logger.info(f"Insufficient data to detect drift for {crop}/{mandi}. Ref: {len(ref_df)}, Cur: {len(cur_df)}")
            continue
            
        # 3. Generate Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df[['modal_price']], current_data=cur_df[['modal_price']])
        
        # 4. Extract Results
        result = report.as_dict()
        dataset_drift = result['metrics'][0]['result']['dataset_drift']
        drift_share = result['metrics'][0]['result']['drift_share']
        
        if dataset_drift:
            logger.warning(f"🚨 DRIFT DETECTED: {crop} @ {mandi} (Drift share: {drift_share:.2f})")
            drifted_pairs.append((crop, mandi))
        else:
            logger.info(f"✅ Stable: {crop} @ {mandi} (Drift share: {drift_share:.2f})")
            
        # Optional: Save HTML report to disk or MLflow
        # report_path = f"drift_report_{crop}_{mandi}.html"
        # report.save_html(report_path)
            
    engine.dispose()
    
    # Trigger retrain for drifted models
    if drifted_pairs:
        logger.info(f"Triggering retraining for {len(drifted_pairs)} drifted models...")
        retrain_all_models.delay(crops=drifted_pairs)
    else:
        logger.info("No data drift detected across all monitored crop/mandi pairs.")
        
    return {"drifted_count": len(drifted_pairs)}
