import logging
from celery import shared_task
import pandas as pd
from scipy.stats import ks_2samp
from typing import List

# Assume retrain task acts as our trigger
# from .retrain import trigger_retraining 

logger = logging.getLogger(__name__)

def fetch_training_prices(crop: str) -> List[float]:
    """Mock database query to fetch price distributions used during training."""
    # Example placeholder: SELECT price FROM training_data WHERE crop = crop
    return [3100.5, 3120.0, 3090.5, 3150.0, 3115.0]

def fetch_recent_prices(crop: str, days: int = 7) -> List[float]:
    """Mock database query to fetch live incoming prices for the last N days."""
    # Example placeholder: SELECT price FROM incoming_data WHERE crop = crop AND date >= NOW() - 7 days
    return [3300.0, 3320.5, 3310.0, 3350.0, 3340.0]

@shared_task
def detect_drift_weekly():
    """
    Weekly drift detection job using Kolmogorov-Smirnov test.
    Compares incoming price distributions vs training data for each crop.
    Triggers retraining if p < 0.05.
    """
    logger.info("Starting weekly drift detection KS test...")
    crops = ["wheat", "rice", "maize"] # Example crops
    
    drift_detected_crops = []
    
    for crop in crops:
        try:
            training_prices = fetch_training_prices(crop)
            recent_prices = fetch_recent_prices(crop)
            
            if not training_prices or not recent_prices:
                logger.warning(f"Insufficient data for KS test on {crop}.")
                continue
            
            # Perform Two-Sample Kolmogorov-Smirnov Test
            statistic, p_value = ks_2samp(training_prices, recent_prices)
            logger.info(f"[{crop}] KS test result: statistic={statistic:.4f}, p_value={p_value:.4f}")
            
            if p_value < 0.05:
                logger.warning(f"[{crop}] Contextual drift detected (p < 0.05).")
                drift_detected_crops.append(crop)
                
        except Exception as e:
            logger.error(f"[{crop}] Error during drift detection: {str(e)}")

    if drift_detected_crops:
        logger.info(f"Triggering retraining pipeline for: {drift_detected_crops}")
        # trigger_retraining.delay(crops=drift_detected_crops)
