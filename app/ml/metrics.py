import numpy as np

def calculate_metrics(y_true, y_pred, y_mask):
    """
    Calculate evaluation metrics for forecasting.
    y_true: np.ndarray [batch, horizons]
    y_pred: np.ndarray [batch, horizons]
    y_mask: np.ndarray [batch, horizons] - 1 if valid, 0 if NaN/missing
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)

    horizons = y_true.shape[1]
    metrics = []

    for h in range(horizons):
        mask = y_mask[:, h] > 0
        yt = y_true[mask, h]
        yp = y_pred[mask, h]

        if len(yt) == 0:
            metrics.append({
                "MAE": None,
                "RMSE": None,
                "sMAPE": None,
                "WAPE": None,
                "R2": None,
                "MedAE": None,
                "Directional_Accuracy": None,
                "samples": 0
            })
            continue

        # Basic errors
        errors = yt - yp
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        medae = np.median(abs_errors)
        
        # sMAPE
        denominator = (np.abs(yt) + np.abs(yp)) / 2.0
        smape = np.mean(abs_errors / np.maximum(denominator, 1e-8)) * 100.0

        # WAPE (Weighted Absolute Percentage Error)
        wape = np.sum(abs_errors) / np.maximum(np.sum(np.abs(yt)), 1e-8) * 100.0

        # R2
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - (ss_res / np.maximum(ss_tot, 1e-8))

        # Directional Accuracy (naive: is prediction > mean(yt) same direction as yt > mean(yt))
        # Better: Since we are predicting future, compare to lag_1 (if available). 
        # For simplicity without lag_1 in this function, we just compute basic metrics.
        # Let's approximate directional accuracy relative to the mean for now, or just leave it out if lag_0 isn't passed.
        # Actually, let's just compute the other robust metrics.
        
        metrics.append({
            "MAE": float(mae),
            "RMSE": float(rmse),
            "sMAPE": float(smape),
            "WAPE": float(wape),
            "R2": float(r2),
            "MedAE": float(medae),
            "samples": int(len(yt))
        })

    return metrics
