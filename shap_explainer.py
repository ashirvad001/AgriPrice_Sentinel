"""
shap_explainer.py
─────────────────
SHAP-based explainability for the BiLSTM crop price forecasting model.

Uses TensorFlow GradientTape for gradient-based feature attribution (equivalent
to Integrated Gradients / DeepExplainer) to compute per-feature importance.
Generates farmer-friendly visualisations and persists SHAP values to PostgreSQL.
"""

import os
import asyncio
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tensorflow as tf
import keras_tuner as kt
from datetime import date, datetime
from forecast_model import build_hypermodel
from database import AsyncSessionLocal, ShapExplanation, init_db

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE NAMES  — 48 features from feature_engineering.py
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    # Raw columns (10)
    "modal_price", "msp", "min_price", "max_price", "arrivals_tonnes",
    "rainfall_mm", "max_temp", "min_temp", "freight_index", "futures_price",
    # Price lags (6)
    "price_lag_7", "price_lag_14", "price_lag_21",
    "price_lag_30", "price_lag_60", "price_lag_90",
    # Price rolling (6)
    "price_rmean_7", "price_rstd_7",
    "price_rmean_14", "price_rstd_14",
    "price_rmean_30", "price_rstd_30",
    # Arrivals lags (3)
    "arrivals_lag_7", "arrivals_lag_14", "arrivals_lag_30",
    # Arrivals rolling (6)
    "arrivals_rmean_7", "arrivals_rstd_7",
    "arrivals_rmean_14", "arrivals_rstd_14",
    "arrivals_rmean_30", "arrivals_rstd_30",
    # Seasonality (4)
    "month_sin", "month_cos", "is_rabi", "is_kharif",
    # MSP features (2)
    "msp_distance", "msp_pct",
    # Rainfall / drought (2)
    "rainfall_roll_10d", "drought_flag",
    # Market (2)
    "freight_momentum", "futures_basis",
    # Derived price (3)
    "price_spread", "price_volatility", "price_momentum_7d",
    # Temperature (2)
    "temp_range", "temp_rmean_7",
    # Week-of-year seasonality (2)
    "week_sin", "week_cos",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FARMER-FRIENDLY LABELS  — plain-language explanations
# ═══════════════════════════════════════════════════════════════════════════════
FARMER_LABELS = {
    # Raw columns
    "modal_price":       "Today's market price (₹/qtl)",
    "msp":               "Government MSP rate",
    "min_price":         "Lowest price today",
    "max_price":         "Highest price today",
    "arrivals_tonnes":   "Crop arrivals at mandi (tonnes)",
    "rainfall_mm":       "Daily rainfall (mm)",
    "max_temp":          "Maximum temperature (°C)",
    "min_temp":          "Minimum temperature (°C)",
    "freight_index":     "Transport cost index",
    "futures_price":     "Futures market price",
    # Price lags
    "price_lag_7":       "Price 1 week ago",
    "price_lag_14":      "Price 2 weeks ago",
    "price_lag_21":      "Price 3 weeks ago",
    "price_lag_30":      "Price 1 month ago",
    "price_lag_60":      "Price 2 months ago",
    "price_lag_90":      "Price 3 months ago",
    # Price rolling
    "price_rmean_7":     "Avg price (last 7 days)",
    "price_rstd_7":      "Price variability (7-day)",
    "price_rmean_14":    "Avg price (last 14 days)",
    "price_rstd_14":     "Price variability (14-day)",
    "price_rmean_30":    "Avg price (last 30 days)",
    "price_rstd_30":     "Price variability (30-day)",
    # Arrivals lags
    "arrivals_lag_7":    "Arrivals 1 week ago",
    "arrivals_lag_14":   "Arrivals 2 weeks ago",
    "arrivals_lag_30":   "Arrivals 1 month ago",
    # Arrivals rolling
    "arrivals_rmean_7":  "Avg arrivals (7-day)",
    "arrivals_rstd_7":   "Arrivals variability (7-day)",
    "arrivals_rmean_14": "Avg arrivals (14-day)",
    "arrivals_rstd_14":  "Arrivals variability (14-day)",
    "arrivals_rmean_30": "Avg arrivals (30-day)",
    "arrivals_rstd_30":  "Arrivals variability (30-day)",
    # Seasonality
    "month_sin":         "Season cycle (sine)",
    "month_cos":         "Season cycle (cosine)",
    "is_rabi":           "Rabi season flag",
    "is_kharif":         "Kharif season flag",
    # MSP
    "msp_distance":      "Price above/below MSP (₹)",
    "msp_pct":           "Price vs MSP (%)",
    # Rainfall / drought
    "rainfall_roll_10d": "Rainfall (last 10 days)",
    "drought_flag":      "Drought risk indicator",
    # Market
    "freight_momentum":  "Transport cost trend (%)",
    "futures_basis":     "Spot vs futures gap (₹)",
    # Derived price
    "price_spread":      "Daily price range (₹)",
    "price_volatility":  "Price volatility index",
    "price_momentum_7d": "Weekly price momentum (%)",
    # Temperature
    "temp_range":        "Day-night temp diff (°C)",
    "temp_rmean_7":      "Avg temperature (7-day, °C)",
    # Week-of-year seasonality
    "week_sin":          "Week-of-year cycle (sine)",
    "week_cos":          "Week-of-year cycle (cosine)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COLOUR PALETTE  — premium chart styling
# ═══════════════════════════════════════════════════════════════════════════════
_POS_COLOR  = "#2ecc71"   # green  — price-increasing factors
_NEG_COLOR  = "#e74c3c"   # red    — price-decreasing factors
_BG_COLOR   = "#1a1a2e"
_TEXT_COLOR  = "#e0e0e0"
_GRID_COLOR = "#2d2d44"
_ACCENT     = "#f1c40f"

plt.rcParams.update({
    "figure.facecolor": _BG_COLOR,
    "axes.facecolor":   _BG_COLOR,
    "axes.edgecolor":   _GRID_COLOR,
    "axes.labelcolor":  _TEXT_COLOR,
    "text.color":       _TEXT_COLOR,
    "xtick.color":      _TEXT_COLOR,
    "ytick.color":      _TEXT_COLOR,
    "grid.color":       _GRID_COLOR,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GRADIENT-BASED ATTRIBUTION  (Integrated-Gradients style)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_integrated_gradients(model, inputs, baseline=None, m_steps=50):
    """
    Compute Integrated Gradients for a batch of inputs.

    This is equivalent to DeepExplainer / GradientExplainer but uses
    pure TensorFlow GradientTape, avoiding shap library compatibility issues.

    Parameters
    ----------
    model    : tf.keras.Model
    inputs   : np.ndarray  — shape (batch, seq_len, n_features)
    baseline : np.ndarray  — same shape as inputs (default: zeros)
    m_steps  : int         — interpolation steps (higher = more accurate)

    Returns
    -------
    attributions : np.ndarray — shape (batch, seq_len, n_features)
    """
    inputs_tf = tf.cast(inputs, tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(inputs_tf)
    else:
        baseline = tf.cast(baseline, tf.float32)

    # Generate interpolation alphas
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)  # (m_steps+1,)

    gradient_sum = tf.zeros_like(inputs_tf)

    for alpha in alphas:
        interpolated = baseline + alpha * (inputs_tf - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = model(interpolated, training=False)
            # Sum across all output neurons for global attribution
            target = tf.reduce_sum(preds, axis=-1)
        grads = tape.gradient(target, interpolated)
        gradient_sum += grads

    # Average gradients and scale by (input - baseline)
    avg_gradients = gradient_sum / tf.cast(m_steps + 1, tf.float32)
    attributions = (inputs_tf - baseline) * avg_gradients

    return attributions.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CropShapExplainer CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class CropShapExplainer:
    """Compute and visualise SHAP-style feature importance for the BiLSTM."""

    def __init__(self, model, background_data: np.ndarray):
        """
        Parameters
        ----------
        model           : tf.keras.Model
        background_data : np.ndarray (n_background, seq_len, n_features)
                          Used to compute the baseline (mean of background).
        """
        self.model = model
        self.baseline = np.mean(background_data, axis=0, keepdims=True)

    # ── compute attributions ─────────────────────────────────────────────────
    def explain(self, X_sample: np.ndarray, m_steps: int = 50) -> np.ndarray:
        """
        Compute feature attributions via Integrated Gradients.

        Returns
        -------
        attributions : np.ndarray  shape (batch, seq_len, n_features)
        """
        # Broadcast baseline to match batch size
        baseline_batch = np.repeat(self.baseline, X_sample.shape[0], axis=0)
        return compute_integrated_gradients(
            self.model, X_sample, baseline=baseline_batch, m_steps=m_steps
        )

    # ── aggregate over time ──────────────────────────────────────────────────
    @staticmethod
    def aggregate_over_time(shap_values: np.ndarray) -> np.ndarray:
        """Mean absolute attribution across time axis → (batch, n_features)."""
        return np.mean(np.abs(shap_values), axis=1)

    # ── (1) Bar chart: top-15 features ───────────────────────────────────────
    def plot_top15_bar(self, shap_values: np.ndarray,
                       save_path: str = "shap_outputs/top15_bar.png"):
        """
        Horizontal bar chart of top-15 features by mean |attribution|
        across all samples, using farmer-friendly labels.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        agg = self.aggregate_over_time(shap_values)        # (batch, n_feat)
        global_importance = np.mean(agg, axis=0)            # (n_feat,)

        n_features = len(global_importance)
        names  = FEATURE_NAMES[:n_features]
        labels = [FARMER_LABELS.get(n, n) for n in names]

        # Sort descending, take top 15
        idx = np.argsort(global_importance)[::-1][:15]
        top_vals   = global_importance[idx][::-1]  # reverse for horizontal layout
        top_labels = [labels[i] for i in idx][::-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(top_vals))

        # Gradient-style coloring (stronger → more saturated green)
        norm_vals = top_vals / (top_vals.max() + 1e-9)
        colors = [plt.cm.Greens(0.35 + 0.6 * v) for v in norm_vals]

        bars = ax.barh(y_pos, top_vals, height=0.55, color=colors, edgecolor="none")

        # Value labels on bars
        for bar, val in zip(bars, top_vals):
            ax.text(bar.get_width() + top_vals.max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color=_TEXT_COLOR)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels, fontsize=10)
        ax.set_xlabel("Mean |SHAP value|  (impact on predicted price)", fontsize=11)
        ax.set_title("🌾  Top 15 Factors Driving Price Prediction",
                      fontsize=14, fontweight="bold", pad=15)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"✅  Bar chart saved → {save_path}")

    # ── (2) Waterfall chart: single prediction ───────────────────────────────
    def plot_waterfall(self, shap_values: np.ndarray, feature_values: np.ndarray,
                       sample_idx: int = 0,
                       save_path: str = "shap_outputs/waterfall.png"):
        """
        Waterfall chart showing each feature's contribution to the final
        predicted price for a single sample.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # For chosen sample, aggregate attribution over time → (n_features,)
        sample_attr = np.mean(shap_values[sample_idx], axis=0)   # signed mean
        n_features = len(sample_attr)

        # Use last timestep of the sample as the representative feature value
        if feature_values.ndim == 3:
            sample_feat = feature_values[sample_idx, -1, :]       # last timestep
        else:
            sample_feat = feature_values[sample_idx]

        names  = FEATURE_NAMES[:n_features]
        labels = [FARMER_LABELS.get(n, n) for n in names]

        # Sort by |attribution| descending, take top 15
        idx = np.argsort(np.abs(sample_attr))[::-1][:15]
        sorted_attr   = sample_attr[idx]
        sorted_labels = [labels[i] for i in idx]
        sorted_feat   = sample_feat[idx] if len(sample_feat) >= n_features else np.zeros(15)

        # Base value = model prediction on the baseline
        base_value = float(self.model.predict(self.baseline, verbose=0).mean())

        # Build waterfall geometry
        cumulative = base_value
        bar_starts = []
        for sv in sorted_attr:
            bar_starts.append(cumulative)
            cumulative += sv

        fig, ax = plt.subplots(figsize=(11, 8))
        y_pos = np.arange(len(sorted_attr))[::-1]  # top feature at top
        colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in sorted_attr]

        # Draw bars
        ax.barh(y_pos, sorted_attr, left=bar_starts, height=0.45,
                color=colors, edgecolor="none", alpha=0.9)

        # Connector lines between consecutive bars
        for i in range(len(sorted_attr) - 1):
            connector_x = bar_starts[i] + sorted_attr[i]
            ax.plot([connector_x, connector_x],
                    [y_pos[i] - 0.28, y_pos[i + 1] + 0.28],
                    color=_GRID_COLOR, linewidth=0.8, linestyle="--")

        # Y-axis labels with value and contribution
        display_labels = []
        for lbl, sv, fv in zip(sorted_labels, sorted_attr, sorted_feat):
            sign = "+" if sv >= 0 else ""
            display_labels.append(f"{lbl}\n(val={fv:.1f}, {sign}{sv:.4f})")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=9)

        # Base value line
        ax.axvline(base_value, color=_ACCENT, linewidth=1.5, linestyle="-",
                   alpha=0.7, label=f"Base value: ₹{base_value:.2f}")

        # Final prediction line
        final_val = base_value + np.sum(sorted_attr)
        ax.axvline(final_val, color="#3498db", linewidth=1.5, linestyle="--",
                   alpha=0.7, label=f"Prediction: ₹{final_val:.2f}")

        ax.set_xlabel("Predicted Price Contribution (₹/quintal)", fontsize=11)
        ax.set_title("🔍  How Each Factor Affects This Price Prediction",
                      fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.5)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"✅  Waterfall chart saved → {save_path}")

    # ── (3) Save to PostgreSQL ───────────────────────────────────────────────
    async def save_to_db(self, crop: str, shap_values: np.ndarray,
                         feature_values: np.ndarray,
                         prediction_date: date = None):
        """
        Persist SHAP explanations for a crop to PostgreSQL.
        Stores all features ranked by importance for frontend display.
        """
        if prediction_date is None:
            prediction_date = date.today()

        agg = self.aggregate_over_time(shap_values)
        global_importance = np.mean(agg, axis=0)

        # Mean feature value (last timestep across all samples)
        if feature_values.ndim == 3:
            mean_feat_vals = np.mean(feature_values[:, -1, :], axis=0)
        else:
            mean_feat_vals = np.mean(feature_values, axis=0)

        n_features = min(len(global_importance), len(FEATURE_NAMES))
        idx_sorted = np.argsort(global_importance)[::-1]

        rows = []
        for rank, i in enumerate(idx_sorted[:n_features], start=1):
            fname = FEATURE_NAMES[i]
            rows.append(ShapExplanation(
                crop=crop,
                prediction_date=prediction_date,
                feature_name=fname,
                shap_value=float(global_importance[i]),
                feature_value=float(mean_feat_vals[i]) if i < len(mean_feat_vals) else None,
                farmer_label=FARMER_LABELS.get(fname, fname),
                rank=rank,
                created_at=datetime.utcnow(),
            ))

        async with AsyncSessionLocal() as session:
            session.add_all(rows)
            await session.commit()
        print(f"✅  Saved {len(rows)} SHAP explanations for '{crop}' to database.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN  — end-to-end demo with synthetic data
# ═══════════════════════════════════════════════════════════════════════════════
async def main():
    print("=" * 60)
    print("  SHAP Explainability — BiLSTM Crop Price Forecaster")
    print("=" * 60)

    # ── Build model ──────────────────────────────────────────────────────────
    hp = kt.HyperParameters()
    hp.Choice("lstm_units", [128])
    hp.Choice("dropout", [0.2])
    hp.Choice("learning_rate", [1e-3])
    model = build_hypermodel(hp, input_shape=(60, 48))
    print(f"✓ Model built  ({model.count_params():,} params)")

    # ── Synthetic data ───────────────────────────────────────────────────────
    np.random.seed(42)
    tf.random.set_seed(42)
    n_bg, n_test, seq_len, n_feat = 50, 10, 60, 48

    X_background = np.random.randn(n_bg, seq_len, n_feat).astype(np.float32)
    X_test       = np.random.randn(n_test, seq_len, n_feat).astype(np.float32)

    # Quick train so attributions are non-trivial
    Y_dummy = np.random.randn(n_bg, 30).astype(np.float32)
    model.fit(X_background, Y_dummy, epochs=3, batch_size=32, verbose=0)
    print("✓ Model quick-trained (3 epochs)")

    # ── Compute Attributions ─────────────────────────────────────────────────
    explainer = CropShapExplainer(model, X_background)
    print("✓ Explainer initialised (Integrated Gradients)")

    shap_values = explainer.explain(X_test, m_steps=30)
    print(f"✓ SHAP values computed  shape={shap_values.shape}")

    # ── Generate Charts ──────────────────────────────────────────────────────
    explainer.plot_top15_bar(shap_values)
    explainer.plot_waterfall(shap_values, X_test, sample_idx=0)

    # ── DB Persistence ───────────────────────────────────────────────────────
    try:
        await init_db()
        await explainer.save_to_db("Wheat", shap_values, X_test)
    except Exception as e:
        print(f"⚠  DB save skipped ({type(e).__name__}: {e})")

    print("\n🎉  Done!  Check shap_outputs/ for charts.")


if __name__ == "__main__":
    asyncio.run(main())
