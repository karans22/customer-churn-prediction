"""
SHAP-based model explainability for churn prediction.
Generates global and local explanation plots.
"""
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from src.features.feature_engineering import generate_raw_data, engineer_features, get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("shap_analysis")


def run_shap_analysis(model_path: str = "data/processed/best_model.pkl",
                      n_samples: int = 500):
    """Generate SHAP summary, beeswarm, waterfall, and feature importance plots."""

    logger.info("Loading model and generating features…")
    model = joblib.load(model_path)
    df = generate_raw_data(n=5000)
    df = engineer_features(df)
    X, y = get_feature_matrix(df)
    X_sample = X.sample(n=n_samples, random_state=42)

    logger.info("Computing SHAP values (TreeExplainer)…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle binary vs multiclass output
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # ── 1. Global Feature Importance ──────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False,
                      color="#E74C3C", max_display=15)
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/shap_feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved: reports/shap_feature_importance.png")

    # ── 2. Beeswarm Summary Plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.title("SHAP Beeswarm — Feature Impact on Churn Probability", fontsize=13)
    plt.tight_layout()
    plt.savefig("reports/shap_beeswarm.png", dpi=150)
    plt.close()
    logger.info("Saved: reports/shap_beeswarm.png")

    # ── 3. Waterfall plot for a high-risk customer ────────────────────────────
    high_risk_idx = sv[:, 0].argmin()   # most negative base → highest churn push
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
        sv[high_risk_idx],
        feature_names=X_sample.columns.tolist(),
        max_display=12,
        show=False,
    )
    plt.title(f"Waterfall — High Risk Customer (idx={high_risk_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_highrisk.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: reports/shap_waterfall_highrisk.png")

    # ── 4. Dependence plot: tenure vs churn ───────────────────────────────────
    plt.figure(figsize=(8, 5))
    shap.dependence_plot("tenure_months", sv, X_sample,
                          interaction_index="monthly_charges", show=False)
    plt.title("SHAP Dependence: Tenure vs Churn (coloured by Monthly Charges)", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/shap_dependence_tenure.png", dpi=150)
    plt.close()
    logger.info("Saved: reports/shap_dependence_tenure.png")

    # ── 5. Top features summary ───────────────────────────────────────────────
    mean_shap = np.abs(sv).mean(axis=0)
    top_features = pd.Series(mean_shap, index=X_sample.columns).sort_values(ascending=False)
    logger.info("\nTop 10 features by mean |SHAP| value:")
    logger.info(top_features.head(10).to_string())
    return top_features


if __name__ == "__main__":
    run_shap_analysis()
