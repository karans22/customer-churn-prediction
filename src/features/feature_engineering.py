"""
Feature engineering pipeline for churn prediction.
Creates domain-driven features from raw customer data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def generate_raw_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Synthetic telco-style customer dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id":       [f"C{i:06d}" for i in range(n)],
        "tenure_months":     rng.integers(1, 120, n),
        "monthly_charges":   rng.uniform(18.0, 120.0, n).round(2),
        "total_charges":     None,   # will be derived
        "num_products":      rng.integers(1, 6, n),
        "support_tickets":   rng.integers(0, 20, n),
        "contract_type":     rng.choice(["Month-to-month", "One year", "Two year"], n,
                                         p=[0.55, 0.25, 0.20]),
        "payment_method":    rng.choice(["Electronic check", "Mailed check",
                                          "Bank transfer", "Credit card"], n),
        "internet_service":  rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20]),
        "tech_support":      rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "online_security":   rng.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "senior_citizen":    rng.integers(0, 2, n),
        "partner":           rng.choice(["Yes", "No"], n),
        "dependents":        rng.choice(["Yes", "No"], n),
    })

    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]
                           * rng.uniform(0.9, 1.1, n)).round(2)
    df["days_since_last_login"] = rng.integers(1, 120, n)
    df["num_complaints_last6m"] = rng.integers(0, 8, n)

    # Simulate churn with realistic drivers
    churn_score = (
        (df["contract_type"] == "Month-to-month").astype(float) * 0.25 +
        (df["tenure_months"] < 12).astype(float) * 0.18 +
        (df["support_tickets"] > 5).astype(float) * 0.12 +
        (df["num_products"] == 1).astype(float) * 0.10 +
        (df["internet_service"] == "Fiber optic").astype(float) * 0.08 +
        (df["tech_support"] == "No").astype(float) * 0.05 +
        rng.uniform(0, 0.15, n)
    ).clip(0, 1)
    df["churn"] = (rng.random(n) < churn_score).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transforms."""
    df = df.copy()

    # ── Tenure features ───────────────────────────────────────────────────────
    df["tenure_band"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 36, 72, 999],
        labels=["New (0-1yr)", "Growing (1-3yr)", "Established (3-6yr)", "Loyal (6yr+)"]
    ).astype(str)

    # ── Spend features ────────────────────────────────────────────────────────
    df["revenue_per_month"] = (df["total_charges"] /
                                df["tenure_months"].replace(0, 1)).round(2)
    df["charge_increase_indicator"] = (
        df["monthly_charges"] > df["revenue_per_month"] * 1.1
    ).astype(int)

    # ── Support / frustration proxies ─────────────────────────────────────────
    df["support_ticket_rate"] = (df["support_tickets"] /
                                  df["tenure_months"].replace(0, 1)).round(4)
    df["high_complaint_flag"] = (df["num_complaints_last6m"] >= 3).astype(int)

    # ── Contract risk composite ───────────────────────────────────────────────
    df["contract_risk"] = (
        (df["contract_type"] == "Month-to-month").astype(int) * 2 +
        (df["payment_method"] == "Electronic check").astype(int)
    )

    # ── Engagement score ──────────────────────────────────────────────────────
    df["engagement_score"] = (
        df["num_products"] * 2 +
        (df["tech_support"] == "Yes").astype(int) +
        (df["online_security"] == "Yes").astype(int) +
        (df["partner"] == "Yes").astype(int) +
        (df["dependents"] == "Yes").astype(int)
    )

    # ── Encode categoricals ───────────────────────────────────────────────────
    df["contract_encoded"] = df["contract_type"].map({
        "Month-to-month": 0, "One year": 1, "Two year": 2
    })
    df["internet_encoded"] = df["internet_service"].map({
        "No": 0, "DSL": 1, "Fiber optic": 2
    })

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) ready for modelling."""
    feature_cols = [
        "tenure_months", "monthly_charges", "total_charges", "num_products",
        "support_tickets", "senior_citizen", "revenue_per_month",
        "charge_increase_indicator", "support_ticket_rate", "high_complaint_flag",
        "contract_risk", "engagement_score", "contract_encoded", "internet_encoded",
        "days_since_last_login", "num_complaints_last6m",
    ]
    X = df[feature_cols].fillna(0)
    y = df["churn"]
    return X, y
