"""
Multi-model trainer with Optuna hyperparameter tuning.
Compares LogReg, RandomForest, XGBoost, LightGBM.
"""
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, classification_report)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.feature_engineering import generate_raw_data, engineer_features, get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("trainer")


def get_baseline_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":            XGBClassifier(n_estimators=300, learning_rate=0.05,
                                            eval_metric="logloss", random_state=42,
                                            use_label_encoder=False),
        "LightGBM":           LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              random_state=42, n_jobs=-1, verbose=-1),
    }


def tune_lgbm(X_train, y_train, n_trials: int = 50) -> dict:
    """Tune LightGBM with Optuna."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best LightGBM AUC: {study.best_value:.4f}")
    return study.best_params


def train_and_compare():
    logger.info("Generating data and engineering features…")
    df = generate_raw_data(n=10000)
    df = engineer_features(df)
    X, y = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y.mean():.1%}")

    mlflow.set_experiment("churn_prediction")
    results = {}

    for name, model in get_baseline_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "auc_roc":   roc_auc_score(y_test, y_proba),
                "f1":        f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall":    recall_score(y_test, y_pred),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name)
            results[name] = metrics
            logger.info(f"{name:25s} | AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f}")

    # Tune best model (LightGBM)
    logger.info("\nTuning LightGBM with Optuna (50 trials)…")
    best_params = tune_lgbm(X_train, y_train, n_trials=50)
    best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

    with mlflow.start_run(run_name="LightGBM_tuned"):
        tuned = LGBMClassifier(**best_params)
        tuned.fit(X_train, y_train)
        y_pred  = tuned.predict(X_test)
        y_proba = tuned.predict_proba(X_test)[:, 1]
        metrics = {
            "auc_roc":   roc_auc_score(y_test, y_proba),
            "f1":        f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(tuned, "LightGBM_tuned")
        results["LightGBM_tuned"] = metrics
        logger.info(f"{'LightGBM_tuned':25s} | AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f}")

    joblib.dump(tuned, "data/processed/best_model.pkl")
    logger.info("\n✅ Best model saved. Full classification report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    return pd.DataFrame(results).T.sort_values("auc_roc", ascending=False)


if __name__ == "__main__":
    print(train_and_compare().to_string())
