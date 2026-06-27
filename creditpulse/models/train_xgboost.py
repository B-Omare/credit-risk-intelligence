"""
Phase 5 — XGBoost credit default classifier.
Handles class imbalance, outputs SHAP-ready model.
"""

import json
import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/loans_ifrs9.parquet")
MODEL_PATH = Path("models/xgboost_model.pkl")
METRICS_PATH = Path("models/xgboost_metrics.json")

FEATURE_COLS = [
    "AGE_YEARS", "EMPLOYMENT_YEARS", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "AMT_ANNUITY", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "LTV_RATIO",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "INCOME_PER_PERSON",
]
TARGET_COL = "TARGET"


def prepare_data(df: pd.DataFrame):
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None

    X = X.fillna(X.median())
    return X, y, available_features


def train(input_path: Path = INPUT_PATH) -> dict:
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
        from sklearn.utils.class_weight import compute_sample_weight

        df = pd.read_parquet(input_path)
        X, y, features = prepare_data(df)

        if y is None:
            raise ValueError("TARGET column not found")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_ratio,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "average_precision": float(average_precision_score(y_test, y_prob)),
            "features_used": features,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "default_rate": float(y.mean()),
            "best_iteration": int(model.best_iteration),
        }

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": features}, f)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics

    except ImportError:
        logger.warning("XGBoost not installed")
        return {"error": "xgboost not installed"}


if __name__ == "__main__":
    train()
