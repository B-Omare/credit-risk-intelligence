"""
CreditPulse - XGBoost Credit Risk Classifier
Phase 5: Trains an XGBoost model to predict default probability.
Includes SHAP explainability for every prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# Features we'll use for modelling
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "ext_source_mean", "ext_source_std",
    "income_to_credit_ratio", "days_employed_pct",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
]


def load_data():
    """Load cleaned data and prepare features."""
    print("Loading data...")
    df = pd.read_parquet(PROCESSED / "loans_clean.parquet")

    # Keep only rows where all features are present
    df = df[FEATURES + ["TARGET"]].dropna()
    print(f"  Rows after dropping NaN: {len(df):,}")

    X = df[FEATURES]
    y = df["TARGET"]

    return X, y


def train_model(X, y):
    """Split data and train XGBoost model."""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Test:  {len(X_test):,} rows")

    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        scale_pos_weight=11,  # ratio of negative to positive class
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Print model performance metrics."""
    print("\n── Model Performance ────────────────────────")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Gini:    {(2 * auc - 1):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred_proba


def generate_shap(model, X_test):
    """Generate SHAP values and save waterfall plot."""
    print("\nGenerating SHAP explanations...")
    
    # Fix for XGBoost 3.x compatibility
    booster = model.get_booster()
    explainer   = shap.TreeExplainer(booster)
    X_sample = X_test.iloc[:100]
    shap_values = explainer(X_sample)

    # Summary plot — feature importance across all borrowers
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample,
                      show=False, plot_size=(10, 6))
    plt.title("CreditPulse — SHAP Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot saved to reports/shap_summary.png")

    # Waterfall plot — single borrower explanation
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("CreditPulse — Single Borrower SHAP Explanation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP waterfall plot saved to reports/shap_waterfall.png")

    return shap_values


def save_model(model):
    """Save trained model to disk."""
    output = MODELS / "xgboost_model.pkl"
    joblib.dump(model, output)
    print(f"\n  Model saved to {output}")


def main():
    X, y = load_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    generate_shap(model, X_test)
    save_model(model)
    print("\nXGBoost training complete!")


if __name__ == "__main__":
    main()