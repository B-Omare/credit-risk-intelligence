"""
CreditPulse - Isolation Forest Fraud Detector
Phase 5: Detects anomalous loan applications that may indicate fraud.
Trained on clean applications — anomalies score low.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "income_to_credit_ratio", "days_employed_pct",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
]


def load_data():
    """Load clean applications only for training."""
    print("Loading data...")
    df = pd.read_parquet(PROCESSED / "loans_clean.parquet")
    df = df[FEATURES + ["TARGET"]].dropna()

    # Train only on non-defaulted loans — these are our "normal" pattern
    df_clean = df[df["TARGET"] == 0].sample(n=10000, random_state=42)
    print(f"  Clean applications for training: {len(df_clean):,}")
    return df, df_clean


def train_model(df_clean):
    """Train Isolation Forest on clean applications."""
    print("\nTraining Isolation Forest...")

    scaler = StandardScaler()
    X_clean = scaler.fit_transform(df_clean[FEATURES])

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.02,  # Expect 2% anomalies
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_clean)
    print("  Training complete.")
    return iso_forest, scaler


def score_applications(iso_forest, scaler, df):
    """Score all applications for anomaly."""
    print("\nScoring all applications...")
    X = scaler.transform(df[FEATURES])

    # Anomaly score: lower = more anomalous
    df = df.copy()
    df["anomaly_score"]  = iso_forest.decision_function(X)
    df["fraud_flag"]     = iso_forest.predict(X)  # -1 = anomaly, 1 = normal
    df["fraud_flag"]     = (df["fraud_flag"] == -1).astype(int)

    flagged = df["fraud_flag"].sum()
    print(f"  Applications flagged as anomalous: {flagged:,} "
          f"({flagged/len(df):.1%})")

    return df


def plot_anomaly_scores(df):
    """Plot distribution of anomaly scores."""
    print("\nPlotting anomaly score distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Score distribution
    ax = axes[0]
    normal = df[df["fraud_flag"] == 0]["anomaly_score"]
    fraud  = df[df["fraud_flag"] == 1]["anomaly_score"]

    ax.hist(normal, bins=50, alpha=0.6, color="#3498DB",
            label=f"Normal ({len(normal):,})", density=True)
    ax.hist(fraud, bins=50, alpha=0.6, color="#E74C3C",
            label=f"Flagged ({len(fraud):,})", density=True)
    ax.set_title("Anomaly Score Distribution", fontweight="bold")
    ax.set_xlabel("Anomaly Score (lower = more suspicious)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Default rate by fraud flag
    ax = axes[1]
    default_rates = df.groupby("fraud_flag")["TARGET"].mean()
    bars = ax.bar(["Normal", "Fraud Flagged"],
                  default_rates.values,
                  color=["#3498DB", "#E74C3C"],
                  alpha=0.8, edgecolor="white")
    ax.set_title("Default Rate: Normal vs Fraud Flagged", fontweight="bold")
    ax.set_ylabel("Default Rate")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, default_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", fontweight="bold")

    plt.suptitle("CreditPulse — Fraud Detection Analysis",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    output = REPORTS / "fraud_detection.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {output}")


def save_model(iso_forest, scaler):
    """Save fraud detector artifacts."""
    joblib.dump(iso_forest, MODELS / "fraud_detector.pkl")
    joblib.dump(scaler,     MODELS / "fraud_scaler.pkl")
    print(f"  Fraud detector saved to models/")


def main():
    df, df_clean    = load_data()
    iso_forest, scaler = train_model(df_clean)
    df_scored       = score_applications(iso_forest, scaler, df)
    plot_anomaly_scores(df_scored)
    save_model(iso_forest, scaler)
    print("\nFraud detection complete!")


if __name__ == "__main__":
    main()