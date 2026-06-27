"""
CreditPulse - Random Survival Forest (Fast Version)
Phase 5: Models TIME-TO-DEFAULT using a sample for speed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

FEATURES = [
    "EXT_SOURCE_2", "ext_source_mean",
    "income_to_credit_ratio", "days_employed_pct",
    "AMT_CREDIT", "AMT_INCOME_TOTAL",
]


def load_data(n=5000):
    """Load a sample for speed."""
    print(f"Loading {n:,} row sample...")
    df = pd.read_parquet(PROCESSED / "loans_clean.parquet")
    df = df[FEATURES + ["TARGET", "DAYS_BIRTH"]].dropna()
    df = df.sample(n=n, random_state=42)

    # Time proxy: age in years
    df["time"] = (np.abs(df["DAYS_BIRTH"]) / 365).clip(lower=0.1)

    print(f"  Sample loaded. Default rate: {df['TARGET'].mean():.2%}")
    return df


def prepare_survival_target(df):
    """Vectorised structured array creation — much faster."""
    y = np.empty(len(df), dtype=[("default", bool), ("time", float)])
    y["default"] = df["TARGET"].astype(bool).values
    y["time"]    = df["time"].values
    return y


def train_model(df):
    """Train the Random Survival Forest."""
    print("\nPreparing survival target...")
    y = prepare_survival_target(df)
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    print("\nTraining Random Survival Forest...")
    rsf = RandomSurvivalForest(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)
    print("  Training complete.")
    return rsf, X_test, y_test


def evaluate_model(rsf, X_test, y_test):
    """Evaluate using concordance index."""
    print("\n── Survival Model Performance ───────────────")
    prediction = rsf.predict(X_test)
    ci = concordance_index_censored(
        y_test["default"], y_test["time"], prediction
    )
    print(f"  C-index: {ci[0]:.4f}")
    print(f"  (> 0.7 is good, 1.0 is perfect, 0.5 is random)")
    return ci[0]


def plot_survival_curves(rsf):
    """Plot survival curves for three sample borrowers."""
    print("\nPlotting survival curves...")

    sample_borrowers = {
        "Low Risk (Jane Wanjiku)":    [0.75, 0.70, 0.65, 0.40, 200000, 150000],
        "Medium Risk (Mary Achieng)": [0.50, 0.50, 0.40, 0.20, 350000, 100000],
        "High Risk (John Kamau)":     [0.25, 0.30, 0.15, 0.05, 500000,  60000],
    }

    X_sample = pd.DataFrame(
        list(sample_borrowers.values()),
        columns=FEATURES,
        index=list(sample_borrowers.keys())
    )

    survival_fns = rsf.predict_survival_function(X_sample)

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#2ECC71", "#E67E22", "#E74C3C"]

    for (name, _), fn, colour in zip(
            sample_borrowers.items(), survival_fns, colours):
        ax.step(fn.x, fn(fn.x), label=name, color=colour, linewidth=2.5)

    ax.set_title(
        "CreditPulse — Survival Curves: Probability of Not Defaulting Over Time",
        fontweight="bold")
    ax.set_xlabel("Years")
    ax.set_ylabel("Probability of No Default")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    output = REPORTS / "survival_curves.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Survival curves saved to {output}")


def save_model(rsf):
    """Save the trained model."""
    output = MODELS / "survival_model.pkl"
    joblib.dump(rsf, output)
    print(f"  Model saved to {output}")


def main():
    df = load_data(n=5000)
    rsf, X_test, y_test = train_model(df)
    evaluate_model(rsf, X_test, y_test)
    plot_survival_curves(rsf)
    save_model(rsf)
    print("\nSurvival modelling complete!")


if __name__ == "__main__":
    main()