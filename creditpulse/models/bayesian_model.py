"""
CreditPulse - Bayesian Credit Scorer (Fast Approximation)
Phase 5: Uses Bootstrap resampling to approximate Bayesian uncertainty.
Output: Full probability distribution per borrower — not just a single score.
This approach gives the same key benefit as PyMC (uncertainty quantification)
in seconds instead of hours.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

FEATURES = ["EXT_SOURCE_2", "income_to_credit_ratio", "days_employed_pct"]
N_BOOTSTRAP = 200  # Number of bootstrap models


def load_data():
    """Load cleaned data."""
    print("Loading data...")
    df = pd.read_parquet(PROCESSED / "loans_clean.parquet")
    df = df[FEATURES + ["TARGET"]].dropna()
    print(f"  Loaded {len(df):,} rows")
    return df


def fit_bootstrap_models(df):
    """
    Fit N_BOOTSTRAP logistic regression models on bootstrap samples.
    Each model gives a slightly different probability estimate.
    The spread across models = our uncertainty about the true probability.
    """
    print(f"\nFitting {N_BOOTSTRAP} bootstrap models...")

    X = df[FEATURES].values
    y = df["TARGET"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = []
    for i in range(N_BOOTSTRAP):
        # Bootstrap sample: sample with replacement
        idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_boot = X_scaled[idx]
        y_boot = y[idx]

        model = LogisticRegression(
            random_state=i,
            class_weight="balanced",
            max_iter=200
        )
        model.fit(X_boot, y_boot)
        models.append(model)

        if (i + 1) % 50 == 0:
            print(f"  Fitted {i + 1}/{N_BOOTSTRAP} models...")

    print("  All bootstrap models fitted.")
    return models, scaler


def predict_with_uncertainty(models, scaler, borrower_features):
    """
    Predict default probability with uncertainty for a single borrower.
    Returns mean probability and 95% credible interval.
    """
    X = np.array([borrower_features])
    X_scaled = scaler.transform(X)

    # Get prediction from each bootstrap model
    probs = [m.predict_proba(X_scaled)[0, 1] for m in models]
    probs = np.array(probs)

    mean_prob = probs.mean()
    ci_lo     = np.percentile(probs, 2.5)
    ci_hi     = np.percentile(probs, 97.5)

    return mean_prob, ci_lo, ci_hi, probs


def plot_uncertainty(probs, borrower_name, mean_prob, ci_lo, ci_hi):
    """Plot the uncertainty distribution for a single borrower."""
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(probs, bins=30, color="#3498DB", alpha=0.7, edgecolor="white")
    ax.axvline(mean_prob, color="#E74C3C", linewidth=2,
               label=f"Mean: {mean_prob:.1%}")
    ax.axvline(ci_lo, color="#E67E22", linewidth=1.5, linestyle="--",
               label=f"95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    ax.axvline(ci_hi, color="#E67E22", linewidth=1.5, linestyle="--")

    ax.fill_betweenx(
        [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
        ci_lo, ci_hi, alpha=0.15, color="#E67E22"
    )

    ax.set_title(f"CreditPulse — Bayesian Uncertainty for {borrower_name}",
                 fontweight="bold")
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Frequency across bootstrap models")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output = REPORTS / f"bayesian_uncertainty.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Uncertainty plot saved to {output}")


def save_artifacts(models, scaler):
    """Save bootstrap models and scaler."""
    joblib.dump(models, MODELS / "bayesian_bootstrap_models.pkl")
    joblib.dump(scaler, MODELS / "bayesian_scaler.pkl")
    print(f"  Models saved to models/")


def main():
    np.random.seed(42)
    df = load_data()
    models, scaler = fit_bootstrap_models(df)

    # Test on three sample borrowers
    test_borrowers = [
        {
            "name": "Jane Wanjiku (Low Risk)",
            "features": [0.75, 0.65, 0.40]  # EXT_SOURCE_2, income_ratio, employed_pct
        },
        {
            "name": "John Kamau (High Risk)",
            "features": [0.25, 0.15, 0.05]
        },
        {
            "name": "Mary Achieng (Medium Risk)",
            "features": [0.50, 0.40, 0.20]
        },
    ]

    print("\n── Bayesian Uncertainty Estimates ───────────────────────")
    for borrower in test_borrowers:
        mean_prob, ci_lo, ci_hi, probs = predict_with_uncertainty(
            models, scaler, borrower["features"]
        )
        print(f"\n  {borrower['name']}")
        print(f"    Mean default probability: {mean_prob:.1%}")
        print(f"    95% credible interval:   [{ci_lo:.1%}, {ci_hi:.1%}]")

    # Plot uncertainty for the medium risk borrower
    mean_prob, ci_lo, ci_hi, probs = predict_with_uncertainty(
        models, scaler, test_borrowers[2]["features"]
    )
    plot_uncertainty(probs, test_borrowers[2]["name"], mean_prob, ci_lo, ci_hi)

    save_artifacts(models, scaler)
    print("\nBayesian modelling complete!")


if __name__ == "__main__":
    main()