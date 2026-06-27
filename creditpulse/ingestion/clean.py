"""
CreditPulse - Data Cleaning & IFRS 9 Schema
Phase 2: Loads raw Home Credit data, cleans it, and adds IFRS 9 stage labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)


def load_raw_data():
    """Load the main application training file."""
    print("Loading raw data...")
    df = pd.read_csv(RAW / "application_train.csv")
    print(f"  Loaded {len(df):,} rows and {df.shape[1]} columns")
    return df


def clean_data(df):
    """Basic cleaning — handle missing values and outliers."""
    print("Cleaning data...")

    # DAYS_EMPLOYED has 365243 as a code for 'unemployed' — replace with 0
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0)

    # Fill missing EXT_SOURCE values with median
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing AMT_ANNUITY with median
    df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].median())

    print(f"  Cleaned. Missing values in TARGET: {df['TARGET'].isna().sum()}")
    return df


def engineer_features(df):
    """Create new features from existing columns."""
    print("Engineering features...")

    # What percentage of their life have they been employed?
    df["days_employed_pct"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # Loan amount relative to income
    df["income_to_credit_ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]

    # Average of credit bureau scores
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["ext_source_mean"] = df[ext_cols].mean(axis=1)
    df["ext_source_std"] = df[ext_cols].std(axis=1)

    print("  Features engineered.")
    return df


def add_ifrs9_stage(df, default_col="TARGET"):
    """
    Add IFRS 9 stage based on default flag.
    Stage 1 = performing, Stage 2 = watch, Stage 3 = non-performing.
    Will be updated with model probabilities in Phase 5.
    """
    print("Adding IFRS 9 stage labels...")

    conditions = [
        df[default_col] == 0,
        df[default_col] == 1,
    ]
    choices = [1, 3]
    df["ifrs9_stage"] = np.select(conditions, choices, default=2)

    print(f"  Stage distribution:\n{df['ifrs9_stage'].value_counts().sort_index()}")
    return df


def save_data(df):
    """Save cleaned data as parquet — faster and smaller than CSV."""
    output_path = PROCESSED / "loans_clean.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = add_ifrs9_stage(df)
    save_data(df)
    print("\nPhase 2 data cleaning complete!")


if __name__ == "__main__":
    main()