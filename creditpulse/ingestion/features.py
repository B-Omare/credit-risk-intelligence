"""
Feature engineering for CreditPulse.
Creates domain-specific features from cleaned loan data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/loans_clean.parquet")
OUTPUT_PATH = Path("data/processed/loans_features.parquet")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Age in years (DAYS_BIRTH is negative)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

    # Employment duration in years
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365).round(1)
        df["EMPLOYMENT_YEARS"] = df["EMPLOYMENT_YEARS"].clip(lower=0)

    # Credit-to-income ratio
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Annuity burden
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Loan-to-value proxy
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["LTV_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # External source composite score
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)

    # Family pressure indicator
    if {"CNT_CHILDREN", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].fillna(1) + 1)

    return df


def engineer(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    logger.info(f"Engineering features from {input_path}")
    df = pd.read_parquet(input_path)
    df = build_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Features saved to {output_path} — {df.shape[1]} total columns")
    return df


if __name__ == "__main__":
    engineer()
