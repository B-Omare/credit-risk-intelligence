"""
Phase 2 — Data ingestion and cleaning.
Transforms raw Home Credit CSV into a clean parquet file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/application_train.csv")
OUTPUT_PATH = Path("data/processed/loans_clean.parquet")


COLS_TO_KEEP = [
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

ANOMALY_REPLACEMENTS = {
    "DAYS_EMPLOYED": 365243,
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, usecols=lambda c: c in COLS_TO_KEEP)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    for col, sentinel in ANOMALY_REPLACEMENTS.items():
        if col in df.columns:
            mask = df[col] == sentinel
            logger.info(f"Replacing {mask.sum()} anomalous values in {col}")
            df.loc[mask, col] = np.nan
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    binary_map = {"Y": 1, "N": 0, "M": 1, "F": 0, "XNA": np.nan}
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    df = pd.get_dummies(df, columns=["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                                      "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                                      "WEEKDAY_APPR_PROCESS_START"], drop_first=True)
    return df


def clean(input_path: Path = RAW_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = load_raw(input_path)
    df = fix_anomalies(df)
    df = impute_missing(df)
    df = encode_categoricals(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved clean data to {output_path} — {len(df):,} rows")
    return df


if __name__ == "__main__":
    clean()
