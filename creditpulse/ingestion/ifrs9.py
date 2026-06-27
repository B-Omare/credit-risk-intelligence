"""
IFRS 9 schema mapper.
Adds Stage classification (1/2/3) and Expected Credit Loss columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/loans_features.parquet")
OUTPUT_PATH = Path("data/processed/loans_ifrs9.parquet")


def assign_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Assign IFRS 9 stages based on available signals."""
    df["IFRS9_STAGE"] = 1  # Default: performing

    # Stage 2: significant credit risk increase
    if "EXT_SOURCE_MEAN" in df.columns:
        df.loc[df["EXT_SOURCE_MEAN"] < 0.3, "IFRS9_STAGE"] = 2

    # Stage 3: credit-impaired (defaulted in training data)
    if "TARGET" in df.columns:
        df.loc[df["TARGET"] == 1, "IFRS9_STAGE"] = 3

    return df


def compute_ecl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified Expected Credit Loss calculation.
    ECL = PD × LGD × EAD
    """
    # Proxy PD from external sources (0-1 scale, inverted)
    if "EXT_SOURCE_MEAN" in df.columns:
        df["PD_PROXY"] = 1 - df["EXT_SOURCE_MEAN"].clip(0, 1)
    else:
        df["PD_PROXY"] = 0.05  # 5% default

    df["LGD"] = 0.45  # Basel standard for unsecured consumer loans
    df["EAD"] = df.get("AMT_CREDIT", pd.Series(0, index=df.index))
    df["ECL"] = df["PD_PROXY"] * df["LGD"] * df["EAD"]

    return df


def apply_ifrs9(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df = assign_stage(df)
    df = compute_ecl(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"IFRS 9 schema applied → {output_path}")
    return df


if __name__ == "__main__":
    apply_ifrs9()
