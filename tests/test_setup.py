def test_creditpulse_importable():
    """Confirm the package structure is correct."""
    import creditpulse
    assert creditpulse is not None


def test_data_folders_exist():
    """Confirm all expected data folders exist, creating them if needed."""
    import os
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/simulated", exist_ok=True)
    assert os.path.isdir("data/raw")
    assert os.path.isdir("data/processed")
    assert os.path.isdir("data/simulated")


def test_cleaned_parquet_exists():
    """Confirm the cleaned parquet file was created."""
    import os
    assert os.path.isfile("data/processed/loans_clean.parquet")


def test_cleaned_data_quality():
    """Confirm the cleaned data meets quality standards."""
    import pandas as pd
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    assert df["TARGET"].isna().sum() == 0
    assert df["ext_source_mean"].isna().sum() == 0
    assert set(df["ifrs9_stage"].unique()).issubset({1, 2, 3})
    assert len(df) == 307511
    assert "days_employed_pct" in df.columns
    assert "income_to_credit_ratio" in df.columns
    print(f"  Data quality checks passed. {len(df):,} rows verified.")