"""
Phase 5 — Survival analysis for time-to-default modelling.
Uses scikit-survival's Random Survival Forest.
"""

import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/loans_ifrs9.parquet")
OUTPUT_PATH = Path("models/survival_model.pkl")

FEATURE_COLS = [
    "AGE_YEARS", "EMPLOYMENT_YEARS", "AMT_INCOME_TOTAL",
    "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN", "CNT_CHILDREN",
]


def train(input_path: Path = INPUT_PATH) -> dict:
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(input_path)
        available = [c for c in FEATURE_COLS if c in df.columns]
        sub = df[available + ["TARGET"]].dropna()
        sub = sub.sample(min(10000, len(sub)), random_state=42)

        # Simulate time-to-event (months until default / censoring)
        np.random.seed(42)
        n = len(sub)
        time_to_event = np.where(
            sub["TARGET"] == 1,
            np.random.exponential(scale=12, size=n).clip(1, 36),
            np.random.uniform(24, 60, size=n),
        )
        sub["time"] = time_to_event.astype(int)
        sub["event"] = sub["TARGET"].astype(bool)

        y = Surv.from_dataframe("event", "time", sub)
        X = sub[available].fillna(sub[available].median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rsf = RandomSurvivalForest(
            n_estimators=100, min_samples_split=10,
            random_state=42, n_jobs=-1,
        )
        rsf.fit(X_train, y_train)
        score = rsf.score(X_test, y_test)
        logger.info(f"Survival Forest C-index: {score:.4f}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump({"model": rsf, "features": available, "c_index": score}, f)

        return {"c_index": float(score), "features": available}

    except ImportError:
        logger.warning("scikit-survival not installed")
        return {"error": "scikit-survival not installed"}


if __name__ == "__main__":
    train()
