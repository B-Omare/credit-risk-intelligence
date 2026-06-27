"""
CreditPulse - FastAPI Microservice
Phase 7: REST API exposing three endpoints:
  - POST /score     — score a single borrower
  - POST /explain   — generate plain-language explanation
  - GET  /portfolio — portfolio-level risk summary
"""

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from creditpulse.nlp.explainer import explain_decision

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="CreditPulse API",
    description="Causal AI credit risk assessment for East African markets",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ───────────────────────────────────────────────────────────
MODELS = Path("models")

try:
    xgb_model      = joblib.load(MODELS / "xgboost_model.pkl")
    bayes_models   = joblib.load(MODELS / "bayesian_bootstrap_models.pkl")
    bayes_scaler   = joblib.load(MODELS / "bayesian_scaler.pkl")
    fraud_detector = joblib.load(MODELS / "fraud_detector.pkl")
    fraud_scaler   = joblib.load(MODELS / "fraud_scaler.pkl")
    print("All models loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    xgb_model = bayes_models = bayes_scaler = None
    fraud_detector = fraud_scaler = None

# ── XGBoost features ──────────────────────────────────────────────────────
XGB_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "ext_source_mean", "ext_source_std",
    "income_to_credit_ratio", "days_employed_pct",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
]

BAYES_FEATURES = [
    "EXT_SOURCE_2", "income_to_credit_ratio", "days_employed_pct"
]

FRAUD_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "income_to_credit_ratio", "days_employed_pct",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
]


# ── Request/Response models ───────────────────────────────────────────────
class BorrowerFeatures(BaseModel):
    EXT_SOURCE_1: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_2: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_3: float = Field(0.5, ge=0, le=1)
    ext_source_mean: float = Field(0.5, ge=0, le=1)
    ext_source_std: float = Field(0.1, ge=0)
    income_to_credit_ratio: float = Field(0.5, ge=0)
    days_employed_pct: float = Field(0.2, ge=0)
    AMT_CREDIT: float = Field(300000, ge=0)
    AMT_INCOME_TOTAL: float = Field(150000, ge=0)
    AMT_ANNUITY: float = Field(25000, ge=0)
    DAYS_BIRTH: float = Field(-12000)
    DAYS_EMPLOYED: float = Field(-2000)
    name: str = Field("Borrower")


class ScoreResponse(BaseModel):
    name: str
    default_probability: float
    bayesian_mean: float
    bayesian_ci_low: float
    bayesian_ci_high: float
    ifrs9_stage: int
    fraud_flag: int
    risk_label: str


class ExplainResponse(BaseModel):
    name: str
    decision: str
    explanation: str
    default_probability: float


# ── Helper functions ──────────────────────────────────────────────────────
def get_ifrs9_stage(prob: float) -> int:
    if prob < 0.05:
        return 1
    elif prob < 0.20:
        return 2
    return 3


def get_risk_label(prob: float) -> str:
    if prob < 0.10:
        return "Low Risk"
    elif prob < 0.30:
        return "Medium Risk"
    return "High Risk"


def predict_xgb(features: BorrowerFeatures) -> float:
    X = pd.DataFrame([[
        features.EXT_SOURCE_1, features.EXT_SOURCE_2,
        features.EXT_SOURCE_3, features.ext_source_mean,
        features.ext_source_std, features.income_to_credit_ratio,
        features.days_employed_pct, features.AMT_CREDIT,
        features.AMT_INCOME_TOTAL, features.AMT_ANNUITY,
        features.DAYS_BIRTH, features.DAYS_EMPLOYED,
    ]], columns=XGB_FEATURES)
    return float(xgb_model.predict_proba(X)[0, 1])


def predict_bayesian(features: BorrowerFeatures):
    X = np.array([[
        features.EXT_SOURCE_2,
        features.income_to_credit_ratio,
        features.days_employed_pct,
    ]])
    X_scaled = bayes_scaler.transform(X)
    probs = np.array([m.predict_proba(X_scaled)[0, 1]
                      for m in bayes_models])
    return float(probs.mean()), float(np.percentile(probs, 2.5)), \
           float(np.percentile(probs, 97.5))


def predict_fraud(features: BorrowerFeatures) -> int:
    X = np.array([[
        features.EXT_SOURCE_1, features.EXT_SOURCE_2,
        features.EXT_SOURCE_3, features.income_to_credit_ratio,
        features.days_employed_pct, features.AMT_CREDIT,
        features.AMT_INCOME_TOTAL, features.AMT_ANNUITY,
    ]])
    X_scaled = fraud_scaler.transform(X)
    pred = fraud_detector.predict(X_scaled)
    return int(pred[0] == -1)


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "CreditPulse API",
        "version": "0.1.0",
        "endpoints": ["/score", "/explain", "/portfolio", "/docs"]
    }


@app.post("/score", response_model=ScoreResponse)
def score_borrower(features: BorrowerFeatures):
    """Score a single borrower — returns default probability + IFRS 9 stage."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    prob          = predict_xgb(features)
    bayes_mean, ci_lo, ci_hi = predict_bayesian(features)
    fraud_flag    = predict_fraud(features)
    ifrs9_stage   = get_ifrs9_stage(prob)
    risk_label    = get_risk_label(prob)

    return ScoreResponse(
        name=features.name,
        default_probability=round(prob, 4),
        bayesian_mean=round(bayes_mean, 4),
        bayesian_ci_low=round(ci_lo, 4),
        bayesian_ci_high=round(ci_hi, 4),
        ifrs9_stage=ifrs9_stage,
        fraud_flag=fraud_flag,
        risk_label=risk_label,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_borrower(features: BorrowerFeatures):
    """Generate a plain-language explanation for a loan decision."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    prob       = predict_xgb(features)
    decision   = "approved" if prob < 0.30 else "declined"

    borrower_data = {
        "name": features.name,
        "ext_source_mean": features.ext_source_mean,
        "income_to_credit_ratio": features.income_to_credit_ratio,
        "days_employed_pct": features.days_employed_pct,
        "mpesa_avg_balance_30d": 5000,
    }
    explanation = explain_decision(borrower_data, decision)

    return ExplainResponse(
        name=features.name,
        decision=decision,
        explanation=explanation,
        default_probability=round(prob, 4),
    )


@app.get("/portfolio")
def portfolio_summary():
    """Return portfolio-level risk summary from processed data."""
    try:
        df = pd.read_parquet("data/processed/loans_clean.parquet")
        total     = len(df)
        default_rate = float(df["TARGET"].mean())

        stage_dist = {
            "stage_1": int((df["ifrs9_stage"] == 1).sum()),
            "stage_2": int((df["ifrs9_stage"] == 2).sum()),
            "stage_3": int((df["ifrs9_stage"] == 3).sum()),
        }

        return {
            "total_loans": total,
            "default_rate": round(default_rate, 4),
            "ifrs9_distribution": stage_dist,
            "avg_credit_amount": round(float(df["AMT_CREDIT"].mean()), 2),
            "avg_income": round(float(df["AMT_INCOME_TOTAL"].mean()), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))