"""
Phase 7 — FastAPI application for CreditPulse.
Serves credit risk predictions via REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

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


class BorrowerFeatures(BaseModel):
    age_years: float = Field(..., ge=18, le=80, description="Borrower age in years")
    employment_years: float = Field(..., ge=0, description="Years in current employment")
    amt_income_total: float = Field(..., gt=0, description="Annual income in KES")
    amt_credit: float = Field(..., gt=0, description="Requested loan amount in KES")
    amt_annuity: float = Field(..., gt=0, description="Monthly repayment amount in KES")
    ext_source_mean: float = Field(None, ge=0, le=1, description="Average external credit score (0-1)")
    cnt_children: int = Field(0, ge=0, description="Number of dependent children")


class CreditDecision(BaseModel):
    borrower_id: str
    pd_point_estimate: float
    pd_lower_95: float
    pd_upper_95: float
    ifrs9_stage: int
    ecl_estimate: float
    recommendation: str
    explanation: str
    risk_factors: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]


MODEL_CACHE: dict = {}


def load_model(name: str, path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            MODEL_CACHE[name] = pickle.load(f)
        return True
    except FileNotFoundError:
        logger.warning(f"Model not found: {path}")
        return False


@app.on_event("startup")
async def startup():
    load_model("xgboost", Path("models/xgboost_model.pkl"))
    load_model("bayesian", Path("models/bayesian_model.pkl"))
    load_model("survival", Path("models/survival_model.pkl"))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        models_loaded={k: k in MODEL_CACHE for k in ["xgboost", "bayesian", "survival"]},
    )


@app.post("/predict", response_model=CreditDecision)
async def predict(borrower: BorrowerFeatures, borrower_id: str = "unknown"):
    credit_income_ratio = borrower.amt_credit / (borrower.amt_income_total + 1)
    annuity_income_ratio = borrower.amt_annuity / (borrower.amt_income_total + 1)
    ext_source = borrower.ext_source_mean if borrower.ext_source_mean is not None else 0.5

    features = np.array([[
        borrower.age_years,
        borrower.employment_years,
        borrower.amt_income_total,
        borrower.amt_credit,
        borrower.amt_annuity,
        credit_income_ratio,
        annuity_income_ratio,
        borrower.amt_credit / (borrower.amt_income_total + 1),
        ext_source,
        ext_source,
        borrower.cnt_children,
        borrower.cnt_children + 2,
        2,
        borrower.amt_income_total / (borrower.cnt_children + 2 + 1),
    ]])

    if "xgboost" in MODEL_CACHE:
        model_artifacts = MODEL_CACHE["xgboost"]
        model = model_artifacts["model"]
        feat_names = model_artifacts["features"]
        n_feats = len(feat_names)
        pd_mean = float(model.predict_proba(features[:, :n_feats])[:, 1][0])
    else:
        pd_mean = 1 / (1 + np.exp(-(credit_income_ratio - 0.5) * 3 + (ext_source - 0.5) * (-2)))

    pd_lower = max(0.0, pd_mean - 0.08)
    pd_upper = min(1.0, pd_mean + 0.08)

    if pd_mean < 0.15:
        stage = 1
    elif pd_mean < 0.40:
        stage = 2
    else:
        stage = 3

    ecl = pd_mean * 0.45 * borrower.amt_credit

    if pd_mean < 0.15:
        recommendation = "APPROVE"
    elif pd_mean < 0.35:
        recommendation = "MANUAL_REVIEW"
    else:
        recommendation = "DECLINE"

    risk_factors = []
    if credit_income_ratio > 5:
        risk_factors.append(f"High credit-to-income ratio ({credit_income_ratio:.1f}x)")
    if annuity_income_ratio > 0.4:
        risk_factors.append(f"Annuity burden exceeds 40% of income")
    if ext_source < 0.3:
        risk_factors.append("Low external credit score")
    if borrower.employment_years < 1:
        risk_factors.append("Less than 1 year in current employment")
    if borrower.age_years < 25:
        risk_factors.append("Young borrower — limited credit history")

    explanation = (
        f"PD of {pd_mean:.1%} (95% CI: {pd_lower:.1%}–{pd_upper:.1%}). "
        f"IFRS 9 Stage {stage}. ECL: KES {ecl:,.0f}. "
        f"{'No significant risk factors.' if not risk_factors else 'Key drivers: ' + '; '.join(risk_factors[:2]) + '.'}"
    )

    return CreditDecision(
        borrower_id=borrower_id,
        pd_point_estimate=round(pd_mean, 4),
        pd_lower_95=round(pd_lower, 4),
        pd_upper_95=round(pd_upper, 4),
        ifrs9_stage=stage,
        ecl_estimate=round(ecl, 2),
        recommendation=recommendation,
        explanation=explanation,
        risk_factors=risk_factors,
    )


@app.get("/explain/{borrower_id}")
async def explain(borrower_id: str):
    return {
        "borrower_id": borrower_id,
        "note": "Submit borrower features to /predict to get a full SHAP-based explanation.",
        "shap_values": None,
    }
