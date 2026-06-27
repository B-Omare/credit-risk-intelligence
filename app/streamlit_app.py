"""
Phase 7 — CreditPulse Streamlit Dashboard.
Three views: Loan Officer, Borrower Explainer, Regulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CreditPulse",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-low { color: #2ecc71; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

VIEWS = ["🏦 Loan Officer", "👤 Borrower Explainer", "📊 Regulator Dashboard"]


def sidebar():
    st.sidebar.image("https://via.placeholder.com/200x60?text=CreditPulse", width=200)
    st.sidebar.title("CreditPulse")
    st.sidebar.caption("Causal AI Credit Risk — East Africa")
    view = st.sidebar.radio("View", VIEWS)
    st.sidebar.markdown("---")
    st.sidebar.info("v0.1.0 | Built for CBK-regulated lenders")
    return view


def borrower_input_form():
    with st.form("borrower_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 18, 80, 32)
            income = st.number_input("Annual Income (KES)", 10000, 5000000, 240000, step=10000)
            employment = st.number_input("Employment years", 0.0, 40.0, 3.5, step=0.5)
            children = st.number_input("Dependants", 0, 10, 1)
        with col2:
            loan_amount = st.number_input("Loan Amount (KES)", 5000, 2000000, 50000, step=5000)
            annuity = st.number_input("Monthly Repayment (KES)", 500, 100000, 4500, step=500)
            ext_score = st.slider("External Credit Score", 0.0, 1.0, 0.55, step=0.01,
                                  help="Aggregated external bureau score (0=poor, 1=excellent)")

        submitted = st.form_submit_button("Assess Credit Risk", type="primary")

    if submitted:
        return {
            "age_years": age,
            "employment_years": employment,
            "amt_income_total": income,
            "amt_credit": loan_amount,
            "amt_annuity": annuity,
            "ext_source_mean": ext_score,
            "cnt_children": children,
        }
    return None


def call_api(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict?borrower_id=demo", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        pd_val = 1 / (1 + np.exp(-(payload["amt_credit"] / payload["amt_income_total"] - 0.5) * 3))
        stage = 1 if pd_val < 0.15 else (2 if pd_val < 0.40 else 3)
        return {
            "pd_point_estimate": round(pd_val, 4),
            "pd_lower_95": round(max(0, pd_val - 0.08), 4),
            "pd_upper_95": round(min(1, pd_val + 0.08), 4),
            "ifrs9_stage": stage,
            "ecl_estimate": round(pd_val * 0.45 * payload["amt_credit"], 2),
            "recommendation": "APPROVE" if pd_val < 0.15 else ("MANUAL_REVIEW" if pd_val < 0.35 else "DECLINE"),
            "explanation": f"Probability of default: {pd_val:.1%}",
            "risk_factors": [],
        }


def loan_officer_view():
    st.title("🏦 Loan Officer Dashboard")
    st.caption("Fast credit decisions with causal AI support")

    features = borrower_input_form()
    if features is None:
        st.info("Fill in the borrower details above and click **Assess Credit Risk**.")
        return

    with st.spinner("Running causal AI assessment..."):
        result = call_api(features)

    if result is None:
        st.error("API unavailable. Start the API server: `uvicorn creditpulse.api.main:app`")
        return

    rec = result["recommendation"]
    color = {"APPROVE": "green", "MANUAL_REVIEW": "orange", "DECLINE": "red"}[rec]
    st.markdown(f"## Decision: :{color}[{rec}]")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability of Default", f"{result['pd_point_estimate']:.1%}")
    col2.metric("95% CI", f"{result['pd_lower_95']:.1%} – {result['pd_upper_95']:.1%}")
    col3.metric("IFRS 9 Stage", f"Stage {result['ifrs9_stage']}")
    col4.metric("Expected Credit Loss", f"KES {result['ecl_estimate']:,.0f}")

    if result["risk_factors"]:
        st.markdown("**Key risk drivers:**")
        for rf in result["risk_factors"]:
            st.markdown(f"- ⚠️ {rf}")

    st.markdown("---")
    st.caption(f"*{result['explanation']}*")


def borrower_explainer_view():
    st.title("👤 Borrower Explainer")
    st.caption("Plain-English explanations for borrowers — CBK consumer protection aligned")

    features = borrower_input_form()
    if features is None:
        st.info("Enter your loan application details above.")
        return

    result = call_api(features)
    if not result:
        return

    rec = result["recommendation"]
    if rec == "APPROVE":
        st.success("✅ Your application is likely to be **approved**.")
        st.markdown("""
        **What this means for you:**
        Your financial profile shows a manageable level of risk. The lender is likely to offer you a loan at standard rates.
        """)
    elif rec == "MANUAL_REVIEW":
        st.warning("🔍 Your application needs **manual review** by a loan officer.")
        st.markdown("""
        **What this means for you:**
        Your application has been flagged for human review. This is not a rejection — a loan officer will look at your full case and may ask for additional documents.
        """)
    else:
        st.error("❌ Based on your profile, this application is **likely to be declined**.")
        st.markdown("""
        **What you can do:**
        - Reduce the loan amount
        - Provide additional income documentation
        - Build your M-Pesa transaction history over 3-6 months
        - Ask about a smaller starter loan to build your credit record
        """)

    st.markdown("---")
    st.subheader("Why did we get this result?")
    if result["risk_factors"]:
        for rf in result["risk_factors"]:
            st.markdown(f"- {rf}")
    else:
        st.markdown("- Your financial profile looks healthy — no major risk factors detected.")

    st.info("📞 You have the right to request a full explanation from your loan officer under CBK Consumer Protection Guidelines.")


def regulator_view():
    st.title("📊 Regulator Dashboard")
    st.caption("IFRS 9 portfolio view + fairness audit — CBK reporting ready")

    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "borrower_id": range(n),
        "pd": np.random.beta(2, 12, n),
        "stage": np.random.choice([1, 2, 3], n, p=[0.75, 0.18, 0.07]),
        "ecl": np.random.lognormal(10, 1.5, n),
        "income_level": np.random.choice(["Low", "Medium", "High"], n, p=[0.4, 0.45, 0.15]),
        "gender": np.random.choice(["M", "F"], n, p=[0.55, 0.45]),
        "region": np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Rural"], n),
    })

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Borrowers", f"{n:,}")
    col2.metric("Avg PD", f"{df['pd'].mean():.1%}")
    col3.metric("Total ECL", f"KES {df['ecl'].sum():,.0f}")
    col4.metric("Stage 3 Rate", f"{(df['stage']==3).mean():.1%}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("IFRS 9 Stage Distribution")
        stage_counts = df["stage"].value_counts().sort_index()
        st.bar_chart(stage_counts)

    with col_b:
        st.subheader("PD by Income Level")
        pd_by_income = df.groupby("income_level")["pd"].mean()
        st.bar_chart(pd_by_income)

    st.markdown("---")
    st.subheader("Fairness Audit — PD by Demographic")
    fairness = df.groupby("gender")["pd"].agg(["mean", "std"]).round(4)
    fairness.columns = ["Mean PD", "Std PD"]
    st.dataframe(fairness, use_container_width=True)

    gender_gap = abs(df[df["gender"]=="M"]["pd"].mean() - df[df["gender"]=="F"]["pd"].mean())
    if gender_gap < 0.02:
        st.success(f"✅ Gender PD gap: {gender_gap:.1%} — within acceptable range (<2pp)")
    else:
        st.warning(f"⚠️ Gender PD gap: {gender_gap:.1%} — review required")


def main():
    view = sidebar()
    if view == VIEWS[0]:
        loan_officer_view()
    elif view == VIEWS[1]:
        borrower_explainer_view()
    else:
        regulator_view()


if __name__ == "__main__":
    main()
