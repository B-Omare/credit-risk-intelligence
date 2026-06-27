"""
CreditPulse - Three-View Streamlit Dashboard
Phase 7: Interactive dashboard for loan officers, executives, and regulators.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditPulse",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-card-front-side.png",
                 width=80)
st.sidebar.title("CreditPulse")
st.sidebar.markdown("*Causal AI Credit Risk Assessment*")
st.sidebar.markdown("---")

view = st.sidebar.radio(
    "Select View",
    ["🏦 Loan Officer", "📊 Executive", "📋 Policy & Compliance"]
)

# ══════════════════════════════════════════════════════════════════════════
# VIEW 1 — LOAN OFFICER
# ══════════════════════════════════════════════════════════════════════════
if view == "🏦 Loan Officer":
    st.title("🏦 Loan Officer View")
    st.markdown("Score an individual borrower and get a plain-language explanation.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Borrower Details")
        name           = st.text_input("Borrower Name", "Jane Wanjiku")
        ext_source_2   = st.slider("Credit Bureau Score", 0.0, 1.0, 0.65,
                                    help="Higher = better credit history")
        income_ratio   = st.slider("Income to Credit Ratio", 0.0, 2.0, 0.70,
                                    help="Higher = more affordable loan")
        employed_pct   = st.slider("Employment Stability", 0.0, 1.0, 0.40,
                                    help="Higher = longer employment history")
        amt_credit     = st.number_input("Loan Amount (KES)", 10000, 2000000, 300000)
        amt_income     = st.number_input("Annual Income (KES)", 10000, 5000000, 180000)

    with col2:
        st.subheader("Risk Assessment")

        if st.button("🔍 Score Borrower", type="primary"):
            payload = {
                "name": name,
                "EXT_SOURCE_1": ext_source_2 * 0.9,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_2 * 1.1 if ext_source_2 * 1.1 <= 1 else 1.0,
                "ext_source_mean": ext_source_2,
                "ext_source_std": 0.08,
                "income_to_credit_ratio": income_ratio,
                "days_employed_pct": employed_pct,
                "AMT_CREDIT": amt_credit,
                "AMT_INCOME_TOTAL": amt_income,
                "AMT_ANNUITY": amt_credit / 12,
                "DAYS_BIRTH": -13000,
                "DAYS_EMPLOYED": -2000,
            }

            with st.spinner("Scoring borrower..."):
                try:
                    # Score
                    score_resp = requests.post(
                        f"{API_URL}/score", json=payload, timeout=30
                    ).json()

                    # Explain
                    explain_resp = requests.post(
                        f"{API_URL}/explain", json=payload, timeout=30
                    ).json()

                    prob  = score_resp["default_probability"]
                    stage = score_resp["ifrs9_stage"]
                    label = score_resp["risk_label"]
                    fraud = score_resp["fraud_flag"]
                    b_low = score_resp["bayesian_ci_low"]
                    b_hi  = score_resp["bayesian_ci_high"]

                    # ── Metrics ───────────────────────────────────────────
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Default Probability", f"{prob:.1%}")
                    m2.metric("IFRS 9 Stage", f"Stage {stage}")
                    m3.metric("Risk Label", label)

                    # ── Fraud flag ────────────────────────────────────────
                    if fraud:
                        st.error("⚠️ Fraud flag raised — refer to fraud team")
                    else:
                        st.success("✅ No fraud signals detected")

                    # ── Bayesian uncertainty ───────────────────────────────
                    st.markdown("**Bayesian Uncertainty Range**")
                    st.progress(prob)
                    st.caption(
                        f"95% credible interval: "
                        f"[{b_low:.1%} — {b_hi:.1%}]"
                    )

                    # ── Risk gauge ────────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(6, 1.5))
                    colour = ("#2ECC71" if prob < 0.10
                              else "#E67E22" if prob < 0.30
                              else "#E74C3C")
                    ax.barh(["Risk"], [prob], color=colour, height=0.4)
                    ax.barh(["Risk"], [1 - prob], left=[prob],
                            color="#EEEEEE", height=0.4)
                    ax.axvline(0.10, color="#2ECC71", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.axvline(0.30, color="#E67E22", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Default Probability")
                    ax.set_title(f"{name} — {label}", fontweight="bold")
                    ax.grid(False)
                    st.pyplot(fig)
                    plt.close()

                    # ── Plain-language explanation ─────────────────────────
                    st.markdown("**Decision Explanation**")
                    decision = explain_resp["decision"].upper()
                    box_colour = (
                        "🟢" if decision == "APPROVED" else "🔴"
                    )
                    st.info(
                        f"{box_colour} **{decision}**\n\n"
                        f"{explain_resp['explanation']}"
                    )

                except Exception as e:
                    st.error(f"API error: {e}. Make sure the API is running.")

# ══════════════════════════════════════════════════════════════════════════
# VIEW 2 — EXECUTIVE
# ══════════════════════════════════════════════════════════════════════════
elif view == "📊 Executive":
    st.title("📊 Executive Portfolio View")
    st.markdown("Portfolio-level risk overview and IFRS 9 staging.")
    st.markdown("---")

    with st.spinner("Loading portfolio data..."):
        try:
            port = requests.get(f"{API_URL}/portfolio", timeout=30).json()

            # ── Portfolio metrics ─────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Loans", f"{port['total_loans']:,}")
            m2.metric("Default Rate", f"{port['default_rate']:.1%}")
            m3.metric("Avg Loan Size",
                      f"KES {port['avg_credit_amount']:,.0f}")
            m4.metric("Avg Income",
                      f"KES {port['avg_income']:,.0f}")

            st.markdown("---")

            # ── IFRS 9 distribution ───────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("IFRS 9 Stage Distribution")
                stages = port["ifrs9_distribution"]
                labels = ["Stage 1\n(Performing)",
                          "Stage 2\n(Watch)",
                          "Stage 3\n(Non-Performing)"]
                values = [stages["stage_1"],
                          stages["stage_2"],
                          stages["stage_3"]]
                colours = ["#2ECC71", "#E67E22", "#E74C3C"]

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(labels, values, color=colours, alpha=0.85,
                              edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 500,
                            f"{val:,}", ha="center", fontweight="bold")
                ax.set_title("Loan Portfolio by IFRS 9 Stage",
                             fontweight="bold")
                ax.set_ylabel("Number of Loans")
                ax.grid(axis="y", alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Portfolio Risk Breakdown")
                total = sum(values)
                for label, val, colour in zip(
                        ["Stage 1", "Stage 2", "Stage 3"],
                        values, colours):
                    pct = val / total
                    st.markdown(f"**{label}**")
                    st.progress(pct)
                    st.caption(f"{val:,} loans ({pct:.1%})")

        except Exception as e:
            st.error(f"Could not load portfolio: {e}")

    # ── Causal scenario ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Causal Scenario Modelling")
    st.markdown(
        "Based on the Difference-in-Differences analysis, "
        "estimate the portfolio impact of a macro shock."
    )

    shock_pct = st.slider(
        "Income shock severity (% of borrowers affected)", 0, 50, 15
    )
    did_effect = -0.1125
    affected   = int(port["total_loans"] * shock_pct / 100)
    extra_defaults = int(affected * abs(did_effect) * 0.08)

    c1, c2, c3 = st.columns(3)
    c1.metric("Borrowers Affected", f"{affected:,}")
    c2.metric("Estimated Extra Defaults", f"{extra_defaults:,}")
    c3.metric("Portfolio Impact",
              f"{extra_defaults / port['total_loans']:.2%}")

# ══════════════════════════════════════════════════════════════════════════
# VIEW 3 — POLICY & COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════
elif view == "📋 Policy & Compliance":
    st.title("📋 Policy & Compliance View")
    st.markdown(
        "Fairness audit results, causal evidence, and regulatory compliance."
    )
    st.markdown("---")

    # ── Fairness metrics ──────────────────────────────────────────────────
    st.subheader("Fairness Audit — Borrower Segments")
    try:
        fairness_df = pd.read_csv("reports/fairness_metrics.csv")
        st.dataframe(fairness_df, use_container_width=True)
    except Exception:
        st.warning("Fairness metrics file not found.")

    st.markdown("---")

    # ── Causal evidence ───────────────────────────────────────────────────
    st.subheader("Causal Evidence Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Difference-in-Differences (COVID Experiment)**")
        st.metric("DiD Coefficient", "-0.1125")
        st.metric("P-value", "0.002")
        st.caption(
            "Income shocks causally increase default risk among "
            "vulnerable occupation borrowers."
        )

    with col2:
        st.markdown("**Regression Discontinuity**")
        st.metric("LATE at Cut-off", "0.0058")
        st.metric("P-value", "0.0028")
        st.caption(
            "Crossing the credit score threshold has a real causal "
            "effect on default outcomes."
        )

    st.markdown("---")

    # ── Model card ────────────────────────────────────────────────────────
    st.subheader("SR 11-7 Model Card")
    try:
        with open("reports/model_card.md", "r", encoding="utf-8") as f:
            model_card = f.read()
        st.markdown(model_card)
    except Exception:
        st.warning("Model card not found.")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    st.subheader("Evidence Charts")
    chart_files = {
        "Causal DAG":               "reports/causal_dag.png",
        "DiD Plot":                 "reports/did_plot.png",
        "Regression Discontinuity": "reports/rd_plot.png",
        "Fairness Audit":           "reports/fairness_audit.png",
        "SHAP Feature Importance":  "reports/shap_summary.png",
        "Survival Curves":          "reports/survival_curves.png",
    }

    cols = st.columns(2)
    for i, (title, path) in enumerate(chart_files.items()):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if Path(path).exists():
                st.image(path)
            else:
                st.caption("Chart not available")