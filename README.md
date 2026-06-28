# CreditPulse

![CI](https://github.com/B-Omare/creditpulse/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-East%20Africa-orange)
![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20Bayesian%20%7C%20Survival%20%7C%20Fraud-purple)

> **Causal AI credit risk assessment for East African digital lending markets.**
![CreditPulse Dashboard](reports/dashboard.png)
> Built for thin-file borrowers, M-Pesa ecosystems, and CBK regulatory compliance.

---

## What is CreditPulse?

CreditPulse is an open-source credit risk intelligence platform that goes beyond
correlation-based scoring. It asks not just *who* will default, but *why* —
using causal inference, Bayesian uncertainty quantification, and explainable AI.

### Key Results

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Classifier | AUC-ROC | 0.7415 |
| XGBoost Classifier | Gini Coefficient | 0.4830 |
| Random Survival Forest | C-index | 0.8130 |
| Difference-in-Differences | Causal coefficient | -0.1125 (p=0.002) |
| Regression Discontinuity | LATE at cut-off | 0.0058 (p=0.0028) |
| Fraud Detector | Applications flagged | 1.75% |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/B-Omare/creditpulse.git
cd creditpulse

# 2. Environment
conda create -n creditpulse python=3.11 -y
conda activate creditpulse
pip install -e .
pip install pandas numpy pyarrow xgboost==2.1.4 scikit-learn joblib shap
pip install scikit-survival dowhy econml statsmodels networkx matplotlib seaborn
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface chromadb bertopic sentence-transformers
pip install fastapi uvicorn streamlit pydantic httpx

# 3. Run the pipeline
python creditpulse/ingestion/clean.py
python creditpulse/causal/dag.py
python creditpulse/causal/diff_in_diff.py
python creditpulse/causal/regression_discontinuity.py
python creditpulse/nlp/rag_pipeline.py
python creditpulse/nlp/topic_model.py
python creditpulse/models/xgboost_model.py
python creditpulse/models/bayesian_model.py
python creditpulse/models/survival_model.py
python creditpulse/models/fraud_detector.py
python creditpulse/explainability/fairness_audit.py
python creditpulse/explainability/model_card.py

# 4. Launch
# Terminal 1
uvicorn creditpulse.api.main:app --reload

# Terminal 2
streamlit run app/streamlit_app.py
```

Then open:
- **Dashboard:** http://localhost:8501
- **API docs:** http://127.0.0.1:8000/docs

For full reproduction instructions see [REPRODUCE.md](REPRODUCE.md).

---

## System Architecture

| Layer | Component | Technology |
|-------|-----------|------------|
| Data | Ingestion & ETL | pandas + Snakemake |
| Data | IFRS 9 Schema | Domain logic |
| Causal | Causal DAG | DoWhy + networkx |
| Causal | DiD Experiment | statsmodels |
| Causal | Regression Discontinuity | statsmodels + matplotlib |
| NLP | RAG Chatbot | LangChain + ChromaDB |
| NLP | Topic Modelling | BERTopic |
| NLP | Decision Explainer | Rule-based engine |
| ML | XGBoost Classifier | XGBoost + SHAP |
| ML | Bayesian Scorer | Bootstrap ensemble |
| ML | Survival Model | scikit-survival |
| ML | Fraud Detector | Isolation Forest |
| Explainability | Fairness Audit | sklearn metrics |
| Explainability | Model Card | SR 11-7 Markdown |
| API | Scoring Service | FastAPI + uvicorn |
| UI | Dashboard | Streamlit |

---

## Repository Structure
---

## Three-View Dashboard

### Loan Officer View
Score individual borrowers with interactive sliders. Returns default
probability, IFRS 9 stage, Bayesian credible interval, fraud flag,
and a plain-language explanation compliant with CBK 2022 regulations.

### Executive View
Portfolio-level IFRS 9 stage distribution, key risk metrics, and a
causal scenario modelling tool — estimate the portfolio impact of a
macro shock using the DiD coefficient.

### Policy & Compliance View
Fairness audit across borrower segments, causal evidence summary
(DiD + RD), and the full SR 11-7 model card.

---

## Regulatory Compliance

| Standard | Status |
|----------|--------|
| IFRS 9 (loan stage classification) | Implemented |
| CBK Digital Credit Regulations 2022 | Borrower explanations implemented |
| SR 11-7 (model risk management) | Model card generated |
| Fairness audit (age, income, occupation) | Implemented |

---

## Citation

```bibtex
@software{omare2026creditpulse,
  author    = {Omare, Brian},
  title     = {CreditPulse: Causal AI Credit Risk Assessment
               for East African Digital Lending Markets},
  year      = {2026},
  url       = {https://github.com/B-Omare/creditpulse},
  version   = {0.1.0}
}
```

---

## Docker

```bash
docker-compose up --build
```

Requires 4GB+ free RAM. Starts both the API (port 8000) and
dashboard (port 8501) in separate containers.

---

*Built with Python 3.11 | Tested on Windows 11 and Ubuntu 24.04*
