# CreditPulse
Causal AI credit risk assessment for East African digital lending markets.

## Setup
```bash
conda create -n creditpulse python=3.11 -y
conda activate creditpulse
pip install -e .
pip install pandas numpy xgboost==2.1.4 scikit-learn joblib pyarrow
pip install fastapi uvicorn streamlit pydantic httpx
pip install dowhy econml statsmodels networkx
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface chromadb bertopic sentence-transformers
pip install shap scikit-survival
```

## Run the Pipeline
```bash
# Phase 2: Clean data
python creditpulse/ingestion/clean.py

# Phase 3: Causal analysis
python creditpulse/causal/dag.py
python creditpulse/causal/diff_in_diff.py
python creditpulse/causal/regression_discontinuity.py

# Phase 4: NLP
python creditpulse/nlp/rag_pipeline.py
python creditpulse/nlp/topic_model.py

# Phase 5: Models
python creditpulse/models/xgboost_model.py
python creditpulse/models/bayesian_model.py
python creditpulse/models/survival_model.py
python creditpulse/models/fraud_detector.py

# Phase 6: Explainability
python creditpulse/explainability/fairness_audit.py
python creditpulse/explainability/model_card.py
```

## Run the Application
```bash
# Terminal 1 — Start the API
uvicorn creditpulse.api.main:app --reload

# Terminal 2 — Start the dashboard
streamlit run app/streamlit_app.py
```

Then open:
- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

## Docker (requires 4GB+ free RAM)
```bash
docker-compose up --build
```

## Structure
- `creditpulse/` — core Python package
- `data/` — raw, processed, and simulated datasets
- `app/` — Streamlit dashboard
- `tests/` — pytest test suite
- `reports/` — charts, model card, fairness metrics

## Results
| Model | Metric | Value |
|-------|--------|-------|
| XGBoost | AUC-ROC | 0.7415 |
| XGBoost | Gini | 0.4830 |
| Survival Forest | C-index | 0.8130 |
| DiD Experiment | P-value | 0.002 |
| RD Analysis | P-value | 0.0028 |