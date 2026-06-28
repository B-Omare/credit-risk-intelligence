# Reproducing CreditPulse Results

This document provides step-by-step instructions to reproduce all results,
charts, models, and the live application from scratch.

**Estimated total time:** 45-60 minutes (excluding data download)
**Operating system tested:** Windows 11, Ubuntu 24.04
**Hardware minimum:** 8GB RAM, 10GB free disk space

---

## Step 1 — Clone and Install (5 minutes)

### 1.1 Clone the repository
```bash
git clone https://github.com/B-Omare/creditpulse.git
cd creditpulse
```

### 1.2 Create and activate the Conda environment
```bash
conda create -n creditpulse python=3.11 -y
conda activate creditpulse
```

### 1.3 Install the package and all dependencies
```bash
pip install -e .
pip install pandas numpy pyarrow
pip install xgboost==2.1.4 scikit-learn joblib shap
pip install scikit-survival
pip install dowhy econml statsmodels networkx matplotlib seaborn
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface chromadb bertopic sentence-transformers
pip install fastapi uvicorn streamlit pydantic httpx
pip install pytest black ruff
```

### 1.4 Verify installation
```bash
python -c "import creditpulse; print('CreditPulse installed successfully')"
```

Expected output:
---

## Step 2 — Download Data (15 minutes)

### 2.1 Set up Kaggle authentication
```bash
# Option A: OAuth login (recommended)
pip install kaggle
kaggle auth login
# Follow the browser prompt to authenticate

# Option B: API token
# Download token from kaggle.com/settings/api
# Save to C:\Users\USERNAME\.kaggle\access_token (Windows)
# or ~/.kaggle/access_token (Linux/Mac)
set KAGGLE_API_TOKEN=your_token_here  # Windows
export KAGGLE_API_TOKEN=your_token_here  # Linux/Mac
```

### 2.2 Accept competition rules
Visit **kaggle.com/competitions/home-credit-default-risk** and
click **Join Competition** to accept the data usage rules.
This is required before the download will succeed.

### 2.3 Download and extract the dataset
```bash
kaggle competitions download -c home-credit-default-risk -p data\raw
```

Then extract (Windows):
```bash
cd data\raw
tar -xf home-credit-default-risk.zip
cd ..\..
```

**Expected files in data/raw/:**
- application_train.csv (~166MB)
- application_test.csv (~74MB)
- bureau.csv (~218MB)
- installments_payments.csv (~1.9GB)
- POS_CASH_balance.csv (~1.3GB)
- previous_application.csv (~226MB)

---

## Step 3 — Run the Full Pipeline (30 minutes)

Run each script in order. Each script prints progress and
confirms completion with a final message.

### Phase 2 — Data Ingestion
```bash
python creditpulse/ingestion/clean.py
```
**Expected output:**

**Expected file:** data/processed/loans_clean.parquet (~31MB)

---

### Phase 3 — Causal Inference
```bash
python creditpulse/causal/dag.py
python creditpulse/causal/diff_in_diff.py
python creditpulse/causal/regression_discontinuity.py
```
**Expected outputs in reports/:**
- causal_dag.png
- did_plot.png
- rd_plot.png

**Expected DiD result:**


**Expected RD result:**

---

### Phase 4 — NLP
```bash
python creditpulse/nlp/rag_pipeline.py
python creditpulse/nlp/topic_model.py
python creditpulse/nlp/explainer.py
```
**Expected outputs:**
- data/chroma_db/ (vector store, ~5MB)
- reports/bertopic_summary.csv (18 topics)

**Expected explainer output (sample):**

---

### Phase 5 — Models
```bash
python creditpulse/models/xgboost_model.py
python creditpulse/models/bayesian_model.py
python creditpulse/models/survival_model.py
python creditpulse/models/fraud_detector.py
```
**Expected model performance:**

| Model | Metric | Expected Value |
|-------|--------|---------------|
| XGBoost | AUC-ROC | 0.74 ± 0.01 |
| XGBoost | Gini | 0.48 ± 0.01 |
| Survival Forest | C-index | 0.81 ± 0.02 |
| Fraud Detector | Flag rate | 1.5% - 2.0% |

**Expected files in models/:**
- xgboost_model.pkl (~1.3MB)
- bayesian_bootstrap_models.pkl (~74KB)
- bayesian_scaler.pkl (~1KB)
- survival_model.pkl (~98MB)
- fraud_detector.pkl (~1.1MB)
- fraud_scaler.pkl (~1KB)

**Expected outputs in reports/:**
- shap_summary.png
- shap_waterfall.png
- bayesian_uncertainty.png
- survival_curves.png
- fraud_detection.png

---

### Phase 6 — Explainability
```bash
python creditpulse/explainability/fairness_audit.py
python creditpulse/explainability/model_card.py
```
**Expected outputs in reports/:**
- fairness_audit.png
- fairness_metrics.csv
- model_card.md (162 lines)

**Expected fairness result (sample):**

---

## Step 4 — Launch the Application (2 minutes)

### 4.1 Start the FastAPI microservice
Open Terminal 1:
```bash
conda activate creditpulse
uvicorn creditpulse.api.main:app --reload
```
**Expected output:**

Verify the API is working:

You should see the Swagger UI with three endpoints.

### 4.2 Start the Streamlit dashboard
Open Terminal 2:
```bash
conda activate creditpulse
streamlit run app/streamlit_app.py
```

Open your browser at:

You should see the CreditPulse dashboard with three views:
- Loan Officer View
- Executive Portfolio View
- Policy & Compliance View

---

## Step 5 — Run the Test Suite
```bash
pytest tests/ -v
```
**Expected output:**

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `conda not recognised` | Conda not in PATH | Open Anaconda Prompt instead of terminal |
| `kaggle: authentication required` | Token not set | Run `set KAGGLE_API_TOKEN=your_token` |
| `ModuleNotFoundError` | Missing library | Run `pip install missing-library-name` |
| `FileNotFoundError: loans_clean.parquet` | Phase 2 not run | Run `python creditpulse/ingestion/clean.py` first |
| `Models not loaded` in API | Phase 5 not run | Run all four model scripts first |
| Docker: not enough memory | 8GB RAM limit | Run API and dashboard directly without Docker |

---

## Expected Final File Structure

After running all phases, your project should contain:

---

## Citation

If you use CreditPulse in your research, please cite: 
---

*Last verified: June 2026 | Python 3.11 | Windows 11 + Ubuntu 24.04*
