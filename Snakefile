# CreditPulse Snakemake Pipeline
# Run with: snakemake --cores 4

configfile: "config.yaml"

rule all:
    input:
        "data/processed/loans_ifrs9.parquet",
        "models/xgboost_model.pkl",
        "reports/model_card.md"

rule ingest_data:
    input:
        "data/raw/application_train.csv"
    output:
        "data/processed/loans_clean.parquet"
    script:
        "creditpulse/ingestion/clean.py"

rule engineer_features:
    input:
        "data/processed/loans_clean.parquet"
    output:
        "data/processed/loans_features.parquet"
    script:
        "creditpulse/ingestion/features.py"

rule apply_ifrs9_schema:
    input:
        "data/processed/loans_features.parquet"
    output:
        "data/processed/loans_ifrs9.parquet"
    script:
        "creditpulse/ingestion/ifrs9.py"

rule simulate_mpesa:
    output:
        "data/simulated/mpesa_transactions.parquet"
    script:
        "creditpulse/ingestion/simulate_mpesa.py"

rule train_xgboost:
    input:
        "data/processed/loans_ifrs9.parquet"
    output:
        "models/xgboost_model.pkl",
        "models/xgboost_metrics.json"
    script:
        "creditpulse/models/train_xgboost.py"

rule train_bayesian:
    input:
        "data/processed/loans_ifrs9.parquet"
    output:
        "models/bayesian_model.pkl"
    script:
        "creditpulse/models/train_bayesian.py"

rule train_survival:
    input:
        "data/processed/loans_ifrs9.parquet"
    output:
        "models/survival_model.pkl"
    script:
        "creditpulse/models/train_survival.py"

rule causal_analysis:
    input:
        "data/processed/loans_ifrs9.parquet"
    output:
        "reports/causal_effects.json"
    script:
        "creditpulse/causal/dowhy_analysis.py"

rule generate_model_card:
    input:
        "models/xgboost_model.pkl",
        "models/xgboost_metrics.json",
        "reports/causal_effects.json"
    output:
        "reports/model_card.md"
    script:
        "creditpulse/explainability/model_card.py"
