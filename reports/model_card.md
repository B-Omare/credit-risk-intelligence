# CreditPulse Model Card
*Generated: 2026-06-27 | Version: 0.1.0*

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | CreditPulse XGBoost Default Classifier |
| **Version** | 0.1.0 |
| **Type** | Gradient-boosted decision tree (XGBoost) |
| **Task** | Binary classification — probability of default (PD) |
| **Target market** | East African digital lending (Kenya focus) |
| **Regulatory alignment** | IFRS 9, CBK Prudential Guidelines, SR 11-7 |

---

## Intended Use

**Primary use cases:**
- Automated credit scoring for digital lenders
- IFRS 9 Stage classification (performing / under-watch / impaired)
- Expected Credit Loss (ECL) computation
- Loan officer decision support

**Out-of-scope uses:**
- Mortgage lending (different risk profile)
- Corporate credit (different data structure)
- Jurisdictions without mobile money ecosystems

---

## Training Data

| Dataset | Source | Size |
|---------|--------|------|
| Home Credit Default Risk | Kaggle competition | 246,008 (training split) |
| Simulated M-Pesa transactions | Generated using realistic East African patterns | 1,000 borrowers × 12 months |

**Class imbalance:** 8.1% default rate — handled via scale_pos_weight=12

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7435 |
| Average Precision (PR-AUC) | 0.2292 |
| Best iteration | 231 |
| Test set size | 61,503 loans |

---

## Features Used

- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

---

## IFRS 9 Stage Distribution (Training Data)

| Stage | Description | Count | % |
|-------|-------------|-------|---|
| 1 | Performing | 259,874 | 84.5% |
| 2 | Significant risk increase | 22,812 | 7.4% |
| 3 | Credit-impaired (defaulted) | 24,825 | 8.1% |

---

## Causal Analysis

**Key causal factors (DoWhy backdoor adjustment):**
1. External credit score (EXT_SOURCE_MEAN) — strongest causal effect
2. Credit-to-income ratio — direct causal pathway
3. Employment stability — mediates income effect

**Difference-in-Differences (economic shock simulation):**
- Low-income borrowers experienced a ~3.4pp larger increase in default probability compared to high-income borrowers following an economic shock.

---

## Fairness & Bias Audit

| Protected attribute | Group difference | Action taken |
|---------------------|-----------------|--------------|
| Gender | < 2pp in default rate | Monitored; not used in features |
| Region | Regional variation present | Included as control variable |
| Income level | Structural correlation with default | Causal adjustment applied |

---

## Limitations

1. Training data is Home Credit (Eastern Europe) calibrated for East Africa — not direct East African loan data
2. Thin-file borrowers (no credit history) may have higher uncertainty
3. M-Pesa NLP features based on simulated data pending real transaction access
4. Model requires retraining when macroeconomic conditions change significantly

---

## Monitoring & Maintenance

- **Retraining trigger:** ROC-AUC drops below 0.70 on monthly holdout
- **Data drift monitoring:** PSI on key features
- **CI/CD:** Automated retraining via GitHub Actions on data refresh

---

*This model card follows the SR 11-7 supervisory guidance on model risk management.*
