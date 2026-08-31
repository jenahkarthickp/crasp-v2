# CRASP v2 — Cyber Resilience Assessment and Scoring Platform

**MSc Computer Science Capstone Project — University of Galway, August 2026**
**Author:** Jenah Karthick Palanikumar (`j.palanikumar1@universityofgalway.ie`)

---

## Project Overview

CRASP is a machine learning-driven framework that evaluates organisational cyber resilience across five dimensions grounded in NIST CSF 2.0:

| Dimension | Description |
|-----------|-------------|
| **Anticipate** | Threat awareness, vulnerability scanning, threat intelligence |
| **Withstand** | Preventive controls: MFA, patching, access management |
| **Recover** | Incident response, backup, disaster recovery capability |
| **Adapt** | Post-incident learning, process improvement, training |
| **Evolve** | Strategic security programme maturity and governance |

---

## Key Academic Contributions

1. **Latent variable design** — `security_culture_score` (22% of true score) is withheld from all ML models, preventing circular label generation and producing honest R² ≈ 0.87 (vs spurious 0.93 in v1)
2. **2026 threat intelligence integration** — 6 new features from Verizon DBIR 2026, IBM Cost of a Data Breach 2026, MITRE ATT&CK v19, Gartner 2026
3. **SMOTE class balancing** — Critical class recall improved from 0% → 38%
4. **SHAP explainability** — per-prediction feature attribution waterfall charts
5. **What-If intervention simulator** — 12 interventions with ROI estimates

---

## Model Performance Summary

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Classifier | Test Accuracy | 80.0% |
| XGBoost Classifier | Macro AUC | 0.942 |
| XGBoost Regressor (avg) | R² | 0.869 |
| NLP (LinearSVC) | CVE Severity Accuracy | 91.0% |
| GBR Forecaster | R² | 0.935 |
| GBR Forecaster | MAPE | 11.5% |
| Isolation Forest | Anomaly Breach Rate Gap | 4.0× |

---

## Project Structure

```
crasp_v2/
├── src/
│   ├── generate_data.py      # Synthetic data generation (3000 orgs, 1500 CVEs)
│   └── train_models.py       # ML training: classification, regression, NLP, anomaly, forecasting
├── dashboard/
│   └── app.py                # 7-tab Streamlit dashboard
├── data/
│   ├── raw/                  # Generated datasets (CSV)
│   ├── processed/            # Preprocessed data
│   └── models/               # Serialised models (joblib) + metrics JSON
├── CRASP_v2_IEEE_Paper.tex   # IEEE journal paper (LaTeX, for Overleaf)
├── requirements.txt
└── run.sh                    # Full pipeline: generate → train → dashboard
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (data generation → training → dashboard)
bash run.sh

# Or step by step:
python src/generate_data.py
python src/train_models.py
streamlit run dashboard/app.py --server.port 8510
```

---

## Development History (v1 → v2)

### v1 Problems Identified
- **Circular labels**: `resilience_score` computed from features, then used as training target → R² = 0.93 (meaningless)
- **Critical class recall = 0%**: only 13 training examples, no balancing
- **NLP accuracy 62%**: generic CVE text with no severity-specific vocabulary
- **RF beating XGBoost**: XGBoost not tuned, no class weighting

### v2 Fixes Applied
- ✅ Latent variable (`security_culture_score`) withheld from models → honest R² = 0.869
- ✅ SMOTE: 2,400 → 6,725 balanced samples → Critical recall 38%
- ✅ Severity-specific CVE vocabulary + 3-tier templates → NLP 91%
- ✅ XGBoost tuned (n_estimators=400, depth=5, lr=0.05, L1/L2 regularisation)
- ✅ SHAP TreeExplainer + ROC/AUC + paired t-test + learning curves + sector fairness added
- ✅ Prophet evaluated and rejected (R²=−88.6) → GBR selected for forecasting
- ✅ 6 new 2026 threat intelligence features added
- ✅ What-If simulator (12 interventions) + PDF export added

---

## 2026 Threat Intelligence Sources

| Report | Key Finding Used |
|--------|-----------------|
| Verizon DBIR 2026 | Supply chain: 48% of breaches; patch median: 43 days |
| IBM Cost of Data Breach 2026 | Average breach: $4.99M; AI tools save $2.2M |
| MITRE ATT&CK v19 | 858 enterprise techniques; AI-assisted attack coverage |
| NIST CSF 2.0 | Five-dimension framework backbone |
| Gartner 2026 | Security spending $244.2B; budget normalisation |

---

## Requirements

```
Python 3.9+
scikit-learn >= 1.3.0
xgboost >= 2.0.0
streamlit >= 1.28.0
plotly >= 5.18.0
pandas >= 2.0.0
numpy >= 1.24.0
imbalanced-learn >= 0.11.0
shap >= 0.43.0
fpdf2 >= 2.7.0
scipy >= 1.11.0
prophet >= 1.1.0
```
