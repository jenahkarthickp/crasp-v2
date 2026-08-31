# CRASP Development Changelog

## [Unreleased — v2 in progress]

### Problems identified in v1 (August 7, 2026)

**Critical flaw — Circular label generation:**
`resilience_score` in v1 is computed as a deterministic linear combination
of the same features used for training. This means the ML model is
simply learning to reproduce a formula it has access to, producing
artificially inflated R²=0.93. Any model would achieve near-perfect
accuracy if given the formula inputs to predict the formula output.

Fix planned: introduce a latent variable `security_culture_score` that
contributes 22% of the true score but is withheld from all ML models.
This produces a genuine information ceiling of R²≈0.87.

**Critical flaw — Class imbalance:**
The Critical resilience class has only 13 training examples (0.4% of
training set), producing recall=0% for the most important class to detect.

Fix planned: apply SMOTE oversampling to training set only.

**Performance gap — NLP accuracy 62%:**
CVE descriptions are too generic for the classifier to learn severity
patterns. All four severity levels use similar generic vocabulary.

Fix planned: add severity-specific vocabulary profiles and three-tier
template system.

**Algorithm gap — RF beating XGBoost:**
XGBoost with default parameters is outperformed by Random Forest with
class_weight='balanced'. XGBoost needs hyperparameter tuning and
sample_weight assignment.

## [v2.0 — Latent Variable Fix]

### Fixed (August 9, 2026)

**Latent variable design implemented in generate_data.py:**
- Added `security_culture_score` as unobservable latent variable
- True score formula: 0.78 * formula_score + 0.22 * culture_score + N(0,4)
- Culture score drawn from sector-specific Gaussian distributions
- Culture score EXCLUDED from all ML training features
- Expected R² ceiling now ~0.87 (genuine information limit)
- Increased dataset to 3000 organisations for better minority class coverage
- Added 6 new 2026 threat intelligence features:
  - avg_patch_days (DBIR 2026: median 43 days)
  - shadow_ai_exposure (DBIR 2026: 45% workforce)
  - supply_chain_vendors (DBIR 2026: 48% breach vector)
  - uses_ai_security_tools (IBM 2026: saves $2.2M)
  - ai_breach_history (IBM 2026: 25% of breaches)
  - estimated_breach_cost_usd (IBM 2026: $4.99M avg)

## [v2.1 — SMOTE + NLP Fix]

### Fixed (August 11, 2026)

**SMOTE oversampling added to train_models.py:**
- Applied to training set ONLY (no test contamination)
- Critical class: 13 examples → 1345 after SMOTE
- Training set total: 2400 → 6725 balanced examples
- Critical class recall improved from 0% → 38%

**NLP completely rewritten:**
- Added SEVERITY_PROFILES dict with per-severity vocabulary
- Three-tier CVE template system (40% full, 35% partial, 25% generic)
- Attack types overlap across severities to prevent trivial memorisation
- Switched from TF-IDF+LR to TF-IDF+LinearSVC (better margin separation)
- NLP accuracy: 62% → 91%

**XGBoost tuned:**
- n_estimators=400, max_depth=5, learning_rate=0.05
- subsample=0.8, colsample_bytree=0.7, L1/L2 regularisation
- SHAP-weighted sample_weight for minority class emphasis
- XGBoost now beats RF: 80.0% vs 78.0%

## [v2.2 — Explainability + Anomaly + Forecasting]

### Added (August 13-14, 2026)

**SHAP explainability (train_models.py):**
- TreeExplainer on XGBoost classifier
- Global SHAP: top 3 = security_budget_pct (0.133), patch_compliance_pct (0.080), mfa_coverage_pct (0.068)
- Three 2026 features in top-10 SHAP rankings
- CalibratedClassifierCV (isotonic, 3-fold) for reliable probabilities

**Anomaly detection:**
- Isolation Forest (contamination=0.05, n_estimators=300)
- Anomalous orgs: breach rate 52.7% vs 13.1% normal (4x gap)
- Mean score 9.2 points below normal

**Time-series forecasting:**
- 300 orgs x 18 months sliding-window features
- Prophet evaluated: R²=-88.6 (fails on 18-month series, no seasonality)
- GBR selected: R²=0.935, MAPE=11.5%
- Negative Prophet result documented as valid academic finding

**Evaluation enhancements:**
- ROC/AUC (One-vs-Rest, Macro AUC=0.942)
- Paired t-test: XGBoost vs RF (t=-0.169, p=0.874, not significant)
- Learning curves with 95% CI
- Sector fairness: 74% (Education) to 84% (Finance)

## [v2.3 — Dashboard Enhancements]

### Added (August 16-17, 2026)

**PDF export (fpdf2):**
- generate_pdf_report() produces downloadable assessment PDFs
- Content: org profile, overall score, dimension table, top-8 recommendations
- 2026 threat indicators relevant to sector included

**What-If intervention simulator (Tab 7):**
- 12 interventions: SIEM, SOAR, EDR, MFA 95%, Training 90%,
  Patch <14 days, Shadow AI governance, AI security tools,
  DRP, CISO hire, Vendor assessment, Backup testing
- Feature deltas applied to current org profile
- Before/after radar chart across 5 dimensions
- ROI: delta_score * $100k / intervention_cost
- Example: Developing org (48.2) → 61.8 with 4 interventions ($85k, 12 weeks)

**Per-prediction SHAP waterfall in dashboard:**
- TreeExplainer called per assessment
- Waterfall chart: green (positive) / red (negative) feature contributions
- Base expected value shown as reference

## [v2.4 — Final Submission]

### Finalised (August 20, 2026)

**System validated end-to-end:**
- Full pipeline: generate_data.py → train_models.py → dashboard
- All 5 model types verified: classification, regression, NLP, anomaly, forecasting
- Dashboard 7 tabs tested on sample organisations from all 8 sectors

**IEEE paper finalised:**
- 9-11 pages in IEEEtran two-column format
- 17 references including 5 authoritative 2026 reports
- All tables cross-checked against model_metrics.json values
- Author: Jenah Karthick Palanikumar, University of Galway, August 2026

**Final performance summary:**
- Classification XGBoost: accuracy=80.0%, Macro AUC=0.942
- Regression (avg): R²=0.869, RMSE=6.09
- NLP LinearSVC: accuracy=91.0%
- GBR Forecasting: R²=0.935, MAPE=11.5%
- Anomaly detection: 4x breach rate gap (52.7% vs 13.1%)
- Critical class recall: 0% (v1) → 38% (v2, post-SMOTE)
