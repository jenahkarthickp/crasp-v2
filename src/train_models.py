"""
CRASP v2 - ML Model Training (Enhanced)

Enhancements over baseline:
  1. SMOTE oversampling for minority classes (Critical, Optimized)
  2. SHAP explainability — global + per-prediction feature attribution
  3. ROC curves + macro AUC for multiclass classification
  4. Statistical significance test (paired t-test XGBoost vs RF)
  5. Learning curves — accuracy vs training set size
  6. Calibrated classifier — reliable probability estimates
  7. Sector bias / fairness analysis — per-sector accuracy
  8. Prophet time-series forecasting (alongside GBR baseline)
  9. Tuned XGBoost hyperparameters with class-weighted sample weights
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve as sklearn_lc,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    IsolationForest, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, roc_curve,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats as scipy_stats
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "models"
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["sector", "size_category", "backup_frequency"]
BOOL_COLS = [
    "has_ciso","uses_external_mssp","has_siem","has_threat_hunting",
    "has_ids_ips","has_dlp","has_edr","has_drp","backup_encryption",
    "has_soar","dynamic_policy_updates","has_devsecops",
    "cloud_security_posture","continuous_improvement_program",
    "board_security_reporting","bug_bounty_program",
    "shadow_ai_exposure","uses_ai_security_tools","ai_breach_history",
]
NUMERIC_COLS = [
    "employees","revenue_million_usd","security_budget_pct",
    "security_budget_usd","security_staff",
    "threat_intel_feeds","vuln_scans_per_month","security_training_pct",
    "avg_detect_hours","phishing_sim_per_year",
    "firewall_layers","mfa_coverage_pct","encryption_at_rest_pct",
    "encryption_in_transit_pct","patch_compliance_pct",
    "network_segmentation_level",
    "backup_tests_per_year","avg_recovery_hours","recovery_time_objective_hours",
    "security_automation_pct","config_changes_per_year",
    "post_incident_reviews_pct","security_metrics_tracked",
    "security_certifications",
    "incidents_last_year","successful_breaches","data_lost_gb",
    "total_downtime_hours",
    "avg_patch_days","supply_chain_vendors","estimated_breach_cost_usd",
]

GOAL_TARGETS = ["score_anticipate","score_withstand","score_recover",
                "score_adapt","score_evolve"]


def prepare_features(df):
    df = df.copy()
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    df["budget_per_employee"]  = df["security_budget_usd"] / (df["employees"] + 1)
    df["staff_per_employee"]   = df["security_staff"] / (df["employees"] + 1) * 100
    df["tools_score"]          = (df["has_siem"] + df["has_soar"] + df["has_ids_ips"]
                                  + df["has_dlp"] + df["has_edr"])
    df["detection_efficiency"] = 1 / (df["avg_detect_hours"] + 1)
    df["recovery_efficiency"]  = 1 / (df["avg_recovery_hours"] + 1)
    df["breach_rate"]          = df["successful_breaches"] / (df["incidents_last_year"] + 1)
    return df


def get_feature_columns(df):
    exclude = [
        "org_id","company_name",
        "score_anticipate","score_withstand","score_recover","score_adapt","score_evolve",
        "overall_resilience_score","resilience_level","resilience_label",
        "security_culture_score",
    ]
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def train_classification(df):
    print("\n" + "="*55)
    print("MODEL 1: CLASSIFICATION — Resilience Level (1–5)")
    print("="*55)

    df_feat   = prepare_features(df)
    feat_cols = get_feature_columns(df_feat)
    X       = df_feat[feat_cols].fillna(0)
    y       = df["resilience_level"]
    sectors = df["sector"]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sectors, test_size=0.2, random_state=42, stratify=y
    )
    scaler         = StandardScaler()
    X_train_s_orig = scaler.fit_transform(X_train)   # pre-SMOTE — keep for calibration & LC
    X_test_s       = scaler.transform(X_test)
    y_train_0      = y_train.values - 1              # 0-indexed for XGBoost

    # ── SMOTE ────────────────────────────────────────────────────────────
    if HAS_SMOTE:
        min_cls = min(Counter(y_train_0).values())
        k = min(5, min_cls - 1)
        if k >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_s, y_train_0_bal = smote.fit_resample(X_train_s_orig, y_train_0)
            print(f"  SMOTE: {len(y_train_0)} → {len(y_train_0_bal)} samples, "
                  f"dist: {dict(sorted(Counter(y_train_0_bal).items()))}")
        else:
            X_train_s, y_train_0_bal = X_train_s_orig, y_train_0
    else:
        X_train_s, y_train_0_bal = X_train_s_orig, y_train_0

    y_train_sk = y_train_0_bal + 1
    sample_w   = compute_sample_weight("balanced", y_train_0_bal)

    # ── XGBoost (primary) ────────────────────────────────────────────────
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    xgb_clf.fit(X_train_s, y_train_0_bal, sample_weight=sample_w)
    xgb_pred = xgb_clf.predict(X_test_s) + 1
    xgb_acc  = accuracy_score(y_test, xgb_pred)
    xgb_cv   = cross_val_score(xgb_clf, X_train_s, y_train_0_bal, cv=5, scoring="accuracy")

    # ── Random Forest ────────────────────────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf_clf.fit(X_train_s, y_train_sk)
    rf_pred = rf_clf.predict(X_test_s)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_cv   = cross_val_score(rf_clf, X_train_s, y_train_sk, cv=5, scoring="accuracy")

    # ── Logistic Regression ──────────────────────────────────────────────
    lr_clf = LogisticRegression(
        max_iter=1000, C=0.5, class_weight="balanced",
        random_state=42, multi_class="multinomial",
    )
    lr_clf.fit(X_train_s, y_train_sk)
    lr_pred = lr_clf.predict(X_test_s)
    lr_acc  = accuracy_score(y_test, lr_pred)
    lr_cv   = cross_val_score(lr_clf, X_train_s, y_train_sk, cv=5, scoring="accuracy")

    print(f"\n  {'Model':<25} {'Test Acc':>10}  {'5-Fold CV Mean':>15}  {'CV Std':>8}")
    print(f"  {'-'*62}")
    print(f"  {'XGBoost (primary)':<25} {xgb_acc*100:>9.1f}%  {xgb_cv.mean()*100:>14.1f}%  {xgb_cv.std()*100:>7.1f}%")
    print(f"  {'Random Forest (baseline)':<25} {rf_acc*100:>9.1f}%  {rf_cv.mean()*100:>14.1f}%  {rf_cv.std()*100:>7.1f}%")
    print(f"  {'Logistic Reg (baseline)':<25} {lr_acc*100:>9.1f}%  {lr_cv.mean()*100:>14.1f}%  {lr_cv.std()*100:>7.1f}%")

    present_labels = sorted(y_test.unique())
    label_names    = ["Critical","Developing","Managed","Advanced","Optimized"]
    present_names  = [label_names[l-1] for l in present_labels]
    print(f"\n  XGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred, labels=present_labels,
                                target_names=present_names, zero_division=0))

    importances  = pd.Series(xgb_clf.feature_importances_, index=feat_cols)
    top_features = importances.nlargest(20).to_dict()
    print(f"\n  Top 10 predictive features:")
    for feat, imp in list(top_features.items())[:10]:
        print(f"    {feat:<40} {imp:.4f}")

    joblib.dump(xgb_clf,  MODEL_DIR / "classifier.pkl")
    joblib.dump(scaler,   MODEL_DIR / "scaler_clf.pkl")
    joblib.dump(feat_cols,MODEL_DIR / "feature_cols.pkl")
    joblib.dump(rf_clf,   MODEL_DIR / "classifier_rf_baseline.pkl")

    metrics = {
        "xgboost": {
            "test_accuracy":    round(xgb_acc, 4),
            "cv_accuracy_mean": round(xgb_cv.mean(), 4),
            "cv_accuracy_std":  round(xgb_cv.std(), 4),
        },
        "random_forest_baseline": {
            "test_accuracy":    round(rf_acc, 4),
            "cv_accuracy_mean": round(rf_cv.mean(), 4),
            "cv_accuracy_std":  round(rf_cv.std(), 4),
        },
        "logistic_regression_baseline": {
            "test_accuracy":    round(lr_acc, 4),
            "cv_accuracy_mean": round(lr_cv.mean(), 4),
            "cv_accuracy_std":  round(lr_cv.std(), 4),
        },
        "xgboost_gain_over_rf":     round((xgb_acc - rf_acc) * 100, 2),
        "feature_importance_top20": {k: round(v, 5) for k, v in top_features.items()},
    }

    # ── ROC / AUC ────────────────────────────────────────────────────────
    print("\n  Computing ROC curves and AUC…")
    classes    = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    xgb_proba  = xgb_clf.predict_proba(X_test_s)
    try:
        auc_macro = roc_auc_score(y_test_bin, xgb_proba,
                                  multi_class="ovr", average="macro")
        print(f"  Macro AUC: {auc_macro:.4f}")
    except Exception:
        auc_macro = 0.0

    roc_curves = {}
    for idx, cls in enumerate(classes):
        name = label_names[cls - 1]
        if idx < y_test_bin.shape[1] and y_test_bin[:, idx].sum() > 0:
            try:
                fpr, tpr, _ = roc_curve(y_test_bin[:, idx], xgb_proba[:, idx])
                cls_auc     = roc_auc_score(y_test_bin[:, idx], xgb_proba[:, idx])
                step = max(1, len(fpr) // 60)
                roc_curves[name] = {
                    "fpr": fpr[::step].tolist(),
                    "tpr": tpr[::step].tolist(),
                    "auc": round(cls_auc, 4),
                }
            except Exception:
                pass
    metrics["auc_macro"]  = round(auc_macro, 4)
    metrics["roc_curves"] = roc_curves

    # ── Paired t-test XGBoost vs RF ──────────────────────────────────────
    t_stat, p_val = scipy_stats.ttest_rel(xgb_cv, rf_cv)
    metrics["ttest_xgb_vs_rf"] = {
        "t_statistic": round(float(t_stat), 4),
        "p_value":     round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
        "interpretation": (
            f"XGBoost CV {'significantly' if p_val < 0.05 else 'NOT significantly'} "
            f"different from RF (p={p_val:.4f})"
        ),
    }
    print(f"  T-test XGB vs RF: t={t_stat:.3f}, p={p_val:.4f} "
          f"({'significant' if p_val < 0.05 else 'not significant'})")

    # ── Learning Curves ───────────────────────────────────────────────────
    print("  Computing learning curves…")
    lc_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, verbosity=0, eval_metric="mlogloss",
    )
    train_sizes_abs, tr_scores, cv_scores = sklearn_lc(
        lc_clf, X_train_s_orig, y_train_0,
        train_sizes=np.linspace(0.1, 1.0, 8), cv=3,
        scoring="accuracy", n_jobs=-1,
    )
    metrics["learning_curves"] = {
        "train_sizes": [int(s) for s in train_sizes_abs],
        "train_mean":  [round(s, 4) for s in tr_scores.mean(1)],
        "train_std":   [round(s, 4) for s in tr_scores.std(1)],
        "cv_mean":     [round(s, 4) for s in cv_scores.mean(1)],
        "cv_std":      [round(s, 4) for s in cv_scores.std(1)],
    }
    print(f"  Learning curve: final CV accuracy = {cv_scores.mean(1)[-1]*100:.1f}%")

    # ── Sector Bias / Fairness ────────────────────────────────────────────
    s_arr = s_test.values
    sector_bias = {}
    for sector in np.unique(s_arr):
        mask = s_arr == sector
        if mask.sum() >= 5:
            sector_bias[sector] = {
                "accuracy":  round(accuracy_score(y_test.values[mask], xgb_pred[mask]), 3),
                "n_samples": int(mask.sum()),
            }
    metrics["sector_bias"] = sector_bias
    print(f"  Sector bias: {sector_bias}")

    # ── SHAP Global Feature Importance ───────────────────────────────────
    shap_importance = {}
    if HAS_SHAP:
        print("  Computing SHAP values…")
        try:
            explainer = shap_lib.TreeExplainer(xgb_clf)
            n_shap    = min(200, len(X_test_s))
            sv        = explainer.shap_values(X_test_s[:n_shap])
            if isinstance(sv, list):
                global_imp = np.mean([np.abs(v).mean(0) for v in sv], axis=0)
            elif sv.ndim == 3:
                global_imp = np.abs(sv).mean(axis=(0, 2))
            else:
                global_imp = np.abs(sv).mean(0)
            shap_importance = (
                pd.Series(global_imp, index=feat_cols)
                .nlargest(20).round(5).to_dict()
            )
            print(f"  SHAP top 3: {list(shap_importance.keys())[:3]}")
        except Exception as e:
            print(f"  SHAP skipped: {e}")
    metrics["shap_global_importance"] = shap_importance

    # ── Calibrated Classifier (reliable probabilities) ────────────────────
    print("  Training calibrated classifier…")
    try:
        n_cal = int(len(X_train_s_orig) * 0.25)
        cal   = CalibratedClassifierCV(xgb_clf, cv="prefit", method="sigmoid")
        cal.fit(X_train_s_orig[:n_cal], y_train_0[:n_cal])
        joblib.dump(cal, MODEL_DIR / "classifier_calibrated.pkl")
        cal_proba  = cal.predict_proba(X_test_s)
        cal_pred   = cal_proba.argmax(axis=1) + 1
        cal_acc    = accuracy_score(y_test, cal_pred)
        metrics["calibrated_accuracy"] = round(cal_acc, 4)
        print(f"  Calibrated accuracy: {cal_acc*100:.1f}%")
    except Exception as e:
        print(f"  Calibration skipped: {e}")

    return metrics, feat_cols, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def train_regression(df, feat_cols, scaler):
    print("\n" + "="*55)
    print("MODEL 2: REGRESSION — Individual Goal Scores")
    print("="*55)
    print("  NOTE: R² ~0.70–0.85 expected (latent culture variable withheld).\n")

    df_feat = prepare_features(df)
    X       = df_feat[feat_cols].fillna(0)
    X_s     = scaler.transform(X)

    all_metrics = {}
    print(f"  {'Goal':<12} {'XGB R²':>8}  {'XGB RMSE':>9}  {'RF R²':>7}  {'RF RMSE':>8}")
    print(f"  {'-'*52}")

    for goal in GOAL_TARGETS:
        y = df[goal]
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y, test_size=0.2, random_state=42
        )
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        xgb_reg.fit(X_train, y_train)
        xgb_pred = xgb_reg.predict(X_test)
        xgb_r2   = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae  = mean_absolute_error(y_test, xgb_pred)

        rf_reg  = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train, y_train)
        rf_pred = rf_reg.predict(X_test)
        rf_r2   = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        short = goal.replace("score_","").capitalize()
        print(f"  {short:<12} {xgb_r2:>8.3f}  {xgb_rmse:>9.2f}  {rf_r2:>7.3f}  {rf_rmse:>8.2f}")

        joblib.dump(xgb_reg, MODEL_DIR / f"regressor_{goal.replace('score_','')}.pkl")
        all_metrics[goal] = {
            "xgboost":      {"r2": round(xgb_r2,4),"rmse": round(xgb_rmse,4),"mae": round(xgb_mae,4)},
            "random_forest":{"r2": round(rf_r2,4), "rmse": round(rf_rmse,4)},
        }

    avg_xgb = np.mean([v["xgboost"]["r2"] for v in all_metrics.values()])
    avg_rf  = np.mean([v["random_forest"]["r2"] for v in all_metrics.values()])
    print(f"\n  Avg XGBoost R²: {avg_xgb:.3f}  |  Avg RF R²: {avg_rf:.3f}")
    print(f"  XGBoost improvement over RF: +{(avg_xgb - avg_rf)*100:.1f}% R²")
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def train_anomaly_detection(df, feat_cols, scaler):
    print("\n" + "="*55)
    print("MODEL 3: ANOMALY DETECTION — Isolation Forest")
    print("="*55)

    df_feat = prepare_features(df)
    X       = df_feat[feat_cols].fillna(0)
    X_s     = scaler.transform(X)

    iso    = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_s)
    preds  = iso.predict(X_s)
    scores = iso.decision_function(X_s)

    df_t = df.copy()
    df_t["is_anomaly"]    = (preds == -1)
    df_t["anomaly_score"] = scores

    anomalies = df_t[df_t["is_anomaly"]]
    normals   = df_t[~df_t["is_anomaly"]]
    br_anom   = (anomalies["successful_breaches"] > 0).mean()
    br_norm   = (normals["successful_breaches"] > 0).mean()

    print(f"  Detected {(preds==-1).sum()} anomalies ({(preds==-1).mean()*100:.1f}%)")
    print(f"  Resilience — Normal: {normals['overall_resilience_score'].mean():.1f}  |  "
          f"Anomaly: {anomalies['overall_resilience_score'].mean():.1f}")
    print(f"  Breach rate — Normal: {br_norm*100:.1f}%  |  Anomaly: {br_anom*100:.1f}%")

    joblib.dump(iso, MODEL_DIR / "anomaly_detector.pkl")
    return {
        "n_anomalies":        int((preds==-1).sum()),
        "pct_anomalies":      round((preds==-1).mean()*100, 1),
        "avg_score_normal":   round(normals["overall_resilience_score"].mean(), 2),
        "avg_score_anomaly":  round(anomalies["overall_resilience_score"].mean(), 2),
        "breach_rate_normal": round(br_norm, 4),
        "breach_rate_anomaly":round(br_anom, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: TIME-SERIES FORECASTING (GBR + optional Prophet)
# ─────────────────────────────────────────────────────────────────────────────
def train_forecasting(df_ts):
    print("\n" + "="*55)
    print("MODEL 4: TIME-SERIES FORECASTING")
    print("="*55)

    records = []
    for org_id, grp in df_ts.groupby("org_id"):
        grp    = grp.sort_values("month_index")
        scores = grp["resilience_score"].values
        budget = grp["budget_pct"].iloc[0]
        if len(scores) < 18:
            continue
        for i in range(6, 18):
            window = scores[max(0, i-12):i]
            if len(window) < 6:
                continue
            records.append({
                "current_score": scores[i-1],
                "mean_6m":       window[-6:].mean(),
                "mean_12m":      window.mean(),
                "trend_3m":      float(window[-1]-window[-3]) if len(window)>=3 else 0,
                "trend_6m":      float(window[-1]-window[-6]) if len(window)>=6 else 0,
                "std_6m":        window[-6:].std(),
                "budget_pct":    budget,
                "min_12m":       window.min(),
                "max_12m":       window.max(),
                "months_ahead":  1,
                "target":        scores[i],
            })

    df_sup        = pd.DataFrame(records)
    feat_cols_ts  = [c for c in df_sup.columns if c != "target"]
    X, y          = df_sup[feat_cols_ts], df_sup["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting (primary)
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    gbr.fit(X_tr, y_tr)
    gbr_pred = gbr.predict(X_te)
    gbr_r2   = r2_score(y_te, gbr_pred)
    gbr_rmse = np.sqrt(mean_squared_error(y_te, gbr_pred))
    gbr_mape = np.mean(np.abs((y_te - gbr_pred) / (y_te + 1e-9))) * 100
    print(f"  GBR   R²={gbr_r2:.3f}  RMSE={gbr_rmse:.2f}  MAPE={gbr_mape:.1f}%")

    joblib.dump(gbr,          MODEL_DIR / "forecaster.pkl")
    joblib.dump(feat_cols_ts, MODEL_DIR / "forecaster_features.pkl")

    result = {
        "gbr": {"r2": round(gbr_r2,4), "rmse": round(gbr_rmse,4), "mape": round(gbr_mape,2)},
        "r2": round(gbr_r2,4), "rmse": round(gbr_rmse,4), "mape": round(gbr_mape,2),
    }

    # ── Prophet comparison (optional) ─────────────────────────────────────
    if HAS_PROPHET:
        print("  Running Prophet comparison…")
        try:
            prophet_results = []
            for org_id, grp in df_ts.groupby("org_id"):
                grp = grp.sort_values("month_index").reset_index(drop=True)
                if len(grp) < 12:
                    continue
                from datetime import datetime, timedelta
                base_date = datetime(2024, 9, 1)
                prophet_df = pd.DataFrame({
                    "ds": [base_date + timedelta(days=30*i) for i in range(len(grp))],
                    "y":  grp["resilience_score"].values,
                })
                m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                            daily_seasonality=False, seasonality_mode="additive",
                            changepoint_prior_scale=0.1)
                m.fit(prophet_df.iloc[:14])
                future   = m.make_future_dataframe(periods=4, freq="MS")
                forecast = m.predict(future)
                y_actual = prophet_df["y"].values[14:]
                y_hat    = forecast["yhat"].values[14:14+len(y_actual)]
                prophet_results.append(r2_score(y_actual, y_hat))

            prophet_r2 = float(np.mean(prophet_results))
            print(f"  Prophet avg R²={prophet_r2:.3f} across {len(prophet_results)} orgs")
            result["prophet"] = {"avg_r2": round(prophet_r2, 4), "n_orgs": len(prophet_results)}
        except Exception as e:
            print(f"  Prophet error: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5: NLP — CVE Severity Classification
# ─────────────────────────────────────────────────────────────────────────────
def train_nlp(df_cve):
    print("\n" + "="*55)
    print("MODEL 5: NLP — CVE Severity Classification")
    print("  Task: predict CRITICAL/HIGH/MEDIUM/LOW from description text")
    print("  Labels: CVSS numeric scores (not keyword matching)")
    print("="*55)

    df = df_cve[df_cve["severity"].isin(["CRITICAL","HIGH","MEDIUM","LOW"])].copy()
    print(f"\n  Label distribution:\n{df['severity'].value_counts().to_string()}")
    print(f"  Total samples: {len(df)}")

    X = df["description"].astype(str)
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LinearSVC (primary)
    svc_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                  stop_words="english", sublinear_tf=True, min_df=2)),
        ("clf", LinearSVC(C=0.8, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    svc_pipe.fit(X_train, y_train)
    svc_pred = svc_pipe.predict(X_test)
    svc_acc  = accuracy_score(y_test, svc_pred)

    # Logistic Regression baseline
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                  stop_words="english", sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)

    # Random Forest baseline
    rf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                  stop_words="english", sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    print(f"\n  {'Model':<35} {'Test Accuracy':>14}")
    print(f"  {'-'*51}")
    print(f"  {'TF-IDF + LinearSVC (primary)':<35} {svc_acc*100:>13.1f}%")
    print(f"  {'TF-IDF + Logistic Reg (baseline)':<35} {lr_acc*100:>13.1f}%")
    print(f"  {'TF-IDF + Random Forest (baseline)':<35} {rf_acc*100:>13.1f}%")
    print(f"\n  LinearSVC Classification Report:")
    print(classification_report(y_test, svc_pred, zero_division=0))

    df.to_csv(PROC_DIR / "nlp_training.csv", index=False)
    joblib.dump(svc_pipe, MODEL_DIR / "nlp_pipeline.pkl")

    return {
        "task": "cvss_severity_prediction",
        "classes": ["CRITICAL","HIGH","MEDIUM","LOW"],
        "tfidf_svc_accuracy": round(svc_acc, 4),
        "tfidf_lr_accuracy":  round(lr_acc, 4),
        "tfidf_rf_accuracy":  round(rf_acc, 4),
        "svc_gain_over_rf":   round((svc_acc - rf_acc)*100, 2),
        "note": "Labels from CVSS scores; 25% ambiguous generic templates add realistic noise",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS DB
# ─────────────────────────────────────────────────────────────────────────────
RECOMMENDATIONS_DB = [
    {"id":"R01","title":"Implement Automated Backup Testing","goal":"Recover",
     "description":"Monthly automated backup restoration tests.",
     "impact_pct":22,"cost_usd":15000,"time_weeks":6,"priority_base":1,
     "condition": lambda o: o.get("backup_tests_per_year",0) < 4},
    {"id":"R02","title":"Deploy SIEM Solution","goal":"Anticipate",
     "description":"SIEM for centralised log collection and threat detection.",
     "impact_pct":20,"cost_usd":30000,"time_weeks":10,"priority_base":1,
     "condition": lambda o: not o.get("has_siem",False)},
    {"id":"R03","title":"Enforce Multi-Factor Authentication","goal":"Withstand",
     "description":"Roll out MFA to all accounts targeting 95%+ coverage.",
     "impact_pct":18,"cost_usd":8000,"time_weeks":4,"priority_base":1,
     "condition": lambda o: o.get("mfa_coverage_pct",100) < 80},
    {"id":"R04","title":"Implement Network Segmentation","goal":"Withstand",
     "description":"Divide network into isolated segments.",
     "impact_pct":15,"cost_usd":30000,"time_weeks":12,"priority_base":2,
     "condition": lambda o: o.get("network_segmentation_level",5) < 3},
    {"id":"R05","title":"Establish Disaster Recovery Plan","goal":"Recover",
     "description":"DRP with defined RTO/RPO targets.",
     "impact_pct":20,"cost_usd":10000,"time_weeks":8,"priority_base":1,
     "condition": lambda o: not o.get("has_drp",False)},
    {"id":"R06","title":"Increase Security Awareness Training","goal":"Anticipate",
     "description":"Raise training to 90%+ with quarterly phishing simulations.",
     "impact_pct":12,"cost_usd":5000,"time_weeks":4,"priority_base":2,
     "condition": lambda o: o.get("security_training_pct",100) < 80},
    {"id":"R07","title":"Deploy Endpoint Detection & Response","goal":"Withstand",
     "description":"EDR across all endpoints for real-time threat detection.",
     "impact_pct":14,"cost_usd":20000,"time_weeks":6,"priority_base":2,
     "condition": lambda o: not o.get("has_edr",False)},
    {"id":"R08","title":"Improve Patch Compliance","goal":"Withstand",
     "description":"Automated patch management to 95%+ within 30 days.",
     "impact_pct":10,"cost_usd":12000,"time_weeks":6,"priority_base":2,
     "condition": lambda o: o.get("patch_compliance_pct",100) < 85},
    {"id":"R09","title":"Deploy SOAR Platform","goal":"Adapt",
     "description":"Security Orchestration, Automation and Response.",
     "impact_pct":16,"cost_usd":40000,"time_weeks":14,"priority_base":3,
     "condition": lambda o: o.get("has_siem",False) and not o.get("has_soar",False)},
    {"id":"R10","title":"Establish Post-Incident Review Process","goal":"Evolve",
     "description":"Mandatory post-incident reviews for all security events.",
     "impact_pct":10,"cost_usd":3000,"time_weeks":2,"priority_base":2,
     "condition": lambda o: o.get("post_incident_reviews_pct",100) < 70},
    {"id":"R11","title":"Add Threat Intelligence Feeds","goal":"Anticipate",
     "description":"Subscribe to 3+ threat intelligence feeds.",
     "impact_pct":12,"cost_usd":10000,"time_weeks":3,"priority_base":2,
     "condition": lambda o: o.get("threat_intel_feeds",0) < 3},
    {"id":"R12","title":"Encrypt Data at Rest","goal":"Withstand",
     "description":"Full-disk and database encryption.",
     "impact_pct":11,"cost_usd":15000,"time_weeks":8,"priority_base":2,
     "condition": lambda o: o.get("encryption_at_rest_pct",100) < 80},
    {"id":"R13","title":"Hire Dedicated CISO","goal":"Evolve",
     "description":"CISO for strategic security leadership.",
     "impact_pct":15,"cost_usd":150000,"time_weeks":12,"priority_base":2,
     "condition": lambda o: not o.get("has_ciso",False) and o.get("employees",0) > 200},
    {"id":"R14","title":"Implement Security Metrics Dashboard","goal":"Evolve",
     "description":"Track 20+ security KPIs with quarterly board reporting.",
     "impact_pct":8,"cost_usd":5000,"time_weeks":4,"priority_base":3,
     "condition": lambda o: o.get("security_metrics_tracked",0) < 15},
    {"id":"R15","title":"Deploy Honeypots for Early Detection","goal":"Anticipate",
     "description":"Decoy systems to detect attackers early.",
     "impact_pct":12,"cost_usd":5000,"time_weeks":2,"priority_base":3,
     "condition": lambda o: o.get("has_siem",False) and o.get("avg_detect_hours",0) > 24},
    {"id":"R16","title":"Establish Shadow AI Governance Policy","goal":"Adapt",
     "description":"DBIR 2026: 45% employees use unapproved AI. Implement AI usage policy and monitoring.",
     "impact_pct":18,"cost_usd":12000,"time_weeks":6,"priority_base":1,
     "condition": lambda o: o.get("shadow_ai_exposure", False)},
    {"id":"R17","title":"Conduct Third-Party Vendor Risk Assessment","goal":"Withstand",
     "description":"DBIR 2026: Supply chain breaches 48% of all breaches. Assess all vendors.",
     "impact_pct":20,"cost_usd":20000,"time_weeks":10,"priority_base":1,
     "condition": lambda o: o.get("supply_chain_vendors", 0) > 10},
    {"id":"R18","title":"Deploy AI-Powered Security Tools","goal":"Anticipate",
     "description":"IBM 2026: AI tools save $2.2M per breach and detect 108 days faster.",
     "impact_pct":22,"cost_usd":35000,"time_weeks":12,"priority_base":2,
     "condition": lambda o: not o.get("uses_ai_security_tools", False)},
]

rec_serializable = [{k: v for k, v in r.items() if k != "condition"} for r in RECOMMENDATIONS_DB]
with open(MODEL_DIR / "recommendations_db.json", "w") as f:
    json.dump(rec_serializable, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CRASP v2 — ML Model Training (Enhanced)")
    print("=" * 60)

    df_orgs = pd.read_csv(DATA_DIR / "organizations.csv")
    df_ts   = pd.read_csv(DATA_DIR / "timeseries.csv")
    df_cve  = pd.read_csv(DATA_DIR / "cve_data.csv")
    print(f"  Loaded: {len(df_orgs)} orgs, {len(df_ts)} time-series, {len(df_cve)} CVEs")

    clf_metrics, feat_cols, scaler = train_classification(df_orgs)
    reg_metrics  = train_regression(df_orgs, feat_cols, scaler)
    anom_metrics = train_anomaly_detection(df_orgs, feat_cols, scaler)
    fore_metrics = train_forecasting(df_ts)
    nlp_metrics  = train_nlp(df_cve)

    all_metrics = {
        "classification":    clf_metrics,
        "regression":        reg_metrics,
        "anomaly_detection": anom_metrics,
        "forecasting":       fore_metrics,
        "nlp":               nlp_metrics,
        "methodology_note": (
            "R² is lower than naive implementations because a latent "
            "'security_culture_score' (22% weight) is withheld from features. "
            "SMOTE balances Critical/Optimized classes. "
            "SHAP provides post-hoc explainability. "
            "ROC/AUC and t-tests provide rigorous evaluation."
        ),
    }
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    avg_r2 = np.mean([v["xgboost"]["r2"] for v in reg_metrics.values()])
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Classification XGBoost Accuracy : {clf_metrics['xgboost']['test_accuracy']*100:.1f}%")
    print(f"  Classification RF Baseline      : {clf_metrics['random_forest_baseline']['test_accuracy']*100:.1f}%")
    print(f"  Macro AUC                       : {clf_metrics.get('auc_macro',0):.4f}")
    print(f"  T-test (XGB vs RF)              : {clf_metrics['ttest_xgb_vs_rf']['interpretation']}")
    print(f"  Regression Avg XGBoost R²       : {avg_r2:.3f}")
    print(f"  Anomaly Detected                : {anom_metrics['n_anomalies']} orgs")
    print(f"  Forecasting GBR MAPE            : {fore_metrics['mape']:.1f}%")
    if "prophet" in fore_metrics:
        print(f"  Forecasting Prophet avg R²      : {fore_metrics['prophet']['avg_r2']:.3f}")
    print(f"  NLP (Severity) SVC Accuracy     : {nlp_metrics['tfidf_svc_accuracy']*100:.1f}%")
    print(f"  NLP LR Baseline                 : {nlp_metrics['tfidf_lr_accuracy']*100:.1f}%")
    print("=" * 60)
    print(f"  Models saved to: {MODEL_DIR}")
