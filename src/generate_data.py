"""
CRASP v2 - Synthetic Data Generation
Generates:
  1. 1000 organizational profiles with 80+ observable features
  2. Time-series resilience scores (200 orgs × 18 months)
  3. CVE vulnerability data — descriptions WITHOUT CVSS in text (for real NLP task)
  4. MITRE ATT&CK techniques

Fix applied: A latent 'security_culture_score' variable (unobservable) contributes 22%
to each goal score, so ML models must predict under genuine uncertainty
rather than just reproducing a known deterministic formula.
"""

import numpy as np
import pandas as pd
import json
import requests
import time
import os
import random
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR / SIZE DEFINITIONS
# Sources:
#   Verizon DBIR 2026 — incident rates, supply chain breach rates by sector
#   IBM Cost of a Data Breach 2026 — breach cost multipliers (Healthcare $6.64M highest)
#   Gartner Security Spending 2026 — budget multipliers ($244.2B global, +13.3% YoY)
# ─────────────────────────────────────────────────────────────────────────────
SECTORS = {
    # baseline: sector resilience norm (0-100)
    # budget_mult: relative security spend vs average (Gartner 2026)
    # incident_rate: breach frequency multiplier vs baseline (DBIR 2026)
    # breach_cost_mult: cost multiplier vs $4.99M global average (IBM 2026)
    # supply_chain_risk: probability extra supply chain exposure (DBIR 2026: 48% avg)
    "Finance":       {"baseline": 74, "budget_mult": 1.35, "incident_rate": 0.78, "breach_cost_mult": 1.10, "supply_chain_risk": 0.55},
    "Healthcare":    {"baseline": 62, "budget_mult": 1.12, "incident_rate": 1.28, "breach_cost_mult": 1.33, "supply_chain_risk": 0.60},
    "Technology":    {"baseline": 71, "budget_mult": 1.25, "incident_rate": 0.88, "breach_cost_mult": 1.05, "supply_chain_risk": 0.70},
    "Retail":        {"baseline": 57, "budget_mult": 0.92, "incident_rate": 1.12, "breach_cost_mult": 0.90, "supply_chain_risk": 0.65},
    "Education":     {"baseline": 49, "budget_mult": 0.72, "incident_rate": 1.32, "breach_cost_mult": 0.80, "supply_chain_risk": 0.40},
    "Government":    {"baseline": 67, "budget_mult": 1.02, "incident_rate": 1.02, "breach_cost_mult": 0.95, "supply_chain_risk": 0.50},
    "Manufacturing": {"baseline": 59, "budget_mult": 0.88, "incident_rate": 1.12, "breach_cost_mult": 0.92, "supply_chain_risk": 0.72},
    "Energy":        {"baseline": 69, "budget_mult": 1.18, "incident_rate": 0.92, "breach_cost_mult": 1.08, "supply_chain_risk": 0.58},
}

SIZE_CATEGORIES = {
    "Small":      {"emp_range": (10, 200),    "rev_range": (1, 20),     "must_ciso": False},
    "Medium":     {"emp_range": (200, 1000),  "rev_range": (20, 200),   "must_ciso": False},
    "Large":      {"emp_range": (1000, 5000), "rev_range": (200, 2000), "must_ciso": True},
    "Enterprise": {"emp_range": (5000, 50000),"rev_range": (2000,50000),"must_ciso": True},
}

COMPANY_NAMES = [
    "Alpha","Beta","Gamma","Delta","Epsilon","Zeta","Theta","Sigma",
    "Omega","Apex","Nexus","Vertex","Zenith","Horizon","Summit",
    "Pinnacle","Crest","Peak","Ridge","Vantage","Meridian","Solaris",
    "Aurora","Orion","Titan","Atlas","Helios","Aegis","Bastion",
    "Citadel","Fortress","Shield","Sentinel","Guardian","Watchman",
    "Paladin","Vanguard","Rampart","Bulwark","Paragon",
]
SUFFIXES = ["Corp","Inc","LLC","Solutions","Systems","Technologies",
            "Group","Partners","Enterprises","Services","Global","Dynamics"]


def generate_company_name():
    return f"{random.choice(COMPANY_NAMES)} {random.choice(SUFFIXES)}"


# ─────────────────────────────────────────────────────────────────────────────
# LATENT CULTURE SCORE  (key fix for academic soundness)
# ─────────────────────────────────────────────────────────────────────────────
def sample_culture_score(sector_name, size_name, budget_pct):
    """
    Represents unobservable factors: team morale, security leadership quality,
    informal knowledge sharing, risk appetite. These cannot be captured by
    survey metrics alone — they are the 'iceberg below the waterline'.

    In real assessments consultants observe this; our models cannot.
    Contributing 22% to true score creates genuine prediction uncertainty.
    """
    base = 50.0
    # Sector culture norms (Finance/Tech tend to have stronger security culture)
    sector_bonus = {"Finance": 10, "Technology": 8, "Energy": 5,
                    "Government": 3, "Healthcare": 0, "Manufacturing": -3,
                    "Retail": -5, "Education": -8}
    base += sector_bonus.get(sector_name, 0)
    # Larger orgs & higher budgets correlate weakly with culture
    if size_name in ("Large", "Enterprise"):
        base += 5
    base += (budget_pct - 9) * 0.8
    # Add genuine randomness — culture has high variance
    return float(np.clip(np.random.normal(base, 18), 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# FORMULA SCORES (observable-feature component, 78% weight)
# ─────────────────────────────────────────────────────────────────────────────
def calc_anticipate_formula(org):
    score = 0
    score += min(org["threat_intel_feeds"] * 5, 25)
    score += min(org["vuln_scans_per_month"] * 2, 16)
    score += org["security_training_pct"] * 0.20
    score += 10 if org["has_siem"] else 0
    score += 10 if org["has_threat_hunting"] else 0
    detect_hrs = org["avg_detect_hours"]
    if detect_hrs <= 1:    score += 15
    elif detect_hrs <= 12: score += 10
    elif detect_hrs <= 48: score += 5
    score += org["phishing_sim_per_year"] * 1.5
    # 2026: AI security tools boost detection (IBM 2026 — AI reduces response time)
    score += 8 if org.get("uses_ai_security_tools", False) else 0
    # 2026: Shadow AI increases unmonitored attack surface (DBIR 2026)
    score -= 6 if org.get("shadow_ai_exposure", False) else 0
    return float(np.clip(score, 0, 100))


def calc_withstand_formula(org):
    score = 0
    score += min(org["firewall_layers"] * 8, 24)
    score += org["mfa_coverage_pct"] * 0.20
    score += org["encryption_at_rest_pct"] * 0.10
    score += org["encryption_in_transit_pct"] * 0.10
    score += org["patch_compliance_pct"] * 0.15
    score += org["network_segmentation_level"] * 4
    score += 5 if org["has_ids_ips"] else 0
    score += 5 if org["has_dlp"] else 0
    score += 5 if org["has_edr"] else 0
    # 2026: Faster patching = stronger defence (DBIR 2026: median now 43 days)
    patch_days = org.get("avg_patch_days", 43)
    if patch_days <= 14:   score += 10
    elif patch_days <= 30: score += 6
    elif patch_days <= 43: score += 3
    else:                  score -= 3
    # 2026: Each third-party vendor is an attack surface (DBIR 2026: 48% supply chain)
    vendors = org.get("supply_chain_vendors", 10)
    score -= min(vendors * 0.3, 10)
    return float(np.clip(score, 0, 100))


def calc_recover_formula(org):
    score = 0
    backup_map = {"Hourly": 25, "Daily": 20, "Weekly": 12, "Monthly": 5, "None": 0}
    score += backup_map.get(org["backup_frequency"], 0)
    score += min(org["backup_tests_per_year"] * 4, 20)
    score += 15 if org["has_drp"] else 0
    score += 10 if org["backup_encryption"] else 0
    rto = org["avg_recovery_hours"]
    if rto <= 4:    score += 20
    elif rto <= 12: score += 14
    elif rto <= 48: score += 8
    else:           score += 2
    return float(np.clip(score, 0, 100))


def calc_adapt_formula(org):
    score = 0
    score += org["security_automation_pct"] * 0.35
    score += 15 if org["has_soar"] else 0
    score += 10 if org["dynamic_policy_updates"] else 0
    score += min(org["config_changes_per_year"] / 10, 15)
    score += 10 if org["has_devsecops"] else 0
    score += 5 if org["cloud_security_posture"] else 0
    # 2026: AI security tools improve automated response (IBM 2026)
    score += 10 if org.get("uses_ai_security_tools", False) else 0
    # 2026: Shadow AI introduces ungoverned risk vectors (DBIR 2026)
    score -= 8 if org.get("shadow_ai_exposure", False) else 0
    return float(np.clip(score, 0, 100))


def calc_evolve_formula(org):
    score = 0
    score += org["post_incident_reviews_pct"] * 0.25
    score += 15 if org["continuous_improvement_program"] else 0
    score += min(org["security_metrics_tracked"] * 1.5, 20)
    score += 10 if org["board_security_reporting"] else 0
    score += 10 if org["security_certifications"] > 0 else 0
    score += min(org["security_certifications"] * 3, 15)
    score += 5 if org["bug_bounty_program"] else 0
    return float(np.clip(score, 0, 100))


def calc_true_score(formula_score, culture_score):
    """
    True resilience = 78% observable metrics + 22% latent culture + noise.
    The 22% latent component is what makes ML prediction non-trivial.
    """
    noise = np.random.normal(0, 4)
    true_score = 0.78 * formula_score + 0.22 * culture_score + noise
    return float(np.clip(true_score, 0, 100))


def assign_resilience_level(overall_score):
    if overall_score < 25:   return 1
    elif overall_score < 50: return 2
    elif overall_score < 75: return 3
    elif overall_score < 90: return 4
    else:                    return 5


LEVEL_LABELS = {1: "Critical", 2: "Developing", 3: "Managed", 4: "Advanced", 5: "Optimized"}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SINGLE ORGANIZATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_organization(org_id):
    sector_name = random.choice(list(SECTORS.keys()))
    sector = SECTORS[sector_name]
    size_name = random.choices(
        list(SIZE_CATEGORIES.keys()), weights=[0.30, 0.35, 0.25, 0.10]
    )[0]
    size = SIZE_CATEGORIES[size_name]

    employees  = int(np.random.uniform(*size["emp_range"]))
    revenue_m  = float(np.random.uniform(*size["rev_range"]))
    budget_pct = float(np.clip(np.random.normal(9, 2.5) * sector["budget_mult"], 2, 18))
    it_budget_m      = revenue_m * np.random.uniform(0.04, 0.12)
    security_budget  = it_budget_m * (budget_pct / 100) * 1_000_000
    staff_ratio      = np.random.normal(1.0, 0.4) * (budget_pct / 9)
    security_staff   = max(1, int(employees * staff_ratio / 100))

    has_ciso = size["must_ciso"] or (random.random() < 0.68 * (budget_pct / 9))
    has_siem = random.random() < (0.4 + budget_pct * 0.04)
    has_soar = has_siem and (random.random() < (0.15 + budget_pct * 0.02))

    org = {
        "org_id": org_id,
        "company_name": generate_company_name(),
        "sector": sector_name,
        "size_category": size_name,
        "employees": employees,
        "revenue_million_usd": round(revenue_m, 2),

        "security_budget_pct": round(budget_pct, 2),
        "security_budget_usd": round(security_budget, 0),
        "security_staff": security_staff,
        "has_ciso": has_ciso,
        "uses_external_mssp": random.random() < (0.3 + budget_pct * 0.02),

        # ANTICIPATE
        "threat_intel_feeds":      max(0, int(np.random.normal(3, 1.5) * (budget_pct / 9))),
        "vuln_scans_per_month":    max(0, int(np.random.normal(4, 2) * (budget_pct / 9))),
        "security_training_pct":   float(np.clip(np.random.normal(75, 15) * (budget_pct / 9), 10, 100)),
        "has_siem":                has_siem,
        "has_threat_hunting":      has_siem and (random.random() < 0.4),
        "avg_detect_hours":        float(np.clip(np.random.exponential(48) / (budget_pct / 9), 1, 720)),
        "phishing_sim_per_year":   max(0, int(np.random.normal(4, 2) * (budget_pct / 9))),

        # WITHSTAND
        "firewall_layers":            max(1, int(np.random.normal(2, 0.8) * (budget_pct / 9))),
        "mfa_coverage_pct":           float(np.clip(np.random.normal(70, 20) * (budget_pct / 9), 5, 100)),
        "encryption_at_rest_pct":     float(np.clip(np.random.normal(65, 20) * (budget_pct / 9), 5, 100)),
        "encryption_in_transit_pct":  float(np.clip(np.random.normal(75, 15) * (budget_pct / 9), 10, 100)),
        "patch_compliance_pct":       float(np.clip(np.random.normal(78, 15) * (budget_pct / 9), 20, 100)),
        "network_segmentation_level": max(0, min(5, int(np.random.normal(2.5, 1) * (budget_pct / 9)))),
        "has_ids_ips":                random.random() < (0.3 + budget_pct * 0.04),
        "has_dlp":                    random.random() < (0.2 + budget_pct * 0.03),
        "has_edr":                    random.random() < (0.35 + budget_pct * 0.04),

        # RECOVER
        "backup_frequency": random.choices(
            ["Hourly", "Daily", "Weekly", "Monthly", "None"],
            weights=[max(0.01, budget_pct*0.02), 0.45, 0.30, 0.15, max(0.01, 0.09 - budget_pct*0.005)]
        )[0],
        "backup_tests_per_year":         max(0, int(np.random.normal(4, 2) * (budget_pct / 9))),
        "has_drp":                        random.random() < (0.3 + budget_pct * 0.05),
        "avg_recovery_hours":             float(np.clip(np.random.exponential(72) / (budget_pct / 9), 1, 720)),
        "recovery_time_objective_hours":  float(np.clip(np.random.normal(24, 12), 1, 168)),
        "backup_encryption":              random.random() < (0.4 + budget_pct * 0.04),

        # ADAPT
        "security_automation_pct":  float(np.clip(np.random.normal(50, 20) * (budget_pct / 9), 0, 100)),
        "has_soar":                  has_soar,
        "dynamic_policy_updates":    random.random() < (0.3 + budget_pct * 0.04),
        "config_changes_per_year":   max(0, int(np.random.normal(80, 40) * (budget_pct / 9))),
        "has_devsecops":             random.random() < (0.2 + budget_pct * 0.03),
        "cloud_security_posture":    random.random() < (0.4 + budget_pct * 0.03),

        # EVOLVE
        "post_incident_reviews_pct":      float(np.clip(np.random.normal(65, 20) * (budget_pct / 9), 0, 100)),
        "continuous_improvement_program": random.random() < (0.3 + budget_pct * 0.05),
        "security_metrics_tracked":       max(0, int(np.random.normal(12, 5) * (budget_pct / 9))),
        "board_security_reporting":       random.random() < (0.3 + budget_pct * 0.04),
        "security_certifications":        max(0, int(np.random.normal(1, 1) * (budget_pct / 9))),
        "bug_bounty_program":             random.random() < (0.1 + budget_pct * 0.015),

        # Incident history
        "incidents_last_year": max(0, int(
            np.random.normal(10, 5) * sector["incident_rate"] / (budget_pct / 9)
        )),
        "successful_breaches":    0,
        "data_lost_gb":           0.0,
        "total_downtime_hours":   0.0,

        # ── 2026 REPORT FEATURES ──────────────────────────────────────────────
        # DBIR 2026: median patch time rose to 43 days; faster = better defence
        "avg_patch_days": float(np.clip(
            np.random.normal(43, 18) / (budget_pct / 9), 5, 180
        )),
        # DBIR 2026: 45% of employees use unapproved shadow AI tools
        "shadow_ai_exposure": random.random() < (0.45 - budget_pct * 0.015),
        # DBIR 2026: 48% of breaches involve supply chain; more vendors = more risk
        "supply_chain_vendors": max(0, int(
            np.random.normal(15, 8) * sector["supply_chain_risk"]
        )),
        # IBM 2026: AI security tools cut breach cost; adoption grows with budget
        "uses_ai_security_tools": random.random() < (0.20 + budget_pct * 0.035),
        # IBM 2026: 25% of malicious breaches are now AI-enabled
        "ai_breach_history": random.random() < (0.25 * sector["incident_rate"]),
        # IBM 2026: Avg breach cost $4.99M globally; sector multiplier applied
        "estimated_breach_cost_usd": round(
            4_990_000 * sector["breach_cost_mult"] * np.random.uniform(0.6, 1.4), 0
        ),
    }

    breach_prob = 0.05 + (org["incidents_last_year"] / 100) + max(0, (5 - budget_pct) * 0.03)
    org["successful_breaches"] = int(np.random.poisson(breach_prob))
    if org["successful_breaches"] > 0:
        org["data_lost_gb"]         = float(np.random.exponential(50) * org["successful_breaches"])
        org["total_downtime_hours"]  = float(np.random.exponential(24) * org["successful_breaches"])

    # Sample latent culture score (unobservable — NOT used as a feature)
    culture = sample_culture_score(sector_name, size_name, budget_pct)

    # TRUE goal scores blend observable formula + latent culture + noise
    org["score_anticipate"] = round(calc_true_score(calc_anticipate_formula(org), culture), 1)
    org["score_withstand"]  = round(calc_true_score(calc_withstand_formula(org), culture), 1)
    org["score_recover"]    = round(calc_true_score(calc_recover_formula(org), culture), 1)
    org["score_adapt"]      = round(calc_true_score(calc_adapt_formula(org), culture), 1)
    org["score_evolve"]     = round(calc_true_score(calc_evolve_formula(org), culture), 1)

    weights = {"anticipate": 0.22, "withstand": 0.25, "recover": 0.22, "adapt": 0.17, "evolve": 0.14}
    overall = (
        org["score_anticipate"] * weights["anticipate"] +
        org["score_withstand"]  * weights["withstand"]  +
        org["score_recover"]    * weights["recover"]    +
        org["score_adapt"]      * weights["adapt"]      +
        org["score_evolve"]     * weights["evolve"]
    )
    sector_adj = (sector["baseline"] - 65) * 0.1
    overall = float(np.clip(overall + sector_adj + np.random.normal(0, 2), 0, 100))
    org["overall_resilience_score"] = round(overall, 1)
    org["resilience_level"]         = assign_resilience_level(overall)
    org["resilience_label"]         = LEVEL_LABELS[org["resilience_level"]]
    # Store culture for research transparency (but models do NOT receive this column)
    org["security_culture_score"]   = round(culture, 1)

    return org


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE 1000 ORGANIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def generate_organizations(n=3000):
    print(f"Generating {n} synthetic organizations...")
    orgs = [generate_organization(i + 1) for i in range(n)]
    df   = pd.DataFrame(orgs)
    path = OUTPUT_DIR / "organizations.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} organizations → {path}")
    print(f"  Score distribution:\n{df['resilience_label'].value_counts().to_string()}")
    print(f"  Culture score vs overall correlation: {df['security_culture_score'].corr(df['overall_resilience_score']):.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TIME-SERIES DATA (200 orgs × 18 months)
# ─────────────────────────────────────────────────────────────────────────────
EVENTS = [
    {"name": "Security Incident",      "delta": -12, "prob": 0.05},
    {"name": "New Regulation",         "delta": -7,  "prob": 0.03},
    {"name": "Staff Departure",        "delta": -10, "prob": 0.04},
    {"name": "New Tool Deployed",      "delta": +12, "prob": 0.04},
    {"name": "Executive Support",      "delta": +15, "prob": 0.02},
    {"name": "Budget Increase",        "delta": +10, "prob": 0.03},
    {"name": "Budget Cut",             "delta": -8,  "prob": 0.03},
    {"name": "Security Training",      "delta": +6,  "prob": 0.05},
    {"name": "Major Breach",           "delta": -20, "prob": 0.01},
    {"name": "Certification Achieved", "delta": +8,  "prob": 0.02},
]


def generate_timeseries(df_orgs, n=200):
    print(f"\nGenerating time-series for {n} organizations (18 months)...")
    sample_orgs = df_orgs.sample(n, random_state=42).reset_index(drop=True)
    records     = []
    base_date   = datetime(2024, 9, 1)

    for _, org in sample_orgs.iterrows():
        current_score = org["overall_resilience_score"]
        budget_pct    = org["security_budget_pct"]

        if budget_pct > 11:   monthly_trend = np.random.uniform(0.3, 1.0)
        elif budget_pct > 7:  monthly_trend = np.random.uniform(-0.3, 0.5)
        else:                 monthly_trend = np.random.uniform(-0.8, 0.1)

        start_score = float(np.clip(
            current_score - (monthly_trend * 18) + np.random.normal(0, 3), 10, 95
        ))
        score = start_score

        for month_idx in range(18):
            month_date = base_date + timedelta(days=30 * month_idx)
            score += monthly_trend + np.random.normal(0, 2)
            if month_date.month in [7, 8]:
                score -= np.random.uniform(0, 1.5)

            event_this_month = None
            for event in EVENTS:
                if random.random() < event["prob"]:
                    score += event["delta"] + np.random.normal(0, 2)
                    event_this_month = event["name"]
                    break
            score = float(np.clip(score, 5, 99))

            records.append({
                "org_id":           org["org_id"],
                "company_name":     org["company_name"],
                "sector":           org["sector"],
                "month_index":      month_idx + 1,
                "date":             month_date.strftime("%Y-%m"),
                "resilience_score": round(score, 1),
                "event":            event_this_month or "",
                "budget_pct":       org["security_budget_pct"],
            })

    df_ts = pd.DataFrame(records)
    path  = OUTPUT_DIR / "timeseries.csv"
    df_ts.to_csv(path, index=False)
    print(f"  Saved {len(df_ts)} time-series records → {path}")
    return df_ts


# ─────────────────────────────────────────────────────────────────────────────
# CVE DATA — descriptions exclude CVSS numbers but use severity-discriminating
# vocabulary so the NLP model has genuine signal to learn from.
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTS = [
    "Apache HTTP Server","Microsoft Exchange Server","OpenSSL","Linux Kernel",
    "WordPress","VMware vCenter","Cisco IOS","Oracle WebLogic","Adobe Acrobat",
    "Google Chrome","Mozilla Firefox","Windows Server","Nginx","Jenkins",
    "GitLab","Kubernetes","Docker","Redis","MongoDB","PostgreSQL","Jira",
    "Atlassian Confluence","SolarWinds Orion","Fortinet FortiOS","Palo Alto PAN-OS",
    "Ivanti Connect Secure","MOVEit Transfer","Progress WS_FTP","Citrix NetScaler",
    "F5 BIG-IP","Pulse Secure VPN","Cisco ASA","Microsoft SharePoint",
]

# Per-severity attack types, CVSS ranges, and discriminating vocabulary.
# NOTE: Several attack types intentionally appear in multiple severity buckets
# (SQL injection can be CRITICAL or HIGH; information disclosure can be any level)
# — this mirrors real CVE data and prevents the NLP model from achieving trivial
# accuracy by memorising attack type → severity mappings.
SEVERITY_PROFILES = {
    "CRITICAL": {
        "cvss_range": (9.0, 10.0),
        "attacks": ["remote code execution", "command injection", "buffer overflow",
                    "authentication bypass", "SQL injection"],  # SQL injection also in HIGH
        "qualifiers": ["unauthenticated", "pre-authentication", "zero-click",
                       "network-accessible without credentials"],
        "impacts": [
            "execute arbitrary code with SYSTEM-level privileges",
            "achieve complete unauthenticated remote code execution",
            "fully compromise the affected host",
            "gain root-level access without any credentials",
            "deploy malicious payloads and achieve persistent access",
            "take complete control of the target system",
        ],
        "contexts": [
            "No authentication or user interaction is required to exploit this vulnerability.",
            "This is remotely exploitable with a network attack vector and zero user interaction.",
            "The vulnerability has been observed being actively exploited in the wild.",
            "An unauthenticated remote attacker can exploit this to achieve arbitrary code execution.",
            "CVSS attack complexity is low; no privileges are required for exploitation.",
            "Exploit code has been publicly released; immediate patching is strongly recommended.",
        ],
    },
    "HIGH": {
        "cvss_range": (7.0, 8.9),
        "attacks": ["privilege escalation", "SQL injection", "path traversal",
                    "information disclosure", "denial of service"],  # overlaps with CRITICAL, MEDIUM, LOW
        "qualifiers": ["authenticated low-privileged", "network-adjacent",
                       "locally authenticated"],
        "impacts": [
            "escalate privileges to local administrator",
            "bypass authentication and gain elevated access to restricted functions",
            "read sensitive credential and configuration files",
            "execute arbitrary SQL queries against the backend database",
            "obtain elevated privileges on the underlying operating system",
            "gain unauthorized access to privileged administrative functionality",
        ],
        "contexts": [
            "Exploitation requires a low-privileged authenticated account.",
            "An attacker with network access and minimal credentials can exploit this issue.",
            "Successful exploitation could allow privilege escalation to admin level.",
            "The vulnerability has low attack complexity and requires low privileges.",
            "An authenticated attacker with a standard user account can leverage this flaw.",
            "Network-adjacent attackers with basic credentials may trigger this vulnerability.",
        ],
    },
    "MEDIUM": {
        "cvss_range": (4.0, 6.9),
        "attacks": ["cross-site scripting", "path traversal", "SQL injection",
                    "information disclosure", "denial of service"],  # overlaps with HIGH and LOW
        "qualifiers": ["authenticated", "requires user interaction",
                       "requires social engineering"],
        "impacts": [
            "inject malicious scripts into the victim's browser session",
            "perform cross-site scripting attacks against authenticated users",
            "disclose partial file system contents via path traversal",
            "redirect users to malicious external sites",
            "access internal network resources via server-side request forgery",
            "expose sensitive application data to authenticated low-privileged users",
        ],
        "contexts": [
            "Exploitation requires the victim to click a maliciously crafted link.",
            "User interaction is required; the attacker must convince a user to visit a crafted page.",
            "The vulnerability requires an authenticated session to exploit.",
            "Impact is limited to confidentiality and integrity; availability is unaffected.",
            "Successful exploitation requires social engineering of a legitimate authenticated user.",
            "Attack complexity is high and requires specific user interaction to succeed.",
        ],
    },
    "LOW": {
        "cvss_range": (1.0, 3.9),
        "attacks": ["information disclosure", "path traversal",
                    "cross-site scripting", "denial of service"],  # overlaps with MEDIUM and HIGH
        "qualifiers": ["physically present", "locally authenticated with high privileges",
                       "requires administrative access"],
        "impacts": [
            "expose non-sensitive software version information to local users",
            "reveal minor system configuration details with negligible security impact",
            "cause a brief disruption to a non-critical auxiliary service component",
            "disclose low-sensitivity metadata to locally authenticated users",
            "enumerate installed software versions via verbose error responses",
        ],
        "contexts": [
            "Exploitation requires physical access to the affected system.",
            "Impact is limited to minor information disclosure with negligible risk.",
            "The vulnerability has very limited attack surface and extremely low exploitability.",
            "No sensitive data or privileged functions are accessible via this flaw.",
            "Exploitation requires high privileges and is constrained to local access only.",
            "The security impact is considered minimal; no sensitive data is exposed.",
        ],
    },
}

# Target distribution (balanced enough for NLP training)
_SEVERITY_WEIGHTS = {"CRITICAL": 0.25, "HIGH": 0.32, "MEDIUM": 0.30, "LOW": 0.13}


def _cvss_to_severity(cvss):
    if cvss >= 9.0:   return "CRITICAL"
    elif cvss >= 7.0: return "HIGH"
    elif cvss >= 4.0: return "MEDIUM"
    else:             return "LOW"


def generate_synthetic_cves(n=1500):
    """
    Generate CVE records with severity-discriminating vocabulary.
    Severity is chosen first (controlled distribution), then attack type and
    vocabulary are drawn from that severity's profile.
    CVSS numeric scores are assigned from the severity's CVSS range and are
    NOT embedded in the text — the NLP model must learn from content only.
    """
    sev_labels  = list(_SEVERITY_WEIGHTS.keys())
    sev_weights = list(_SEVERITY_WEIGHTS.values())

    records = []
    for i in range(n):
        year     = random.choice([2023, 2024, 2025])
        cve_num  = random.randint(10000, 99999)
        product  = random.choice(PRODUCTS)

        # Choose severity first → controlled distribution
        severity = random.choices(sev_labels, weights=sev_weights)[0]
        profile  = SEVERITY_PROFILES[severity]

        cvss     = round(random.uniform(*profile["cvss_range"]), 1)
        attack   = random.choice(profile["attacks"])
        qual     = random.choice(profile["qualifiers"])
        impact   = random.choice(profile["impacts"])
        context  = random.choice(profile["contexts"])

        tier = random.random()
        generic_contexts = [
            "The issue was discovered during an internal security audit.",
            "Affected versions should be updated to the latest release.",
            "Vendors have released patches addressing this vulnerability.",
            "Users are advised to apply the available security update.",
            "This vulnerability affects multiple versions of the software.",
            "A patch is available in the vendor's latest security advisory.",
        ]
        generic_impacts = [
            "gain unintended access", "cause an unexpected error",
            "affect system integrity", "access restricted resources",
            "trigger unexpected application behaviour",
        ]
        if tier < 0.25:
            # Tier 3 — fully generic: no severity signals at all
            desc = (
                f"A {attack} vulnerability in {product} could allow attackers to "
                f"{random.choice(generic_impacts)}. {random.choice(generic_contexts)}"
            )
        elif tier < 0.60:
            # Tier 2 — partial: attack type + impact only (no qualifier, no context)
            desc = (
                f"A {attack} vulnerability exists in {product} that may allow "
                f"attackers to {impact}. {random.choice(generic_contexts)}"
            )
        else:
            # Tier 1 — full severity-specific vocabulary
            templates = [
                f"A {attack} vulnerability in {product} allows {qual} attackers to {impact}. {context}",
                f"{product} is affected by a {attack} flaw. {qual.capitalize()} adversaries can {impact}. {context}",
                f"An improperly secured {attack} function in {product} permits {qual} exploitation to {impact}. {context}",
                f"A flaw in the {random.choice(['authentication','parsing','input validation','session management'])} "
                f"component of {product} leads to {attack}, allowing {qual} attackers to {impact}. {context}",
            ]
            desc = random.choice(templates)

        records.append({
            "cve_id":         f"CVE-{year}-{cve_num}",
            "description":    desc,
            "published_date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "cvss_score":     cvss,
            "severity":       severity,
        })
    return records


def fetch_cve_data(target=1500):
    print("\nFetching CVE data from NVD API...")
    records = []
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    try:
        while len(records) < target:
            params = {
                "resultsPerPage": 100,
                "startIndex": len(records),
                "pubStartDate": "2023-01-01T00:00:00.000",
                "pubEndDate":   "2025-12-31T23:59:59.999",
            }
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  NVD API {resp.status_code} — using synthetic fallback")
                break
            data  = resp.json()
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                break
            for v in vulns:
                cve  = v.get("cve", {})
                cid  = cve.get("id", "")
                desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
                pub  = cve.get("published", "")[:10]
                metrics   = cve.get("metrics", {})
                cvss      = None
                severity  = "UNKNOWN"
                for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if key in metrics and metrics[key]:
                        m         = metrics[key][0]
                        cvss_data = m.get("cvssData", {})
                        cvss      = cvss_data.get("baseScore")
                        severity  = m.get("baseSeverity", cvss_data.get("baseSeverity", "UNKNOWN"))
                        break
                if cid and desc and cvss:
                    # Strip any explicit score mentions from real descriptions
                    clean_desc = desc[:500].replace(str(cvss), "").strip()
                    records.append({
                        "cve_id": cid, "description": clean_desc,
                        "published_date": pub, "cvss_score": cvss, "severity": severity,
                    })
            print(f"  Fetched {len(records)} CVEs...")
            time.sleep(0.7)
            if len(records) >= target:
                break
    except Exception as e:
        print(f"  NVD error: {e}")

    if len(records) < 50:
        print("  Generating synthetic CVEs (NVD unavailable)...")
        records = generate_synthetic_cves(target)

    df   = pd.DataFrame(records[:target])
    path = OUTPUT_DIR / "cve_data.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} CVEs → {path}")
    print(f"  Severity distribution:\n{df['severity'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_mitre_attack():
    print("\nFetching MITRE ATT&CK data...")
    records = []
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            objects = resp.json().get("objects", [])
            for obj in objects:
                if obj.get("type") != "attack-pattern":
                    continue
                tech_id = next((
                    r.get("external_id", "") for r in obj.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"
                ), "")
                if not tech_id:
                    continue
                records.append({
                    "technique_id":   tech_id,
                    "name":           obj.get("name", ""),
                    "description":    obj.get("description", "")[:400],
                    "tactics":        ", ".join(p["phase_name"] for p in obj.get("kill_chain_phases", [])),
                    "platforms":      ", ".join(obj.get("x_mitre_platforms", [])),
                    "is_subtechnique": "." in tech_id,
                })
            print(f"  Downloaded {len(records)} MITRE ATT&CK techniques")
    except Exception as e:
        print(f"  MITRE download error: {e}")

    if len(records) < 50:
        print("  Generating synthetic MITRE fallback...")
        records = _synthetic_mitre()

    df   = pd.DataFrame(records)
    path = OUTPUT_DIR / "mitre_attack.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} MITRE techniques → {path}")
    return df


def _synthetic_mitre():
    return [
        {"technique_id":"T1566.001","name":"Spearphishing Attachment","tactics":"initial-access",
         "description":"Adversaries send spearphishing emails with malicious attachments.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":True},
        {"technique_id":"T1059.001","name":"PowerShell","tactics":"execution",
         "description":"Adversaries abuse PowerShell commands for execution.",
         "platforms":"Windows","is_subtechnique":True},
        {"technique_id":"T1486","name":"Data Encrypted for Impact","tactics":"impact",
         "description":"Adversaries encrypt data on target systems (ransomware).",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1190","name":"Exploit Public-Facing Application","tactics":"initial-access",
         "description":"Adversaries exploit weaknesses in Internet-facing applications.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1021.001","name":"Remote Desktop Protocol","tactics":"lateral-movement",
         "description":"Adversaries use Valid Accounts to log into systems via RDP.",
         "platforms":"Windows","is_subtechnique":True},
        {"technique_id":"T1110","name":"Brute Force","tactics":"credential-access",
         "description":"Adversaries use brute force techniques to gain account access.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1027","name":"Obfuscated Files","tactics":"defense-evasion",
         "description":"Adversaries make executables or files difficult to analyze.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1071","name":"Application Layer Protocol","tactics":"command-and-control",
         "description":"Adversaries communicate using OSI application layer protocols.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1003","name":"OS Credential Dumping","tactics":"credential-access",
         "description":"Adversaries attempt to dump credentials for account login information.",
         "platforms":"Windows,Linux","is_subtechnique":False},
        {"technique_id":"T1082","name":"System Information Discovery","tactics":"discovery",
         "description":"Adversaries attempt to get detailed OS and hardware information.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1041","name":"Exfiltration Over C2 Channel","tactics":"exfiltration",
         "description":"Adversaries steal data by exfiltrating over a C2 channel.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1055","name":"Process Injection","tactics":"defense-evasion",
         "description":"Adversaries inject code into processes to evade defenses.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1053.005","name":"Scheduled Task","tactics":"persistence",
         "description":"Adversaries abuse Windows Task Scheduler for task scheduling.",
         "platforms":"Windows","is_subtechnique":True},
        {"technique_id":"T1078","name":"Valid Accounts","tactics":"defense-evasion",
         "description":"Adversaries obtain and abuse credentials of existing accounts.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1562.001","name":"Disable Security Tools","tactics":"defense-evasion",
         "description":"Adversaries disable or modify security tools to avoid detection.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":True},
        {"technique_id":"T1047","name":"Windows Management Instrumentation","tactics":"execution",
         "description":"Adversaries abuse WMI to execute malicious commands.",
         "platforms":"Windows","is_subtechnique":False},
        {"technique_id":"T1218","name":"System Binary Proxy Execution","tactics":"defense-evasion",
         "description":"Adversaries bypass defenses by proxying execution through system binaries.",
         "platforms":"Windows","is_subtechnique":False},
        {"technique_id":"T1105","name":"Ingress Tool Transfer","tactics":"command-and-control",
         "description":"Adversaries transfer tools or files from an external system.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1083","name":"File and Directory Discovery","tactics":"discovery",
         "description":"Adversaries enumerate files and directories in specific locations.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
        {"technique_id":"T1485","name":"Data Destruction","tactics":"impact",
         "description":"Adversaries destroy data and files to interrupt system availability.",
         "platforms":"Windows,Linux,macOS","is_subtechnique":False},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CRASP v2 — Synthetic Data Generation")
    print("=" * 60)

    df_orgs = generate_organizations(3000)
    df_ts   = generate_timeseries(df_orgs, 300)
    df_cve  = fetch_cve_data(1500)
    df_mit  = fetch_mitre_attack()

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print(f"  Organizations : {len(df_orgs)}")
    print(f"  Time-series   : {len(df_ts)} records ({len(df_ts['org_id'].unique())} orgs)")
    print(f"  CVE entries   : {len(df_cve)}")
    print(f"  MITRE techs   : {len(df_mit)}")
    print("=" * 60)
