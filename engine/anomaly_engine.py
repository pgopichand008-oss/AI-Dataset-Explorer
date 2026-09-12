"""
FILE: engine/anomaly_engine.py

Anomaly Investigation Engine.
Performs deterministic, evidence-grounded investigation of numerical and categorical
anomalies across datasets using IQR, Z-score, Isolation Forest, and frequency analysis.

Strictly preserves data honesty:
- 100% deterministic, offline, and JSON-serializable.
- Zero external API / LLM calls.
- NEVER modifies, drops, clips, or replaces values in the original dataset.
- Does not assert causal claims or speculate on business meaning.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from engine.evidence import build_evidence


SEVERITY_RANKS = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
}

SEVERITY_BASE_SCORES = {
    "CRITICAL": 90.0,
    "HIGH": 75.0,
    "MEDIUM": 55.0,
    "LOW": 35.0,
    "INFO": 15.0,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        if pd.isna(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _calculate_finding_score(
    severity: str,
    affected_pct: float = 0.0,
    method_agreement: bool = False,
) -> float:
    """Deterministically compute a bounded finding score in [0.0, 100.0]."""
    base = SEVERITY_BASE_SCORES.get(severity, 35.0)
    mag_adj = min(6.0, (max(0.0, affected_pct) / 100.0) * 6.0)
    agreement_adj = 4.0 if method_agreement else 0.0
    return round(max(0.0, min(100.0, base + mag_adj + agreement_adj)), 2)


# ============================================================
# NUMERICAL ANOMALY INVESTIGATION
# ============================================================

def _investigate_numerical_column(
    col: str,
    series: pd.Series,
    row_count: int,
) -> Optional[Dict[str, Any]]:
    """
    Investigate a numerical series using IQR and Z-score methods.
    Does not modify the original data.
    """
    s_clean = series.dropna()
    n_valid = len(s_clean)
    if n_valid < 4:
        return None

    methods_triggered: List[str] = []
    evidence_lines: List[str] = []

    # 1. IQR Analysis
    q1 = float(s_clean.quantile(0.25))
    q3 = float(s_clean.quantile(0.75))
    iqr = q3 - q1

    if iqr > 0:
        lb = round(q1 - 1.5 * iqr, 4)
        ub = round(q3 + 1.5 * iqr, 4)
        iqr_mask = (s_clean < lb) | (s_clean > ub)
    else:
        # Zero IQR: safe handling
        lb = round(q1, 4)
        ub = round(q3, 4)
        iqr_mask = s_clean != q1

    iqr_count = int(iqr_mask.sum())
    iqr_pct = round((iqr_count / row_count) * 100, 2) if row_count > 0 else 0.0

    if iqr_count > 0:
        methods_triggered.append("IQR")
        evidence_lines.append(
            f"{iqr_count} of {row_count:,} observations ({iqr_pct:.1f}%) were outside IQR fences [{lb}, {ub}] (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})."
        )

    # 2. Z-Score Analysis
    mean_val = float(s_clean.mean())
    std_val = float(s_clean.std())

    z_count = 0
    z_pct = 0.0
    if std_val > 0 and not pd.isna(std_val):
        z_scores = (s_clean - mean_val) / std_val
        z_mask = z_scores.abs() > 3.0
        z_count = int(z_mask.sum())
        z_pct = round((z_count / row_count) * 100, 2) if row_count > 0 else 0.0

        if z_count > 0:
            methods_triggered.append("Z-score")
            evidence_lines.append(
                f"{z_count} of {row_count:,} observations ({z_pct:.1f}%) had |z| > 3.0 (mean={mean_val:.2f}, std={std_val:.2f}, threshold=3.0)."
            )

    if not methods_triggered:
        return None

    # Determine consensus and primary metrics
    if len(methods_triggered) > 1:
        method_name = "IQR + Z-score"
        max_cnt = max(iqr_count, z_count)
        max_pct = max(iqr_pct, z_pct)
        why = f"Values lie significantly beyond standard interquartile fences [{lb}, {ub}] and exceed 3 standard deviations from the mean."
        agreement = True
    elif "IQR" in methods_triggered:
        method_name = "IQR"
        max_cnt = iqr_count
        max_pct = iqr_pct
        why = f"{iqr_count} observations fall outside the standard 1.5x IQR bounds [{lb}, {ub}]."
        agreement = False
    else:
        method_name = "Z-score"
        max_cnt = z_count
        max_pct = z_pct
        why = f"{z_count} observations deviate by more than 3 standard deviations from the empirical mean ({mean_val:.2f})."
        agreement = False

    # Severity classification
    if max_pct >= 10.0:
        severity = "HIGH"
    elif max_pct >= 3.0 or agreement:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    score = _calculate_finding_score(severity, affected_pct=max_pct, method_agreement=agreement)

    return {
        "column": col,
        "method": method_name,
        "anomaly_count": max_cnt,
        "affected_percentage": max_pct,
        "severity": severity,
        "evidence": evidence_lines,
        "why_unusual": why,
        "impact": f"Extreme observations in '{col}' may disproportionately pull the mean, inflate variance estimates, and skew parametric algorithms.",
        "action": f"Inspect extreme records in '{col}' to verify validity and evaluate whether robust scaling, log-transformation, or clipping is appropriate.",
        "_score": score,
        "_magnitude": max_pct,
        "_methods": methods_triggered,
    }


def _investigate_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Apply Isolation Forest on suitable multivariate datasets (N >= 30, numeric_cols >= 2).
    Uses deterministic random_state=42. Does not mutate the input DataFrame.
    """
    row_count = len(df)
    if row_count < 30 or len(numeric_cols) < 2:
        return None

    # Safely prepare numeric sub-matrix without in-place mutation
    sub_df = df[numeric_cols].copy(deep=False)
    # Filter out all-null rows or fill with column medians for isolation modeling only
    clean_matrix = sub_df.fillna(sub_df.median(numeric_only=True))

    # Guard against zero variance across all columns
    if clean_matrix.std(numeric_only=True).sum() == 0:
        return None

    try:
        iso = IsolationForest(contamination="auto", random_state=42)
        iso.fit(clean_matrix)
        scores = iso.score_samples(clean_matrix)
        anomaly_mask = scores <= -0.70
        anomaly_count = int(anomaly_mask.sum())
        anomaly_pct = round((anomaly_count / row_count) * 100, 2)

        if anomaly_count > 0:
            severity = "HIGH" if anomaly_pct >= 10.0 else "MEDIUM"
            score = _calculate_finding_score(severity, affected_pct=anomaly_pct)

            return {
                "column": "Multivariate (" + ", ".join(numeric_cols[:3]) + (f" + {len(numeric_cols)-3} more" if len(numeric_cols) > 3 else "") + ")",
                "method": "Isolation Forest",
                "anomaly_count": anomaly_count,
                "affected_percentage": anomaly_pct,
                "severity": severity,
                "evidence": [
                    f"{anomaly_count} of {row_count:,} observations ({anomaly_pct:.1f}%) were flagged as multivariate isolation anomalies (contamination='auto', random_state=42)."
                ],
                "why_unusual": "Flagged by Isolation Forest as structurally isolated points relative to the joint multi-feature density distribution.",
                "impact": "Multivariate outlier observations can destabilize joint covariance matrices, clustering partitions, and dimensional embeddings.",
                "action": "Inspect the multi-feature profiles of flagged records to identify unusual cross-variable combinations.",
                "_score": score,
                "_magnitude": anomaly_pct,
                "_methods": ["Isolation Forest"],
            }
    except Exception:
        return None

    return None


# ============================================================
# CATEGORICAL FREQUENCY ANOMALIES
# ============================================================

def _investigate_categorical_series(
    col: str,
    series: pd.Series,
    row_count: int,
) -> List[Dict[str, Any]]:
    """
    Investigate categorical frequency anomalies (rare categories, singletons, dominant classes).
    Does not mutate input series.
    """
    findings: List[Dict[str, Any]] = []
    s_clean = series.dropna()
    n_valid = len(s_clean)
    if n_valid < 4:
        return findings

    val_counts = s_clean.value_counts()
    unique_cnt = len(val_counts)

    # 1. Singleton Categories (count == 1 when sample size is meaningful >= 20)
    singletons = val_counts[val_counts == 1]
    if len(singletons) > 0 and n_valid >= 20:
        first_singleton = str(singletons.index[0])
        cnt = 1
        pct = round((1 / n_valid) * 100, 2)
        findings.append({
            "column": col,
            "method": "frequency",
            "category": first_singleton,
            "count": cnt,
            "percentage": pct,
            "severity": "LOW",
            "evidence": f"Category '{first_singleton}' in column '{col}' occurs exactly once ({pct:.2f}% of {n_valid:,} observations).",
            "why_unusual": f"Singleton frequency anomaly ({len(singletons)} unique singleton value(s) in attribute).",
            "impact": "Single-instance categories create unrepresented levels in cross-validation splits and increase risk of unseen level errors during scoring.",
            "action": f"Review singleton category '{first_singleton}' to assess whether it represents a typing error, code mismatch, or should be grouped into an 'Other' bucket.",
            "_score": 35.0,
            "_magnitude": pct,
        })

    # 2. Rare Categories (<= 1.0% when n_valid >= 50, excluding singletons already reported)
    rare = val_counts[(val_counts > 1) & ((val_counts / n_valid) <= 0.01)]
    if len(rare) > 0 and n_valid >= 50:
        rare_cat = str(rare.index[0])
        cnt = int(rare.iloc[0])
        pct = round((cnt / n_valid) * 100, 2)
        findings.append({
            "column": col,
            "method": "frequency",
            "category": rare_cat,
            "count": cnt,
            "percentage": pct,
            "severity": "LOW",
            "evidence": f"Category '{rare_cat}' in column '{col}' occurs {cnt:,} times ({pct:.2f}% of observations, <= 1.0% threshold).",
            "why_unusual": f"Rare frequency anomaly ({len(rare)} rare category level(s) detected below 1.0% prevalence).",
            "impact": "Infrequent categories have wide confidence intervals and can create sparse representation in linear or tree models.",
            "action": f"Evaluate category consolidation for '{rare_cat}' before downstream encoding.",
            "_score": 30.0,
            "_magnitude": pct,
        })

    # 3. Unusually Dominant Categories (>= 95.0% when n_valid >= 20)
    top_cat = str(val_counts.index[0])
    top_cnt = int(val_counts.iloc[0])
    top_pct = round((top_cnt / n_valid) * 100, 2)

    if top_pct >= 95.0 and n_valid >= 20 and unique_cnt > 1:
        findings.append({
            "column": col,
            "method": "frequency",
            "category": top_cat,
            "count": top_cnt,
            "percentage": top_pct,
            "severity": "MEDIUM",
            "evidence": f"Category '{top_cat}' heavily dominates column '{col}', accounting for {top_cnt:,} of {n_valid:,} records ({top_pct:.1f}%).",
            "why_unusual": f"Near-constant frequency concentration ({top_pct:.1f}% in a single level).",
            "impact": "Near-zero categorical variance provides almost no discriminatory power and can cause near-zero variance issues in predictive models.",
            "action": f"Assess whether near-constant column '{col}' should be retained or removed from predictive feature sets.",
            "_score": 55.0,
            "_magnitude": top_pct,
        })

    return findings


# ============================================================
# MAIN PUBLIC API
# ============================================================

def investigate_anomalies(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    max_findings: int = 10,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a comprehensive, deterministic anomaly investigation over a dataset.

    Reuses Phase 1 evidence where available and evaluates:
    - Numerical outliers via IQR and Z-scores
    - Model-based multivariate anomalies via Isolation Forest
    - Categorical frequency anomalies (singletons, rare, and dominant levels)
    - Type coercion and invalid string anomalies

    Strictly preserves data honesty:
    - Never mutates, drops, or alters the original dataset.
    - 100% deterministic and JSON-serializable.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Pre-built Evidence dict from engine.evidence.build_evidence, or raw DataFrame.
    max_findings : int
        Maximum number of findings to return in the findings list (default: 10).
    dataset_name : Optional[str]
        Optional label for the dataset.

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-safe anomaly investigation package.
    """
    # 1. Resolve Evidence & Underlying DataFrame safely (no mutation)
    raw_df: Optional[pd.DataFrame] = None

    if isinstance(evidence_or_df, pd.DataFrame):
        raw_df = evidence_or_df
        evidence = build_evidence(evidence_or_df, dataset_name=dataset_name)
    elif isinstance(evidence_or_df, dict):
        evidence = evidence_or_df
    elif evidence_or_df is None:
        evidence = build_evidence(None, dataset_name=dataset_name)
    else:
        evidence = build_evidence(None, dataset_name=dataset_name)

    name = (
        dataset_name
        or evidence.get("metadata", {}).get("dataset_name")
        or evidence.get("dataset_name")
        or "Dataset"
    )

    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))
    num_cols = list(structure.get("numerical_columns", []))
    cat_cols = list(structure.get("categorical_columns", []))

    # Handle Empty Dataset / Single-Row Edge Cases
    if row_count == 0 or col_count == 0:
        return {
            "dataset_name": name,
            "summary": {
                "total_anomalies": 0,
                "affected_columns": 0,
                "affected_rows": 0,
                "affected_percentage": 0.0,
                "methods_used": [],
            },
            "findings": [],
            "categorical_findings": [],
            "recommendations": ["Dataset contains 0 usable records or columns; no anomalies can be investigated."],
            "generated_from": "evidence",
        }

    if row_count == 1:
        return {
            "dataset_name": name,
            "summary": {
                "total_anomalies": 0,
                "affected_columns": 0,
                "affected_rows": 0,
                "affected_percentage": 0.0,
                "methods_used": [],
            },
            "findings": [],
            "categorical_findings": [],
            "recommendations": ["Single-row dataset does not provide enough observations to compute statistical anomaly boundaries."],
            "generated_from": "evidence",
        }

    methods_used_set: Set[str] = set()
    candidate_findings: List[Dict[str, Any]] = []
    categorical_findings: List[Dict[str, Any]] = []
    affected_cols_set: Set[str] = set()
    total_anomaly_count = 0

    # 2. Numerical Anomaly Investigation
    if raw_df is not None:
        # Investigate directly using the DataFrame
        for col in num_cols:
            if col in raw_df.columns:
                series = raw_df[col]
                res = _investigate_numerical_column(col, series, row_count)
                if res:
                    candidate_findings.append(res)
                    affected_cols_set.add(col)
                    total_anomaly_count += res["anomaly_count"]
                    for m in res["_methods"]:
                        methods_used_set.add(m)

        # Isolation Forest (Multivariate)
        if len(num_cols) >= 2 and row_count >= 30:
            iso_res = _investigate_isolation_forest(raw_df, num_cols)
            if iso_res:
                candidate_findings.append(iso_res)
                total_anomaly_count += iso_res["anomaly_count"]
                for m in iso_res["_methods"]:
                    methods_used_set.add(m)

    else:
        # Investigate using pre-computed Evidence Layer
        quality = evidence.get("quality", {})
        outlier_findings = quality.get("outlier_findings", [])
        num_stats = evidence.get("numerical_statistics", {})

        for f in outlier_findings:
            col = str(f.get("attribute", ""))
            cnt = _safe_int(f.get("count", 0))
            pct = _safe_float(f.get("percentage", 0.0))
            lb = f.get("lower_bound")
            ub = f.get("upper_bound")

            if cnt > 0 and col:
                methods_used_set.add("IQR")
                affected_cols_set.add(col)
                total_anomaly_count += cnt

                evidence_lines = [
                    f"{cnt} of {row_count:,} observations ({pct:.1f}%) were outside IQR fences [{lb}, {ub}]."
                ]
                methods_triggered = ["IQR"]

                # Check if Z-score can be corroborated from numerical_statistics
                stat = num_stats.get(col, {})
                mean_val = stat.get("mean")
                std_val = stat.get("std")
                max_val = stat.get("max")
                min_val = stat.get("min")

                has_z = False
                if std_val is not None and std_val > 0 and mean_val is not None:
                    if max_val is not None and (max_val - mean_val) / std_val > 3.0:
                        has_z = True
                    if min_val is not None and (mean_val - min_val) / std_val > 3.0:
                        has_z = True

                if has_z:
                    methods_triggered.append("Z-score")
                    methods_used_set.add("Z-score")
                    evidence_lines.append(
                        f"Extreme values exceed 3.0 standard deviations (mean={mean_val:.2f}, std={std_val:.2f})."
                    )

                method_label = "IQR + Z-score" if has_z else "IQR"
                severity = "HIGH" if pct >= 10.0 else ("MEDIUM" if pct >= 3.0 or has_z else "LOW")
                score = _calculate_finding_score(severity, affected_pct=pct, method_agreement=has_z)

                candidate_findings.append({
                    "column": col,
                    "method": method_label,
                    "anomaly_count": cnt,
                    "affected_percentage": pct,
                    "severity": severity,
                    "evidence": evidence_lines,
                    "why_unusual": f"Observations lie outside standard interquartile bounds [{lb}, {ub}].",
                    "impact": f"Extreme values in '{col}' can pull the sample mean and increase parametric modeling variance.",
                    "action": f"Inspect extreme records in '{col}' to verify authenticity before model fitting.",
                    "_score": score,
                    "_magnitude": pct,
                    "_methods": methods_triggered,
                })

    # 3. Invalid Values / Type Coercion Findings
    columns_detail = evidence.get("columns", [])
    for c in columns_detail:
        col = c.get("name")
        inv = c.get("invalid_values", {})
        inv_cnt = _safe_int(inv.get("invalid_count", inv.get("non_numeric_count", 0)))
        if inv_cnt > 0 and col:
            methods_used_set.add("Type Coercion")
            affected_cols_set.add(col)
            total_anomaly_count += inv_cnt
            pct = round((inv_cnt / row_count) * 100, 2)
            severity = "HIGH" if pct >= 10.0 else "MEDIUM"
            score = _calculate_finding_score(severity, affected_pct=pct)

            candidate_findings.append({
                "column": col,
                "method": "Type Coercion",
                "anomaly_count": inv_cnt,
                "affected_percentage": pct,
                "severity": severity,
                "evidence": [
                    f"{inv_cnt:,} values ({pct:.1f}%) could not be parsed as valid numeric entries.",
                    f"Sample unparseable string(s): {inv.get('invalid_examples', inv.get('sample_invalid', []))[:3]}.",
                ],
                "why_unusual": "Failed numeric conversion due to unexpected string or symbol entries in numerical attribute.",
                "impact": "Causes runtime conversion errors and prevents mathematical aggregations.",
                "action": f"Review invalid entries in column '{col}' and sanitize source formats.",
                "_score": score,
                "_magnitude": pct,
                "_methods": ["Type Coercion"],
            })

    # 4. Categorical Frequency Anomaly Investigation
    if raw_df is not None:
        for col in cat_cols:
            if col in raw_df.columns:
                series = raw_df[col]
                cat_res = _investigate_categorical_series(col, series, row_count)
                if cat_res:
                    categorical_findings.extend(cat_res)
                    affected_cols_set.add(col)
                    methods_used_set.add("frequency")
    else:
        cat_stats = evidence.get("categorical_statistics", {})
        for col, stat in cat_stats.items():
            if isinstance(stat, dict):
                dom_pct = _safe_float(stat.get("dominance_percentage", 0.0))
                top_cat = stat.get("top_category")
                top_cnt = _safe_int(stat.get("top_frequency", 0))
                unique_cnt = _safe_int(stat.get("unique_count", 0))

                if dom_pct >= 95.0 and row_count >= 20 and unique_cnt > 1:
                    methods_used_set.add("frequency")
                    affected_cols_set.add(col)
                    categorical_findings.append({
                        "column": col,
                        "method": "frequency",
                        "category": str(top_cat),
                        "count": top_cnt,
                        "percentage": dom_pct,
                        "severity": "MEDIUM",
                        "evidence": f"Category '{top_cat}' in column '{col}' accounts for {dom_pct:.1f}% of observations ({top_cnt:,} of {row_count:,}).",
                        "why_unusual": f"Extreme categorical dominance ({dom_pct:.1f}% in single level).",
                        "impact": "Provides near-zero variance for predictive modeling.",
                        "action": f"Review whether column '{col}' provides useful variance for analysis.",
                        "_score": 55.0,
                        "_magnitude": dom_pct,
                    })

    # 5. Deterministic Sorting & Ranking of Numerical Findings
    def _sort_key(x: Dict[str, Any]) -> Tuple[float, int, float, str]:
        score = _safe_float(x.get("_score", 0.0))
        sev = x.get("severity", "INFO")
        sev_rank = SEVERITY_RANKS.get(sev, 5)
        mag = _safe_float(x.get("_magnitude", 0.0))
        title = str(x.get("column", ""))
        return (-score, sev_rank, -mag, title)

    candidate_findings.sort(key=_sort_key)

    # Clean internal score keys and assign sequential 1-based ranks
    clean_findings: List[Dict[str, Any]] = []
    for rank_idx, finding in enumerate(candidate_findings, start=1):
        clean_findings.append({
            "rank": rank_idx,
            "column": finding.get("column", ""),
            "method": finding.get("method", ""),
            "anomaly_count": finding.get("anomaly_count", 0),
            "affected_percentage": finding.get("affected_percentage", 0.0),
            "severity": finding.get("severity", "LOW"),
            "evidence": finding.get("evidence", []),
            "why_unusual": finding.get("why_unusual", ""),
            "impact": finding.get("impact", ""),
            "action": finding.get("action", ""),
        })

    # Clean internal keys for categorical findings
    clean_categorical_findings: List[Dict[str, Any]] = []
    categorical_findings.sort(key=lambda x: (-_safe_float(x.get("_score", 0.0)), str(x.get("column", ""))))
    for cf in categorical_findings:
        clean_categorical_findings.append({
            "column": cf.get("column", ""),
            "method": cf.get("method", "frequency"),
            "category": cf.get("category", ""),
            "count": cf.get("count", 0),
            "percentage": cf.get("percentage", 0.0),
            "severity": cf.get("severity", "LOW"),
            "evidence": cf.get("evidence", ""),
            "why_unusual": cf.get("why_unusual", ""),
            "impact": cf.get("impact", ""),
            "action": cf.get("action", ""),
        })

    # Truncate to max_findings
    if max_findings is not None and max_findings > 0:
        returned_findings = clean_findings[:max_findings]
    else:
        returned_findings = clean_findings

    # 6. Overall Summary Metrics
    # Compute estimated affected row count bounded by row_count
    max_single_col_anomalies = max([f["anomaly_count"] for f in clean_findings] or [0])
    affected_rows = min(row_count, max_single_col_anomalies)
    overall_affected_pct = round((affected_rows / row_count) * 100, 2) if row_count > 0 else 0.0

    methods_sorted = sorted(list(methods_used_set))

    # 7. Actionable Recommendations
    recommendations: List[str] = []
    if returned_findings:
        for f in returned_findings[:3]:
            recommendations.append(f["action"])
    if clean_categorical_findings:
        recommendations.append(clean_categorical_findings[0]["action"])

    if not recommendations:
        recommendations.append(
            "No severe numerical outliers or unusual categorical frequency patterns detected; proceed with baseline modeling."
        )

    return {
        "dataset_name": name,
        "summary": {
            "total_anomalies": total_anomaly_count,
            "affected_columns": len(affected_cols_set),
            "affected_rows": affected_rows,
            "affected_percentage": overall_affected_pct,
            "methods_used": methods_sorted,
        },
        "findings": returned_findings,
        "categorical_findings": clean_categorical_findings,
        "recommendations": recommendations,
        "generated_from": "evidence",
    }
