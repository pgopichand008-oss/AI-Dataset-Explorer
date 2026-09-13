"""
FILE: engine/priority_engine.py

Insight Priority Engine.
Converts factual findings from Phase 1 (Evidence Layer) and Phase 2 (Health Engine)
into deterministic, prioritized, and actionable insights.

Strictly preserves data honesty:
- All numerical values are derived directly from computed evidence and health metrics.
- No invented statistics, column names, or percentages.
- No causal assertions or speculative business claims.
- 100% deterministic, reproducible, and JSON-serializable.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health


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


def _calculate_priority_score(
    severity: str,
    magnitude_pct: float = 0.0,
    is_blocking: bool = False,
    scope_factor: float = 0.0,
) -> float:
    """
    Deterministically calculate a bounded priority score in [0.0, 100.0].

    Parameters:
    -----------
    severity : str
        One of 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'.
    magnitude_pct : float
        Percentage of rows/cells/columns affected (0.0 to 100.0).
    is_blocking : bool
        True if the issue halts or invalidates machine learning or standard analysis.
    scope_factor : float
        Additional weight for dataset-wide vs localized single-column issues (0.0 to 5.0).
    """
    base = SEVERITY_BASE_SCORES.get(severity, 35.0)

    # Magnitude adjustment: up to +6.0 points based on proportion affected
    magnitude_adjustment = min(6.0, (max(0.0, magnitude_pct) / 100.0) * 6.0)

    # Blocking adjustment: +4.0 points if blocking
    blocking_adjustment = 4.0 if is_blocking else 0.0

    # Scope adjustment: up to +3.0 points
    scope_adj = min(3.0, max(0.0, scope_factor))

    score = base + magnitude_adjustment + blocking_adjustment + scope_adj
    return round(max(0.0, min(100.0, score)), 2)


def _detect_empty_or_single_row(
    evidence: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Detect empty dataset or single-row dataset."""
    insights: List[Dict[str, Any]] = []
    handled_keys: Set[str] = set()

    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))

    if row_count == 0 or col_count == 0:
        handled_keys.add("empty_dataset")
        insights.append({
            "title": "Empty Dataset (0 Usable Records or Columns)",
            "category": "structure",
            "severity": "CRITICAL",
            "evidence": {
                "row_count": row_count,
                "column_count": col_count,
            },
            "interpretation": f"The dataset contains {row_count} row(s) and {col_count} column(s).",
            "impact": "No analytical, statistical, or machine learning operations can be conducted on an empty dataset.",
            "priority_score": 100.0,
            "priority_rank": 0,
            "recommended_action": "Provide a dataset containing populated tabular records and columns.",
            "source": "evidence",
            "_magnitude": 100.0,
        })
        return insights, handled_keys

    if row_count == 1:
        handled_keys.add("single_row")
        insights.append({
            "title": "Single-Row Dataset (Insufficient Sample for Analysis)",
            "category": "sample_size",
            "severity": "CRITICAL",
            "evidence": {
                "row_count": 1,
                "column_count": col_count,
            },
            "interpretation": "The dataset contains exactly 1 record.",
            "impact": "Variance, standard deviation, correlation, and predictive modeling cannot be computed from a single row.",
            "priority_score": 95.0,
            "priority_rank": 0,
            "recommended_action": "Collect additional observational records before conducting statistical or machine learning analysis.",
            "source": "evidence",
            "_magnitude": 100.0,
        })
        return insights, handled_keys

    # Small sample size (2 to 29 rows)
    if row_count < 10:
        handled_keys.add("sample_size")
        insights.append({
            "title": f"Very Small Sample Size ({row_count} rows)",
            "category": "sample_size",
            "severity": "HIGH",
            "evidence": {
                "row_count": row_count,
                "minimum_recommended": 30,
            },
            "interpretation": f"The dataset contains only {row_count} rows, which is substantially below standard empirical thresholds.",
            "impact": "High sampling variance and extreme risk of unrepresentative statistical metrics and modeling overfitting.",
            "priority_score": _calculate_priority_score("HIGH", magnitude_pct=80.0, is_blocking=False, scope_factor=2.0),
            "priority_rank": 0,
            "recommended_action": "Acquire additional data records to improve statistical confidence and validation reliability.",
            "source": "evidence",
            "_magnitude": 80.0,
        })
    elif row_count < 30:
        handled_keys.add("sample_size")
        insights.append({
            "title": f"Limited Sample Size ({row_count} rows)",
            "category": "sample_size",
            "severity": "MEDIUM",
            "evidence": {
                "row_count": row_count,
                "minimum_recommended": 30,
            },
            "interpretation": f"The dataset has {row_count} rows, which is below the recommended threshold of 30 observations for modeling.",
            "impact": "Statistical inferences have limited statistical power and machine learning cross-validation splits will be small.",
            "priority_score": _calculate_priority_score("MEDIUM", magnitude_pct=50.0, is_blocking=False, scope_factor=1.0),
            "priority_rank": 0,
            "recommended_action": "Exercise caution when evaluating statistical tests, and prioritize simpler models or data collection.",
            "source": "evidence",
            "_magnitude": 50.0,
        })

    return insights, handled_keys


def _detect_structural_findings(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect structural anomalies such as duplicate column names."""
    insights: List[Dict[str, Any]] = []
    structure = evidence.get("structure", {})

    if structure.get("has_duplicate_columns"):
        dup_names = list(structure.get("duplicate_column_names", []))
        insights.append({
            "title": f"Duplicate Column Names Detected ({len(dup_names)} duplicated)",
            "category": "structure",
            "severity": "CRITICAL",
            "evidence": {
                "duplicate_column_names": dup_names,
                "duplicate_count": len(dup_names),
            },
            "interpretation": f"The dataset contains duplicate column header names: {', '.join(dup_names)}.",
            "impact": "Duplicate column headers cause indexing ambiguity, data overwrite errors, and pipeline crashes.",
            "priority_score": _calculate_priority_score("CRITICAL", magnitude_pct=60.0, is_blocking=True, scope_factor=3.0),
            "priority_rank": 0,
            "recommended_action": "Rename duplicate column headers with unique names prior to downstream data transformations.",
            "source": "evidence",
            "_magnitude": 60.0,
        })

    return insights


def _detect_missingness(
    evidence: Dict[str, Any],
    empty_cols_set: Set[str],
) -> List[Dict[str, Any]]:
    """
    Detect missingness findings:
    - Empty (all-missing) columns
    - Column-level missingness
    - Dataset-wide missingness
    """
    insights: List[Dict[str, Any]] = []

    structure = evidence.get("structure", {})
    completeness = evidence.get("completeness", {})
    quality = evidence.get("quality", {})

    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))
    total_missing_cells = _safe_int(completeness.get("missing_cells", 0))
    total_missing_pct = _safe_float(completeness.get("missing_percentage", 0.0))
    columns_with_missing = completeness.get("columns_with_missing", [])
    empty_columns = quality.get("empty_columns", [])
    col_missing_counts = completeness.get("column_missing_counts", {})
    col_missing_percentages = completeness.get("column_missing_percentages", {})

    if row_count == 0:
        return insights

    # 1. Empty / all-missing columns
    for col in empty_columns:
        empty_cols_set.add(col)
        insights.append({
            "title": f"Empty Column: '{col}' (100% Missing)",
            "category": "missingness",
            "severity": "HIGH",
            "evidence": {
                "column": col,
                "missing_count": row_count,
                "missing_percentage": 100.0,
            },
            "interpretation": f"Column '{col}' contains 0 non-null values across all {row_count:,} rows.",
            "impact": "Carries no analytical information or variance; cannot be used in predictive modeling or statistical testing.",
            "priority_score": _calculate_priority_score("HIGH", magnitude_pct=100.0, is_blocking=False, scope_factor=1.5),
            "priority_rank": 0,
            "recommended_action": f"Remove empty column '{col}' from the dataset or verify data pipeline ingestion.",
            "source": "evidence",
            "_magnitude": 100.0,
        })

    # 2. Column-level missingness (excluding completely empty columns)
    for item in columns_with_missing:
        if isinstance(item, dict):
            col = item.get("column")
            missing_cnt = _safe_int(item.get("missing_count", 0))
            missing_pct = _safe_float(item.get("missing_percentage", 0.0))
        elif isinstance(item, str):
            col = item
            missing_cnt = _safe_int(col_missing_counts.get(col, 0))
            missing_pct = _safe_float(col_missing_percentages.get(col, 0.0))
        else:
            continue

        if not col or col in empty_cols_set:
            continue

        if missing_pct <= 0.0:
            continue

        if missing_pct > 40.0:
            severity = "CRITICAL"
            is_blocking = True
        elif missing_pct > 20.0:
            severity = "HIGH"
            is_blocking = False
        elif missing_pct > 5.0:
            severity = "MEDIUM"
            is_blocking = False
        else:
            severity = "LOW"
            is_blocking = False

        score = _calculate_priority_score(severity, magnitude_pct=missing_pct, is_blocking=is_blocking)

        insights.append({
            "title": f"Missing Values in Column '{col}' ({missing_pct:.1f}%)",
            "category": "missingness",
            "severity": severity,
            "evidence": {
                "column": col,
                "missing_count": missing_cnt,
                "missing_percentage": missing_pct,
                "total_rows": row_count,
            },
            "interpretation": f"Column '{col}' has {missing_cnt:,} missing values ({missing_pct:.1f}% of {row_count:,} rows).",
            "impact": f"Reduces sample representation for '{col}'-dependent queries and requires imputation or exclusion before model training.",
            "priority_score": score,
            "priority_rank": 0,
            "recommended_action": f"Investigate the missingness pattern for '{col}' and apply domain-appropriate imputation or filter rules.",
            "source": "evidence",
            "_magnitude": missing_pct,
        })

    # 3. Dataset-wide missingness (if widespread across multiple attributes)
    non_empty_missing_cols = [
        (item.get("column") if isinstance(item, dict) else item)
        for item in columns_with_missing
    ]
    non_empty_missing_cols = [c for c in non_empty_missing_cols if c and c not in empty_cols_set]

    if total_missing_pct >= 20.0 and len(non_empty_missing_cols) > 1:
        severity = "CRITICAL" if total_missing_pct >= 40.0 else "HIGH"
        insights.append({
            "title": f"Widespread Dataset Missingness ({total_missing_pct:.1f}% Overall)",
            "category": "missingness",
            "severity": severity,
            "evidence": {
                "total_missing_cells": total_missing_cells,
                "missing_percentage": total_missing_pct,
                "affected_columns_count": len(non_empty_missing_cols),
                "total_columns": col_count,
            },
            "interpretation": f"{total_missing_pct:.1f}% of all cells ({total_missing_cells:,} total) across {len(non_empty_missing_cols)} columns are missing.",
            "impact": "Broad sparsity limits joint feature analyses and reduces complete-case sample sizes for multi-variable models.",
            "priority_score": _calculate_priority_score(severity, magnitude_pct=total_missing_pct, is_blocking=(total_missing_pct >= 40.0), scope_factor=3.0),
            "priority_rank": 0,
            "recommended_action": "Formulate a systematic data imputation strategy across the affected feature set.",
            "source": "evidence",
            "_magnitude": total_missing_pct,
        })

    return insights


def _detect_duplicates(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect duplicate rows."""
    insights: List[Dict[str, Any]] = []
    structure = evidence.get("structure", {})
    duplicates = evidence.get("duplicates", {})

    row_count = _safe_int(structure.get("row_count", 0))
    dup_rows = _safe_int(duplicates.get("duplicate_rows", 0))
    dup_pct = _safe_float(duplicates.get("duplicate_row_percentage", 0.0))

    if dup_rows > 0 and row_count > 0:
        if dup_pct >= 25.0:
            severity = "CRITICAL"
            is_blocking = True
        elif dup_pct >= 10.0:
            severity = "HIGH"
            is_blocking = False
        elif dup_pct >= 2.0:
            severity = "MEDIUM"
            is_blocking = False
        else:
            severity = "LOW"
            is_blocking = False

        score = _calculate_priority_score(severity, magnitude_pct=dup_pct, is_blocking=is_blocking, scope_factor=2.0)

        insights.append({
            "title": f"Duplicate Rows Detected ({dup_rows:,} rows, {dup_pct:.1f}%)",
            "category": "duplicates",
            "severity": severity,
            "evidence": {
                "duplicate_rows": dup_rows,
                "duplicate_row_percentage": dup_pct,
                "total_rows": row_count,
            },
            "interpretation": f"The dataset contains {dup_rows:,} duplicate rows ({dup_pct:.1f}% of all {row_count:,} records).",
            "impact": "Duplicate records artificially inflate sample weights, distort statistical metrics, and can cause train-test leakage.",
            "priority_score": score,
            "priority_rank": 0,
            "recommended_action": "Inspect duplicate rows to confirm whether they are redundant ingestions or legitimate repeated events, and deduplicate if appropriate.",
            "source": "evidence",
            "_magnitude": dup_pct,
        })

    return insights


def _detect_constant_columns(
    evidence: Dict[str, Any],
    empty_cols_set: Set[str],
) -> List[Dict[str, Any]]:
    """Detect zero-variance / constant columns (excluding empty columns)."""
    insights: List[Dict[str, Any]] = []
    quality = evidence.get("quality", {})
    structure = evidence.get("structure", {})
    columns = evidence.get("columns", [])

    row_count = _safe_int(structure.get("row_count", 0))
    constant_columns = quality.get("constant_columns", [])

    for col in constant_columns:
        if col in empty_cols_set:
            continue

        sample_val = None
        for c in columns:
            if c.get("name") == col:
                samples = c.get("sample_values", [])
                if samples:
                    sample_val = samples[0]
                break

        insights.append({
            "title": f"Constant Column: '{col}' (Single Unique Value)",
            "category": "quality",
            "severity": "MEDIUM",
            "evidence": {
                "column": col,
                "unique_count": 1,
                "constant_value": sample_val,
                "total_rows": row_count,
            },
            "interpretation": f"Column '{col}' contains only 1 unique value across all {row_count:,} records.",
            "impact": "Provides zero statistical variance or predictive signal; causes singular matrix issues in linear models.",
            "priority_score": _calculate_priority_score("MEDIUM", magnitude_pct=50.0, is_blocking=False),
            "priority_rank": 0,
            "recommended_action": f"Remove constant column '{col}' from predictive feature sets.",
            "source": "evidence",
            "_magnitude": 50.0,
        })

    return insights


def _detect_invalid_values(
    evidence: Dict[str, Any],
    empty_cols_set: Set[str],
) -> List[Dict[str, Any]]:
    """Detect unparseable / non-numeric values in numeric columns."""
    insights: List[Dict[str, Any]] = []
    columns = evidence.get("columns", [])
    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))

    for c in columns:
        col = c.get("name")
        if not col or col in empty_cols_set:
            continue

        inv = c.get("invalid_values", {})
        invalid_cnt = _safe_int(inv.get("invalid_count", inv.get("non_numeric_count", 0)))

        if invalid_cnt > 0:
            pct = round((invalid_cnt / row_count) * 100, 2) if row_count > 0 else 0.0
            severity = "HIGH" if pct > 10.0 else "MEDIUM"
            is_blocking = pct > 20.0
            score = _calculate_priority_score(severity, magnitude_pct=pct, is_blocking=is_blocking)

            sample_invalid = inv.get("invalid_examples", inv.get("sample_invalid", []))

            insights.append({
                "title": f"Invalid Values in Column '{col}' ({invalid_cnt:,} non-numeric)",
                "category": "validity",
                "severity": severity,
                "evidence": {
                    "column": col,
                    "invalid_count": invalid_cnt,
                    "invalid_percentage": pct,
                    "sample_invalid": sample_invalid,
                },
                "interpretation": f"Column '{col}' contains {invalid_cnt:,} values ({pct:.1f}%) that cannot be converted to numeric format.",
                "impact": "Prevents mathematical aggregations, causes type coercion failures, and disrupts algorithmic training.",
                "priority_score": score,
                "priority_rank": 0,
                "recommended_action": f"Inspect invalid string entries in column '{col}' and clean or convert them to standard numeric formats.",
                "source": "evidence",
                "_magnitude": pct,
            })

    return insights


def _detect_outliers(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect significant numerical outliers."""
    insights: List[Dict[str, Any]] = []
    quality = evidence.get("quality", {})
    num_stats = evidence.get("numerical_statistics", {})
    outlier_findings = quality.get("outlier_findings", [])
    outlier_columns = quality.get("outlier_columns", [])

    findings_by_attr: Dict[str, Dict[str, Any]] = {}
    for f in outlier_findings:
        attr = f.get("attribute")
        if attr:
            findings_by_attr[attr] = f

    for col in outlier_columns:
        stat = num_stats.get(col, {})
        f_info = findings_by_attr.get(col, {})

        out_cnt = _safe_int(f_info.get("count", 0))
        out_pct = _safe_float(f_info.get("percentage", 0.0))

        if out_pct >= 10.0:
            severity = "HIGH"
        elif out_pct >= 3.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        score = _calculate_priority_score(severity, magnitude_pct=out_pct)

        insights.append({
            "title": f"Significant Outliers in Column '{col}' ({out_pct:.1f}%)" if out_pct > 0 else f"Outliers Detected in Column '{col}'",
            "category": "outliers",
            "severity": severity,
            "evidence": {
                "column": col,
                "outlier_count": out_cnt,
                "outlier_percentage": out_pct,
                "lower_bound": f_info.get("lower_bound"),
                "upper_bound": f_info.get("upper_bound"),
                "min": stat.get("min"),
                "max": stat.get("max"),
                "mean": stat.get("mean"),
                "median": stat.get("median"),
            },
            "interpretation": f"Column '{col}' exhibits {out_cnt:,} values ({out_pct:.1f}%) lying beyond IQR fences." if out_pct > 0 else f"Column '{col}' contains outlier values beyond normal statistical boundaries.",
            "impact": "Extreme values can disproportionately influence parametric models, inflate standard errors, and skew predictions.",
            "priority_score": score,
            "priority_rank": 0,
            "recommended_action": f"Inspect extreme values in '{col}' to verify validity and apply robust scaling, clipping, or transformation.",
            "source": "evidence",
            "_magnitude": out_pct,
        })

    return insights


def _detect_correlations(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect strong correlations and multicollinearity."""
    insights: List[Dict[str, Any]] = []
    correlations = evidence.get("correlations", {})

    if not correlations.get("supported"):
        return insights

    multi_pairs = correlations.get("potential_multicollinearity", [])
    strong_pairs = correlations.get("strongest_pairs", [])

    seen_pairs: Set[Tuple[str, str]] = set()
    high_pairs: List[Dict[str, Any]] = []

    for item in (multi_pairs + strong_pairs):
        a1 = item.get("attribute_1")
        a2 = item.get("attribute_2")
        r = _safe_float(item.get("correlation", 0.0))

        if not a1 or not a2:
            continue

        pair_key = tuple(sorted([str(a1), str(a2)]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        if abs(r) >= 0.80:
            high_pairs.append({
                "attribute_1": a1,
                "attribute_2": a2,
                "correlation": r,
            })

    # Sort high pairs by absolute correlation descending
    high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    for pair in high_pairs[:3]:
        r = pair["correlation"]
        a1 = pair["attribute_1"]
        a2 = pair["attribute_2"]
        abs_r = abs(r)
        direction = "positive" if r > 0 else "negative"

        if abs_r >= 0.95:
            severity = "HIGH"
        elif abs_r >= 0.85:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        score = _calculate_priority_score(severity, magnitude_pct=(abs_r * 100 - 80) * 5.0)

        insights.append({
            "title": f"Strong Correlation: '{a1}' and '{a2}' (r = {r:.2f})",
            "category": "correlation",
            "severity": severity,
            "evidence": {
                "attribute_1": a1,
                "attribute_2": a2,
                "correlation": round(r, 4),
            },
            "interpretation": f"Attributes '{a1}' and '{a2}' display a strong {direction} correlation of {r:.2f}.",
            "impact": "High collinearity can destabilize regression coefficient estimates and duplicate feature weight in algorithms.",
            "priority_score": score,
            "priority_rank": 0,
            "recommended_action": f"Evaluate redundancy between '{a1}' and '{a2}'; consider feature reduction, regularization, or domain filtering.",
            "source": "evidence",
            "_magnitude": round(abs_r * 100, 1),
        })

    return insights


def _detect_ml_readiness(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect machine learning readiness blockers and warnings."""
    insights: List[Dict[str, Any]] = []
    structure = evidence.get("structure", {})
    ml = evidence.get("ml_readiness", {})

    row_count = _safe_int(structure.get("row_count", 0))
    if row_count <= 1:
        return insights

    ml_score = _safe_float(ml.get("ml_score", 0.0))
    ml_status = str(ml.get("ml_status", "NOT READY"))
    can_train = bool(ml.get("can_train", False))
    warnings = ml.get("warnings", [])
    target_cands = ml.get("target_candidates", [])

    if not can_train or ml_status in ["NOT READY", "Critical"]:
        severity = "CRITICAL" if ml_score < 40.0 else "HIGH"
        insights.append({
            "title": f"Machine Learning Training Blocked (Score: {ml_score:.0f}/100)",
            "category": "ml_readiness",
            "severity": severity,
            "evidence": {
                "ml_score": ml_score,
                "ml_status": ml_status,
                "can_train": can_train,
                "warnings": warnings,
            },
            "interpretation": f"The dataset is categorized as '{ml_status}' with an ML readiness score of {ml_score:.0f}/100.",
            "impact": "Automated model training cannot proceed reliably due to unresolved data quality, target, or sample size constraints.",
            "priority_score": _calculate_priority_score(severity, magnitude_pct=100.0 - ml_score, is_blocking=True),
            "priority_rank": 0,
            "recommended_action": "Remediate highlighted data quality warnings and designate a valid target attribute before modeling.",
            "source": "evidence",
            "_magnitude": round(100.0 - ml_score, 1),
        })
    elif warnings:
        insights.append({
            "title": f"ML Readiness Warnings ({len(warnings)} issue(s) flagged)",
            "category": "ml_readiness",
            "severity": "MEDIUM",
            "evidence": {
                "ml_score": ml_score,
                "ml_status": ml_status,
                "warnings": warnings,
            },
            "interpretation": f"ML readiness scored {ml_score:.0f}/100 with warnings: {'; '.join(warnings[:2])}.",
            "impact": "Supervised modeling can proceed, but unaddressed warnings may impair model stability and predictive accuracy.",
            "priority_score": _calculate_priority_score("MEDIUM", magnitude_pct=50.0),
            "priority_rank": 0,
            "recommended_action": "Review ML warnings and apply recommended feature preprocessing before model training.",
            "source": "evidence",
            "_magnitude": 50.0,
        })
    elif not target_cands and not ml.get("selected_target"):
        insights.append({
            "title": "No Clear Target Variable Detected",
            "category": "ml_readiness",
            "severity": "LOW",
            "evidence": {
                "target_candidates_count": 0,
                "ml_score": ml_score,
            },
            "interpretation": "No high-confidence target variable was automatically detected among dataset columns.",
            "impact": "Supervised learning requires an explicitly designated label attribute.",
            "priority_score": _calculate_priority_score("LOW", magnitude_pct=30.0),
            "priority_rank": 0,
            "recommended_action": "Explicitly specify the desired target attribute if supervised learning is required.",
            "source": "evidence",
            "_magnitude": 30.0,
        })

    return insights


def _detect_health_dimension_findings(
    health: Optional[Dict[str, Any]],
    handled_keys: Set[str],
) -> List[Dict[str, Any]]:
    """
    Detect health dimension alerts when a health dimension is severely compromised
    and not already reflected by specific granular findings.
    """
    insights: List[Dict[str, Any]] = []
    if not health or not isinstance(health, dict):
        return insights

    dims = health.get("dimensions", {})
    if not isinstance(dims, dict):
        return insights

    for dim_name, dim_info in dims.items():
        if not isinstance(dim_info, dict):
            continue

        score = _safe_float(dim_info.get("score", 100.0))
        status = str(dim_info.get("status", "GOOD"))

        # Only create a health dimension insight if POOR and not already covered
        if status == "POOR" and score < 50.0:
            if dim_name in handled_keys:
                continue

            dim_title = dim_name.replace("_", " ").title()
            insights.append({
                "title": f"Poor {dim_title} Health ({score:.1f}/100)",
                "category": "health",
                "severity": "HIGH",
                "evidence": {
                    "dimension": dim_name,
                    "score": score,
                    "status": status,
                    "reason": dim_info.get("reason", ""),
                },
                "interpretation": f"Dataset {dim_title} scored {score:.1f}/100, classified as {status}.",
                "impact": f"Low {dim_title} impairs general reliability and downstream analytical trust.",
                "priority_score": _calculate_priority_score("HIGH", magnitude_pct=100.0 - score),
                "priority_rank": 0,
                "recommended_action": f"Review {dim_title} indicators and address underlying data hygiene issues.",
                "source": "health",
                "_magnitude": round(100.0 - score, 1),
            })

    return insights


# ============================================================
# MAIN PUBLIC API
# ============================================================

def build_prioritized_insights(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    health: Optional[Dict[str, Any]] = None,
    max_insights: int = 10,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a deterministic, prioritized list of dataset findings and actionable insights.

    Accepts either a pre-computed Evidence Layer dict or a raw pandas DataFrame.
    Reuses Phase 1 evidence and Phase 2 health assessments.
    Strictly preserves data honesty without inventing values.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Pre-built Evidence dict from engine.evidence.build_evidence, or raw DataFrame.
    health : Optional[Dict[str, Any]]
        Pre-built Health dict from engine.health.calculate_health. If None and needed,
        it will be generated safely from the resolved evidence.
    max_insights : int
        Maximum number of top-ranked insights to return in the insights list (default: 10).
    dataset_name : Optional[str]
        Optional label for the dataset.

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable dictionary with insights and summary.
    """
    # 1. Resolve Evidence Object (do not mutate caller's inputs)
    if isinstance(evidence_or_df, pd.DataFrame) or evidence_or_df is None:
        evidence = build_evidence(evidence_or_df, dataset_name=dataset_name)
    elif isinstance(evidence_or_df, dict):
        evidence = evidence_or_df
    else:
        evidence = build_evidence(None, dataset_name=dataset_name)

    # 2. Resolve Health Object
    if health is None or not isinstance(health, dict):
        health = calculate_health(evidence, dataset_name=dataset_name)

    resolved_dataset_name = dataset_name or evidence.get("dataset_name") or health.get("dataset_name")

    # 3. Detect Findings Across Categories
    empty_cols_set: Set[str] = set()
    raw_insights: List[Dict[str, Any]] = []

    # Check for empty / single-row / small sample dataset
    early_insights, handled_keys = _detect_empty_or_single_row(evidence)
    raw_insights.extend(early_insights)

    # If completely empty, no other findings apply
    if "empty_dataset" in handled_keys:
        all_insights = raw_insights
    else:
        # Structural anomalies
        raw_insights.extend(_detect_structural_findings(evidence))

        # Missingness & Empty Columns
        raw_insights.extend(_detect_missingness(evidence, empty_cols_set))

        # Duplicates
        raw_insights.extend(_detect_duplicates(evidence))

        # Constant columns (deduplicated against empty columns)
        raw_insights.extend(_detect_constant_columns(evidence, empty_cols_set))

        # Invalid values
        raw_insights.extend(_detect_invalid_values(evidence, empty_cols_set))

        # Outliers
        raw_insights.extend(_detect_outliers(evidence))

        # Strong correlations & Multicollinearity
        raw_insights.extend(_detect_correlations(evidence))

        # ML readiness blockers & warnings
        raw_insights.extend(_detect_ml_readiness(evidence))

        # Health dimension overrides (deduplicated)
        raw_insights.extend(_detect_health_dimension_findings(health, handled_keys))

        all_insights = raw_insights

    # 4. Healthy Dataset Fallback
    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))

    if not all_insights and row_count > 1 and col_count > 0:
        all_insights.append({
            "title": "Dataset Is Structurally Healthy and Complete",
            "category": "health",
            "severity": "INFO",
            "evidence": {
                "row_count": row_count,
                "column_count": col_count,
                "missing_percentage": 0.0,
                "duplicate_rows": 0,
            },
            "interpretation": f"The dataset contains {row_count:,} records across {col_count:,} columns with 0% missing cells and no duplicate records.",
            "impact": "Provides a clean, high-integrity foundation for exploratory analysis, reporting, and statistical modeling.",
            "priority_score": 15.0,
            "priority_rank": 0,
            "recommended_action": "Proceed directly with statistical exploration, feature engineering, or predictive analysis.",
            "source": "evidence",
            "_magnitude": 0.0,
        })

    # 5. Deterministic Deduplication Check
    unique_insights: List[Dict[str, Any]] = []
    seen_signatures: Set[str] = set()

    for ins in all_insights:
        cat = ins.get("category", "")
        ev = ins.get("evidence", {})
        col = ev.get("column", "")
        title = ins.get("title", "")
        sig = f"{cat}::{col}::{title}"

        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_insights.append(ins)

    # 6. Deterministic Sorting & Ranking
    def _sort_key(x: Dict[str, Any]) -> Tuple[float, int, float, str]:
        score = _safe_float(x.get("priority_score", 0.0))
        sev = x.get("severity", "INFO")
        sev_rank = SEVERITY_RANKS.get(sev, 5)
        mag = _safe_float(x.get("_magnitude", 0.0))
        title = str(x.get("title", ""))
        return (-score, sev_rank, -mag, title)

    unique_insights.sort(key=_sort_key)

    # Clean up internal sorting keys and assign sequential 1-based ranks
    for rank_idx, ins in enumerate(unique_insights, start=1):
        ins["priority_rank"] = rank_idx
        if "_magnitude" in ins:
            del ins["_magnitude"]

    # 7. Compute Summary Across All Findings
    total_findings = len(unique_insights)
    crit_cnt = sum(1 for x in unique_insights if x.get("severity") == "CRITICAL")
    high_cnt = sum(1 for x in unique_insights if x.get("severity") == "HIGH")
    med_cnt = sum(1 for x in unique_insights if x.get("severity") == "MEDIUM")
    low_cnt = sum(1 for x in unique_insights if x.get("severity") == "LOW")
    info_cnt = sum(1 for x in unique_insights if x.get("severity") == "INFO")
    top_priority_title = unique_insights[0]["title"] if unique_insights else None

    # Truncate to max_insights if requested
    if max_insights is not None and max_insights > 0:
        returned_insights = unique_insights[:max_insights]
    else:
        returned_insights = unique_insights

    return {
        "insights": returned_insights,
        "summary": {
            "total_findings": total_findings,
            "critical_count": crit_cnt,
            "high_count": high_cnt,
            "medium_count": med_cnt,
            "low_count": low_cnt,
            "info_count": info_cnt,
            "top_priority": top_priority_title,
        },
        "dataset_name": resolved_dataset_name,
        "generated_from": "evidence",
    }
