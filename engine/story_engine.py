"""
FILE: engine/story_engine.py

Data Story Mode Backend Engine.
Transforms factual evidence from Phase 1 (Evidence Layer), Phase 2 (Health Engine),
and Phase 3 (Insight Priority Engine) into a coherent, structured, and actionable narrative.

Strictly preserves data honesty:
- 100% deterministic, offline, and JSON-serializable.
- Zero external API / LLM calls.
- Every claim is grounded directly in computed evidence, health, and priority metrics.
- No invented domain descriptions, statistics, or causal assertions.
"""

import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights


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


# ============================================================
# STORY BUILDER HELPERS
# ============================================================

def _build_overview(
    name: str,
    structure: Dict[str, Any],
    completeness: Dict[str, Any],
    duplicates: Dict[str, Any],
    ml_readiness: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build dataset overview dictionary and key characteristics."""
    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))
    num_cols = list(structure.get("numerical_columns", []))
    cat_cols = list(structure.get("categorical_columns", []))
    dt_cols = list(structure.get("datetime_columns", []))

    missing_cells = _safe_int(completeness.get("missing_cells", 0))
    missing_pct = _safe_float(completeness.get("missing_percentage", 0.0))
    dup_rows = _safe_int(duplicates.get("duplicate_rows", 0))
    dup_pct = _safe_float(duplicates.get("duplicate_row_percentage", 0.0))
    ml_score = _safe_float(ml_readiness.get("ml_score", 0.0))
    ml_status = str(ml_readiness.get("ml_status", "NOT READY"))

    if row_count == 0 or col_count == 0:
        summary = f"Dataset '{name}' contains 0 usable records or columns and is completely empty."
        key_chars = [
            "Dataset is empty with no records or columns.",
            "No analytical, statistical, or predictive operations can be conducted.",
        ]
        return {"summary": summary, "key_characteristics": key_chars}, key_chars

    if row_count == 1:
        summary = f"Dataset '{name}' contains exactly 1 row across {col_count:,} column(s), representing an isolated record."
        key_chars = [
            f"Dimensions: 1 row × {col_count:,} columns.",
            "Single observation prevents variance, correlation, and predictive evaluation.",
        ]
        return {"summary": summary, "key_characteristics": key_chars}, key_chars

    type_parts = []
    if num_cols:
        type_parts.append(f"{len(num_cols)} numerical")
    if cat_cols:
        type_parts.append(f"{len(cat_cols)} categorical")
    if dt_cols:
        type_parts.append(f"{len(dt_cols)} datetime")
    type_str = f" ({', '.join(type_parts)})" if type_parts else ""

    summary = f"Dataset '{name}' contains {row_count:,} rows and {col_count:,} columns{type_str}."

    key_chars = [
        f"Dimensions: {row_count:,} rows × {col_count:,} columns.",
    ]
    if num_cols or cat_cols or dt_cols:
        key_chars.append(
            f"Attributes: {len(num_cols)} numerical, {len(cat_cols)} categorical, {len(dt_cols)} datetime."
        )

    if missing_cells == 0:
        key_chars.append("Completeness: 100% complete records (0 missing values).")
    else:
        key_chars.append(
            f"Completeness: {missing_cells:,} missing cells ({missing_pct:.1f}% of total data)."
        )

    if dup_rows == 0:
        key_chars.append("Duplicates: 0 duplicate rows (all records are unique).")
    else:
        key_chars.append(
            f"Duplicates: {dup_rows:,} duplicate rows ({dup_pct:.1f}% of records)."
        )

    if row_count > 1 and "ml_score" in ml_readiness:
        key_chars.append(
            f"ML Readiness: Scored {ml_score:.0f}/100 ({ml_status})."
        )

    return {"summary": summary, "key_characteristics": key_chars}, key_chars


def _build_health_summary(
    health: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract deterministic health summary, dimensional strengths/weaknesses, and risks."""
    overall = health.get("overall", {})
    health_score = _safe_float(overall.get("score", 0.0))
    health_status = str(overall.get("status", "NOT SUITABLE"))
    health_summary = str(overall.get("summary", ""))

    dims = health.get("dimensions", {})
    dim_list: List[Dict[str, Any]] = []

    for d_name, d_info in dims.items():
        if isinstance(d_info, dict):
            dim_list.append({
                "name": d_name,
                "score": _safe_float(d_info.get("score", 0.0)),
                "status": str(d_info.get("status", "POOR")),
                "reason": str(d_info.get("reason", "")),
            })

    # Strongest dimensions: score descending, then name ascending
    strongest = sorted(dim_list, key=lambda x: (-x["score"], x["name"]))
    # Weakest dimensions: score ascending, then name ascending
    weakest = sorted(dim_list, key=lambda x: (x["score"], x["name"]))

    health_risks = list(health.get("risks", []))

    return {
        "score": health_score,
        "status": health_status,
        "summary": health_summary,
        "strongest_dimensions": strongest[:3] if dim_list else [],
        "weakest_dimensions": weakest[:3] if dim_list else [],
        "health_risks": health_risks,
    }


def _extract_patterns(
    evidence: Dict[str, Any],
) -> List[str]:
    """Extract factual patterns from Evidence."""
    patterns: List[str] = []

    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))
    if row_count <= 1:
        return ["Insufficient observations to identify empirical or statistical patterns."]

    correlations = evidence.get("correlations", {})
    strongest_pairs = correlations.get("strongest_pairs", [])
    cat_stats = evidence.get("categorical_statistics", {})
    num_stats = evidence.get("numerical_statistics", {})
    completeness = evidence.get("completeness", {})

    # 1. Correlation Patterns
    for pair in strongest_pairs[:3]:
        r = _safe_float(pair.get("correlation", 0.0))
        a1 = pair.get("attribute_1")
        a2 = pair.get("attribute_2")
        if abs(r) >= 0.50 and a1 and a2:
            direction = "positive" if r > 0 else "negative"
            patterns.append(
                f"Attributes '{a1}' and '{a2}' exhibit a strong {direction} linear correlation of {r:.2f}."
            )

    # 2. Dominant Category Patterns
    for col, stat in cat_stats.items():
        if isinstance(stat, dict):
            dom_pct = _safe_float(stat.get("dominance_percentage", 0.0))
            top_cat = stat.get("top_category")
            if dom_pct >= 50.0 and top_cat is not None:
                patterns.append(
                    f"Column '{col}' is heavily dominated by category '{top_cat}' ({dom_pct:.1f}% of observations)."
                )

    # 3. Distribution Skewness Patterns
    for col, stat in num_stats.items():
        if isinstance(stat, dict):
            skew = stat.get("skewness")
            if skew is not None:
                skew_val = _safe_float(skew, 0.0)
                if abs(skew_val) >= 1.5:
                    skew_type = "right-skewed (positive tail)" if skew_val > 0 else "left-skewed (negative tail)"
                    patterns.append(
                        f"Column '{col}' displays a {skew_type} distribution (skewness = {skew_val:.2f})."
                    )

    # 4. Missingness Concentration Patterns
    missing_pct = _safe_float(completeness.get("missing_percentage", 0.0))
    cols_with_missing = completeness.get("columns_with_missing", [])
    col_missing_pcts = completeness.get("column_missing_percentages", {})
    if missing_pct > 0.0 and cols_with_missing:
        # Find highest missing column
        top_miss_col = None
        top_miss_pct = 0.0
        for item in cols_with_missing:
            col_name = item.get("column") if isinstance(item, dict) else item
            pct = _safe_float(item.get("missing_percentage")) if isinstance(item, dict) else _safe_float(col_missing_pcts.get(col_name, 0.0))
            if pct > top_miss_pct:
                top_miss_pct = pct
                top_miss_col = col_name

        if top_miss_col and top_miss_pct > 15.0:
            patterns.append(
                f"Missingness is concentrated in {len(cols_with_missing)} column(s), led by '{top_miss_col}' ({top_miss_pct:.1f}% missing)."
            )

    if not patterns:
        patterns.append(
            "No strong linear correlations or dominant categorical distributions were identified in the available evidence."
        )

    return patterns


def _extract_anomalies(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract factual anomalies and outliers from Evidence/Quality."""
    anomalies: List[Dict[str, Any]] = []

    quality = evidence.get("quality", {})
    outlier_findings = quality.get("outlier_findings", [])
    outlier_columns = quality.get("outlier_columns", [])
    columns = evidence.get("columns", [])

    seen_cols: Set[str] = set()

    # 1. Outlier Findings
    for f in outlier_findings:
        col = f.get("attribute")
        if col:
            seen_cols.add(col)
            anomalies.append({
                "column": col,
                "outlier_count": _safe_int(f.get("count", 0)),
                "outlier_percentage": _safe_float(f.get("percentage", 0.0)),
                "method": "IQR",
                "lower_bound": f.get("lower_bound"),
                "upper_bound": f.get("upper_bound"),
            })

    # If findings list was empty but outlier_columns exists
    for col in outlier_columns:
        if col not in seen_cols:
            seen_cols.add(col)
            anomalies.append({
                "column": col,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "method": "IQR",
                "lower_bound": None,
                "upper_bound": None,
            })

    # 2. Invalid / Non-Numeric Values in Columns
    for c in columns:
        col = c.get("name")
        inv = c.get("invalid_values", {})
        inv_cnt = _safe_int(inv.get("invalid_count", inv.get("non_numeric_count", 0)))
        if inv_cnt > 0 and col:
            anomalies.append({
                "column": col,
                "invalid_count": inv_cnt,
                "invalid_percentage": _safe_float(inv.get("invalid_percentage", 0.0)),
                "method": "Type Coercion",
                "examples": inv.get("invalid_examples", inv.get("sample_invalid", [])),
            })

    # Sort deterministically by column name
    anomalies.sort(key=lambda x: str(x.get("column", "")))
    return anomalies


def _extract_analytical_risks(
    evidence: Dict[str, Any],
    health: Dict[str, Any],
) -> List[str]:
    """Translate actual evidence and health findings into factual analytical risks."""
    risks: List[str] = []
    seen: Set[str] = set()

    def _add_risk(r: str):
        if r and r not in seen:
            seen.add(r)
            risks.append(r)

    structure = evidence.get("structure", {})
    completeness = evidence.get("completeness", {})
    duplicates = evidence.get("duplicates", {})
    quality = evidence.get("quality", {})
    correlations = evidence.get("correlations", {})
    ml = evidence.get("ml_readiness", {})

    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))
    missing_cells = _safe_int(completeness.get("missing_cells", 0))
    missing_pct = _safe_float(completeness.get("missing_percentage", 0.0))
    dup_rows = _safe_int(duplicates.get("duplicate_rows", 0))
    dup_pct = _safe_float(duplicates.get("duplicate_row_percentage", 0.0))
    outlier_cols = quality.get("outlier_columns", [])
    empty_cols = quality.get("empty_columns", [])
    const_cols = quality.get("constant_columns", [])
    has_dup_cols = structure.get("has_duplicate_columns", False)
    dup_col_names = structure.get("duplicate_column_names", [])

    can_train = ml.get("can_train", False)
    ml_status = ml.get("ml_status", "")
    ml_score = _safe_float(ml.get("ml_score", 0.0))

    # Empty / Single-row dataset
    if row_count == 0 or col_count == 0:
        _add_risk("Empty dataset prevents any statistical estimation, visualization, or predictive modeling.")
        return risks
    if row_count == 1:
        _add_risk("Single observation precludes variance estimation, hypothesis testing, and machine learning.")
        return risks

    # Duplicate column headers
    if has_dup_cols:
        _add_risk(
            f"Duplicate column names ({', '.join(dup_col_names)}) introduce attribute ambiguity and pipeline overwrite errors."
        )

    # Empty columns
    if empty_cols:
        _add_risk(
            f"Empty column(s) ({', '.join(empty_cols)}) contain 0% usable values and provide zero predictive variance."
        )

    # Missingness
    if missing_cells > 0:
        _add_risk(
            f"Missing values ({missing_pct:.1f}% across {len(completeness.get('columns_with_missing', []))} column(s)) reduce complete-case sample sizes and may introduce imputation bias."
        )

    # Duplicates
    if dup_rows > 0:
        _add_risk(
            f"Duplicate records ({dup_rows:,} rows, {dup_pct:.1f}%) artificially inflate sample weights and risk train-test data leakage."
        )

    # Small sample size
    if row_count < 10:
        _add_risk(
            f"Very small sample size ({row_count} rows) induces extreme sampling variance and high risk of statistical distortion."
        )
    elif row_count < 30:
        _add_risk(
            f"Small sample size ({row_count} rows) limits statistical power and increases estimation error in predictive algorithms."
        )

    # Outliers
    if outlier_cols:
        _add_risk(
            f"Statistical outliers detected in {len(outlier_cols)} column(s) can disproportionately bias parametric estimators and distance metrics."
        )

    # Constant columns
    valid_const = [c for c in const_cols if c not in empty_cols]
    if valid_const:
        _add_risk(
            f"Constant column(s) ({', '.join(valid_const)}) carry zero information and may cause singular matrix issues in linear models."
        )

    # Multicollinearity
    multi_pairs = correlations.get("potential_multicollinearity", [])
    if multi_pairs:
        p = multi_pairs[0]
        _add_risk(
            f"High correlation between '{p.get('attribute_1')}' and '{p.get('attribute_2')}' (r = {_safe_float(p.get('correlation', 0.0)):.2f}) indicates potential multicollinearity."
        )

    # ML readiness
    if row_count > 1 and (not can_train or ml_status in ["NOT READY", "Critical"]):
        _add_risk(
            f"Current data readiness constraints ({ml_status}, score {ml_score:.0f}/100) impede reliable automated model training."
        )

    # Incorporate health risks
    for hr in health.get("risks", []):
        _add_risk(str(hr))

    return risks


def _extract_recommended_next_steps(
    important_findings: List[Dict[str, Any]],
    health: Dict[str, Any],
) -> List[str]:
    """Derive prioritized next steps from findings and health recommendations."""
    steps: List[str] = []
    seen: Set[str] = set()

    def _add_step(s: str):
        if s and s not in seen:
            seen.add(s)
            steps.append(s)

    # 1. From top prioritized findings
    for finding in important_findings:
        action = finding.get("recommended_action")
        if action:
            _add_step(action)

    # 2. From health recommended actions
    for action in health.get("recommended_actions", []):
        _add_step(str(action))

    if not steps:
        steps.append("Proceed directly with exploratory data analysis, feature engineering, or modeling.")

    return steps


# ============================================================
# MAIN PUBLIC API
# ============================================================

def build_data_story(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    health: Optional[Dict[str, Any]] = None,
    prioritized_insights: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    max_insights: int = 5,
) -> Dict[str, Any]:
    """
    Construct a coherent, structured, and factual Data Story about the dataset.

    Reuses Phase 1 (Evidence Layer), Phase 2 (Health Engine), and Phase 3
    (Insight Priority Engine) without duplicate calculations.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Pre-built Evidence dict from engine.evidence.build_evidence, or raw DataFrame.
    health : Optional[Dict[str, Any]]
        Pre-built Health dict from engine.health.calculate_health. If None, generated safely.
    prioritized_insights : Optional[Dict[str, Any]]
        Pre-built Priority dict from engine.priority_engine.build_prioritized_insights. If None, generated safely.
    dataset_name : Optional[str]
        Optional label for the dataset.
    max_insights : int
        Maximum number of top prioritized findings to include (default: 5).

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable dictionary containing the complete Data Story.
    """
    # 1. Resolve Evidence Object (do not mutate caller's inputs)
    if isinstance(evidence_or_df, pd.DataFrame) or evidence_or_df is None:
        evidence = build_evidence(evidence_or_df, dataset_name=dataset_name)
    elif isinstance(evidence_or_df, dict):
        evidence = evidence_or_df
    else:
        evidence = build_evidence(None, dataset_name=dataset_name)

    # 2. Resolve Health Object (reuse caller's if provided)
    if health is None or not isinstance(health, dict):
        health = calculate_health(evidence, dataset_name=dataset_name)

    # 3. Resolve Prioritized Insights (reuse caller's if provided)
    if prioritized_insights is None or not isinstance(prioritized_insights, dict):
        prioritized_insights = build_prioritized_insights(
            evidence,
            health=health,
            max_insights=max_insights,
            dataset_name=dataset_name,
        )

    # 4. Resolve Dataset Identity & Attributes
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
    dt_cols = list(structure.get("datetime_columns", []))

    dataset_dict = {
        "name": name,
        "rows": row_count,
        "columns": col_count,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": dt_cols,
    }

    # 5. Section 1: Overview
    overview_dict, key_characteristics = _build_overview(
        name=name,
        structure=structure,
        completeness=evidence.get("completeness", {}),
        duplicates=evidence.get("duplicates", {}),
        ml_readiness=evidence.get("ml_readiness", {}),
    )

    # 6. Section 2 & 3: Health Summary & Top Insights
    health_dict = _build_health_summary(health)

    raw_insights = prioritized_insights.get("insights", [])
    important_findings: List[Dict[str, Any]] = []
    for ins in raw_insights[:max_insights]:
        important_findings.append({
            "title": ins.get("title", ""),
            "category": ins.get("category", ""),
            "severity": ins.get("severity", ""),
            "evidence": copy.deepcopy(ins.get("evidence", {})),
            "interpretation": ins.get("interpretation", ""),
            "impact": ins.get("impact", ""),
            "priority_score": _safe_float(ins.get("priority_score", 0.0)),
            "priority_rank": _safe_int(ins.get("priority_rank", 0)),
            "recommended_action": ins.get("recommended_action", ""),
        })

    # 7. Section 4: Patterns
    patterns = _extract_patterns(evidence)

    # 8. Section 5: Anomalies
    anomalies = _extract_anomalies(evidence)

    # 9. Section 6: Analytical Risks
    analytical_risks = _extract_analytical_risks(evidence, health)

    # 10. Section 7: Recommended Next Steps
    recommended_next_steps = _extract_recommended_next_steps(important_findings, health)

    # 11. Section 8: Story Sections Flow
    story_sections = [
        {
            "title": "Dataset Overview",
            "summary": overview_dict["summary"],
            "items": key_characteristics,
        },
        {
            "title": "Health Check",
            "summary": health_dict["summary"],
            "items": [
                f"Overall Score: {health_dict['score']:.1f}/100 ({health_dict['status']}).",
                f"Strongest Dimension: {health_dict['strongest_dimensions'][0]['name']} ({health_dict['strongest_dimensions'][0]['score']:.1f}/100)" if health_dict['strongest_dimensions'] else "No dimensions evaluated.",
                f"Weakest Dimension: {health_dict['weakest_dimensions'][0]['name']} ({health_dict['weakest_dimensions'][0]['score']:.1f}/100)" if health_dict['weakest_dimensions'] else "No dimensions evaluated.",
            ],
        },
        {
            "title": "What Matters Most",
            "summary": f"{len(important_findings)} key finding(s) prioritized for immediate review.",
            "items": [f.get("title", "") for f in important_findings],
        },
        {
            "title": "Patterns",
            "summary": f"{len(patterns)} empirical pattern(s) identified.",
            "items": patterns,
        },
        {
            "title": "Anomalies",
            "summary": f"{len(anomalies)} anomaly/outlier group(s) observed.",
            "items": [
                f"Column '{a.get('column')}': {a.get('outlier_count', a.get('invalid_count', 0))} {a.get('method', 'anomaly')} finding(s)"
                for a in anomalies
            ] if anomalies else ["No severe statistical anomalies or invalid data types detected."],
        },
        {
            "title": "Analytical Risks",
            "summary": f"{len(analytical_risks)} analytical risk(s) flagged for exploration and modeling.",
            "items": analytical_risks,
        },
        {
            "title": "Recommended Next Steps",
            "summary": f"{len(recommended_next_steps)} prioritized remediation step(s).",
            "items": recommended_next_steps,
        },
    ]

    return {
        "dataset": dataset_dict,
        "overview": overview_dict,
        "health": health_dict,
        "important_findings": important_findings,
        "patterns": patterns,
        "anomalies": anomalies,
        "analytical_risks": analytical_risks,
        "recommended_next_steps": recommended_next_steps,
        "story_sections": story_sections,
        "generated_from": "evidence",
    }
