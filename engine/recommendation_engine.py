"""
FILE: engine/recommendation_engine.py

Next Best Analysis Agent.
Inspects factual dataset evidence from Phase 1 (Evidence Layer), Phase 2 (Health Engine),
and Phase 3 (Insight Priority Engine) to recommend adaptive, prioritized, and actionable
next analytical investigations.

Strictly preserves data honesty:
- 100% deterministic, offline, and JSON-serializable.
- Zero external API / LLM calls.
- Every recommendation is supported by factual, uninvented evidence.
- Adapts ranking and recommendations dynamically based on actual dataset characteristics.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights


PRIORITY_BASE_SCORES = {
    "CRITICAL": 90.0,
    "HIGH": 75.0,
    "MEDIUM": 55.0,
    "LOW": 35.0,
    "INFO": 15.0,
}

PRIORITY_RANKS = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
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


def _calculate_score(
    priority: str,
    magnitude_pct: float = 0.0,
    is_blocking: bool = False,
    scope_factor: float = 0.0,
) -> float:
    """Deterministically compute an adaptive priority score in [0.0, 100.0]."""
    base = PRIORITY_BASE_SCORES.get(priority, 35.0)
    mag_adj = min(6.0, (max(0.0, magnitude_pct) / 100.0) * 6.0)
    blocking_adj = 4.0 if is_blocking else 0.0
    scope_adj = min(3.0, max(0.0, scope_factor))
    return round(max(0.0, min(100.0, base + mag_adj + blocking_adj + scope_adj)), 2)


# ============================================================
# RECOMMENDATION DETECTORS
# ============================================================

def _recommend_missingness(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend missingness pattern investigation if meaningful missing values exist."""
    completeness = evidence.get("completeness", {})
    structure = evidence.get("structure", {})
    quality = evidence.get("quality", {})

    row_count = _safe_int(structure.get("row_count", 0))
    missing_cells = _safe_int(completeness.get("missing_cells", 0))
    missing_pct = _safe_float(completeness.get("missing_percentage", 0.0))
    cols_with_missing = completeness.get("columns_with_missing", [])
    col_missing_counts = completeness.get("column_missing_counts", {})
    col_missing_pcts = completeness.get("column_missing_percentages", {})
    empty_cols = quality.get("empty_columns", [])

    if missing_cells == 0 or row_count == 0:
        return None

    evidence_items: List[str] = [
        f"Overall dataset contains {missing_cells:,} missing cells ({missing_pct:.1f}% of total data) across {len(cols_with_missing)} column(s)."
    ]

    # Add top affected columns
    sorted_cols: List[Tuple[str, float, int]] = []
    for item in cols_with_missing:
        col_name = item.get("column") if isinstance(item, dict) else str(item)
        pct = _safe_float(item.get("missing_percentage")) if isinstance(item, dict) else _safe_float(col_missing_pcts.get(col_name, 0.0))
        cnt = _safe_int(item.get("missing_count")) if isinstance(item, dict) else _safe_int(col_missing_counts.get(col_name, 0))
        sorted_cols.append((col_name, pct, cnt))

    sorted_cols.sort(key=lambda x: x[1], reverse=True)
    for c_name, c_pct, c_cnt in sorted_cols[:3]:
        evidence_items.append(f"Column '{c_name}' has {c_pct:.1f}% missing values ({c_cnt:,} of {row_count:,} rows).")

    if empty_cols:
        evidence_items.append(f"Completely empty column(s) (100% missing): {', '.join(empty_cols)}.")

    if missing_pct >= 40.0 or len(empty_cols) > 0:
        priority = "CRITICAL"
        is_blocking = True
    elif missing_pct >= 20.0:
        priority = "HIGH"
        is_blocking = False
    elif missing_pct >= 5.0:
        priority = "MEDIUM"
        is_blocking = False
    else:
        priority = "LOW"
        is_blocking = False

    score = _calculate_score(priority, magnitude_pct=missing_pct, is_blocking=is_blocking, scope_factor=2.5)

    return {
        "name": "Investigate Missingness Patterns",
        "category": "missingness",
        "why": "The dataset contains meaningful missing values that reduce effective sample size and can distort joint distributions.",
        "evidence": evidence_items,
        "expected_value": "Identify whether missingness is systematic (MNAR/MAR) or completely at random (MCAR) and select appropriate imputation strategies.",
        "priority": priority,
        "action": "Analyze missingness correlations across columns and establish targeted imputation or exclusion rules.",
        "_score": score,
        "_magnitude": missing_pct,
    }


def _recommend_anomalies(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend numerical and categorical anomaly investigation if outliers exist."""
    quality = evidence.get("quality", {})
    structure = evidence.get("structure", {})
    columns = evidence.get("columns", [])

    row_count = _safe_int(structure.get("row_count", 0))
    outlier_findings = quality.get("outlier_findings", [])
    outlier_columns = quality.get("outlier_columns", [])
    total_outliers = _safe_int(quality.get("total_outlier_values", 0))

    # Check for invalid values in columns
    invalid_cols = [
        c for c in columns
        if _safe_int(c.get("invalid_values", {}).get("invalid_count", 0)) > 0
    ]

    if not outlier_columns and not outlier_findings and not invalid_cols:
        return None

    evidence_items: List[str] = []
    max_outlier_pct = 0.0

    if outlier_findings:
        evidence_items.append(
            f"Detected {total_outliers:,} potential outlier values across {len(outlier_columns)} numerical column(s)."
        )
        for f in outlier_findings[:3]:
            attr = f.get("attribute")
            cnt = _safe_int(f.get("count", 0))
            pct = _safe_float(f.get("percentage", 0.0))
            if pct > max_outlier_pct:
                max_outlier_pct = pct
            evidence_items.append(
                f"Column '{attr}' contains {cnt:,} outliers ({pct:.1f}% of observations) outside IQR boundaries."
            )
    elif outlier_columns:
        evidence_items.append(f"Potential outlier values detected in column(s): {', '.join(outlier_columns)}.")

    for c in invalid_cols[:2]:
        inv_cnt = _safe_int(c.get("invalid_values", {}).get("invalid_count", 0))
        evidence_items.append(f"Column '{c.get('name')}' contains {inv_cnt:,} invalid non-numeric string value(s).")

    if max_outlier_pct >= 10.0 or len(invalid_cols) > 0:
        priority = "HIGH"
    elif max_outlier_pct >= 3.0 or len(outlier_columns) > 1:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    score = _calculate_score(priority, magnitude_pct=max_outlier_pct, is_blocking=False, scope_factor=1.5)

    return {
        "name": "Investigate Numerical and Categorical Anomalies",
        "category": "anomaly",
        "why": "Several attributes contain extreme outliers or invalid type entries that may distort statistical estimators and model weights.",
        "evidence": evidence_items,
        "expected_value": "Determine whether extreme values represent genuine real-world heavy tails, measurement errors, or data corruption.",
        "priority": priority,
        "action": "Inspect extreme observations using robust statistical thresholds (IQR/z-scores) and decide whether clipping, transformation, or filtering is warranted.",
        "_score": score,
        "_magnitude": max_outlier_pct,
    }


def _recommend_correlations(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend correlation and collinearity investigation if strong correlations exist."""
    correlations = evidence.get("correlations", {})
    if not correlations.get("supported"):
        return None

    strongest_pairs = correlations.get("strongest_pairs", [])
    multi_pairs = correlations.get("potential_multicollinearity", [])

    # Filter for pairs with |r| >= 0.50
    high_pairs: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for p in (multi_pairs + strongest_pairs):
        a1 = p.get("attribute_1")
        a2 = p.get("attribute_2")
        r = _safe_float(p.get("correlation", 0.0))
        if not a1 or not a2:
            continue
        key = tuple(sorted([str(a1), str(a2)]))
        if key not in seen and abs(r) >= 0.50:
            seen.add(key)
            high_pairs.append({"a1": a1, "a2": a2, "r": r})

    if not high_pairs:
        return None

    high_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    top_r = abs(high_pairs[0]["r"])

    evidence_items: List[str] = []
    for item in high_pairs[:3]:
        r_val = item["r"]
        direction = "positive" if r_val > 0 else "negative"
        evidence_items.append(
            f"'{item['a1']}' and '{item['a2']}' show a strong {direction} statistical association (correlation r = {r_val:.2f})."
        )

    if multi_pairs:
        evidence_items.append(
            f"Multicollinearity alert: {len(multi_pairs)} attribute pair(s) exhibit high correlation (|r| >= 0.85)."
        )

    if top_r >= 0.90:
        priority = "HIGH"
    elif top_r >= 0.70:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    score = _calculate_score(priority, magnitude_pct=top_r * 100 - 40, is_blocking=False, scope_factor=1.5)

    return {
        "name": "Investigate Strong Variable Relationships and Collinearity",
        "category": "correlation",
        "why": "High pairwise correlations indicate strong statistical associations and potential feature redundancy.",
        "evidence": evidence_items,
        "expected_value": "Understand key numerical relationships and identify redundant attributes to avoid variance inflation in downstream models.",
        "priority": priority,
        "action": "Inspect scatter plots and correlation matrices between strongly correlated pairs, and evaluate dimensionality reduction or feature selection.",
        "_score": score,
        "_magnitude": top_r * 100,
    }


def _recommend_category_distribution(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend categorical distribution analysis if categorical variables exist."""
    structure = evidence.get("structure", {})
    cat_stats = evidence.get("categorical_statistics", {})
    row_count = _safe_int(structure.get("row_count", 0))
    cat_cols = list(structure.get("categorical_columns", []))

    if not cat_cols or row_count <= 1:
        return None

    evidence_items: List[str] = [
        f"Dataset contains {len(cat_cols)} categorical attribute(s) across {row_count:,} records."
    ]

    has_dominance = False
    has_high_cardinality = False

    for col, stat in cat_stats.items():
        if isinstance(stat, dict):
            dom_pct = _safe_float(stat.get("dominance_percentage", 0.0))
            top_cat = stat.get("top_category")
            unique_cnt = _safe_int(stat.get("unique_count", 0))

            if dom_pct >= 60.0 and top_cat is not None:
                has_dominance = True
                evidence_items.append(
                    f"Column '{col}' is heavily concentrated in category '{top_cat}' ({dom_pct:.1f}% share)."
                )
            if unique_cnt > 30 and row_count > 50:
                has_high_cardinality = True
                evidence_items.append(
                    f"Column '{col}' exhibits high cardinality with {unique_cnt:,} distinct categories."
                )

    if has_dominance or has_high_cardinality:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    score = _calculate_score(priority, magnitude_pct=50.0 if has_dominance else 20.0, is_blocking=False, scope_factor=1.0)

    return {
        "name": "Analyze Categorical Distributions and Imbalance",
        "category": "category_distribution",
        "why": "Categorical variables exhibit specific frequency patterns, class concentration, or cardinality levels that shape analytical stratification.",
        "evidence": evidence_items,
        "expected_value": "Understand class balance, rare category presence, and appropriate encoding requirements for modeling.",
        "priority": priority,
        "action": "Review category frequency tables and evaluate whether grouping rare levels or rebalancing categories is necessary.",
        "_score": score,
        "_magnitude": 50.0 if has_dominance else 20.0,
    }


def _recommend_target_relationship(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Recommend target relationship analysis ONLY if a valid, reliable target is detected or specified.
    Do NOT recommend if no target exists.
    """
    ml = evidence.get("ml_readiness", {})
    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))

    if row_count <= 1:
        return None

    target_candidate = ml.get("evaluated_target") or ml.get("best_target_candidate")
    target_candidates = ml.get("target_candidates", [])

    if not target_candidate and not target_candidates:
        return None

    target_col = target_candidate or target_candidates[0].get("column")
    if not target_col:
        return None

    task_type = ml.get("task_type", "predictive modeling")

    evidence_items: List[str] = [
        f"Designated target candidate identified: '{target_col}' ({task_type})."
    ]
    if target_candidates:
        cand_info = target_candidates[0]
        conf = cand_info.get("confidence")
        if conf is not None:
            evidence_items.append(f"Target detection confidence is {conf}%.")

    score = _calculate_score("HIGH", magnitude_pct=75.0, is_blocking=False, scope_factor=2.0)

    return {
        "name": f"Analyze Feature Relationships with Target '{target_col}'",
        "category": "target_relationship",
        "why": f"Understanding how input features relate to target variable '{target_col}' is essential for feature selection and predictive modeling.",
        "evidence": evidence_items,
        "expected_value": "Identify the strongest predictors of the target variable and discover potential non-linear relationships or label leakage.",
        "priority": "HIGH",
        "action": f"Compute bivariate statistics and feature importance against target '{target_col}'.",
        "_score": score,
        "_magnitude": 75.0,
    }


def _recommend_ml_pathway(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend ML preparation if blocked, or ML modeling if ready."""
    structure = evidence.get("structure", {})
    ml = evidence.get("ml_readiness", {})

    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))

    if row_count <= 1 or col_count <= 1:
        return None

    can_train = ml.get("can_train", False)
    ml_score = _safe_float(ml.get("ml_score", 0.0))
    ml_status = str(ml.get("ml_status", "NOT READY"))
    warnings = ml.get("warnings", [])

    if not can_train or ml_status in ["NOT READY", "Critical"]:
        # Recommend resolving blockers
        evidence_items: List[str] = [
            f"Current ML readiness score is {ml_score:.0f}/100 ({ml_status}).",
            "Supervised modeling is blocked by data quality, sample size, or target suitability constraints.",
        ]
        for w in warnings[:2]:
            evidence_items.append(f"Blocker: {w}")

        score = _calculate_score("HIGH", magnitude_pct=100.0 - ml_score, is_blocking=True, scope_factor=2.0)

        return {
            "name": "Resolve Machine Learning Readiness Blockers",
            "category": "ml_readiness",
            "why": "Current dataset readiness constraints prevent reliable and unbiased model training.",
            "evidence": evidence_items,
            "expected_value": "Overcome structural and quality blockers so the dataset can support reliable predictive modeling.",
            "priority": "HIGH",
            "action": "Remediate highlighted data hygiene and target suitability blockers before training machine learning models.",
            "_score": score,
            "_magnitude": round(100.0 - ml_score, 1),
        }
    else:
        # Recommend predictive modeling
        evidence_items: List[str] = [
            f"Dataset achieved an ML readiness score of {ml_score:.0f}/100 ({ml_status}).",
            "Sufficient observations and feature attributes exist to support supervised model estimation.",
        ]
        score = _calculate_score("MEDIUM", magnitude_pct=ml_score, is_blocking=False, scope_factor=1.5)

        return {
            "name": "Conduct Baseline Predictive Machine Learning Modeling",
            "category": "machine_learning",
            "why": "The dataset is sufficiently prepared for baseline predictive exploration and model comparison.",
            "evidence": evidence_items,
            "expected_value": "Establish cross-validated baseline performance across candidate models and assess feature predictive power.",
            "priority": "MEDIUM",
            "action": "Train candidate baseline estimators using k-fold cross-validation and evaluate holdout performance metrics.",
            "_score": score,
            "_magnitude": ml_score,
        }


def _recommend_data_quality_structure(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recommend structural and quality investigations for duplicates, constant cols, or header collisions."""
    structure = evidence.get("structure", {})
    duplicates = evidence.get("duplicates", {})
    quality = evidence.get("quality", {})
    columns = evidence.get("columns", [])

    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))
    dup_rows = _safe_int(duplicates.get("duplicate_rows", 0))
    dup_pct = _safe_float(duplicates.get("duplicate_row_percentage", 0.0))
    constant_cols = quality.get("constant_columns", [])
    has_dup_cols = structure.get("has_duplicate_columns", False)
    dup_col_names = structure.get("duplicate_column_names", [])

    # Empty dataset or single row
    if row_count == 0 or col_count == 0:
        return {
            "name": "Provide Populated Tabular Data",
            "category": "data_quality_structure",
            "why": "The dataset contains 0 records or columns.",
            "evidence": [f"Row count: {row_count}, Column count: {col_count}."],
            "expected_value": "Enable statistical and analytical functionality by ingesting non-empty data.",
            "priority": "CRITICAL",
            "action": "Upload or ingest a populated CSV or tabular dataset.",
            "_score": 100.0,
            "_magnitude": 100.0,
        }

    if row_count == 1:
        return {
            "name": "Collect Additional Observational Records",
            "category": "data_quality_structure",
            "why": "The dataset contains only a single record.",
            "evidence": ["Dataset contains exactly 1 row; variance and correlation cannot be evaluated."],
            "expected_value": "Provide sufficient degrees of freedom for statistical and empirical analysis.",
            "priority": "CRITICAL",
            "action": "Gather additional observational rows before conducting quantitative analysis.",
            "_score": 95.0,
            "_magnitude": 100.0,
        }

    evidence_items: List[str] = []
    is_critical = False
    is_high = False

    if has_dup_cols:
        is_critical = True
        evidence_items.append(f"Duplicate column header names detected: {', '.join(dup_col_names)}.")

    if dup_rows > 0:
        if dup_pct >= 25.0:
            is_critical = True
        elif dup_pct >= 10.0:
            is_high = True
        evidence_items.append(
            f"Dataset contains {dup_rows:,} duplicate rows ({dup_pct:.1f}% of all {row_count:,} records)."
        )

    if constant_cols:
        is_high = True
        evidence_items.append(
            f"Constant column(s) with zero variance: {', '.join(constant_cols)}."
        )

    if row_count < 10:
        is_high = True
        evidence_items.append(f"Sample size is extremely small ({row_count} rows, recommended minimum is 30).")

    if not evidence_items:
        return None

    priority = "CRITICAL" if is_critical else ("HIGH" if is_high else "MEDIUM")
    score = _calculate_score(priority, magnitude_pct=dup_pct, is_blocking=is_critical, scope_factor=2.0)

    return {
        "name": "Audit and Remediate Structural Quality Issues",
        "category": "data_quality_structure",
        "why": "Structural defects such as duplicate records, constant columns, or duplicate headers corrupt analytical integrity.",
        "evidence": evidence_items,
        "expected_value": "Ensure dataset integrity, prevent indexing collisions, and remove non-informative features.",
        "priority": priority,
        "action": "Deduplicate identical records, rename duplicate column headers, and remove zero-variance attributes.",
        "_score": score,
        "_magnitude": dup_pct,
    }


# ============================================================
# MAIN PUBLIC API
# ============================================================

def recommend_next_analysis(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    health: Optional[Dict[str, Any]] = None,
    prioritized_insights: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    max_recommendations: int = 5,
) -> Dict[str, Any]:
    """
    Recommend prioritized, evidence-driven next analytical investigations for a dataset.

    Reuses Phase 1 (Evidence), Phase 2 (Health), and Phase 3 (Priority Insights)
    without duplicating raw metric calculations or profiling operations.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Pre-built Evidence dict from engine.evidence.build_evidence, or raw DataFrame.
    health : Optional[Dict[str, Any]]
        Pre-built Health dict from engine.health.calculate_health. If None, built safely.
    prioritized_insights : Optional[Dict[str, Any]]
        Pre-built Priority dict from engine.priority_engine.build_prioritized_insights. If None, built safely.
    dataset_name : Optional[str]
        Optional label for the dataset.
    max_recommendations : int
        Maximum number of recommendations to return (default: 5).

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-safe dictionary containing ranked recommendations and summary.
    """
    # 1. Resolve Evidence Object
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
            max_insights=max_recommendations,
            dataset_name=dataset_name,
        )

    name = (
        dataset_name
        or evidence.get("metadata", {}).get("dataset_name")
        or evidence.get("dataset_name")
        or "Dataset"
    )

    # 4. Generate Candidate Recommendations
    candidates: List[Dict[str, Any]] = []

    # Check structural issues (empty, single-row, duplicates, constant cols)
    struct_rec = _recommend_data_quality_structure(evidence)
    if struct_rec:
        candidates.append(struct_rec)

    # Missingness
    miss_rec = _recommend_missingness(evidence)
    if miss_rec:
        candidates.append(miss_rec)

    # Anomalies
    anom_rec = _recommend_anomalies(evidence)
    if anom_rec:
        candidates.append(anom_rec)

    # Correlations (only if |r| >= 0.50 exists)
    corr_rec = _recommend_correlations(evidence)
    if corr_rec:
        candidates.append(corr_rec)

    # Categorical distributions
    cat_rec = _recommend_category_distribution(evidence)
    if cat_rec:
        candidates.append(cat_rec)

    # Target relationship (only if target exists)
    target_rec = _recommend_target_relationship(evidence)
    if target_rec:
        candidates.append(target_rec)

    # ML pathway
    ml_rec = _recommend_ml_pathway(evidence)
    if ml_rec:
        candidates.append(ml_rec)

    # Clean / Healthy Dataset Fallback
    structure = evidence.get("structure", {})
    row_count = _safe_int(structure.get("row_count", 0))
    col_count = _safe_int(structure.get("column_count", 0))

    if not candidates and row_count > 1 and col_count > 0:
        candidates.append({
            "name": "Perform Exploratory Data Analysis and Feature Engineering",
            "category": "exploratory_analysis",
            "why": "The dataset is structurally clean with complete records and no severe quality issues detected.",
            "evidence": [
                f"Dataset contains {row_count:,} complete records across {col_count:,} columns with 0% missing cells.",
                "Zero duplicate rows and no severe quality penalties detected.",
            ],
            "expected_value": "Uncover descriptive feature interactions, assess distribution shapes, and formulate analytical hypotheses.",
            "priority": "INFO",
            "action": "Proceed with bivariate visualizations, domain feature transformations, and subgroup comparisons.",
            "_score": 20.0,
            "_magnitude": 0.0,
        })

    # 5. Adaptive Deterministic Sorting & Ranking
    # Primary: _score DESC
    # Secondary: Priority rank ASC (CRITICAL < HIGH < MEDIUM < LOW < INFO)
    # Tertiary: Magnitude DESC
    # Quaternary: Name ASC (deterministic tie-breaker)
    def _sort_key(x: Dict[str, Any]) -> Tuple[float, int, float, str]:
        score = _safe_float(x.get("_score", 0.0))
        prio = x.get("priority", "INFO")
        prio_rank = PRIORITY_RANKS.get(prio, 5)
        mag = _safe_float(x.get("_magnitude", 0.0))
        title = str(x.get("name", ""))
        return (-score, prio_rank, -mag, title)

    candidates.sort(key=_sort_key)

    # Clean internal score keys and assign sequential 1-based ranks
    clean_recommendations: List[Dict[str, Any]] = []
    for rank_idx, rec in enumerate(candidates, start=1):
        clean_rec = {
            "rank": rank_idx,
            "name": rec.get("name", ""),
            "category": rec.get("category", ""),
            "why": rec.get("why", ""),
            "evidence": rec.get("evidence", []),
            "expected_value": rec.get("expected_value", ""),
            "priority": rec.get("priority", "MEDIUM"),
            "action": rec.get("action", ""),
        }
        clean_recommendations.append(clean_rec)

    # 6. Truncate to max_recommendations
    if max_recommendations is not None and max_recommendations > 0:
        final_recommendations = clean_recommendations[:max_recommendations]
    else:
        final_recommendations = clean_recommendations

    # 7. Summary
    rec_count = len(clean_recommendations)
    top_rec = final_recommendations[0]["name"] if final_recommendations else None
    distinct_categories = sorted(list({r["category"] for r in clean_recommendations}))

    return {
        "dataset_name": name,
        "recommendations": final_recommendations,
        "summary": {
            "recommendation_count": rec_count,
            "top_recommendation": top_rec,
            "categories": distinct_categories,
        },
        "generated_from": "evidence",
    }
