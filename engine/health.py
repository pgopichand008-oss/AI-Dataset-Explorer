"""
engine/health.py

Data Health / Readiness Score Engine for AI-Dataset-Explorer.

Responsibilities:
- Converts factual Evidence Layer output into a structured, deterministic health and readiness assessment.
- Evaluates six objective dimensions:
  1. Structural Health
  2. Completeness
  3. Validity
  4. Consistency
  5. Statistical Health
  6. Machine Learning Readiness
- Calculates an exact weighted overall score and classifies into:
  - READY
  - NEEDS ATTENTION
  - NOT SUITABLE
- Formulates factual strengths, weaknesses, risks, and recommended actions.
- Guarantees 100% JSON-serializable output with zero hallucinations, zero model training,
  and zero mutation of input data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from engine.evidence import build_evidence


# ============================================================
# DIMENSION 1: STRUCTURAL HEALTH
# ============================================================

def _assess_structural_health(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate structural health based on row/column existence, duplicate column
    names, feature type availability, and empty/constant column counts.
    """
    structure = evidence.get("structure", {})
    quality = evidence.get("quality", {})

    row_count = int(structure.get("row_count", 0))
    column_count = int(structure.get("column_count", 0))
    has_dup_cols = bool(structure.get("has_duplicate_columns", False))
    dup_col_names = structure.get("duplicate_column_names", [])
    num_cols = structure.get("numerical_columns", [])
    cat_cols = structure.get("categorical_columns", [])

    empty_cols = quality.get("empty_columns", [])
    constant_cols = quality.get("constant_columns", [])
    # Distinct constant columns that are not completely empty
    genuine_constant = [c for c in constant_cols if c not in empty_cols]

    if row_count == 0 or column_count == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "Dataset contains no usable records or columns.",
        }

    base = 100.0
    penalties: List[str] = []

    if row_count == 1:
        base -= 30.0
        penalties.append("single record (1 row)")
    elif row_count < 10:
        base -= 15.0
        penalties.append(f"very small sample size ({row_count} rows)")
    elif row_count < 30:
        base -= 5.0
        penalties.append(f"small sample size ({row_count} rows)")

    if has_dup_cols:
        base -= 25.0
        penalties.append(f"duplicate column names ({', '.join(dup_col_names)})")

    if empty_cols:
        penalty = min(len(empty_cols) * 15.0, 30.0)
        base -= penalty
        penalties.append(f"{len(empty_cols)} completely empty column(s)")

    if genuine_constant:
        penalty = min(len(genuine_constant) * 10.0, 20.0)
        base -= penalty
        penalties.append(f"{len(genuine_constant)} constant column(s)")

    if not num_cols and not cat_cols:
        base -= 40.0
        penalties.append("no identifiable numerical or categorical feature columns")

    score = max(0.0, min(100.0, round(base, 2)))
    status = "GOOD" if score >= 80.0 else ("WARNING" if score >= 50.0 else "POOR")

    if not penalties:
        reason = f"Solid tabular structure with {row_count} rows and {column_count} distinct columns."
    else:
        reason = f"Structural health affected by {', '.join(penalties)}."

    return {
        "score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# DIMENSION 2: COMPLETENESS
# ============================================================

def _assess_completeness(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate dataset completeness based strictly on factual missing counts and percentages.

    Scale:
      0% missing       -> 100
      >0% to <=5%      -> 90
      >5% to <=10%     -> 80
      >10% to <=20%    -> 65
      >20% to <=40%    -> 45
      >40% to <=60%    -> 25
      >60%             -> 10
    """
    completeness = evidence.get("completeness", {})
    structure = evidence.get("structure", {})

    total_cells = int(completeness.get("total_cells", 0))
    missing_cells = int(completeness.get("missing_cells", 0))
    missing_pct = float(completeness.get("missing_percentage", 0.0))
    cols_with_missing = completeness.get("columns_with_missing", [])
    col_missing_pcts = completeness.get("column_missing_percentages", {})

    if total_cells == 0 or structure.get("row_count", 0) == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "Completeness cannot be assessed because the dataset contains zero cells.",
        }

    if missing_pct == 0.0:
        score = 100.0
        reason = "Dataset is 100% complete with 0 missing cells."
    elif missing_pct <= 5.0:
        score = 90.0
        reason = f"High completeness with only {missing_pct:.2f}% missing values across {len(cols_with_missing)} column(s)."
    elif missing_pct <= 10.0:
        score = 80.0
        reason = f"Good completeness with {missing_pct:.2f}% missing values across {len(cols_with_missing)} column(s)."
    elif missing_pct <= 20.0:
        score = 65.0
        reason = f"Moderate missingness of {missing_pct:.2f}% across {len(cols_with_missing)} column(s)."
    elif missing_pct <= 40.0:
        score = 45.0
        reason = f"Substantial missingness of {missing_pct:.2f}% requires cleaning or imputation."
    elif missing_pct <= 60.0:
        score = 25.0
        reason = f"High missingness of {missing_pct:.2f}% impairs analytical reliability."
    else:
        score = 10.0
        reason = f"Severe missingness of {missing_pct:.2f}% across the dataset."

    # Cap score at 80 if any column has 100% missing values
    has_all_missing_col = any(pct == 100.0 for pct in col_missing_pcts.values())
    if has_all_missing_col and score > 80.0:
        score = 80.0
        reason += " (Score capped at 80 due to completely empty column)."

    status = "GOOD" if score >= 80.0 else ("WARNING" if score >= 50.0 else "POOR")

    return {
        "score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# DIMENSION 3: VALIDITY
# ============================================================

def _assess_validity(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate dataset validity based on invalid/mixed non-numeric values,
    all-missing columns, constant columns, and quality findings.
    Statistical outliers are not automatically penalized as invalid.
    """
    structure = evidence.get("structure", {})
    quality = evidence.get("quality", {})
    columns = evidence.get("columns", [])

    row_count = int(structure.get("row_count", 0))
    column_count = int(structure.get("column_count", 0))

    if row_count == 0 or column_count == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "Validity cannot be evaluated on an empty dataset.",
        }

    base = 100.0
    penalties: List[str] = []

    # Check for invalid values in numeric-expected columns
    invalid_cols = [c for c in columns if c.get("invalid_values", {}).get("invalid_count", 0) > 0]
    if invalid_cols:
        total_inv = sum(c["invalid_values"]["invalid_count"] for c in invalid_cols)
        penalty = min(len(invalid_cols) * 15.0, 35.0)
        base -= penalty
        penalties.append(f"{total_inv} invalid non-numeric value(s) in {len(invalid_cols)} column(s)")

    empty_cols = quality.get("empty_columns", [])
    if empty_cols:
        penalty = min(len(empty_cols) * 15.0, 30.0)
        base -= penalty
        penalties.append(f"{len(empty_cols)} completely empty column(s)")

    constant_cols = quality.get("constant_columns", [])
    genuine_const = [c for c in constant_cols if c not in empty_cols]
    if genuine_const:
        penalty = min(len(genuine_const) * 10.0, 20.0)
        base -= penalty
        penalties.append(f"{len(genuine_const)} constant column(s)")

    # Outliers: mild observation penalty only if outlier columns exist (never treat as invalid data)
    outlier_cols = quality.get("outlier_columns", [])
    if outlier_cols:
        penalty = min(len(outlier_cols) * 2.0, 10.0)
        base -= penalty
        penalties.append(f"{len(outlier_cols)} column(s) with statistical outliers")

    score = max(0.0, min(100.0, round(base, 2)))
    status = "GOOD" if score >= 80.0 else ("WARNING" if score >= 50.0 else "POOR")

    if not penalties:
        reason = "All column values conform to expected data types and structural validity rules."
    else:
        reason = f"Validity affected by: {'; '.join(penalties)}."

    return {
        "score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# DIMENSION 4: CONSISTENCY
# ============================================================

def _assess_consistency(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate dataset consistency using duplicate rows, duplicate column names,
    type clashes, and invariant data.
    """
    structure = evidence.get("structure", {})
    duplicates = evidence.get("duplicates", {})
    quality = evidence.get("quality", {})
    columns = evidence.get("columns", [])

    row_count = int(structure.get("row_count", 0))
    column_count = int(structure.get("column_count", 0))

    if row_count == 0 or column_count == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "Consistency cannot be evaluated on an empty dataset.",
        }

    base = 100.0
    penalties: List[str] = []

    dup_rows = int(duplicates.get("duplicate_rows", 0))
    dup_pct = float(duplicates.get("duplicate_row_percentage", 0.0))

    if dup_rows > 0:
        if dup_pct > 30.0:
            base -= 35.0
            penalties.append(f"high duplicate row rate ({dup_rows} rows, {dup_pct:.2f}%)")
        elif dup_pct > 10.0:
            base -= 20.0
            penalties.append(f"moderate duplicate rows ({dup_rows} rows, {dup_pct:.2f}%)")
        else:
            base -= 10.0
            penalties.append(f"minor duplicate rows ({dup_rows} rows, {dup_pct:.2f}%)")

    has_dup_cols = bool(structure.get("has_duplicate_columns", False))
    dup_col_names = structure.get("duplicate_column_names", [])
    if has_dup_cols:
        base -= 20.0
        penalties.append(f"duplicate column names ({', '.join(dup_col_names)})")

    invalid_cols = [c for c in columns if c.get("invalid_values", {}).get("invalid_count", 0) > 0]
    if invalid_cols:
        base -= 15.0
        penalties.append("type inconsistency between numeric expectations and string values")

    empty_cols = quality.get("empty_columns", [])
    genuine_const = [c for c in quality.get("constant_columns", []) if c not in empty_cols]
    if genuine_const:
        penalty = min(len(genuine_const) * 5.0, 15.0)
        base -= penalty
        penalties.append(f"{len(genuine_const)} constant column(s)")

    score = max(0.0, min(100.0, round(base, 2)))
    status = "GOOD" if score >= 80.0 else ("WARNING" if score >= 50.0 else "POOR")

    if not penalties:
        reason = "Dataset records and schema are consistent with no duplicate rows, duplicate columns, or type clashes."
    else:
        reason = f"Consistency issues: {'; '.join(penalties)}."

    return {
        "score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# DIMENSION 5: STATISTICAL HEALTH
# ============================================================

def _assess_statistical_health(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate whether the dataset provides enough statistical signal for meaningful
    analysis. Categorical-only datasets are evaluated on category diversity rather
    than penalized for lacking numerical columns.
    """
    structure = evidence.get("structure", {})
    num_stats = evidence.get("numerical_statistics", {})
    cat_stats = evidence.get("categorical_statistics", {})
    correlations = evidence.get("correlations", {})

    row_count = int(structure.get("row_count", 0))
    num_cols = structure.get("numerical_columns", [])
    cat_cols = structure.get("categorical_columns", [])

    if row_count == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "No statistical signals available in an empty dataset.",
        }

    if row_count == 1:
        return {
            "score": 40.0,
            "status": "POOR",
            "reason": "Single-row dataset provides zero variance; distribution metrics and correlations cannot be calculated.",
        }

    base = 100.0
    penalties: List[str] = []
    signals: List[str] = []

    if num_cols:
        # Check numerical variance
        all_constant_num = all(
            num_stats.get(c, {}).get("std", 1.0) == 0.0
            for c in num_cols
            if c in num_stats
        )
        if all_constant_num:
            base -= 30.0
            penalties.append("all numerical columns exhibit zero variance (constant)")
        else:
            signals.append(f"{len(num_cols)} numerical column(s) with active variance")

        # Check correlations
        if len(num_cols) >= 2:
            if correlations.get("supported", False):
                pairs = correlations.get("strongest_pairs", [])
                if pairs:
                    top_corr = pairs[0].get("correlation", 0.0)
                    signals.append(f"pairwise correlation available (top r = {top_corr:.2f})")
                else:
                    base -= 5.0
                    penalties.append("correlations supported but no non-trivial pairs found")
            else:
                base -= 10.0
                penalties.append("pairwise correlations unavailable")
    else:
        # Categorical-only dataset: evaluate on categorical richness without penalizing missing numerical columns
        signals.append("categorical-only dataset")
        all_const_cat = all(
            cat_stats.get(c, {}).get("unique_count", 0) <= 1
            for c in cat_cols
            if c in cat_stats
        )
        if all_const_cat:
            base -= 40.0
            penalties.append("all categorical columns contain <= 1 unique value")
        else:
            signals.append(f"{len(cat_cols)} categorical attribute(s) with valid distribution")

    # Sample size limitation
    if row_count < 10:
        base -= 15.0
        penalties.append(f"very small sample size ({row_count} rows)")
    elif row_count < 30:
        base -= 10.0
        penalties.append(f"small sample size ({row_count} rows)")

    score = max(0.0, min(100.0, round(base, 2)))
    status = "GOOD" if score >= 80.0 else ("WARNING" if score >= 50.0 else "POOR")

    if not penalties:
        reason = f"Healthy statistical foundation: {'; '.join(signals)}."
    else:
        reason = f"Statistical health limitations: {'; '.join(penalties)}."

    return {
        "score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# DIMENSION 6: ML READINESS
# ============================================================

def _assess_ml_readiness(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the Evidence Layer's ML readiness assessment into the health schema.
    Does not recalculate ML metrics.
    """
    ml_readiness = evidence.get("ml_readiness", {})
    structure = evidence.get("structure", {})

    row_count = int(structure.get("row_count", 0))
    ml_score = float(ml_readiness.get("ml_score", 0.0))
    ml_status = str(ml_readiness.get("ml_status", "NOT READY"))
    can_train = bool(ml_readiness.get("can_train", False))
    target_candidates = ml_readiness.get("target_candidates", [])
    warnings = ml_readiness.get("warnings", [])

    if row_count == 0:
        return {
            "score": 0.0,
            "status": "POOR",
            "reason": "Machine learning analysis is unavailable on an empty dataset.",
        }

    # Status mapping
    if ml_status in ["READY", "Highly Ready"] or ml_score >= 80.0:
        status = "GOOD"
    elif ml_status in ["CAUTION", "Mostly Ready", "Needs Cleaning"] or ml_score >= 50.0:
        status = "WARNING"
    else:
        status = "POOR"

    parts: List[str] = [f"ML readiness score is {ml_score:.0f}/100 ({ml_status})."]

    if target_candidates:
        parts.append(f"Target candidate detected: '{target_candidates[0]['column']}'.")
    else:
        parts.append("No clear target variable detected.")

    if row_count < 30:
        parts.append(f"Sample size ({row_count} rows) is below the recommended modeling minimum.")

    if not can_train and ml_status != "READY":
        parts.append("Supervised modeling is not recommended at this stage.")

    reason = " ".join(parts)

    return {
        "score": ml_score,
        "status": status,
        "reason": reason,
    }


# ============================================================
# SYNTHESIS: STRENGTHS, WEAKNESSES, RISKS, ACTIONS
# ============================================================

def _synthesize_observations(
    evidence: Dict[str, Any],
    dimensions: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Formulate factual strengths, weaknesses, risks, and recommended actions
    strictly grounded in the evidence.
    """
    structure = evidence.get("structure", {})
    completeness = evidence.get("completeness", {})
    duplicates = evidence.get("duplicates", {})
    quality = evidence.get("quality", {})
    correlations = evidence.get("correlations", {})
    ml_readiness = evidence.get("ml_readiness", {})
    columns = evidence.get("columns", [])

    row_count = int(structure.get("row_count", 0))
    column_count = int(structure.get("column_count", 0))
    missing_cells = int(completeness.get("missing_cells", 0))
    missing_pct = float(completeness.get("missing_percentage", 0.0))
    dup_rows = int(duplicates.get("duplicate_rows", 0))
    dup_pct = float(duplicates.get("duplicate_row_percentage", 0.0))
    has_dup_cols = bool(structure.get("has_duplicate_columns", False))
    dup_col_names = structure.get("duplicate_column_names", [])

    strengths: List[str] = []
    weaknesses: List[str] = []
    risks: List[str] = []
    recommended_actions: List[str] = []

    # Handle empty dataset edge case
    if row_count == 0 or column_count == 0:
        weaknesses.append("The dataset contains 0 usable records or columns.")
        risks.append("No analytical, statistical, or predictive operations can be conducted.")
        recommended_actions.append("Upload a dataset containing valid tabular rows and columns.")
        return strengths, weaknesses, risks, recommended_actions

    # Completeness
    if missing_cells == 0:
        strengths.append("The dataset has 0% missing values across all cells.")
    else:
        weaknesses.append(
            f"The dataset contains {missing_cells:,} missing cells ({missing_pct:.2f}% of total values)."
        )
        recommended_actions.append(
            f"Review missing-value patterns across {len(completeness.get('columns_with_missing', []))} affected column(s) and apply appropriate imputation."
        )

    # Duplicates
    if dup_rows == 0:
        strengths.append("No duplicate records detected; all rows are unique.")
    else:
        weaknesses.append(
            f"{dup_rows:,} duplicate row(s) detected ({dup_pct:.2f}% of records)."
        )
        recommended_actions.append(
            f"Deduplicate the {dup_rows} identical rows to prevent bias and double-counting in analysis."
        )

    # Structure & Column Names
    if not has_dup_cols:
        strengths.append("All column names are unique.")
    else:
        weaknesses.append(f"Duplicate column names detected: {', '.join(dup_col_names)}.")
        risks.append("Duplicate column names can cause ambiguity in attribute selection and model training.")
        recommended_actions.append(f"Rename duplicate columns ({', '.join(dup_col_names)}) to unique attribute names.")

    # Empty & Constant Columns
    empty_cols = quality.get("empty_columns", [])
    if empty_cols:
        weaknesses.append(f"{len(empty_cols)} column(s) contain 100% missing values: {', '.join(empty_cols)}.")
        recommended_actions.append(f"Drop completely empty column(s): {', '.join(empty_cols)}.")

    genuine_const = [c for c in quality.get("constant_columns", []) if c not in empty_cols]
    if genuine_const:
        weaknesses.append(f"{len(genuine_const)} column(s) have constant values (zero variance): {', '.join(genuine_const)}.")
        recommended_actions.append(f"Drop or exclude constant column(s) ({', '.join(genuine_const)}) from predictive models.")

    # Invalid non-numeric values
    invalid_cols = [c for c in columns if c.get("invalid_values", {}).get("invalid_count", 0) > 0]
    if invalid_cols:
        for c in invalid_cols:
            inv = c["invalid_values"]
            weaknesses.append(
                f"Column '{c['name']}' contains {inv['invalid_count']} non-numeric invalid value(s) (e.g. {inv['invalid_examples']})."
            )
            recommended_actions.append(
                f"Clean non-numeric values in '{c['name']}' or coerce them to numeric before mathematical modeling."
            )

    # Sample Size Risks
    if row_count < 10:
        risks.append(f"Very small dataset ({row_count} rows); high risk of statistical sampling error.")
    elif row_count < 30:
        risks.append(f"Small dataset ({row_count} rows); machine-learning models may suffer from high variance or overfitting.")

    # Statistical Signal
    num_cols = structure.get("numerical_columns", [])
    if len(num_cols) >= 2 and correlations.get("supported", False):
        pairs = correlations.get("strongest_pairs", [])
        if pairs:
            top_pair = pairs[0]
            strengths.append(
                f"Active numerical correlation detected between '{top_pair['attribute_1']}' and '{top_pair['attribute_2']}' (r = {top_pair['correlation']:.2f})."
            )

    # ML Readiness
    candidates = ml_readiness.get("target_candidates", [])
    if candidates:
        top_cand = candidates[0]
        strengths.append(
            f"Potential target column identified: '{top_cand['column']}' ({top_cand.get('problem_type', 'ML')}, confidence: {top_cand.get('confidence', 0)}%)."
        )
    elif row_count >= 30 and column_count >= 2:
        recommended_actions.append("Select or define an explicit target variable if supervised machine learning is desired.")

    if not recommended_actions:
        recommended_actions.append("Dataset is structurally healthy and ready for exploratory or predictive analysis.")

    return strengths, weaknesses, risks, recommended_actions


# ============================================================
# MAIN HEALTH ENGINE API
# ============================================================

def calculate_health(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    target_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the comprehensive Data Health and Readiness Score from dataset evidence.

    Accepts either an already-built Evidence dictionary or a raw DataFrame.
    Evaluates 6 objective dimensions:
    - structural_health  (weight: 15%)
    - completeness       (weight: 20%)
    - validity           (weight: 20%)
    - consistency        (weight: 15%)
    - statistical_health (weight: 15%)
    - ml_readiness       (weight: 15%)

    Parameters:
    -----------
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None]
        Either a pre-calculated evidence dict from engine.evidence.build_evidence,
        or a raw pandas DataFrame.
    target_column: Optional[str]
        Optional target column to pass if building evidence on the fly.
    dataset_name: Optional[str]
        Optional dataset name label.

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable health dictionary.
    """
    # 1. Resolve Evidence Object
    if isinstance(evidence_or_df, pd.DataFrame) or evidence_or_df is None:
        evidence = build_evidence(evidence_or_df, target_column=target_column, dataset_name=dataset_name)
    elif isinstance(evidence_or_df, dict):
        evidence = evidence_or_df
    else:
        evidence = build_evidence(None, dataset_name=dataset_name)

    # 2. Evaluate Dimensions
    structural = _assess_structural_health(evidence)
    completeness = _assess_completeness(evidence)
    validity = _assess_validity(evidence)
    consistency = _assess_consistency(evidence)
    statistical = _assess_statistical_health(evidence)
    ml_readiness = _assess_ml_readiness(evidence)

    # 3. Calculate Weighted Overall Score
    # Weights:
    # structural_health  = 15%
    # completeness       = 20%
    # validity           = 20%
    # consistency        = 15%
    # statistical_health = 15%
    # ml_readiness       = 15%
    overall_score = round(
        structural["score"] * 0.15 +
        completeness["score"] * 0.20 +
        validity["score"] * 0.20 +
        consistency["score"] * 0.15 +
        statistical["score"] * 0.15 +
        ml_readiness["score"] * 0.15,
        2
    )

    # 4. Classify Overall Status with Safety Overrides
    structure = evidence.get("structure", {})
    row_count = int(structure.get("row_count", 0))
    column_count = int(structure.get("column_count", 0))

    if row_count == 0 or column_count == 0:
        overall_status = "NOT SUITABLE"
        overall_score = 0.0
        summary = "The dataset contains no usable records or columns and is not suitable for analysis."
    elif row_count <= 1:
        overall_status = "NOT SUITABLE"
        summary = f"The dataset contains only {row_count} record(s), preventing statistical or predictive analysis."
    else:
        if overall_score >= 80.0:
            overall_status = "READY"
            summary = f"Dataset is healthy (overall score {overall_score:.2f}/100) with solid structure, complete records, and active analytical signals."
        elif overall_score >= 50.0:
            overall_status = "NEEDS ATTENTION"
            summary = f"Dataset requires attention (overall score {overall_score:.2f}/100) due to identified quality, missingness, or structural concerns."
        else:
            overall_status = "NOT SUITABLE"
            summary = f"Dataset is not suitable for reliable analysis (overall score {overall_score:.2f}/100) without significant remediation."

        # Safety override 1: If structural health is critically poor (< 50), do not classify as READY
        if structural["score"] < 50.0 or structural["status"] == "POOR":
            if overall_status == "READY":
                overall_status = "NEEDS ATTENTION"
                summary = f"Dataset scored {overall_score:.2f}/100 but status is capped at NEEDS ATTENTION due to poor structural health."

        # Safety override 2: If completeness is critically poor (score <= 25, missingness > 40%), cap at NEEDS ATTENTION
        if completeness["score"] <= 25.0:
            if overall_status == "READY":
                overall_status = "NEEDS ATTENTION"
                summary = f"Dataset scored {overall_score:.2f}/100 but status is capped at NEEDS ATTENTION due to severe missingness."

    # 5. Synthesize Qualitative Sections Grounded in Evidence
    dimensions_dict = {
        "structural_health": structural,
        "completeness": completeness,
        "validity": validity,
        "consistency": consistency,
        "statistical_health": statistical,
        "ml_readiness": ml_readiness,
    }

    strengths, weaknesses, risks, recommended_actions = _synthesize_observations(
        evidence,
        dimensions_dict,
    )

    # 6. Extract Evidence Summary
    comp_data = evidence.get("completeness", {})
    dup_data = evidence.get("duplicates", {})
    qual_data = evidence.get("quality", {})
    ml_data = evidence.get("ml_readiness", {})

    evidence_summary = {
        "row_count": row_count,
        "column_count": column_count,
        "missing_percentage": float(comp_data.get("missing_percentage", 0.0)),
        "duplicate_rows": int(dup_data.get("duplicate_rows", 0)),
        "quality_score": int(qual_data.get("quality_score", 0)),
        "ml_score": int(ml_data.get("ml_score", 0)),
    }

    return {
        "overall": {
            "score": overall_score,
            "status": overall_status,
            "summary": summary,
        },
        "dimensions": dimensions_dict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risks": risks,
        "recommended_actions": recommended_actions,
        "evidence_summary": evidence_summary,
    }
