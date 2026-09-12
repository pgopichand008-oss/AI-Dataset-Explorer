"""
engine/evidence.py

Evidence Layer for AI-Dataset-Explorer.

Responsibilities:
- Build a single, deterministic, immutable factual evidence foundation from a dataset.
- Calculate verified facts using Python, Pandas, NumPy, and scikit-learn.
- Reuse existing profiling, quality, and ML readiness logic without duplication.
- Ensure strict resilience against edge cases: empty dataframes, single rows,
  duplicate column names, constant columns, all-missing columns, mixed values.
- Guarantee JSON-serializable output containing no DataFrame objects, raw traces,
  or invented statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils import safe_percentage, infer_column_role
from engine.profiling import (
    build_basic_profile,
    build_column_intelligence,
    build_numerical_summary,
    build_categorical_summary,
)
from engine.quality import (
    detect_outliers,
    build_quality_findings,
    calculate_quality_score,
    calculate_ml_readiness,
)
from change_engine import _invalid_value_summary

try:
    from engine.ml_engine import (
        detect_target_candidates,
        assess_ml_readiness as assess_ml_readiness_engine,
        get_best_target_candidate,
    )
    ML_ENGINE_AVAILABLE = True
except Exception:
    ML_ENGINE_AVAILABLE = False


# ============================================================
# SERIALIZATION & SANITIZATION HELPERS
# ============================================================

def _to_serializable(value: Any) -> Any:
    """
    Recursively sanitize values to ensure complete JSON serializability.
    Converts NumPy / Pandas scalars, NaNs, Infs, and Timestamps to native Python.
    """
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or np.isnan(value) or math.isinf(value) or np.isinf(value):
            return None
        return round(float(value), 4)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]

    if isinstance(value, pd.Series):
        return [_to_serializable(v) for v in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [_to_serializable(v) for v in value.to_dict(orient="records")]

    # Check for pandas/numpy NA types
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


# ============================================================
# COLUMN DEDUPLICATION HELPER FOR ROBUST PROCESSING
# ============================================================

def _prepare_working_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Create a safe working copy of the DataFrame where duplicate column names
    are uniquely suffixed for internal calculation, preventing ambiguous Series/DataFrame
    indexing errors while keeping exact 1-to-1 tracking of original names.

    Returns:
    - work_df: DataFrame with guaranteed unique column names.
    - col_map: Dict mapping work_column_name -> original_column_name.
    - duplicate_names: List of column names that had duplicates.
    """
    original_cols = [str(c) for c in df.columns]
    seen: Dict[str, int] = {}
    work_cols: List[str] = []
    duplicate_names: List[str] = []

    for col in original_cols:
        count = seen.get(col, 0)
        seen[col] = count + 1
        if count == 1:
            duplicate_names.append(col)

    # If duplicates were found, apply indexed suffix to duplicate instances
    name_occurrences: Dict[str, int] = {}
    for col in original_cols:
        if seen[col] > 1:
            occ = name_occurrences.get(col, 0)
            name_occurrences[col] = occ + 1
            work_cols.append(f"{col}_{occ}")
        else:
            work_cols.append(col)

    work_df = df.copy(deep=False)
    work_df.columns = work_cols
    col_map = {w: o for w, o in zip(work_cols, original_cols)}

    return work_df, col_map, duplicate_names


# ============================================================
# DETERMINISTIC CORRELATIONS EXTRACTOR
# ============================================================

def _extract_correlations(numeric_df: pd.DataFrame, min_rows: int = 3) -> Tuple[bool, Dict[str, Dict[str, Optional[float]]], List[Dict[str, Any]]]:
    """
    Calculate pairwise Pearson correlations deterministically without duplication.
    """
    if len(numeric_df.columns) < 2 or len(numeric_df) < min_rows:
        return False, {}, []

    try:
        corr_matrix = numeric_df.corr()
    except Exception:
        return False, {}, []

    cols = list(corr_matrix.columns)
    matrix_dict: Dict[str, Dict[str, Optional[float]]] = {}
    pairs: List[Dict[str, Any]] = []

    for i, col1 in enumerate(cols):
        matrix_dict[col1] = {}
        for j, col2 in enumerate(cols):
            val = corr_matrix.iloc[i, j]
            if pd.isna(val) or math.isnan(val):
                matrix_dict[col1][col2] = None
            else:
                matrix_dict[col1][col2] = round(float(val), 4)

            if j > i and pd.notna(val) and not math.isnan(val):
                pairs.append({
                    "attribute_1": col1,
                    "attribute_2": col2,
                    "correlation": round(float(val), 4),
                    "absolute_correlation": round(abs(float(val)), 4),
                })

    pairs.sort(key=lambda x: x["absolute_correlation"], reverse=True)
    return True, matrix_dict, pairs[:25]


# ============================================================
# CORE EVIDENCE LAYER
# ============================================================

def build_evidence(
    df: Optional[pd.DataFrame],
    target_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct the factual Evidence Layer for a given DataFrame.

    All numerical results are computed deterministically using Pandas, NumPy,
    and scikit-learn. No values are invented or estimated by language models.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataset to analyze. Original DataFrame is never mutated.
    target_column: Optional[str]
        Explicit target column name if defined by the user.
    dataset_name: Optional[str]
        Optional label or filename for the dataset.

    Returns:
    --------
    Dict[str, Any]
        Structured, JSON-serializable evidence package containing:
        - metadata
        - structure
        - completeness
        - duplicates
        - columns (per-column factual profiles)
        - numerical_statistics
        - categorical_statistics
        - quality
        - correlations
        - ml_readiness
        - limitations
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    name = dataset_name or "unnamed_dataset"

    # ------------------------------------------------------------
    # HANDLE EMPTY OR NONE DATAFRAME
    # ------------------------------------------------------------
    if df is None:
        return {
            "metadata": {
                "dataset_name": name,
                "evidence_version": "1.0.0",
                "generated_at": timestamp,
            },
            "structure": {
                "row_count": 0,
                "column_count": 0,
                "column_names": [],
                "has_duplicate_columns": False,
                "duplicate_column_names": [],
                "data_types": {},
                "numerical_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
                "boolean_columns": [],
                "memory_bytes": 0,
            },
            "completeness": {
                "total_cells": 0,
                "missing_cells": 0,
                "missing_percentage": 0.0,
                "complete_cells": 0,
                "complete_percentage": 100.0,
                "columns_with_missing": [],
                "column_missing_counts": {},
                "column_missing_percentages": {},
            },
            "duplicates": {
                "duplicate_rows": 0,
                "duplicate_row_percentage": 0.0,
                "is_unique": True,
            },
            "columns": [],
            "numerical_statistics": {},
            "categorical_statistics": {},
            "quality": {
                "quality_score": 0,
                "quality_status": "Poor",
                "empty_columns": [],
                "constant_columns": [],
                "high_cardinality_columns": [],
                "outlier_columns": [],
                "total_outlier_values": 0,
                "outlier_findings": [],
                "quality_findings": [{"Issue": "Missing Dataset", "Severity": "High", "Details": "No dataset was supplied."}],
            },
            "correlations": {
                "supported": False,
                "columns": [],
                "matrix": {},
                "strongest_pairs": [],
            },
            "ml_readiness": {
                "ml_score": 0,
                "ml_status": "NOT READY",
                "can_train": False,
                "target_candidates": [],
                "best_target_candidate": None,
                "warnings": ["Dataset is None."],
                "strengths": [],
                "evaluated_target": None,
            },
            "limitations": ["Dataset was not provided (None)."],
        }

    # ------------------------------------------------------------
    # PREPARE SAFE WORKING DATAFRAME
    # ------------------------------------------------------------
    work_df, col_map, duplicate_names = _prepare_working_dataframe(df)
    has_dup_cols = len(duplicate_names) > 0
    limitations: List[str] = []

    if has_dup_cols:
        limitations.append(
            f"Duplicate column name(s) detected: {duplicate_names}. "
            "Individual column occurrences were disambiguated for internal calculation."
        )

    # ------------------------------------------------------------
    # 1. STRUCTURE & PROFILING (Reuses engine/profiling.py)
    # ------------------------------------------------------------
    basic_profile = build_basic_profile(work_df)
    row_count = int(basic_profile["rows"])
    column_count = int(basic_profile["columns"])
    missing_cells = int(basic_profile["missing_cells"])
    duplicate_rows = int(basic_profile["duplicate_rows"])
    work_numeric_df = basic_profile["numeric_df"]

    total_cells = row_count * column_count
    missing_pct = safe_percentage(missing_cells, total_cells)
    dup_row_pct = safe_percentage(duplicate_rows, row_count)

    if row_count == 0:
        limitations.append("Dataset is empty (contains 0 records).")
    elif row_count == 1:
        limitations.append("Dataset contains only 1 record; statistical variance and correlations cannot be calculated.")
    elif row_count < 30:
        limitations.append(f"Small dataset ({row_count} rows); statistical distributions and ML metrics should be interpreted cautiously.")

    if column_count == 0:
        limitations.append("Dataset contains no columns.")

    # Memory usage
    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        memory_bytes = 0

    # Classify column types safely
    raw_col_names = [str(c) for c in df.columns]
    data_types: Dict[str, str] = {}
    numerical_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []

    for i, w_col in enumerate(work_df.columns):
        orig_name = col_map[w_col]
        series = work_df.iloc[:, i]
        dtype_str = str(series.dtype)
        data_types[w_col] = dtype_str

        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(orig_name)
            categorical_cols.append(orig_name)
        elif pd.api.types.is_numeric_dtype(series):
            numerical_cols.append(orig_name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(orig_name)
        else:
            # Check if object column is date-like via infer_column_role
            role, _, _, _ = infer_column_role(work_df, w_col)
            if role == "Date / Time":
                datetime_cols.append(orig_name)
            else:
                categorical_cols.append(orig_name)

    if not numerical_cols and column_count > 0:
        limitations.append("No numerical columns found; numerical statistics and correlation analysis are unavailable.")

    if not categorical_cols and column_count > 0:
        limitations.append("No categorical columns found.")

    # ------------------------------------------------------------
    # 2. COLUMN-LEVEL FACTUAL INTELLIGENCE
    # ------------------------------------------------------------
    (
        column_intel_df,
        identifier_columns,
        date_columns,
        target_candidates_raw,
    ) = build_column_intelligence(work_df, row_count)

    columns_detail: List[Dict[str, Any]] = []
    col_missing_counts: Dict[str, int] = {}
    col_missing_percentages: Dict[str, float] = {}
    cols_with_missing: List[str] = []

    for i, w_col in enumerate(work_df.columns):
        orig_name = col_map[w_col]
        series = work_df.iloc[:, i]

        col_miss = int(series.isna().sum())
        col_miss_pct = round(safe_percentage(col_miss, row_count), 2)
        col_missing_counts[orig_name] = col_miss
        col_missing_percentages[orig_name] = col_miss_pct

        if col_miss > 0:
            cols_with_missing.append(orig_name)

        unique_count = int(series.nunique(dropna=True))
        unique_pct = round(safe_percentage(unique_count, row_count), 2)

        is_all_missing = (col_miss == row_count and row_count > 0)
        is_constant = (unique_count <= 1 and col_miss == 0 and row_count > 0) or is_all_missing

        if is_all_missing:
            limitations.append(f"Column '{orig_name}' contains 100% missing values.")
        elif is_constant:
            limitations.append(f"Column '{orig_name}' is constant ({unique_count} unique non-null value).")

        # Inferred role from column_intel_df
        role_info = {}
        if not column_intel_df.empty:
            match = column_intel_df[column_intel_df["Attribute"] == w_col]
            if not match.empty:
                row_dict = match.iloc[0].to_dict()
                role_info = {
                    "role": str(row_dict.get("Role", "Other")),
                    "confidence": round(float(row_dict.get("Confidence", 0.5)), 2),
                    "reason": str(row_dict.get("Reason", "")),
                    "is_target": row_dict.get("Potential Target") == "Yes",
                }

        if not role_info:
            role, conf, is_target, reason = infer_column_role(work_df, w_col)
            role_info = {
                "role": role,
                "confidence": round(float(conf), 2),
                "reason": reason,
                "is_target": is_target,
            }

        # Check for mixed/invalid values in columns expected to be numeric
        invalid_summary = {"invalid_count": 0, "valid_count": 0, "invalid_examples": []}
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            inv = _invalid_value_summary(series)
            if inv["valid_count"] > 0 and inv["invalid_count"] > 0:
                invalid_summary = inv
                limitations.append(
                    f"Column '{orig_name}' contains {inv['invalid_count']} invalid non-numeric value(s) "
                    f"such as {inv['invalid_examples']}."
                )

        # Non-null sample values (up to 5)
        non_null_series = series.dropna()
        sample_vals = [
            _to_serializable(v)
            for v in non_null_series.head(5).tolist()
        ]

        columns_detail.append({
            "index": i,
            "name": orig_name,
            "internal_key": w_col,
            "data_type": str(series.dtype),
            "inferred_role": role_info["role"],
            "role_confidence": role_info["confidence"],
            "role_reason": role_info["reason"],
            "missing_count": col_miss,
            "missing_percentage": col_miss_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "is_constant": is_constant,
            "is_all_missing": is_all_missing,
            "is_target_candidate": role_info["is_target"],
            "is_potential_identifier": w_col in identifier_columns,
            "invalid_values": invalid_summary,
            "sample_values": sample_vals,
        })

    # ------------------------------------------------------------
    # 3. NUMERICAL STATISTICS (Reuses build_numerical_summary)
    # ------------------------------------------------------------
    numerical_stats: Dict[str, Dict[str, Optional[float]]] = {}
    if not work_numeric_df.empty and row_count > 0:
        raw_num_sum = build_numerical_summary(work_numeric_df)
        if not raw_num_sum.empty:
            for w_col in work_numeric_df.columns:
                orig_name = col_map[w_col]
                s = work_numeric_df[w_col].dropna()
                valid_n = len(s)

                if valid_n == 0:
                    numerical_stats[orig_name] = {
                        "count": 0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "q25": None,
                        "median": None,
                        "q75": None,
                        "max": None,
                        "range": None,
                        "skewness": None,
                    }
                else:
                    mean_val = float(s.mean())
                    std_val = float(s.std()) if valid_n > 1 else 0.0
                    min_val = float(s.min())
                    q25_val = float(s.quantile(0.25))
                    med_val = float(s.median())
                    q75_val = float(s.quantile(0.75))
                    max_val = float(s.max())
                    rng_val = max_val - min_val
                    skew_val = float(s.skew()) if valid_n >= 3 else None

                    numerical_stats[orig_name] = {
                        "count": valid_n,
                        "mean": round(mean_val, 4),
                        "std": round(std_val, 4) if std_val is not None else None,
                        "min": round(min_val, 4),
                        "q25": round(q25_val, 4),
                        "median": round(med_val, 4),
                        "q75": round(q75_val, 4),
                        "max": round(max_val, 4),
                        "range": round(rng_val, 4),
                        "skewness": round(skew_val, 4) if skew_val is not None and not (math.isnan(skew_val) or np.isnan(skew_val)) else None,
                    }

    # ------------------------------------------------------------
    # 4. CATEGORICAL STATISTICS (Reuses build_categorical_summary)
    # ------------------------------------------------------------
    categorical_stats: Dict[str, Dict[str, Any]] = {}
    work_cat_cols = [w for w in work_df.columns if col_map[w] in categorical_cols]
    if work_cat_cols and row_count > 0:
        raw_cat_sum = build_categorical_summary(work_df, work_cat_cols, row_count)
        for w_col in work_cat_cols:
            orig_name = col_map[w_col]
            s = work_df[w_col].dropna()
            counts = s.value_counts()
            top_cat = str(counts.index[0]) if not counts.empty else None
            top_freq = int(counts.iloc[0]) if not counts.empty else 0
            dom_pct = round(safe_percentage(top_freq, row_count), 2)

            top_counts_dict = {
                str(k): int(v) for k, v in counts.head(10).items()
            }

            categorical_stats[orig_name] = {
                "count": len(s),
                "unique_count": int(s.nunique()),
                "top_category": top_cat,
                "top_frequency": top_freq,
                "dominance_percentage": dom_pct,
                "category_counts": top_counts_dict,
            }

    # ------------------------------------------------------------
    # 5. QUALITY FINDINGS & SCORING (Reuses engine/quality.py)
    # ------------------------------------------------------------
    outlier_columns_work, total_outlier_values, anomaly_df = detect_outliers(
        work_df,
        work_numeric_df,
    )
    outlier_columns = [col_map.get(w, w) for w in outlier_columns_work]

    outlier_findings: List[Dict[str, Any]] = []
    if not anomaly_df.empty:
        for _, a_row in anomaly_df.iterrows():
            outlier_findings.append({
                "attribute": col_map.get(a_row["Attribute"], a_row["Attribute"]),
                "anomaly_type": a_row.get("Anomaly Type", "Outliers"),
                "count": int(a_row.get("Count", 0)),
                "percentage": float(a_row.get("Percentage", 0.0)),
                "lower_bound": round(float(a_row.get("Lower Bound", 0.0)), 4),
                "upper_bound": round(float(a_row.get("Upper Bound", 0.0)), 4),
            })

    (
        quality_findings_raw,
        empty_columns_work,
        constant_columns_work,
        high_cardinality_work,
    ) = build_quality_findings(
        work_df,
        row_count,
        column_count,
        missing_cells,
        duplicate_rows,
        outlier_columns_work,
    )

    empty_columns = [col_map.get(w, w) for w in empty_columns_work]
    constant_columns = [col_map.get(w, w) for w in constant_columns_work]
    high_cardinality_columns = [col_map.get(w, w) for w in high_cardinality_work]

    quality_score, quality_status = calculate_quality_score(
        row_count,
        column_count,
        missing_cells,
        duplicate_rows,
        outlier_columns_work,
        constant_columns_work,
    )

    # ------------------------------------------------------------
    # 6. CORRELATIONS
    # ------------------------------------------------------------
    corr_supported, corr_matrix, strongest_corr_pairs = _extract_correlations(
        work_numeric_df,
        min_rows=3,
    )

    remapped_corr_matrix: Dict[str, Dict[str, Optional[float]]] = {}
    if corr_supported:
        for w1, sub in corr_matrix.items():
            o1 = col_map.get(w1, w1)
            remapped_corr_matrix[o1] = {col_map.get(w2, w2): v for w2, v in sub.items()}

        remapped_pairs = []
        for p in strongest_corr_pairs:
            remapped_pairs.append({
                "attribute_1": col_map.get(p["attribute_1"], p["attribute_1"]),
                "attribute_2": col_map.get(p["attribute_2"], p["attribute_2"]),
                "correlation": p["correlation"],
                "absolute_correlation": p["absolute_correlation"],
            })
        strongest_corr_pairs = remapped_pairs

    # ------------------------------------------------------------
    # 7. ML READINESS & TARGET DISCOVERY (Reuses ml_engine / quality)
    # ------------------------------------------------------------
    ml_candidates: List[Dict[str, Any]] = []
    best_candidate_name: Optional[str] = None
    ml_score: int = 0
    ml_status: str = "NOT READY"
    can_train: bool = False
    ml_warnings: List[str] = []
    ml_strengths: List[str] = []
    evaluated_target: Optional[str] = None

    if ML_ENGINE_AVAILABLE and row_count > 0 and column_count > 1:
        try:
            candidates = detect_target_candidates(work_df)
            for cand in candidates:
                ml_candidates.append({
                    "column": col_map.get(cand["column"], cand["column"]),
                    "confidence": int(cand.get("confidence", 0)),
                    "problem_type": cand.get("problem_type"),
                    "reason": cand.get("reason", ""),
                })

            if ml_candidates:
                best_candidate_name = ml_candidates[0]["column"]

            # Use designated or detected target
            target_to_eval = None
            if target_column:
                for w_c, o_c in col_map.items():
                    if o_c == target_column:
                        target_to_eval = w_c
                        evaluated_target = target_column
                        break
            elif best_candidate_name:
                for w_c, o_c in col_map.items():
                    if o_c == best_candidate_name:
                        target_to_eval = w_c
                        evaluated_target = best_candidate_name
                        break

            ml_eval = assess_ml_readiness_engine(
                work_df,
                target_column=target_to_eval,
                identifier_columns=identifier_columns,
            )

            ml_score = int(ml_eval.get("score", 0))
            ml_status = str(ml_eval.get("status", "NOT READY"))
            can_train = bool(ml_eval.get("can_train", False))
            ml_warnings = [str(w) for w in ml_eval.get("warnings", [])]
            ml_strengths = [str(s) for s in ml_eval.get("strengths", [])]

        except Exception as ml_err:
            ml_warnings.append(f"Advanced ML assessment deferred: {str(ml_err)}")
            ml_score, ml_status = calculate_ml_readiness(
                row_count,
                missing_cells,
                duplicate_rows,
                outlier_columns_work,
                constant_columns_work,
                identifier_columns,
                column_count,
            )
            can_train = (ml_score >= 50 and row_count >= 30)
    else:
        ml_score, ml_status = calculate_ml_readiness(
            row_count,
            missing_cells,
            duplicate_rows,
            outlier_columns_work,
            constant_columns_work,
            identifier_columns,
            column_count,
        )
        can_train = (ml_score >= 50 and row_count >= 30)
        if row_count < 30:
            ml_warnings.append(f"Insufficient rows for machine learning ({row_count} rows).")
        if column_count <= 1:
            ml_warnings.append("Insufficient columns for feature-target modeling.")

    # ------------------------------------------------------------
    # ASSEMBLE COMPLETE EVIDENCE PAYLOAD
    # ------------------------------------------------------------
    evidence: Dict[str, Any] = {
        "metadata": {
            "dataset_name": name,
            "evidence_version": "1.0.0",
            "generated_at": timestamp,
        },
        "structure": {
            "row_count": row_count,
            "column_count": column_count,
            "column_names": raw_col_names,
            "has_duplicate_columns": has_dup_cols,
            "duplicate_column_names": duplicate_names,
            "data_types": {col_map.get(k, k): v for k, v in data_types.items()},
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "memory_bytes": memory_bytes,
        },
        "completeness": {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": round(missing_pct, 2),
            "complete_cells": total_cells - missing_cells,
            "complete_percentage": round(100.0 - missing_pct, 2),
            "columns_with_missing": cols_with_missing,
            "column_missing_counts": col_missing_counts,
            "column_missing_percentages": col_missing_percentages,
        },
        "duplicates": {
            "duplicate_rows": duplicate_rows,
            "duplicate_row_percentage": round(dup_row_pct, 2),
            "is_unique": duplicate_rows == 0,
        },
        "columns": columns_detail,
        "numerical_statistics": numerical_stats,
        "categorical_statistics": categorical_stats,
        "quality": {
            "quality_score": quality_score,
            "quality_status": quality_status,
            "empty_columns": empty_columns,
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "outlier_columns": outlier_columns,
            "total_outlier_values": total_outlier_values,
            "outlier_findings": outlier_findings,
            "quality_findings": quality_findings_raw,
        },
        "correlations": {
            "supported": corr_supported,
            "columns": [col_map.get(w, w) for w in work_numeric_df.columns] if corr_supported else [],
            "matrix": remapped_corr_matrix,
            "strongest_pairs": strongest_corr_pairs,
        },
        "ml_readiness": {
            "ml_score": ml_score,
            "ml_status": ml_status,
            "can_train": can_train,
            "target_candidates": ml_candidates,
            "best_target_candidate": best_candidate_name,
            "warnings": ml_warnings,
            "strengths": ml_strengths,
            "evaluated_target": evaluated_target,
        },
        "limitations": limitations,
    }

    # Final deep sanitization pass guarantees 100% JSON safety
    return _to_serializable(evidence)
