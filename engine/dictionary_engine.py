"""
FILE: engine/dictionary_engine.py

AI Data Dictionary / Column Intelligence Backend.

Responsibilities:
- Transform raw column information or Evidence Layer output into structured,
  factual, explainable column-level intelligence.
- Deterministic column type classification (numerical, categorical, datetime,
  boolean, text, unknown).
- Semantic role assignment (identifier, target candidate, numerical measure,
  categorical feature, datetime, text, constant, all missing, high cardinality, unknown).
- Factual missingness, uniqueness, and type-specific descriptive statistics.
- Cautious, evidence-grounded interpretations without fabricated business semantics.
- Detection of factual quality concerns (missingness, constant, outliers, high cardinality).
- Factual analytical and ML usefulness guidance.
- JSON serializability and strict DataFrame immutability.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from engine.evidence import build_evidence
from utils import safe_percentage, infer_column_role


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def _to_serializable(val: Any) -> Any:
    """
    Recursively sanitize values to ensure complete JSON serializability.
    Converts NumPy / Pandas scalars, NaNs, Infs, and Timestamps to native Python.
    """
    if val is None:
        return None

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or np.isnan(val) or math.isinf(val) or np.isinf(val):
            return None
        return round(float(val), 4)

    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()

    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}

    if isinstance(val, (list, tuple, set)):
        return [_to_serializable(v) for v in val]

    if isinstance(val, pd.Series):
        return [_to_serializable(v) for v in val.tolist()]

    if isinstance(val, pd.DataFrame):
        return [_to_serializable(v) for v in val.to_dict(orient="records")]

    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    return str(val)


# ============================================================
# COLUMN TYPE DETECTION
# ============================================================

def _detect_column_type(
    col_name: str,
    col_info: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Tuple[str, float]:
    """
    Deterministically classify the column type.
    Supported types: 'numerical', 'categorical', 'datetime', 'boolean', 'text', 'unsupported', 'unknown'.
    Returns (detected_type, confidence).
    """
    data_type_str = str(col_info.get("data_type", "")).lower()
    inferred_role = str(col_info.get("inferred_role", ""))
    samples = col_info.get("sample_values", [])

    # 1. Check native DataFrame dtype if available
    if df is not None and col_name in df.columns:
        series = df[col_name]
        if pd.api.types.is_complex_dtype(series):
            return "unsupported", 0.40
        if pd.api.types.is_bool_dtype(series):
            return "boolean", 0.95
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", 0.95
        if pd.api.types.is_numeric_dtype(series):
            return "numerical", 0.95
        if pd.api.types.is_string_dtype(series):
            str_samples = [str(s) for s in series.dropna().head(100).tolist()]
            if str_samples:
                avg_len = sum(len(s) for s in str_samples) / len(str_samples)
                has_spaces = any(" " in s for s in str_samples)
                if avg_len >= 50 or (avg_len >= 25 and has_spaces):
                    return "text", 0.85
            return "categorical", 0.90

    # 2. Check inferred role or string representation of dtype
    if "complex" in data_type_str or "period" in data_type_str:
        return "unsupported", 0.40

    if "bool" in data_type_str:
        return "boolean", 0.95

    if "datetime" in data_type_str or inferred_role == "Date / Time":
        return "datetime", 0.95

    if any(num_t in data_type_str for num_t in ("int", "float", "double")):
        return "numerical", 0.95

    # 3. Object / String / Categorical analysis
    if any(cat_t in data_type_str for cat_t in ("object", "string", "category")):
        # Check if sample strings are long text
        str_samples = [str(s) for s in samples if s is not None]
        if str_samples:
            avg_len = sum(len(s) for s in str_samples) / len(str_samples)
            has_spaces = any(" " in s for s in str_samples)
            if avg_len >= 50 or (avg_len >= 25 and has_spaces):
                return "text", 0.85

        return "categorical", 0.90

    # Fallback
    return "unknown", 0.50


# ============================================================
# SEMANTIC ROLE ASSIGNMENT
# ============================================================

def _determine_semantic_role(
    col_name: str,
    detected_type: str,
    col_info: Dict[str, Any],
    row_count: int,
) -> Tuple[str, float]:
    """
    Deterministically assign a semantic role based on factual evidence.
    Roles:
    - 'all missing'
    - 'constant'
    - 'identifier'
    - 'target candidate'
    - 'datetime'
    - 'numerical measure'
    - 'text'
    - 'high cardinality'
    - 'categorical feature'
    - 'unknown'
    """
    is_all_missing = col_info.get("is_all_missing", False)
    is_constant = col_info.get("is_constant", False)
    missing_count = col_info.get("missing_count", 0)
    missing_pct = col_info.get("missing_percentage", 0.0)
    unique_count = col_info.get("unique_count", 0)
    unique_pct = col_info.get("unique_percentage", 0.0)
    is_target_cand = col_info.get("is_target_candidate", False)
    is_potential_id = col_info.get("is_potential_identifier", False)
    inferred_role = str(col_info.get("inferred_role", ""))

    col_lower = str(col_name).lower()
    target_keywords = {"target", "label", "outcome", "class", "churn", "default", "result", "y"}

    # 1. All missing
    if is_all_missing or (row_count > 0 and missing_count == row_count) or missing_pct >= 100.0:
        return "all missing", 0.99

    # 2. Constant
    if is_constant or (row_count > 0 and unique_count <= 1 and missing_count == 0):
        return "constant", 0.98

    # 3. Identifier
    if is_potential_id or inferred_role == "Identifier" or (unique_pct >= 95.0 and unique_count > 10 and row_count > 10):
        return "identifier", 0.92

    # 4. Target candidate by name / explicit flag
    if col_lower in target_keywords or (is_target_cand and any(kw in col_lower for kw in ("target", "label", "class", "churn"))):
        return "target candidate", 0.90

    # 5. Datetime
    if detected_type == "datetime" or inferred_role == "Date / Time":
        return "datetime", 0.95

    # 6. Numerical measure
    if detected_type == "numerical":
        if is_target_cand and (col_lower in target_keywords or "target" in col_lower):
            return "target candidate", 0.85
        return "numerical measure", 0.90

    # 7. Text
    if detected_type == "text":
        return "text", 0.85

    # 8. High cardinality categorical
    if detected_type in ("categorical", "boolean") or inferred_role == "Categorical":
        if unique_count > 50 and unique_pct > 30.0:
            return "high cardinality", 0.85
        return "categorical feature", 0.90

    # 9. Target candidate fallback
    if is_target_cand:
        return "target candidate", 0.80

    # 10. Unknown
    return "unknown", 0.50


# ============================================================
# CAUTIOUS INTERPRETATION (POSSIBLE MEANING)
# ============================================================

def _generate_possible_meaning(
    col_name: str,
    detected_type: str,
    semantic_role: str,
    col_info: Dict[str, Any],
) -> str:
    """
    Generate cautious, evidence-grounded interpretations.
    Never invents unverified business semantics like 'Customer age' or 'Sales revenue'.
    Always begins with 'Possible interpretation: '.
    """
    unique_count = col_info.get("unique_count", 0)
    unique_pct = col_info.get("unique_percentage", 0.0)

    if semantic_role == "all missing":
        return "Possible interpretation: unpopulated column containing entirely missing values."

    if semantic_role == "constant":
        return "Possible interpretation: constant column containing a single uniform non-null value across all records."

    if semantic_role == "identifier":
        return (
            f"Possible interpretation: identifier-like column based on very high uniqueness ratio "
            f"({unique_pct:.1f}% unique values)."
        )

    if semantic_role == "target candidate":
        return (
            "Possible interpretation: potential target candidate based on discrete class distribution "
            "or target suitability characteristics."
        )

    if semantic_role == "datetime":
        return "Possible interpretation: time-related field because values are detected as dates or timestamps."

    if semantic_role == "text":
        return "Possible interpretation: unstructured or variable-length text field."

    if semantic_role == "high cardinality":
        return (
            f"Possible interpretation: high-cardinality categorical feature with "
            f"{unique_count} distinct discrete levels."
        )

    if semantic_role == "categorical feature":
        if detected_type == "boolean":
            return "Possible interpretation: binary boolean indicator representing two discrete states."
        return (
            f"Possible interpretation: categorical feature representing {unique_count} "
            f"discrete categorical levels."
        )

    if semantic_role == "numerical measure":
        return (
            "Possible interpretation: numerical measure because the column contains "
            "continuous or discrete quantitative values."
        )

    return "Possible interpretation: unclassified column with ambiguous or mixed data types."


# ============================================================
# QUALITY CONCERNS DETECTION
# ============================================================

def _collect_quality_concerns(
    col_name: str,
    detected_type: str,
    semantic_role: str,
    col_info: Dict[str, Any],
    stats: Dict[str, Any],
    outlier_findings: List[Dict[str, Any]],
    row_count: int,
) -> List[str]:
    """
    Deterministically identify factual quality concerns grounded in evidence.
    """
    concerns: List[str] = []
    missing_count = col_info.get("missing_count", 0)
    missing_pct = col_info.get("missing_percentage", 0.0)
    unique_count = col_info.get("unique_count", 0)
    unique_pct = col_info.get("unique_percentage", 0.0)
    is_constant = col_info.get("is_constant", False)
    is_all_missing = col_info.get("is_all_missing", False)
    invalid_info = col_info.get("invalid_values", {})

    if row_count == 0:
        concerns.append("Dataset is empty (0 records); unable to evaluate column distribution.")
        return concerns

    if row_count == 1:
        concerns.append("Single observation dataset; statistical variance cannot be computed.")

    # 1. Missingness
    if is_all_missing or missing_pct >= 100.0:
        concerns.append("All values are missing (100.0% missing cells).")
    elif missing_pct >= 50.0:
        concerns.append(f"High missingness: {missing_pct:.1f}% of values are missing ({missing_count:,} cells).")
    elif missing_pct >= 10.0:
        concerns.append(f"Moderate missingness: {missing_pct:.1f}% of values are missing ({missing_count:,} cells).")

    # 2. Constant / zero variance
    if is_constant or (unique_count <= 1 and row_count > 1 and missing_count == 0):
        concerns.append("Constant column with zero variance across all rows.")
    elif detected_type == "numerical" and stats.get("std") == 0.0 and row_count > 1:
        concerns.append("Zero standard deviation; values do not exhibit numerical variation.")

    # 3. High uniqueness / identifier
    if semantic_role == "identifier" or (unique_pct >= 95.0 and unique_count > 10 and row_count > 10):
        concerns.append(
            f"Potential identifier with high uniqueness ({unique_pct:.1f}% unique values); "
            "risk of data leakage if used in ML."
        )

    # 4. High cardinality categorical
    if detected_type == "categorical" and unique_count > 50 and unique_pct > 30.0:
        concerns.append(f"High cardinality: contains {unique_count} distinct categories.")

    # 5. Invalid / mixed non-numeric values
    if invalid_info and invalid_info.get("invalid_count", 0) > 0:
        inv_c = invalid_info["invalid_count"]
        examples = invalid_info.get("invalid_examples", [])
        ex_str = f" such as {examples[:3]}" if examples else ""
        concerns.append(f"Contains {inv_c} invalid or mixed non-numeric value(s){ex_str}.")

    # 6. Outliers from evidence quality findings or statistical IQR
    outlier_found = False
    for of in outlier_findings:
        attr = of.get("attribute") or of.get("Attribute") or of.get("column")
        if attr == col_name:
            c = of.get("count") or of.get("Count") or of.get("anomaly_count", 0)
            p = of.get("percentage") or of.get("Percentage") or of.get("affected_percentage", 0.0)
            concerns.append(f"Outliers detected: {c} value(s) ({p:.1f}%) outside 1.5*IQR boundaries.")
            outlier_found = True
            break

    if not outlier_found and detected_type == "numerical" and row_count >= 4:
        q25 = stats.get("q25")
        q75 = stats.get("q75")
        min_v = stats.get("min")
        max_v = stats.get("max")
        if q25 is not None and q75 is not None and min_v is not None and max_v is not None:
            iqr = q75 - q25
            if iqr > 0:
                low_b = q25 - 1.5 * iqr
                high_b = q75 + 1.5 * iqr
                if min_v < low_b or max_v > high_b:
                    concerns.append("Outliers detected: values exist outside 1.5*IQR boundaries.")

    return concerns


# ============================================================
# ANALYTICAL AND ML USEFULNESS GUIDANCE
# ============================================================

def _determine_analytical_usefulness(
    detected_type: str,
    semantic_role: str,
    stats: Dict[str, Any],
    row_count: int,
) -> str:
    """Provide factual, deterministic analytical usefulness guidance."""
    if row_count == 0:
        return "No analytical usefulness available for empty dataset."

    if semantic_role == "all missing":
        return "Not analytically useful until missing data is collected or imputed."

    if semantic_role == "constant":
        return "Very low analytical usefulness due to lack of variation across records."

    if semantic_role == "identifier":
        return "Useful for primary key record identification and table joins; limited direct analytical value for aggregation."

    if semantic_role == "datetime":
        return "Useful for temporal analysis, trend evaluation, seasonality detection, and time-series aggregation."

    if detected_type == "numerical":
        if stats.get("std") == 0.0:
            return "Limited analytical usefulness due to zero variance across observations."
        return "Useful for descriptive statistics, distribution profiling, percentile analysis, and numerical correlation studies."

    if detected_type == "boolean":
        return "Useful for binary filtering, proportion estimation, and cross-tabulation."

    if detected_type == "text":
        return "Useful for string length profiling, keyword extraction, and natural language processing."

    if detected_type == "categorical":
        if semantic_role == "high cardinality":
            return "Useful for high-granularity grouping; may require bucketing long-tail categories for macro analysis."
        return "Useful for group comparisons, segmentation, frequency distribution, and categorical cross-tabulation."

    return "Limited analytical usefulness without data cleaning and schema validation."


def _determine_ml_usefulness(
    detected_type: str,
    semantic_role: str,
    col_info: Dict[str, Any],
    row_count: int,
) -> str:
    """Provide factual, deterministic ML usefulness guidance."""
    if row_count == 0:
        return "Unsuitable for machine learning due to empty dataset."

    missing_pct = col_info.get("missing_percentage", 0.0)
    is_target_cand = col_info.get("is_target_candidate", False)

    if semantic_role == "all missing":
        return "Unsuitable for machine learning in its current state because all values are missing."

    if semantic_role == "constant":
        return "Likely unsuitable as a predictive feature because it is constant with zero variance."

    if semantic_role == "identifier":
        return "Potentially an identifier; exclude or use caution to prevent data leakage and memorization."

    if semantic_role == "target candidate" or is_target_cand:
        return "Target candidate based on existing target suitability evidence."

    if missing_pct >= 50.0:
        return f"High missingness ({missing_pct:.1f}%); requires substantial imputation or removal before model training."

    if semantic_role == "datetime" or detected_type == "datetime":
        return "Can be engineered into temporal features (e.g., day of week, month, elapsed time) for modeling."

    if detected_type == "boolean":
        return "Can be used directly as a binary indicator feature (0/1)."

    if detected_type == "numerical":
        return "Potentially useful as a numerical predictive feature (may benefit from normalization or scaling)."

    if detected_type == "text":
        return "Requires text vectorization (e.g., TF-IDF, tokenization, or embeddings) before use in predictive models."

    if detected_type == "categorical":
        if semantic_role == "high cardinality":
            return "High-cardinality categorical feature; requires dimensionality reduction or target/frequency encoding."
        return "Potentially useful as a categorical feature after appropriate one-hot or ordinal encoding."

    return "Evaluate data quality and distribution before including in model training."


# ============================================================
# STATISTICAL EXTRACTION
# ============================================================

def _extract_column_statistics(
    col_name: str,
    detected_type: str,
    evidence: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Extract factual statistics appropriate for the column type.
    Reuses Evidence calculations wherever available.
    """
    num_stats = evidence.get("numerical_statistics", {})
    cat_stats = evidence.get("categorical_statistics", {})

    if detected_type == "numerical":
        if col_name in num_stats:
            return dict(num_stats[col_name])
        if df is not None and col_name in df.columns:
            s = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(s) == 0:
                return {
                    "count": 0, "mean": None, "std": None, "min": None,
                    "q25": None, "median": None, "q75": None, "max": None,
                    "range": None, "skewness": None,
                }
            std_val = float(s.std()) if len(s) > 1 else 0.0
            skew_val = float(s.skew()) if len(s) >= 3 else None
            return {
                "count": len(s),
                "mean": round(float(s.mean()), 4),
                "std": round(std_val, 4) if std_val is not None else None,
                "min": round(float(s.min()), 4),
                "q25": round(float(s.quantile(0.25)), 4),
                "median": round(float(s.median()), 4),
                "q75": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "range": round(float(s.max() - s.min()), 4),
                "skewness": round(skew_val, 4) if skew_val is not None and not math.isnan(skew_val) else None,
            }
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "q25": None, "median": None, "q75": None, "max": None,
            "range": None, "skewness": None,
        }

    if detected_type in ("categorical", "boolean", "text"):
        if col_name in cat_stats:
            d = dict(cat_stats[col_name])
            d["cardinality"] = d.get("unique_count", 0)
            return d
        if df is not None and col_name in df.columns:
            s = df[col_name].dropna()
            counts = s.value_counts()
            top_cat = str(counts.index[0]) if not counts.empty else None
            top_freq = int(counts.iloc[0]) if not counts.empty else 0
            dom_pct = round(safe_percentage(top_freq, len(df)), 2) if len(df) > 0 else 0.0
            top_counts = {str(k): int(v) for k, v in counts.head(10).items()}
            return {
                "count": len(s),
                "unique_count": int(s.nunique()),
                "cardinality": int(s.nunique()),
                "top_category": top_cat,
                "top_frequency": top_freq,
                "dominance_percentage": dom_pct,
                "category_counts": top_counts,
            }
        return {
            "count": 0, "unique_count": 0, "cardinality": 0,
            "top_category": None, "top_frequency": 0,
            "dominance_percentage": 0.0, "category_counts": {},
        }

    if detected_type == "datetime":
        if df is not None and col_name in df.columns:
            try:
                dt_s = pd.to_datetime(df[col_name], errors="coerce").dropna()
                if not dt_s.empty:
                    min_dt = dt_s.min()
                    max_dt = dt_s.max()
                    range_days = round((max_dt - min_dt).total_seconds() / 86400.0, 2)
                    return {
                        "count": len(dt_s),
                        "min_date": min_dt.isoformat(),
                        "max_date": max_dt.isoformat(),
                        "range_days": range_days,
                        "unique_dates": int(dt_s.nunique()),
                    }
            except Exception:
                pass
        return {
            "count": 0, "min_date": None, "max_date": None,
            "range_days": None, "unique_dates": 0,
        }

    return {}


# ============================================================
# MAIN PUBLIC API
# ============================================================

def build_data_dictionary(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    dataset_name: Optional[str] = None,
    max_columns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construct a deterministic AI Data Dictionary / Column Intelligence package.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Pre-built Evidence dict from engine.evidence.build_evidence, raw DataFrame, or None.
    dataset_name : Optional[str]
        Optional name or label for the dataset.
    max_columns : Optional[int]
        Maximum number of columns to return in the 'columns' list. If None, returns all.

    Returns:
    --------
    Dict[str, Any]
        JSON-serializable data dictionary package containing:
        - dataset_name
        - column_count
        - returned_column_count
        - truncated
        - columns
        - summary
        - generated_from
    """
    # 1. Resolve Evidence Object and Source
    raw_df: Optional[pd.DataFrame] = None

    if isinstance(evidence_or_df, dict) and "structure" in evidence_or_df:
        evidence = evidence_or_df
        source = "evidence"
        name = dataset_name or evidence.get("metadata", {}).get("dataset_name", "unnamed_dataset")
    elif isinstance(evidence_or_df, pd.DataFrame):
        raw_df = evidence_or_df
        # Check if df contains complex columns or unsupported types that cause numpy quantile/describe to fail
        has_complex = any(pd.api.types.is_complex_dtype(evidence_or_df[c]) for c in evidence_or_df.columns)
        if has_complex:
            safe_df = evidence_or_df.copy(deep=False)
            for c in evidence_or_df.columns:
                if pd.api.types.is_complex_dtype(evidence_or_df[c]):
                    safe_df[c] = evidence_or_df[c].astype(str)
            evidence = build_evidence(safe_df, dataset_name=dataset_name)
        else:
            evidence = build_evidence(raw_df, dataset_name=dataset_name)
        source = "dataframe"
        name = dataset_name or evidence.get("metadata", {}).get("dataset_name", "unnamed_dataset")
    elif evidence_or_df is None:
        evidence = build_evidence(None, dataset_name=dataset_name)
        source = "none"
        name = dataset_name or "unnamed_dataset"
    else:
        evidence = build_evidence(None, dataset_name=dataset_name)
        source = "none"
        name = dataset_name or "unnamed_dataset"

    structure = evidence.get("structure", {})
    row_count = structure.get("row_count", 0)
    evidence_cols = evidence.get("columns", [])
    outlier_findings = evidence.get("quality", {}).get("outlier_findings", [])

    # 2. Build Column Intelligence Entries
    columns_intelligence: List[Dict[str, Any]] = []

    for col_info in evidence_cols:
        col_name = str(col_info.get("name", ""))

        # Type detection
        detected_type, type_conf = _detect_column_type(col_name, col_info, df=raw_df)

        # Semantic role
        semantic_role, role_conf = _determine_semantic_role(col_name, detected_type, col_info, row_count)

        # Combined confidence
        confidence = round(float(role_conf), 2)

        # Missingness & Uniqueness
        missing_dict = {
            "count": int(col_info.get("missing_count", 0)),
            "percentage": float(round(col_info.get("missing_percentage", 0.0), 2)),
        }
        unique_dict = {
            "count": int(col_info.get("unique_count", 0)),
            "percentage": float(round(col_info.get("unique_percentage", 0.0), 2)),
        }

        # Statistics
        stats = _extract_column_statistics(col_name, detected_type, evidence, df=raw_df)

        # Cautious interpretation
        possible_meaning = _generate_possible_meaning(col_name, detected_type, semantic_role, col_info)

        # Quality concerns
        quality_concerns = _collect_quality_concerns(
            col_name, detected_type, semantic_role, col_info, stats, outlier_findings, row_count
        )

        # Analytical & ML usefulness
        analytical_usefulness = _determine_analytical_usefulness(
            detected_type, semantic_role, stats, row_count
        )
        ml_usefulness = _determine_ml_usefulness(
            detected_type, semantic_role, col_info, row_count
        )

        column_entry = {
            "name": col_name,
            "detected_type": detected_type,
            "semantic_role": semantic_role,
            "confidence": confidence,
            "missing": missing_dict,
            "unique": unique_dict,
            "statistics": stats,
            "possible_meaning": possible_meaning,
            "quality_concerns": quality_concerns,
            "analytical_usefulness": analytical_usefulness,
            "ml_usefulness": ml_usefulness,
            "is_target_candidate": bool(col_info.get("is_target_candidate", False)),
            "is_constant": bool(col_info.get("is_constant", False)),
            "is_all_missing": bool(col_info.get("is_all_missing", False)),
        }

        columns_intelligence.append(column_entry)

    # 3. Truncate by max_columns if specified
    total_columns_count = len(columns_intelligence)
    if max_columns is not None and max_columns > 0:
        returned_columns = columns_intelligence[:max_columns]
        is_truncated = len(returned_columns) < total_columns_count
    else:
        returned_columns = columns_intelligence
        is_truncated = False

    # 4. Construct Dataset Summary
    summary = {
        "total_columns": total_columns_count,
        "numeric_columns": sum(1 for c in columns_intelligence if c["detected_type"] == "numerical"),
        "categorical_columns": sum(1 for c in columns_intelligence if c["detected_type"] == "categorical"),
        "datetime_columns": sum(1 for c in columns_intelligence if c["detected_type"] == "datetime"),
        "boolean_columns": sum(1 for c in columns_intelligence if c["detected_type"] == "boolean"),
        "columns_with_missing": sum(1 for c in columns_intelligence if c["missing"]["count"] > 0),
        "high_cardinality_columns": sum(1 for c in columns_intelligence if c["semantic_role"] == "high cardinality"),
        "constant_columns": sum(1 for c in columns_intelligence if c["semantic_role"] == "constant"),
        "all_missing_columns": sum(1 for c in columns_intelligence if c["semantic_role"] == "all missing"),
        "identifier_like_columns": sum(1 for c in columns_intelligence if c["semantic_role"] == "identifier"),
        "target_candidates": sum(
            1 for c in columns_intelligence
            if c["semantic_role"] == "target candidate" or c.get("is_target_candidate", False)
        ),
    }

    # 5. Assemble Final Payload
    output = {
        "dataset_name": name,
        "column_count": total_columns_count,
        "returned_column_count": len(returned_columns),
        "truncated": is_truncated,
        "columns": returned_columns,
        "summary": summary,
        "generated_from": source,
    }

    return _to_serializable(output)

