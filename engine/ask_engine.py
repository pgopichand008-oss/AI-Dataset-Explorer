"""
FILE: engine/ask_engine.py

Ask-the-Dataset Backend Engine.
Answers natural-language questions about datasets using deterministic calculations
from Pandas, NumPy, scikit-learn, and the existing Phase 1-10 intelligence engines.

Pipeline:
USER QUESTION -> NORMALIZATION -> INTENT DETECTION -> COLUMN IDENTIFICATION ->
ACTUAL CALCULATION -> EVIDENCE -> ANSWER -> (OPTIONAL) GEMINI EXPLANATION

Core Principles:
1. Deterministic Truth: Pandas, NumPy, and existing engines provide the factual answer.
   Gemini may help explain or clarify, but is NEVER the source of numerical truth.
2. Anti-Fabrication: Never invent statistics, correlations, counts, or column names.
3. Input Immutability: Never modify the caller's DataFrame or Evidence dictionary.
4. JSON Safety: All outputs are 100% JSON-serializable.
5. Zero Leakage: No tracebacks, internal file paths, localhost URLs, or API keys
   are ever returned or exposed in user-facing messages.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story
from engine.recommendation_engine import recommend_next_analysis
from engine.anomaly_engine import investigate_anomalies
from engine.dictionary_engine import build_data_dictionary
from engine.reconsideration import reconsider_dataset

try:
    from config import get_gemini_client, get_gemini_model
except ImportError:
    def get_gemini_client():
        return None

    def get_gemini_model():
        return "gemini-3.6-flash"


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

    if hasattr(val, "isoformat"):
        try:
            return val.isoformat()
        except Exception:
            pass

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
# NORMALIZATION & MATCHING HELPERS
# ============================================================

def _normalize_question(text: Any) -> str:
    """Normalize question text safely."""
    if not isinstance(text, str):
        return ""
    s = text.strip().lower()
    return " ".join(s.split())


def _clean_token_string(text: str) -> str:
    """Normalize string into space-separated alphanumeric tokens."""
    t = text.strip().lower()
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _identify_columns(
    question_norm: str,
    available_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Match candidate column names against the normalized question.
    Returns (matched_columns, ambiguous_candidates).
    """
    if not question_norm or not available_columns:
        return [], []

    q_clean = _clean_token_string(question_norm)
    matched: List[str] = []

    # Map cleaned token strings to actual column names
    col_clean_map: Dict[str, str] = {}
    for c in available_columns:
        c_clean = _clean_token_string(str(c))
        if c_clean:
            col_clean_map[c_clean] = str(c)

    # Check matches sorted by length descending so longer column names take precedence
    sorted_cleans = sorted(col_clean_map.keys(), key=lambda x: len(x), reverse=True)
    matched_cleans: Set[str] = set()

    for c_clean in sorted_cleans:
        pattern = r"(?:^|\s)" + re.escape(c_clean) + r"(?:$|\s|[?!.,])"
        if re.search(pattern, q_clean):
            if not any(c_clean in m for m in matched_cleans):
                matched.append(col_clean_map[c_clean])
                matched_cleans.add(c_clean)

    return matched, []


def _has_word(query: str, word: str) -> bool:
    """Check if a word exists with word boundaries in query."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, query))


# ============================================================
# DETERMINISTIC CALCULATION HANDLERS
# ============================================================

def _handle_dataset_size(
    q: str,
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    struct = evidence.get("structure", {})
    rows = struct.get("row_count")
    cols = struct.get("column_count")
    if rows is None and df is not None:
        rows = len(df)
    if cols is None and df is not None:
        cols = len(df.columns)
    rows = rows or 0
    cols = cols or 0

    if "column" in q and "row" not in q:
        ans = f"The dataset has {cols} column(s)."
        op = "column_count"
        val = cols
    elif "row" in q or "record" in q or "observation" in q:
        ans = f"The dataset has {rows:,} row(s)."
        op = "row_count"
        val = rows
    else:
        ans = f"The dataset has {rows:,} row(s) and {cols} column(s)."
        op = "dimensions"
        val = {"rows": rows, "columns": cols}

    return {
        "status": "answered",
        "intent": {"name": "DATASET_SIZE", "confidence": 0.98},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": op},
        "result": {"value": val},
        "answer": ans,
        "evidence": {"row_count": rows, "column_count": cols},
        "interpretation": "Factual tabular dimensions derived from dataset profiling.",
    }


def _handle_duplicates(
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    dup_section = evidence.get("duplicates", {})
    dup_rows = dup_section.get("duplicate_rows")
    dup_pct = dup_section.get("duplicate_row_percentage")

    if dup_rows is None and df is not None:
        dup_rows = int(df.duplicated().sum())
        dup_pct = round(float((dup_rows / len(df)) * 100), 2) if len(df) > 0 else 0.0

    dup_rows = dup_rows or 0
    dup_pct = dup_pct or 0.0

    if dup_rows == 0:
        ans = "There are 0 duplicate rows in the dataset (0.0%)."
    else:
        ans = f"There are {dup_rows:,} duplicate row(s) in the dataset ({dup_pct:.1f}%)."

    return {
        "status": "answered",
        "intent": {"name": "DUPLICATES", "confidence": 0.98},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "duplicate_count"},
        "result": {"value": dup_rows, "percentage": dup_pct},
        "answer": ans,
        "evidence": {"duplicate_rows": dup_rows, "duplicate_row_percentage": dup_pct},
        "interpretation": "Row-level duplicate observation count and percentage.",
    }


def _handle_missing_values(
    q: str,
    matched_cols: List[str],
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    comp = evidence.get("completeness", {})
    col_missing_pcts = comp.get("column_missing_percentages", {})
    col_missing_counts = comp.get("column_missing_counts", {})
    row_count = evidence.get("structure", {}).get("row_count", len(df) if df is not None else 0)

    # 1. Question about a specific column's missingness
    if matched_cols:
        col = matched_cols[0]
        cnt = col_missing_counts.get(col)
        pct = col_missing_pcts.get(col)
        if cnt is None and df is not None and col in df.columns:
            cnt = int(df[col].isna().sum())
            pct = round(float((cnt / len(df)) * 100), 2) if len(df) > 0 else 0.0
        cnt = cnt or 0
        pct = pct or 0.0

        ans = f"Column '{col}' has {cnt} missing value(s) ({pct:.1f}%)."
        return {
            "status": "answered",
            "intent": {"name": "MISSING_VALUES", "confidence": 0.95},
            "entities": {"columns": [col], "values": []},
            "calculation": {"operation": "missing_percentage", "column": col},
            "result": {"missing_count": cnt, "missing_percentage": pct},
            "answer": ans,
            "evidence": {"column": col, "missing_count": cnt, "missing_percentage": pct, "row_count": row_count},
            "interpretation": f"Missing value rate for column '{col}'.",
        }

    # 2. Question about the highest / most missing column
    if any(w in q for w in ["highest", "most", "maximum", "max", "greatest"]):
        if col_missing_pcts:
            top_col = max(col_missing_pcts.items(), key=lambda x: x[1])[0]
            top_pct = col_missing_pcts[top_col]
            top_cnt = col_missing_counts.get(top_col, 0)
            ans = f"Column '{top_col}' has the highest missingness with {top_cnt} missing value(s) ({top_pct:.1f}%)."
            return {
                "status": "answered",
                "intent": {"name": "MISSING_VALUES", "confidence": 0.96},
                "entities": {"columns": [top_col], "values": []},
                "calculation": {"operation": "highest_missing_column", "column": top_col},
                "result": {"column": top_col, "missing_count": top_cnt, "missing_percentage": top_pct},
                "answer": ans,
                "evidence": {"column": top_col, "missing_count": top_cnt, "missing_percentage": top_pct},
                "interpretation": "Column with the largest fraction of missing values.",
            }

    # 3. General missing values question across columns
    cols_with_missing = [col for col, pct in col_missing_pcts.items() if pct > 0]
    total_missing_cells = comp.get("missing_cells", 0)
    total_pct = comp.get("missing_percentage", 0.0)

    if cols_with_missing:
        ans = (
            f"The dataset contains {total_missing_cells} missing cell(s) ({total_pct:.1f}% overall). "
            f"Columns with missing values: {', '.join(cols_with_missing)}."
        )
    else:
        ans = "No missing values were detected in any column across the dataset."

    return {
        "status": "answered",
        "intent": {"name": "MISSING_VALUES", "confidence": 0.95},
        "entities": {"columns": cols_with_missing, "values": []},
        "calculation": {"operation": "missing_summary"},
        "result": {
            "total_missing_cells": total_missing_cells,
            "overall_missing_percentage": total_pct,
            "columns_with_missing": cols_with_missing,
        },
        "answer": ans,
        "evidence": {"columns_with_missing": cols_with_missing, "total_missing_cells": total_missing_cells},
        "interpretation": "Overview of missingness across all features.",
    }


def _handle_numeric_summary(
    q: str,
    matched_cols: List[str],
    available_numeric_cols: List[str],
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    # Determine operation
    op = "mean"
    if _has_word(q, "median"):
        op = "median"
    elif any(_has_word(q, w) for w in ["minimum", "min", "lowest", "smallest"]):
        op = "min"
    elif any(_has_word(q, w) for w in ["maximum", "max", "highest", "largest"]):
        op = "max"
    elif any(_has_word(q, w) for w in ["sum", "total"]):
        op = "sum"
    elif _has_word(q, "std") or "standard deviation" in q:
        op = "std"
    elif any(_has_word(q, w) for w in ["variance", "var"]):
        op = "variance"
    elif _has_word(q, "range"):
        op = "range"

    # Resolve column
    if not matched_cols:
        if len(available_numeric_cols) > 1:
            return {
                "status": "needs_clarification",
                "answer": f"Which numerical column would you like me to calculate the {op} for?",
                "candidates": available_numeric_cols,
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.8},
            }
        elif len(available_numeric_cols) == 1:
            col = available_numeric_cols[0]
        else:
            return {
                "status": "not_available",
                "answer": f"No numerical columns are available in this dataset to compute {op}.",
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.5},
            }
    else:
        col = matched_cols[0]

    # Verify column existence and check for all-missing values
    if df is not None and col in df.columns:
        if df[col].isna().all():
            return {
                "status": "not_available",
                "answer": f"Column '{col}' contains no valid numerical values to compute {op}.",
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.95},
                "entities": {"columns": [col], "values": []},
            }

    if col not in available_numeric_cols:
        return {
            "status": "unsupported",
            "answer": f"Column '{col}' is not numerical, so {op} cannot be computed.",
            "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.9},
            "entities": {"columns": [col], "values": []},
        }

    val = None
    valid_count = 0

    # Calculate via DataFrame if available
    if df is not None and col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        valid_count = len(series)
        if valid_count == 0:
            return {
                "status": "not_available",
                "answer": f"Column '{col}' contains no valid numerical values to compute {op}.",
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.95},
                "entities": {"columns": [col], "values": []},
            }

        if op == "mean":
            val = float(series.mean())
        elif op == "median":
            val = float(series.median())
        elif op == "min":
            val = float(series.min())
        elif op == "max":
            val = float(series.max())
        elif op == "sum":
            val = float(series.sum())
        elif op == "std":
            val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        elif op == "variance":
            val = float(series.var(ddof=1)) if len(series) > 1 else 0.0
        elif op == "range":
            val = float(series.max() - series.min())

    # Fallback to pre-computed Evidence
    elif "numerical_statistics" in evidence and col in evidence["numerical_statistics"]:
        stats = evidence["numerical_statistics"][col]
        valid_count = stats.get("count", 0)
        if valid_count == 0:
            return {
                "status": "not_available",
                "answer": f"Column '{col}' contains no valid numerical values to compute {op}.",
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.95},
                "entities": {"columns": [col], "values": []},
            }
        if op == "mean":
            val = stats.get("mean")
        elif op == "median":
            val = stats.get("50%") or stats.get("median")
        elif op == "min":
            val = stats.get("min")
        elif op == "max":
            val = stats.get("max")
        elif op == "std":
            val = stats.get("std")
        elif op == "variance":
            val = stats.get("variance")
        elif op == "range":
            if stats.get("max") is not None and stats.get("min") is not None:
                val = stats.get("max") - stats.get("min")

    if val is None:
        return {
            "status": "not_available",
            "answer": f"Could not compute {op} for column '{col}'.",
            "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.8},
        }

    val = round(val, 4)
    if isinstance(val, float) and val.is_integer():
        fmt_val = f"{int(val):,}"
    else:
        fmt_val = f"{val:,.2f}"

    ans = f"The {op} of '{col}' is {fmt_val} based on {valid_count} valid observation(s)."

    return {
        "status": "answered",
        "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.95},
        "entities": {"columns": [col], "values": []},
        "calculation": {"operation": op, "column": col},
        "result": {"value": val},
        "answer": ans,
        "evidence": {"column": col, "operation": op, "value": val, "valid_count": valid_count},
        "interpretation": f"Factual statistical {op} computed over available observations in '{col}'.",
    }


def _handle_categorical_distribution(
    q: str,
    matched_cols: List[str],
    available_cat_cols: List[str],
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    if not matched_cols:
        if len(available_cat_cols) > 1:
            return {
                "status": "needs_clarification",
                "answer": "Which categorical column would you like to inspect?",
                "candidates": available_cat_cols,
                "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.8},
            }
        elif len(available_cat_cols) == 1:
            col = available_cat_cols[0]
        else:
            return {
                "status": "not_available",
                "answer": "No categorical columns are available in this dataset to analyze categories.",
                "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.5},
            }
    else:
        col = matched_cols[0]

    is_unique_req = any(w in q for w in ["how many unique", "unique count", "number of unique", "distinct count"])

    # Calculate via DataFrame
    if df is not None and col in df.columns:
        series = df[col].dropna()
        valid_cnt = len(series)
        unique_cnt = series.nunique()
        val_counts = series.value_counts()

        if is_unique_req:
            ans = f"Column '{col}' has {unique_cnt} unique value(s) across {valid_cnt} valid observation(s)."
            return {
                "status": "answered",
                "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.95},
                "entities": {"columns": [col], "values": []},
                "calculation": {"operation": "unique_count", "column": col},
                "result": {"unique_count": unique_cnt, "valid_count": valid_cnt},
                "answer": ans,
                "evidence": {"column": col, "unique_count": unique_cnt, "valid_count": valid_cnt},
                "interpretation": f"Cardinality of distinct categories in '{col}'.",
            }

        top_cat = val_counts.index[0] if len(val_counts) > 0 else None
        top_cnt = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
        top_pct = round((top_cnt / valid_cnt) * 100, 2) if valid_cnt > 0 else 0.0

        dist = [
            {"category": str(k), "count": int(v), "percentage": round((v / valid_cnt) * 100, 2)}
            for k, v in val_counts.items()
        ]

        ans = f"The most common category in '{col}' is '{top_cat}' with {top_cnt} occurrence(s) ({top_pct:.1f}%)."

        return {
            "status": "answered",
            "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.95},
            "entities": {"columns": [col], "values": [top_cat]},
            "calculation": {"operation": "most_common", "column": col},
            "result": {"top_category": top_cat, "count": top_cnt, "percentage": top_pct, "distribution": dist[:5]},
            "answer": ans,
            "evidence": {"column": col, "top_category": top_cat, "top_frequency": top_cnt, "total_valid": valid_cnt},
            "interpretation": f"Empirical frequency distribution of categories in '{col}'.",
        }

    # Fallback via Evidence
    elif "categorical_statistics" in evidence and col in evidence["categorical_statistics"]:
        stats = evidence["categorical_statistics"][col]
        top_cat = stats.get("top_category")
        top_cnt = stats.get("top_frequency", 0)
        top_pct = stats.get("top_percentage", 0.0)
        unique_cnt = stats.get("unique_count", 0)

        if is_unique_req:
            ans = f"Column '{col}' has {unique_cnt} unique category value(s)."
            return {
                "status": "answered",
                "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.95},
                "entities": {"columns": [col], "values": []},
                "calculation": {"operation": "unique_count", "column": col},
                "result": {"unique_count": unique_cnt},
                "answer": ans,
                "evidence": {"column": col, "unique_count": unique_cnt},
                "interpretation": f"Unique category count for '{col}'.",
            }

        ans = f"The most common category in '{col}' is '{top_cat}' with {top_cnt} occurrence(s) ({top_pct:.1f}%)."
        return {
            "status": "answered",
            "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.95},
            "entities": {"columns": [col], "values": [top_cat]},
            "calculation": {"operation": "most_common", "column": col},
            "result": {"top_category": top_cat, "count": top_cnt, "percentage": top_pct},
            "answer": ans,
            "evidence": {"column": col, "top_category": top_cat, "top_frequency": top_cnt},
            "interpretation": f"Frequency distribution for '{col}'.",
        }

    return {
        "status": "not_available",
        "answer": f"Categorical statistics are not available for '{col}'.",
        "intent": {"name": "CATEGORICAL_DISTRIBUTION", "confidence": 0.7},
    }


def _handle_grouped_aggregation(
    q: str,
    matched_cols: List[str],
    available_cat_cols: List[str],
    available_num_cols: List[str],
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if df is None:
        return {
            "status": "not_available",
            "answer": "Grouped aggregation requires the raw dataset DataFrame, but only pre-computed evidence was provided.",
            "intent": {"name": "GROUPED_AGGREGATION", "confidence": 0.8},
        }

    group_col = None
    metric_col = None

    for c in matched_cols:
        if c in available_cat_cols and group_col is None:
            group_col = c
        elif c in available_num_cols and metric_col is None:
            metric_col = c

    if group_col is None and len(available_cat_cols) == 1:
        group_col = available_cat_cols[0]
    if metric_col is None and len(available_num_cols) == 1:
        metric_col = available_num_cols[0]

    if not group_col or not metric_col:
        candidates = []
        if not group_col:
            candidates.extend(available_cat_cols)
        if not metric_col:
            candidates.extend(available_num_cols)
        return {
            "status": "needs_clarification",
            "answer": "Please specify both the categorical group column and the numerical metric column for the grouped analysis.",
            "candidates": candidates,
            "intent": {"name": "GROUPED_AGGREGATION", "confidence": 0.75},
        }

    agg_name = "mean"
    if any(w in q for w in ["sum", "total"]):
        agg_name = "sum"
    elif any(w in q for w in ["minimum", "min", "lowest"]):
        agg_name = "min"
    elif any(w in q for w in ["maximum", "max", "highest", "largest"]) and "average" not in q and "mean" not in q:
        agg_name = "max"
    elif any(w in q for w in ["count"]):
        agg_name = "count"

    sub_df = df[[group_col, metric_col]].dropna()
    if len(sub_df) == 0:
        return {
            "status": "not_available",
            "answer": f"No overlapping valid records found between '{group_col}' and '{metric_col}'.",
            "intent": {"name": "GROUPED_AGGREGATION", "confidence": 0.9},
        }

    grouped = sub_df.groupby(group_col)[metric_col].agg(agg_name)
    if len(grouped) == 0:
        return {
            "status": "not_available",
            "answer": "Grouped aggregation returned no groups.",
            "intent": {"name": "GROUPED_AGGREGATION", "confidence": 0.9},
        }

    is_highest = not any(w in q for w in ["lowest", "min", "minimum"])
    top_group = grouped.idxmax() if is_highest else grouped.idxmin()
    top_val = float(round(grouped[top_group], 4))

    groups_res = [
        {"group": str(k), "value": float(round(v, 4))}
        for k, v in grouped.items()
    ]

    adj = "highest" if is_highest else "lowest"
    ans = f"'{top_group}' has the {adj} {agg_name} {metric_col} at {top_val:,.2f}."

    return {
        "status": "answered",
        "intent": {"name": "GROUPED_AGGREGATION", "confidence": 0.96},
        "entities": {"columns": [group_col, metric_col], "values": [str(top_group)]},
        "calculation": {
            "operation": f"grouped_{agg_name}",
            "group_by": group_col,
            "metric": metric_col,
            "aggregation": agg_name,
        },
        "result": {
            "top_group": str(top_group),
            "top_value": top_val,
            "groups": groups_res,
        },
        "answer": ans,
        "evidence": {
            "group_by": group_col,
            "metric": metric_col,
            "aggregation": agg_name,
            "groups_count": len(grouped),
        },
        "interpretation": f"Aggregated {agg_name} of '{metric_col}' grouped by '{group_col}'.",
    }


def _handle_correlation(
    matched_cols: List[str],
    available_num_cols: List[str],
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    if len(matched_cols) < 2:
        pairs = evidence.get("correlations", {}).get("strongest_pairs", [])
        if pairs:
            top_p = pairs[0]
            col1 = top_p.get("feature_a", top_p.get("attribute_1", "feature_1"))
            col2 = top_p.get("feature_b", top_p.get("attribute_2", "feature_2"))
            corr_val = top_p.get("correlation", 0.0)
            ans = f"The strongest correlation is between '{col1}' and '{col2}' with a Pearson coefficient of {corr_val:.2f}."
            return {
                "status": "answered",
                "intent": {"name": "CORRELATION", "confidence": 0.92},
                "entities": {"columns": [col1, col2], "values": []},
                "calculation": {"operation": "strongest_correlation"},
                "result": {"strongest_pairs": pairs[:5]},
                "answer": ans,
                "evidence": {"strongest_pair": top_p},
                "interpretation": "Strongest linear correlation pairs computed across numerical features.",
            }
        elif len(available_num_cols) < 2:
            return {
                "status": "not_available",
                "answer": "At least two numerical columns are required to calculate correlations.",
                "intent": {"name": "CORRELATION", "confidence": 0.9},
            }
        else:
            return {
                "status": "needs_clarification",
                "answer": "Which two numerical columns would you like to compute the correlation between?",
                "candidates": available_num_cols,
                "intent": {"name": "CORRELATION", "confidence": 0.8},
            }

    col1, col2 = matched_cols[0], matched_cols[1]
    if col1 not in available_num_cols or col2 not in available_num_cols:
        return {
            "status": "unsupported",
            "answer": f"Correlation requires numerical columns, but '{col1}' or '{col2}' is not numerical.",
            "intent": {"name": "CORRELATION", "confidence": 0.9},
        }

    corr = None
    valid_cnt = 0

    if df is not None and col1 in df.columns and col2 in df.columns:
        pair_df = df[[col1, col2]].dropna()
        valid_cnt = len(pair_df)
        if valid_cnt < 2:
            return {
                "status": "not_available",
                "answer": f"Insufficient overlapping data ({valid_cnt} pair(s)) to compute correlation between '{col1}' and '{col2}'.",
                "intent": {"name": "CORRELATION", "confidence": 0.95},
                "entities": {"columns": [col1, col2], "values": []},
            }
        c = pair_df[col1].corr(pair_df[col2])
        if pd.isna(c):
            corr = 0.0
        else:
            corr = round(float(c), 4)

    elif "correlations" in evidence:
        matrix = evidence["correlations"].get("matrix", {})
        if col1 in matrix and col2 in matrix[col1]:
            corr = matrix[col1][col2]

    if corr is None:
        return {
            "status": "not_available",
            "answer": f"Could not determine correlation between '{col1}' and '{col2}'.",
            "intent": {"name": "CORRELATION", "confidence": 0.8},
        }

    ans = f"The Pearson correlation between '{col1}' and '{col2}' is {corr:.2f}."

    return {
        "status": "answered",
        "intent": {"name": "CORRELATION", "confidence": 0.96},
        "entities": {"columns": [col1, col2], "values": []},
        "calculation": {"operation": "pearson_correlation", "columns": [col1, col2]},
        "result": {"column_a": col1, "column_b": col2, "correlation": corr, "valid_pair_count": valid_cnt},
        "answer": ans,
        "evidence": {"column_a": col1, "column_b": col2, "method": "pearson", "correlation": corr, "valid_pair_count": valid_cnt},
        "interpretation": f"Linear association metric between '{col1}' and '{col2}'. No causal relationship is implied.",
    }


def _handle_outliers(
    evidence: Dict[str, Any],
    anomalies: Optional[Dict[str, Any]],
    raw_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    anom_data = anomalies
    if anom_data is None:
        try:
            if raw_df is not None:
                anom_data = investigate_anomalies(raw_df)
            else:
                anom_data = investigate_anomalies(evidence)
        except Exception:
            anom_data = None

    findings = anom_data.get("findings", []) if isinstance(anom_data, dict) else []

    if findings:
        cols_with_outliers = list({f.get("column", "") for f in findings if f.get("column")})
        first = findings[0]
        cnt = first.get("anomaly_count", first.get("count", 0))
        pct = first.get("affected_percentage", first.get("percentage", 0.0))
        col = first.get("column", "")
        method = first.get("method", "statistical check")

        ans = (
            f"Potential anomalies were identified in {len(cols_with_outliers)} column(s): {', '.join(cols_with_outliers)}. "
            f"For example, '{col}' contains {cnt} potential anomaly observation(s) ({pct:.1f}%) detected via {method}."
        )
    else:
        ans = "No potential anomalies or extreme outliers were detected across the numerical features."
        cols_with_outliers = []

    return {
        "status": "answered",
        "intent": {"name": "OUTLIERS", "confidence": 0.95},
        "entities": {"columns": cols_with_outliers, "values": []},
        "calculation": {"operation": "outlier_detection"},
        "result": {"columns": cols_with_outliers, "findings": findings[:5]},
        "answer": ans,
        "evidence": {"findings": findings[:5]},
        "interpretation": "Statistical anomaly detection results. Flagged values represent potential outliers rather than confirmed errors.",
    }


def _handle_health(
    health: Optional[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    hl = health
    if hl is None:
        try:
            hl = calculate_health(evidence)
        except Exception:
            hl = None

    if isinstance(hl, dict):
        overall = hl.get("overall", {}) if isinstance(hl.get("overall"), dict) else {}
        score = overall.get("score", hl.get("health_score"))
        status = overall.get("status", hl.get("health_status", "UNKNOWN"))
        strengths = hl.get("strengths", [])
        risks = hl.get("risks", [])

        score_str = f"{score:.1f}/100" if score is not None else "N/A"
        ans = f"The dataset health status is {status} with an overall health score of {score_str}."

        return {
            "status": "answered",
            "intent": {"name": "HEALTH", "confidence": 0.98},
            "entities": {"columns": [], "values": []},
            "calculation": {"operation": "health_assessment"},
            "result": {"score": score, "status": status, "strengths": strengths, "risks": risks},
            "answer": ans,
            "evidence": {"score": score, "status": status, "risks": risks},
            "interpretation": f"Automated composite health evaluation categorized as {status}.",
        }

    return {
        "status": "not_available",
        "answer": "Health assessment metrics could not be calculated.",
        "intent": {"name": "HEALTH", "confidence": 0.7},
    }


def _handle_ml_readiness(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    ml = evidence.get("ml_readiness", {})
    score = ml.get("ml_score")
    status = ml.get("ml_status", "UNKNOWN")
    can_train = ml.get("can_train", False)
    warnings_list = ml.get("warnings", [])
    candidates = ml.get("target_candidates", [])

    score_str = f"{score:.1f}/100" if score is not None else "N/A"
    ready_phrase = "ready for initial machine learning modeling" if can_train else "blocked from reliable machine learning training"
    ans = f"The dataset is {ready_phrase} (ML Readiness Score: {score_str}, status: {status})."

    return {
        "status": "answered",
        "intent": {"name": "ML_READINESS", "confidence": 0.95},
        "entities": {"columns": candidates, "values": []},
        "calculation": {"operation": "ml_readiness_assessment"},
        "result": {"score": score, "status": status, "can_train": can_train, "warnings": warnings_list},
        "answer": ans,
        "evidence": {"ml_score": score, "ml_status": status, "can_train": can_train, "warnings": warnings_list},
        "interpretation": "Evaluation of dataset readiness for predictive modeling tasks.",
    }


def _handle_data_quality(
    evidence: Dict[str, Any],
    health: Optional[Dict[str, Any]],
    prioritized_insights: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Resolve prioritized insights if not provided
    if prioritized_insights is None:
        try:
            prioritized_insights = build_prioritized_insights(evidence, health=health)
        except Exception:
            prioritized_insights = None

    issues = []
    if isinstance(prioritized_insights, dict):
        for ins in prioritized_insights.get("insights", [])[:5]:
            issues.append(f"[{ins.get('severity', 'INFO')}] {ins.get('title', '')}")
    elif isinstance(health, dict):
        issues.extend(health.get("risks", []))

    if not issues and "quality" in evidence:
        q_score = evidence["quality"].get("quality_score")
        issues.append(f"Quality score evaluated at {q_score}/100 with no severe defects flagged.")

    if issues:
        ans = f"Key data quality considerations: {'; '.join(issues[:3])}."
    else:
        ans = "No major data quality defects were identified."

    return {
        "status": "answered",
        "intent": {"name": "DATA_QUALITY", "confidence": 0.95},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "data_quality_review"},
        "result": {"issues": issues},
        "answer": ans,
        "evidence": {"issues": issues},
        "interpretation": "Consolidated data quality inspection findings.",
    }


def _handle_column_intelligence(
    matched_cols: List[str],
    available_cols: List[str],
    df: Optional[pd.DataFrame],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    if not matched_cols:
        return {
            "status": "needs_clarification",
            "answer": "Which column would you like to inspect?",
            "candidates": available_cols,
            "intent": {"name": "COLUMN_INTELLIGENCE", "confidence": 0.8},
        }

    col = matched_cols[0]
    try:
        dict_data = build_data_dictionary(evidence, dataset_name="dataset")
        cols_dict = {c.get("name"): c for c in dict_data.get("columns", []) if isinstance(c, dict)}
        col_entry = cols_dict.get(col, {})
    except Exception:
        col_entry = {}

    dtype = col_entry.get("detected_type", col_entry.get("data_type", "unknown"))
    role = col_entry.get("semantic_role", "Attribute")
    conf = col_entry.get("confidence", 0.0)
    missing_pct = col_entry.get("missing", {}).get("percentage", col_entry.get("missing_percentage", 0.0))
    unique_cnt = col_entry.get("unique", {}).get("count", col_entry.get("unique_count", 0))

    role_title = str(role).title()

    ans = (
        f"Column '{col}' is classified as {role_title} (detected type: {dtype}, confidence: {conf*100:.0f}%). "
        f"It contains {unique_cnt} unique values with {missing_pct:.1f}% missingness."
    )

    return {
        "status": "answered",
        "intent": {"name": "COLUMN_INTELLIGENCE", "confidence": 0.95},
        "entities": {"columns": [col], "values": []},
        "calculation": {"operation": "column_profiling", "column": col},
        "result": col_entry,
        "answer": ans,
        "evidence": col_entry,
        "interpretation": f"Automated semantic and structural dictionary intelligence for '{col}'.",
    }


def _handle_patterns(
    story: Optional[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    st = story
    if st is None:
        try:
            st = build_data_story(evidence)
        except Exception:
            st = None

    patterns = st.get("patterns", []) if isinstance(st, dict) else []
    if patterns:
        ans = f"Prominent patterns observed: {'; '.join(patterns[:3])}."
    else:
        ans = "No strong linear correlations or severe categorical distribution imbalances were observed."

    return {
        "status": "answered",
        "intent": {"name": "PATTERNS", "confidence": 0.94},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "pattern_extraction"},
        "result": {"patterns": patterns},
        "answer": ans,
        "evidence": {"patterns": patterns},
        "interpretation": "Empirical patterns extracted from correlation and distribution analyses without causal claims.",
    }


def _handle_changes(
    reconsideration: Optional[Dict[str, Any]],
    workflow_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cs = None
    if isinstance(reconsideration, dict):
        cs = reconsideration.get("change_summary")
    elif isinstance(workflow_state, dict):
        cs = workflow_state.get("stages", {}).get("COMPARE", {}).get("result")

    if cs is None:
        return {
            "status": "not_available",
            "answer": "No previous dataset is available for comparison.",
            "intent": {"name": "CHANGES", "confidence": 0.95},
        }

    has_changes = cs.get("has_changes", False)
    if not has_changes:
        ans = "No structural, quality, or statistical changes were detected between the datasets."
    else:
        ans = "Dataset changes were detected across structural and statistical dimensions."

    return {
        "status": "answered",
        "intent": {"name": "CHANGES", "confidence": 0.96},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "change_detection"},
        "result": cs,
        "answer": ans,
        "evidence": cs,
        "interpretation": "Comparison results between baseline and current dataset versions.",
    }


def _handle_reconsideration(
    reconsideration: Optional[Dict[str, Any]],
    workflow_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    recons = None
    if isinstance(reconsideration, dict):
        recons = reconsideration.get("reconsiderations")
    elif isinstance(workflow_state, dict):
        recons = workflow_state.get("stages", {}).get("RECONSIDER", {}).get("result", {}).get("reconsiderations")

    if recons is None:
        return {
            "status": "not_available",
            "answer": "No previous findings are available for reconsideration.",
            "intent": {"name": "RECONSIDERATION", "confidence": 0.95},
        }

    if recons:
        r_lines = [f"{r.get('title')}: {r.get('classification')}" for r in recons[:3]]
        ans = f"Previous finding classifications: {'; '.join(r_lines)}."
    else:
        ans = "No previous findings required re-classification."

    return {
        "status": "answered",
        "intent": {"name": "RECONSIDERATION", "confidence": 0.96},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "findings_reconsideration"},
        "result": {"reconsiderations": recons},
        "answer": ans,
        "evidence": {"reconsiderations": recons},
        "interpretation": "Formal classification of prior conclusions under updated evidence.",
    }


def _handle_next_analysis(
    recommendations: Optional[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    recs = recommendations
    if recs is None:
        try:
            recs = recommend_next_analysis(evidence)
        except Exception:
            recs = None

    rec_list = []
    if isinstance(recs, dict):
        rec_list = recs.get("recommendations", [])
    elif isinstance(recs, list):
        rec_list = recs

    if rec_list:
        top_rec = rec_list[0]
        name = top_rec.get("name", "Deep dive profiling")
        why = top_rec.get("why", "")
        pri = top_rec.get("priority", "HIGH")
        ans = f"Top recommended next analysis: '{name}' (Priority: {pri}). {why}"
    else:
        ans = "Recommended next step: Proceed with exploratory data analysis and feature distribution review."

    return {
        "status": "answered",
        "intent": {"name": "NEXT_ANALYSIS", "confidence": 0.96},
        "entities": {"columns": [], "values": []},
        "calculation": {"operation": "next_best_analysis"},
        "result": {"recommendations": rec_list[:3]},
        "answer": ans,
        "evidence": {"recommendations": rec_list[:3]},
        "interpretation": "Prioritized sequence of next analytical steps grounded in evidence.",
    }


# ============================================================
# GEMINI INTENT & EXPLANATION HELPERS
# ============================================================

def _call_gemini_intent_classification(
    question: str,
    available_columns: List[str],
) -> Optional[Dict[str, Any]]:
    client = get_gemini_client()
    if client is None:
        return None

    model = get_gemini_model()
    prompt = (
        "You are an analytical intent classifier.\n"
        f"User question: {question}\n"
        f"Available dataset columns: {available_columns}\n\n"
        "Return ONLY a strict JSON object with this format:\n"
        "{\n"
        '  "intent": "DATASET_SIZE | MISSING_VALUES | DUPLICATES | NUMERIC_SUMMARY | CATEGORICAL_DISTRIBUTION | GROUPED_AGGREGATION | CORRELATION | OUTLIERS | HEALTH | ML_READINESS | DATA_QUALITY | COLUMN_INTELLIGENCE | PATTERNS | CHANGES | RECONSIDERATION | NEXT_ANALYSIS | UNKNOWN",\n'
        '  "columns": ["col1"],\n'
        '  "group_by": null,\n'
        '  "aggregation": "mean | sum | min | max | median | std | variance | count | null",\n'
        '  "confidence": 0.9\n'
        "}\n"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", "")
        if not text:
            return None
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```[a-zA-Z]*\n", "", clean_text)
            clean_text = re.sub(r"\n```$", "", clean_text).strip()

        data = json.loads(clean_text)
        if not isinstance(data, dict):
            return None

        # STRICT VALIDATION: Reject any unknown column
        returned_cols = data.get("columns", [])
        if isinstance(returned_cols, list):
            for rc in returned_cols:
                if rc not in available_columns:
                    return None
        else:
            data["columns"] = []

        gb = data.get("group_by")
        if gb and gb not in available_columns:
            return None

        # Validate aggregation
        valid_aggs = {"mean", "sum", "min", "max", "median", "std", "variance", "count", None}
        if data.get("aggregation") not in valid_aggs:
            return None

        # Validate intent
        valid_intents = {
            "DATASET_SIZE", "MISSING_VALUES", "DUPLICATES", "NUMERIC_SUMMARY",
            "CATEGORICAL_DISTRIBUTION", "GROUPED_AGGREGATION", "CORRELATION",
            "OUTLIERS", "HEALTH", "ML_READINESS", "DATA_QUALITY",
            "COLUMN_INTELLIGENCE", "PATTERNS", "CHANGES", "RECONSIDERATION",
            "NEXT_ANALYSIS", "UNKNOWN"
        }
        if data.get("intent") not in valid_intents:
            return None

        return data
    except Exception:
        return None


# ============================================================
# MAIN PUBLIC API
# ============================================================

def ask_dataset(
    question: Any,
    dataset: Optional[pd.DataFrame] = None,
    evidence: Optional[Dict[str, Any]] = None,
    health: Optional[Dict[str, Any]] = None,
    prioritized_insights: Optional[Dict[str, Any]] = None,
    story: Optional[Dict[str, Any]] = None,
    anomalies: Optional[Dict[str, Any]] = None,
    reconsideration: Optional[Dict[str, Any]] = None,
    recommendations: Optional[Dict[str, Any]] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    use_gemini: bool = False,
) -> Dict[str, Any]:
    """
    Ask a natural language question about the dataset and receive an authoritative,
    evidence-grounded answer.
    """
    warnings: List[str] = []
    generated_from: List[str] = ["engine.ask_engine"]

    # 1. QUESTION NORMALIZATION & VALIDATION
    if not isinstance(question, str) or not question.strip():
        return _to_serializable({
            "status": "invalid_question",
            "question": str(question) if question is not None else "",
            "normalized_question": "",
            "intent": {"name": "UNKNOWN", "confidence": 0.0},
            "entities": {"columns": [], "values": []},
            "calculation": {},
            "result": {},
            "answer": "Please provide a valid, non-empty question as a string.",
            "evidence": {},
            "interpretation": "",
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": ["Question was empty, non-string, or whitespace only."],
            "generated_from": generated_from,
        })

    norm_q = _normalize_question(question)

    # 2. RESOLVE DATASET & EVIDENCE SAFELY (Never mutate)
    raw_df: Optional[pd.DataFrame] = None
    if isinstance(dataset, pd.DataFrame):
        raw_df = dataset

    if isinstance(workflow_state, dict):
        generated_from.append("engine.workflow")
        det_res = workflow_state.get("stages", {}).get("DETECT", {}).get("result", {}) or {}
        if health is None:
            health = workflow_state.get("health") or (workflow_state.get("stages", {}).get("ANALYZE", {}).get("result", {}) or {}).get("health")
        if prioritized_insights is None:
            prioritized_insights = workflow_state.get("priorities") or (workflow_state.get("stages", {}).get("ANALYZE", {}).get("result", {}) or {}).get("priorities")
        if story is None:
            story = workflow_state.get("story") or (workflow_state.get("stages", {}).get("ANALYZE", {}).get("result", {}) or {}).get("story")
        if anomalies is None:
            anomalies = workflow_state.get("anomalies") or (workflow_state.get("stages", {}).get("ANALYZE", {}).get("result", {}) or {}).get("anomalies")
        if reconsideration is None:
            reconsideration = workflow_state.get("reconsideration") or (workflow_state.get("stages", {}).get("RECONSIDER", {}).get("result", {}) or {})
        if recommendations is None:
            recommendations = workflow_state.get("next_analysis") or (workflow_state.get("stages", {}).get("RE-PLAN", {}).get("result", {}) or {}).get("next_analyses")

        if evidence is None and det_res:
            evidence = {
                "structure": {
                    "dataset_name": workflow_state.get("dataset_name", "dataset"),
                    "row_count": det_res.get("rows", det_res.get("row_count", 0)),
                    "column_count": det_res.get("columns", det_res.get("column_count", 0)),
                    "column_names": det_res.get("column_names", []),
                    "numerical_columns": det_res.get("numeric_columns", det_res.get("numerical_columns", [])),
                    "categorical_columns": det_res.get("categorical_columns", []),
                    "datetime_columns": det_res.get("datetime_columns", []),
                },
                "completeness": {"missing_cells": det_res.get("missing_cells", 0), "column_missing_percentages": {}, "column_missing_counts": {}},
                "duplicates": {"duplicate_rows": det_res.get("duplicate_rows", 0), "duplicate_row_percentage": det_res.get("duplicate_row_percentage", 0.0)},
            }

    if evidence is None and raw_df is not None:
        evidence = build_evidence(raw_df, dataset_name=dataset_name or "dataset")
        generated_from.append("engine.evidence")
    elif isinstance(evidence, dict):
        evidence = copy.deepcopy(evidence)
        generated_from.append("engine.evidence")

    if evidence is None and raw_df is None:
        return _to_serializable({
            "status": "not_available",
            "question": question,
            "normalized_question": norm_q,
            "intent": {"name": "UNKNOWN", "confidence": 0.0},
            "entities": {"columns": [], "values": []},
            "calculation": {},
            "result": {},
            "answer": "No dataset or evidence was provided to answer questions.",
            "evidence": {},
            "interpretation": "",
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": ["No dataset or evidence input."],
            "generated_from": generated_from,
        })

    struct = evidence.get("structure", {}) if isinstance(evidence, dict) else {}
    avail_cols = list(
        struct.get("column_names")
        or (list(raw_df.columns) if raw_df is not None else [])
    )
    avail_num_cols = list(
        struct.get("numerical_columns")
        or struct.get("numeric_columns")
        or (list(raw_df.select_dtypes(include=[np.number]).columns) if raw_df is not None else [])
    )
    avail_cat_cols = list(
        struct.get("categorical_columns")
        or (list(raw_df.select_dtypes(include=["object", "category"]).columns) if raw_df is not None else [])
    )
    avail_dt_cols = list(
        struct.get("datetime_columns")
        or (list(raw_df.select_dtypes(include=["datetime"]).columns) if raw_df is not None else [])
    )

    # 3. ENTITY & COLUMN IDENTIFICATION
    matched_cols, _ = _identify_columns(norm_q, avail_cols)

    # Missing column check (user explicitly asking for an un-found column)
    if any(w in norm_q for w in ["average ", "mean ", "median ", "minimum of ", "maximum of ", "sum of "]) and not matched_cols:
        target_candidate = None
        for prefix in ["average ", "mean ", "median ", "minimum of ", "maximum of ", "sum of "]:
            if prefix in norm_q:
                target_candidate = norm_q.split(prefix)[-1].strip(" ?.,")
                break
        if target_candidate and target_candidate not in {"it", "them", "all", "this", "values"} and target_candidate not in avail_cols:
            return _to_serializable({
                "status": "unsupported",
                "question": question,
                "normalized_question": norm_q,
                "intent": {"name": "NUMERIC_SUMMARY", "confidence": 0.5},
                "entities": {"columns": [target_candidate], "values": []},
                "calculation": {},
                "result": {},
                "answer": f"Column '{target_candidate}' was not found in the dataset. Available columns are: {', '.join(avail_cols)}.",
                "evidence": {},
                "interpretation": "Referenced column does not exist in the dataset.",
                "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
                "warnings": [f"Column '{target_candidate}' not found."],
                "generated_from": generated_from,
            })

    # Optional Gemini Intent Resolution when Gemini is enabled and column matching was empty
    gemini_resolved = False
    if use_gemini and not matched_cols:
        gemini_intent = _call_gemini_intent_classification(question, avail_cols)
        if gemini_intent and isinstance(gemini_intent, dict):
            g_cols = gemini_intent.get("columns", [])
            if g_cols:
                matched_cols = g_cols
                gemini_resolved = True

    # 4. DETERMINISTIC INTENT DETECTION & CALCULATION

    # Grouped Aggregation check
    has_by = any(w in norm_q for w in [" by ", " per ", " for each "])
    has_which_group = any(w in norm_q for w in ["which ", "what "]) and any(w in norm_q for w in ["has the highest", "has highest", "has the lowest", "has lowest", "most common", "has the most", "has the least"])

    if has_by or (has_which_group and len(matched_cols) >= 2):
        ans_dict = _handle_grouped_aggregation(norm_q, matched_cols, avail_cat_cols, avail_num_cols, raw_df)
        if ans_dict["status"] in ["answered", "needs_clarification"]:
            generated_from.append("pandas.groupby")
            ans_dict.update({
                "question": question,
                "normalized_question": norm_q,
                "generation": {"mode": "gemini" if gemini_resolved else "deterministic", "gemini_used": gemini_resolved, "fallback": False},
                "warnings": warnings,
                "generated_from": generated_from,
            })
            return _to_serializable(ans_dict)

    # Dataset Size
    if any(w in norm_q for w in ["how many rows", "number of rows", "total rows", "row count", "how many records", "how many observations", "how many columns", "number of columns", "total columns", "column count", "how many features", "how big", "dataset size", "dimensions", "dataset shape", "shape of"]):
        ans_dict = _handle_dataset_size(norm_q, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Duplicates
    if any(w in norm_q for w in ["duplicate", "duplicates", "duplicated"]):
        ans_dict = _handle_duplicates(raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Missing Values
    if any(w in norm_q for w in ["missing", "null", "nan", "empty values", "completeness"]):
        ans_dict = _handle_missing_values(norm_q, matched_cols, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "gemini" if gemini_resolved else "deterministic", "gemini_used": gemini_resolved, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Correlation
    if any(w in norm_q for w in ["correlation", "correlations", "correlated", "relationship between", "collinearity"]):
        ans_dict = _handle_correlation(matched_cols, avail_num_cols, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Outliers / Anomalies
    if any(w in norm_q for w in ["outlier", "outliers", "anomaly", "anomalies", "unusual values"]):
        ans_dict = _handle_outliers(evidence, anomalies, raw_df)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Health
    if any(w in norm_q for w in ["healthy", "data health", "how healthy", "health score", "health status", "health of"]):
        ans_dict = _handle_health(health, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # ML Readiness
    if any(w in norm_q for w in ["machine learning", "ml ready", "ml readiness", "can i train", "train a model", "predictive model", "ready for ml"]):
        ans_dict = _handle_ml_readiness(evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Data Quality
    if any(w in norm_q for w in ["quality", "quality issues", "quality problems", "data defects"]):
        ans_dict = _handle_data_quality(evidence, health, prioritized_insights)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Column Intelligence / Dictionary
    if any(w in norm_q for w in ["what does", "tell me about", "what do you know about", "dictionary for", "column info", "represent"]):
        ans_dict = _handle_column_intelligence(matched_cols, avail_cols, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Patterns
    if any(w in norm_q for w in ["pattern", "patterns", "trend", "trends"]):
        ans_dict = _handle_patterns(story, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Changes
    if any(w in norm_q for w in ["what changed", "changes between", "differences between", "difference between", "comparison between old and new"]):
        ans_dict = _handle_changes(reconsideration, workflow_state)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Reconsideration
    if any(w in norm_q for w in ["previous findings", "still hold", "still valid", "reconsideration", "previous conclusions"]):
        ans_dict = _handle_reconsideration(reconsideration, workflow_state)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Next Analysis
    if any(w in norm_q for w in ["what should i analyze next", "next analysis", "what to analyze next", "next steps", "recommendations"]):
        ans_dict = _handle_next_analysis(recommendations, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Categorical Distribution (e.g. most common, unique count)
    if any(w in norm_q for w in ["most common", "most frequent", "top category", "occurs most", "frequencies", "distribution of", "unique count", "how many unique", "distinct"]):
        ans_dict = _handle_categorical_distribution(norm_q, matched_cols, avail_cat_cols, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "gemini" if gemini_resolved else "deterministic", "gemini_used": gemini_resolved, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # Datetime questions (checked before general numeric range)
    is_dt_col = any(c in avail_dt_cols for c in matched_cols)
    if is_dt_col or any(w in norm_q for w in ["date range", "time range", "earliest date", "latest date"]):
        dt_col = matched_cols[0] if is_dt_col else (avail_dt_cols[0] if avail_dt_cols else None)
        if dt_col and raw_df is not None and dt_col in raw_df.columns:
            s = pd.to_datetime(raw_df[dt_col], errors="coerce").dropna()
            if len(s) > 0:
                min_d = str(s.min())
                max_d = str(s.max())
                ans = f"Column '{dt_col}' spans from {min_d} to {max_d} across {len(s)} observations."
                return _to_serializable({
                    "status": "answered",
                    "question": question,
                    "normalized_question": norm_q,
                    "intent": {"name": "COLUMN_STATS", "confidence": 0.9},
                    "entities": {"columns": [dt_col], "values": []},
                    "calculation": {"operation": "datetime_range", "column": dt_col},
                    "result": {"min": min_d, "max": max_d, "count": len(s)},
                    "answer": ans,
                    "evidence": {"column": dt_col, "min": min_d, "max": max_d},
                    "interpretation": f"Temporal range of observation dates in '{dt_col}'.",
                    "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
                    "warnings": warnings,
                    "generated_from": generated_from,
                })

    # Numerical Summary (checked with word boundaries for short keywords)
    has_num_agg = (
        _has_word(norm_q, "average")
        or _has_word(norm_q, "mean")
        or _has_word(norm_q, "median")
        or _has_word(norm_q, "minimum")
        or _has_word(norm_q, "min")
        or _has_word(norm_q, "maximum")
        or _has_word(norm_q, "max")
        or _has_word(norm_q, "highest")
        or _has_word(norm_q, "lowest")
        or _has_word(norm_q, "sum")
        or _has_word(norm_q, "std")
        or "standard deviation" in norm_q
        or _has_word(norm_q, "variance")
        or _has_word(norm_q, "var")
        or _has_word(norm_q, "range")
    )

    if has_num_agg:
        ans_dict = _handle_numeric_summary(norm_q, matched_cols, avail_num_cols, raw_df, evidence)
        ans_dict.update({
            "question": question,
            "normalized_question": norm_q,
            "generation": {"mode": "gemini" if gemini_resolved else "deterministic", "gemini_used": gemini_resolved, "fallback": False},
            "warnings": warnings,
            "generated_from": generated_from,
        })
        return _to_serializable(ans_dict)

    # 5. UNSUPPORTED / UNKNOWN FALLBACK
    return _to_serializable({
        "status": "unsupported",
        "question": question,
        "normalized_question": norm_q,
        "intent": {"name": "UNKNOWN", "confidence": 0.0},
        "entities": {"columns": matched_cols, "values": []},
        "calculation": {},
        "result": {},
        "answer": "I couldn't determine how to answer this question from the dataset. Try asking about missing values, averages, correlations, anomalies, health, or data quality.",
        "evidence": {},
        "interpretation": "Question could not be resolved to a supported analytical intent.",
        "generation": {"mode": "deterministic", "gemini_used": False, "fallback": False},
        "warnings": ["Unrecognized or unsupported analytical query."],
        "generated_from": generated_from,
    })
