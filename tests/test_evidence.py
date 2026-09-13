"""
tests/test_evidence.py

Comprehensive test suite for Phase 1 — Evidence Layer (engine/evidence.py).

Tests:
1. Normal dataset (factual statistical assertions, structure, completeness, correlations)
2. Missing values (exact count and percentage calculations)
3. Duplicate rows (exact duplicate count and percentage)
4. Numerical columns (pure numerical data, statistics, ranges, standard deviation)
5. Categorical columns (category frequencies, dominance %, uniqueness)
6. Datetime column (timestamp detection, role inference)
7. Empty DataFrame (both 0-row/0-col and 0-row/N-col cases)
8. Single-row DataFrame (variance & correlation limitation safety)
9. All-missing column (100% missing detection, empty columns quality finding)
10. Constant column (single-value detection, constant columns quality finding)
11. Mixed/invalid values (detecting invalid non-numeric entries in numeric columns)
12. Duplicate column names (graceful disambiguation, no ambiguous truth value errors)
13. None dataset (graceful handling, safe fallback structure)
14. JSON serializability (guaranteeing deep JSON safety for future AI reporting)
"""

import json
import math
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence


# ============================================================
# 1. NORMAL DATASET
# ============================================================

def test_normal_dataset():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "department": ["HR", "Engineering", "Engineering", "Marketing", "HR"],
    })

    evidence = build_evidence(df, dataset_name="employee_sample")

    # Structure assertions
    structure = evidence["structure"]
    assert structure["row_count"] == 5
    assert structure["column_count"] == 3
    assert structure["column_names"] == ["age", "salary", "department"]
    assert "age" in structure["numerical_columns"]
    assert "salary" in structure["numerical_columns"]
    assert "department" in structure["categorical_columns"]
    assert structure["has_duplicate_columns"] is False
    assert structure["duplicate_column_names"] == []

    # Completeness assertions (zero missing)
    completeness = evidence["completeness"]
    assert completeness["total_cells"] == 15
    assert completeness["missing_cells"] == 0
    assert completeness["missing_percentage"] == 0.0
    assert completeness["complete_cells"] == 15
    assert completeness["complete_percentage"] == 100.0
    assert completeness["columns_with_missing"] == []

    # Duplicate rows assertions (zero duplicates)
    duplicates = evidence["duplicates"]
    assert duplicates["duplicate_rows"] == 0
    assert duplicates["duplicate_row_percentage"] == 0.0
    assert duplicates["is_unique"] is True

    # Numerical statistics assertions (exact mathematical calculations)
    num_stats = evidence["numerical_statistics"]
    assert num_stats["age"]["count"] == 5
    assert num_stats["age"]["mean"] == 35.0
    assert num_stats["age"]["median"] == 35.0
    assert num_stats["age"]["min"] == 25.0
    assert num_stats["age"]["max"] == 45.0
    assert num_stats["age"]["range"] == 20.0

    assert num_stats["salary"]["count"] == 5
    assert num_stats["salary"]["mean"] == 70000.0
    assert num_stats["salary"]["median"] == 70000.0
    assert num_stats["salary"]["min"] == 50000.0
    assert num_stats["salary"]["max"] == 90000.0

    # Categorical statistics assertions
    cat_stats = evidence["categorical_statistics"]
    assert cat_stats["department"]["count"] == 5
    assert cat_stats["department"]["unique_count"] == 3
    assert cat_stats["department"]["top_category"] in ["HR", "Engineering"]
    assert cat_stats["department"]["top_frequency"] == 2
    assert cat_stats["department"]["dominance_percentage"] == 40.0

    # Correlation assertions (age and salary are perfectly linearly correlated: r = 1.0)
    correlations = evidence["correlations"]
    assert correlations["supported"] is True
    assert len(correlations["strongest_pairs"]) >= 1
    top_pair = correlations["strongest_pairs"][0]
    assert top_pair["correlation"] == 1.0
    assert top_pair["absolute_correlation"] == 1.0


# ============================================================
# 2. MISSING VALUES
# ============================================================

def test_missing_values():
    # Exactly 2 missing out of 10 rows in 'age' (20.0%), and 3 missing in 'score' (30.0%)
    df = pd.DataFrame({
        "age": [20, None, 22, 23, 24, None, 26, 27, 28, 29],
        "score": [10.0, 20.0, None, 40.0, None, 60.0, None, 80.0, 90.0, 100.0],
        "category": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    })

    evidence = build_evidence(df)

    completeness = evidence["completeness"]
    assert completeness["total_cells"] == 30
    assert completeness["missing_cells"] == 5
    assert completeness["missing_percentage"] == round(5 / 30 * 100, 2)  # 16.67%
    assert completeness["complete_cells"] == 25

    # Column-level missing assertions
    assert completeness["column_missing_counts"]["age"] == 2
    assert completeness["column_missing_percentages"]["age"] == 20.0
    assert completeness["column_missing_counts"]["score"] == 3
    assert completeness["column_missing_percentages"]["score"] == 30.0
    assert completeness["column_missing_counts"]["category"] == 0
    assert completeness["column_missing_percentages"]["category"] == 0.0

    assert "age" in completeness["columns_with_missing"]
    assert "score" in completeness["columns_with_missing"]
    assert "category" not in completeness["columns_with_missing"]

    # Numerical statistics calculate on non-null values only
    num_stats = evidence["numerical_statistics"]
    assert num_stats["age"]["count"] == 8
    assert num_stats["score"]["count"] == 7


# ============================================================
# 3. DUPLICATE ROWS
# ============================================================

def test_duplicate_rows():
    # 6 rows total: 3 unique rows, each duplicated once -> exactly 3 duplicate rows (50.0%)
    df = pd.DataFrame({
        "item": ["apple", "banana", "cherry", "apple", "banana", "cherry"],
        "price": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    })

    evidence = build_evidence(df)

    duplicates = evidence["duplicates"]
    assert duplicates["duplicate_rows"] == 3
    assert duplicates["duplicate_row_percentage"] == 50.0
    assert duplicates["is_unique"] is False

    quality = evidence["quality"]
    # Verify duplicate rows finding was registered in quality findings
    dup_findings = [f for f in quality["quality_findings"] if f.get("Issue") == "Duplicate Rows"]
    assert len(dup_findings) == 1
    assert "3 duplicate rows" in dup_findings[0]["Details"]


# ============================================================
# 4. NUMERICAL COLUMNS
# ============================================================

def test_numerical_columns():
    df = pd.DataFrame({
        "x": [10, 20, 30],
        "y": [100.0, 200.0, 300.0],
        "z": [-5, 0, 5],
    })

    evidence = build_evidence(df)

    structure = evidence["structure"]
    assert set(structure["numerical_columns"]) == {"x", "y", "z"}
    assert structure["categorical_columns"] == []
    assert structure["datetime_columns"] == []

    stats = evidence["numerical_statistics"]
    assert stats["x"]["mean"] == 20.0
    assert stats["x"]["std"] == 10.0
    assert stats["x"]["min"] == 10.0
    assert stats["x"]["max"] == 30.0

    assert stats["y"]["mean"] == 200.0
    assert stats["z"]["mean"] == 0.0
    assert stats["z"]["range"] == 10.0


# ============================================================
# 5. CATEGORICAL COLUMNS
# ============================================================

def test_categorical_columns():
    df = pd.DataFrame({
        "fruit": ["apple", "banana", "apple", "apple", "cherry"],
        "status": ["active", "active", "inactive", "active", "active"],
    })

    evidence = build_evidence(df)

    structure = evidence["structure"]
    assert "fruit" in structure["categorical_columns"]
    assert "status" in structure["categorical_columns"]
    assert structure["numerical_columns"] == []

    cat_stats = evidence["categorical_statistics"]
    assert cat_stats["fruit"]["top_category"] == "apple"
    assert cat_stats["fruit"]["top_frequency"] == 3
    assert cat_stats["fruit"]["dominance_percentage"] == 60.0
    assert cat_stats["fruit"]["unique_count"] == 3

    assert cat_stats["status"]["top_category"] == "active"
    assert cat_stats["status"]["top_frequency"] == 4
    assert cat_stats["status"]["dominance_percentage"] == 80.0
    assert cat_stats["status"]["unique_count"] == 2


# ============================================================
# 6. DATETIME COLUMN
# ============================================================

def test_datetime_column():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "metric": [100, 110, 105, 115],
    })

    evidence = build_evidence(df)

    structure = evidence["structure"]
    assert "timestamp" in structure["datetime_columns"]
    assert "metric" in structure["numerical_columns"]

    # Verify column detail role inference
    col_dict = {c["name"]: c for c in evidence["columns"]}
    assert col_dict["timestamp"]["inferred_role"] == "Date / Time"


# ============================================================
# 7. EMPTY DATAFRAME
# ============================================================

def test_empty_dataframe_completely_empty():
    df = pd.DataFrame()
    evidence = build_evidence(df)

    assert evidence["structure"]["row_count"] == 0
    assert evidence["structure"]["column_count"] == 0
    assert evidence["structure"]["column_names"] == []
    assert evidence["completeness"]["total_cells"] == 0
    assert evidence["completeness"]["missing_cells"] == 0
    assert evidence["duplicates"]["duplicate_rows"] == 0
    assert evidence["numerical_statistics"] == {}
    assert evidence["categorical_statistics"] == {}
    assert evidence["correlations"]["supported"] is False
    assert any("empty" in lim.lower() for lim in evidence["limitations"])


def test_empty_dataframe_with_columns():
    df = pd.DataFrame(columns=["a", "b", "c"])
    evidence = build_evidence(df)

    assert evidence["structure"]["row_count"] == 0
    assert evidence["structure"]["column_count"] == 3
    assert evidence["structure"]["column_names"] == ["a", "b", "c"]
    assert evidence["completeness"]["total_cells"] == 0
    assert evidence["correlations"]["supported"] is False
    assert any("empty" in lim.lower() for lim in evidence["limitations"])


# ============================================================
# 8. SINGLE-ROW DATAFRAME
# ============================================================

def test_single_row_dataframe():
    df = pd.DataFrame({
        "val_num": [42.0],
        "val_str": ["single"],
    })

    evidence = build_evidence(df)

    assert evidence["structure"]["row_count"] == 1
    assert evidence["structure"]["column_count"] == 2
    assert evidence["duplicates"]["duplicate_rows"] == 0

    # Variance and std are not computable for n=1 (reported safely as 0.0 or None)
    num_stats = evidence["numerical_statistics"]["val_num"]
    assert num_stats["count"] == 1
    assert num_stats["mean"] == 42.0
    assert num_stats["std"] == 0.0
    assert num_stats["min"] == 42.0
    assert num_stats["max"] == 42.0

    # Correlations cannot be computed with 1 row
    assert evidence["correlations"]["supported"] is False
    assert any("1 record" in lim for lim in evidence["limitations"])


# ============================================================
# 9. ALL-MISSING COLUMN
# ============================================================

def test_all_missing_column():
    df = pd.DataFrame({
        "all_null": [np.nan, np.nan, np.nan, np.nan],
        "valid": [1, 2, 3, 4],
    })

    evidence = build_evidence(df)

    completeness = evidence["completeness"]
    assert completeness["column_missing_counts"]["all_null"] == 4
    assert completeness["column_missing_percentages"]["all_null"] == 100.0

    col_dict = {c["name"]: c for c in evidence["columns"]}
    assert col_dict["all_null"]["is_all_missing"] is True
    assert col_dict["valid"]["is_all_missing"] is False

    # Check quality findings
    quality = evidence["quality"]
    assert "all_null" in quality["empty_columns"]
    assert any("all_null" in lim and "100%" in lim for lim in evidence["limitations"])


# ============================================================
# 10. CONSTANT COLUMN
# ============================================================

def test_constant_column():
    df = pd.DataFrame({
        "constant_num": [99, 99, 99, 99, 99],
        "variable_num": [1, 2, 3, 4, 5],
    })

    evidence = build_evidence(df)

    col_dict = {c["name"]: c for c in evidence["columns"]}
    assert col_dict["constant_num"]["is_constant"] is True
    assert col_dict["constant_num"]["unique_count"] == 1
    assert col_dict["variable_num"]["is_constant"] is False

    quality = evidence["quality"]
    assert "constant_num" in quality["constant_columns"]
    assert any("constant_num" in lim and "constant" in lim.lower() for lim in evidence["limitations"])


# ============================================================
# 11. MIXED / INVALID VALUES
# ============================================================

def test_mixed_invalid_values():
    # Numeric column containing text representations ("N/A", "Unknown") alongside numbers
    df = pd.DataFrame({
        "salary": [50000, "Unknown", 70000, "N/A", 90000],
        "department": ["IT", "HR", "IT", "HR", "Finance"],
    })

    evidence = build_evidence(df)

    col_dict = {c["name"]: c for c in evidence["columns"]}
    salary_col = col_dict["salary"]

    assert salary_col["invalid_values"]["invalid_count"] == 2
    assert salary_col["invalid_values"]["valid_count"] == 3
    assert set(salary_col["invalid_values"]["invalid_examples"]) == {"Unknown", "N/A"}

    # Pure categorical column must NOT be marked as having invalid values
    dept_col = col_dict["department"]
    assert dept_col["invalid_values"]["invalid_count"] == 0

    assert any("salary" in lim and "invalid" in lim.lower() for lim in evidence["limitations"])


# ============================================================
# 12. DUPLICATE COLUMN NAMES
# ============================================================

def test_duplicate_column_names():
    # Two columns named 'metric' with different values
    df = pd.DataFrame(
        [[10, 20, "Alpha"], [30, 40, "Beta"], [50, 60, "Gamma"]],
        columns=["metric", "metric", "team"]
    )

    # Function must not raise AttributeError or ambiguous truth value ValueError
    evidence = build_evidence(df)

    structure = evidence["structure"]
    assert structure["row_count"] == 3
    assert structure["column_count"] == 3
    assert structure["has_duplicate_columns"] is True
    assert "metric" in structure["duplicate_column_names"]

    # Check both occurrences were tracked in columns list
    col_names = [c["name"] for c in evidence["columns"]]
    assert col_names == ["metric", "metric", "team"]

    # Check limitations records the duplicate columns
    assert any("duplicate column" in lim.lower() for lim in evidence["limitations"])

    # Verify JSON serialization works without error
    serialized = json.dumps(evidence)
    assert len(serialized) > 0


# ============================================================
# 13. NONE DATAFRAME
# ============================================================

def test_none_dataframe():
    evidence = build_evidence(None)

    assert evidence["structure"]["row_count"] == 0
    assert evidence["structure"]["column_count"] == 0
    assert evidence["limitations"] == ["Dataset was not provided (None)."]


# ============================================================
# 14. JSON SERIALIZABILITY TEST
# ============================================================

def test_json_serializability_across_all_cases():
    test_frames = [
        pd.DataFrame({"a": [1, 2, 3], "b": [4.5, np.nan, 6.7]}),
        pd.DataFrame({"dates": pd.to_datetime(["2026-01-01", "2026-01-02"])}),
        pd.DataFrame({"cat": ["x", "y", "z"]}),
        pd.DataFrame(),
        pd.DataFrame([[1, 2]], columns=["dup", "dup"]),
    ]

    for df in test_frames:
        evidence = build_evidence(df)
        json_output = json.dumps(evidence)
        assert isinstance(json_output, str)
        parsed = json.loads(json_output)
        assert isinstance(parsed, dict)
