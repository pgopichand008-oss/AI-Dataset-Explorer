"""
FILE: tests/test_dictionary_engine.py

Comprehensive unit test suite for Phase 7: AI Data Dictionary / Column Intelligence Backend.
Tests all required scenarios:
1. Normal mixed dataset
2. Numerical-only dataset
3. Categorical-only dataset
4. Datetime dataset
5. Mixed invalid values
6. Missing values handling
7. All-missing column
8. Constant column
9. High-cardinality column
10. Identifier-like column
11. Duplicate rows
12. Empty DataFrame (0x0)
13. Empty DataFrame with columns (0xN)
14. Single-row dataset
15. None input
16. Evidence input
17. DataFrame input
18. max_columns truncation
19. JSON serializability
20. Deterministic output
21. Input DataFrame immutability
22. No fabricated semantic meaning
23. Reuse of supplied Evidence
24. Unusual/unsupported dtype
25. Duplicate column names
26. Target candidate detection
27. Boolean column
28. Free-text column
29. Outlier quality concern
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.dictionary_engine import build_data_dictionary


# ============================================================
# 1. NORMAL MIXED DATASET
# ============================================================

def test_normal_mixed_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "id": [f"ID_{i:04d}" for i in range(100)],
        "age": np.random.randint(20, 60, size=100),
        "income": np.random.normal(50000, 5000, size=100),
        "department": np.random.choice(["Sales", "HR", "Eng", "Finance"], size=100),
        "is_active": np.random.choice([True, False], size=100),
    })

    result = build_data_dictionary(df, dataset_name="corp_records")

    assert "dataset_name" in result
    assert result["dataset_name"] == "corp_records"
    assert "column_count" in result
    assert result["column_count"] == 5
    assert "columns" in result
    assert len(result["columns"]) == 5
    assert "summary" in result
    assert "generated_from" in result

    # Validate each column entry schema
    for col in result["columns"]:
        assert "name" in col
        assert "detected_type" in col
        assert "semantic_role" in col
        assert "confidence" in col
        assert isinstance(col["confidence"], (int, float))
        assert 0.0 <= col["confidence"] <= 1.0
        assert "missing" in col
        assert "count" in col["missing"]
        assert "percentage" in col["missing"]
        assert "unique" in col
        assert "count" in col["unique"]
        assert "percentage" in col["unique"]
        assert "statistics" in col
        assert "possible_meaning" in col
        assert col["possible_meaning"].startswith("Possible interpretation: ")
        assert "quality_concerns" in col
        assert isinstance(col["quality_concerns"], list)
        assert "analytical_usefulness" in col
        assert "ml_usefulness" in col


# ============================================================
# 2. NUMERICAL-ONLY DATASET
# ============================================================

def test_numerical_only_dataset():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.5, 3.8, 4.2, 5.9],
        "feature_2": [10, 20, 30, 40, 50],
    })

    result = build_data_dictionary(df)
    assert result["column_count"] == 2
    assert result["summary"]["numeric_columns"] == 2
    assert result["summary"]["categorical_columns"] == 0

    col1 = result["columns"][0]
    assert col1["detected_type"] == "numerical"
    assert col1["semantic_role"] == "numerical measure"
    assert "mean" in col1["statistics"]
    assert "std" in col1["statistics"]
    assert "min" in col1["statistics"]
    assert "max" in col1["statistics"]


# ============================================================
# 3. CATEGORICAL-ONLY DATASET
# ============================================================

def test_categorical_only_dataset():
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "green", "blue"],
        "size": ["S", "M", "L", "M", "S"],
    })

    result = build_data_dictionary(df)
    assert result["column_count"] == 2
    assert result["summary"]["categorical_columns"] == 2
    assert result["summary"]["numeric_columns"] == 0

    col1 = result["columns"][0]
    assert col1["detected_type"] == "categorical"
    assert col1["semantic_role"] == "categorical feature"
    assert "cardinality" in col1["statistics"]
    assert "top_category" in col1["statistics"]
    assert "top_frequency" in col1["statistics"]


# ============================================================
# 4. DATETIME DATASET
# ============================================================

def test_datetime_dataset():
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=10, freq="D"),
        "str_date": ["2023-05-01", "2023-05-02", "2023-05-03"] * 3 + ["2023-05-04"],
    })

    result = build_data_dictionary(df)
    col1 = result["columns"][0]
    assert col1["detected_type"] == "datetime"
    assert col1["semantic_role"] == "datetime"
    assert "min_date" in col1["statistics"]
    assert "max_date" in col1["statistics"]
    assert "2023-01-01" in str(col1["statistics"]["min_date"])


# ============================================================
# 5. MIXED INVALID VALUES
# ============================================================

def test_mixed_invalid_values():
    df = pd.DataFrame({
        "amount": ["100", "250", "UNKNOWN", "400", "ERROR"],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    # Check that invalid values trigger quality concerns
    assert any("invalid" in c.lower() or "mixed" in c.lower() for c in col["quality_concerns"])


# ============================================================
# 6. MISSING VALUES HANDLING
# ============================================================

def test_missing_values():
    df = pd.DataFrame({
        "feature": [1.0, np.nan, 3.0, np.nan, 5.0],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["missing"]["count"] == 2
    assert col["missing"]["percentage"] == 40.0
    assert result["summary"]["columns_with_missing"] == 1
    assert any("missingness" in c.lower() for c in col["quality_concerns"])


# ============================================================
# 7. ALL-MISSING COLUMN
# ============================================================

def test_all_missing_column():
    df = pd.DataFrame({
        "empty_col": [np.nan, None, np.nan, None],
        "valid_col": [1, 2, 3, 4],
    })

    result = build_data_dictionary(df)
    empty_col = result["columns"][0]
    assert empty_col["semantic_role"] == "all missing"
    assert empty_col["is_all_missing"] is True
    assert empty_col["confidence"] >= 0.95
    assert any("all values are missing" in c.lower() for c in empty_col["quality_concerns"])
    assert result["summary"]["all_missing_columns"] == 1


# ============================================================
# 8. CONSTANT COLUMN
# ============================================================

def test_constant_column():
    df = pd.DataFrame({
        "const_num": [42, 42, 42, 42, 42],
        "const_str": ["FIXED", "FIXED", "FIXED", "FIXED", "FIXED"],
    })

    result = build_data_dictionary(df)
    for col in result["columns"]:
        assert col["semantic_role"] == "constant"
        assert col["is_constant"] is True
        assert col["confidence"] >= 0.95
        assert any("constant" in c.lower() for c in col["quality_concerns"])
    assert result["summary"]["constant_columns"] == 2


# ============================================================
# 9. HIGH CARDINALITY COLUMN
# ============================================================

def test_high_cardinality_column():
    # 60 distinct categories in 100 rows (> 50 count, > 30% ratio)
    cats = [f"Category_{i}" for i in range(60)] + ["Category_0"] * 40
    df = pd.DataFrame({"high_card": cats})

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["semantic_role"] == "high cardinality"
    assert result["summary"]["high_cardinality_columns"] >= 1
    assert any("high cardinality" in c.lower() for c in col["quality_concerns"])


# ============================================================
# 10. IDENTIFIER-LIKE COLUMN
# ============================================================

def test_identifier_like_column():
    df = pd.DataFrame({
        "user_uuid": [f"USR_{i:05d}" for i in range(100)],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["semantic_role"] == "identifier"
    assert col["confidence"] >= 0.90
    assert any("identifier" in c.lower() for c in col["quality_concerns"])
    assert result["summary"]["identifier_like_columns"] >= 1


# ============================================================
# 11. DUPLICATE ROWS
# ============================================================

def test_duplicate_rows():
    df = pd.DataFrame({
        "x": [1, 2, 1, 2],
        "y": ["A", "B", "A", "B"],
    })

    result = build_data_dictionary(df)
    assert result["column_count"] == 2
    assert len(result["columns"]) == 2


# ============================================================
# 12. EMPTY DATAFRAME (0x0)
# ============================================================

def test_empty_dataframe():
    df = pd.DataFrame()
    result = build_data_dictionary(df)

    assert result["column_count"] == 0
    assert result["columns"] == []
    assert result["summary"]["total_columns"] == 0


# ============================================================
# 13. EMPTY DATAFRAME WITH COLUMNS (0xN)
# ============================================================

def test_empty_dataframe_with_columns():
    df = pd.DataFrame(columns=["col_a", "col_b", "col_c"])
    result = build_data_dictionary(df)

    assert result["column_count"] == 3
    assert len(result["columns"]) == 3
    for col in result["columns"]:
        assert col["missing"]["count"] == 0
        assert col["unique"]["count"] == 0


# ============================================================
# 14. SINGLE-ROW DATASET
# ============================================================

def test_single_row_dataset():
    df = pd.DataFrame({
        "num": [10.5],
        "cat": ["Solo"],
    })

    result = build_data_dictionary(df)
    assert result["column_count"] == 2
    assert len(result["columns"]) == 2
    for col in result["columns"]:
        assert any("single" in c.lower() for c in col["quality_concerns"])


# ============================================================
# 15. NONE INPUT
# ============================================================

def test_none_input():
    result = build_data_dictionary(None)

    assert result["column_count"] == 0
    assert result["columns"] == []
    assert result["generated_from"] == "none"
    assert result["summary"]["total_columns"] == 0


# ============================================================
# 16. EVIDENCE INPUT
# ============================================================

def test_evidence_input():
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    evidence = build_evidence(df, dataset_name="ev_test")

    result = build_data_dictionary(evidence)
    assert result["generated_from"] == "evidence"
    assert result["dataset_name"] == "ev_test"
    assert result["column_count"] == 1


# ============================================================
# 17. DATAFRAME INPUT
# ============================================================

def test_dataframe_input():
    df = pd.DataFrame({"x": [10, 20, 30]})
    result = build_data_dictionary(df, dataset_name="df_test")

    assert result["generated_from"] == "dataframe"
    assert result["dataset_name"] == "df_test"
    assert result["column_count"] == 1


# ============================================================
# 18. MAX_COLUMNS TRUNCATION
# ============================================================

def test_max_columns():
    df = pd.DataFrame({
        f"col_{i}": range(10) for i in range(5)
    })

    result = build_data_dictionary(df, max_columns=2)
    assert result["column_count"] == 5
    assert result["returned_column_count"] == 2
    assert len(result["columns"]) == 2
    assert result["truncated"] is True
    assert result["columns"][0]["name"] == "col_0"
    assert result["columns"][1]["name"] == "col_1"


# ============================================================
# 19. JSON SERIALIZABILITY
# ============================================================

def test_json_serializability():
    df = pd.DataFrame({
        "num": [1.1, np.nan, 3.3, np.inf],
        "cat": ["A", "B", None, "D"],
        "dt": pd.date_range("2023-01-01", periods=4),
        "bool_val": [True, False, True, False],
    })

    result = build_data_dictionary(df)
    # Must dump and reload cleanly without non-serializable objects
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized["column_count"] == 4
    assert len(deserialized["columns"]) == 4


# ============================================================
# 20. DETERMINISTIC OUTPUT
# ============================================================

def test_deterministic_output():
    df = pd.DataFrame({
        "score": [95, 80, 70, 85, 90],
        "grade": ["A", "B", "C", "B", "A"],
    })

    res1 = build_data_dictionary(df)
    res2 = build_data_dictionary(df)

    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


# ============================================================
# 21. INPUT IMMUTABILITY
# ============================================================

def test_input_immutability():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["X", "Y", "Z"],
    })
    df_copy = df.copy(deep=True)

    _ = build_data_dictionary(df)

    pd.testing.assert_frame_equal(df, df_copy)


# ============================================================
# 22. NO FABRICATED SEMANTIC MEANING
# ============================================================

def test_no_fabricated_semantic_meaning():
    df = pd.DataFrame({
        "col_a": [25, 30, 45, 50],
        "col_b": [60000, 75000, 90000, 110000],
    })

    result = build_data_dictionary(df)
    forbidden_terms = ["customer age", "employee salary", "sales revenue", "annual income"]

    for col in result["columns"]:
        meaning = col["possible_meaning"].lower()
        assert meaning.startswith("possible interpretation: ")
        for term in forbidden_terms:
            assert term not in meaning


# ============================================================
# 23. REUSE OF SUPPLIED EVIDENCE
# ============================================================

def test_reuse_of_supplied_evidence():
    df = pd.DataFrame({"metric": [10, 20, 30, 40, 50]})
    evidence = build_evidence(df, dataset_name="prebuilt_ev")

    result = build_data_dictionary(evidence)
    assert result["dataset_name"] == "prebuilt_ev"
    assert result["columns"][0]["missing"]["count"] == evidence["columns"][0]["missing_count"]
    assert result["columns"][0]["unique"]["count"] == evidence["columns"][0]["unique_count"]


# ============================================================
# 24. UNUSUAL / UNSUPPORTED DTYPE
# ============================================================

def test_unusual_unsupported_dtype():
    df = pd.DataFrame({
        "complex_val": [complex(1, 2), complex(3, 4), complex(5, 6)],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["detected_type"] in ("unknown", "unsupported")


# ============================================================
# 25. DUPLICATE COLUMN NAMES
# ============================================================

def test_duplicate_column_names():
    df = pd.DataFrame([[1, 2]], columns=["val", "val"])

    result = build_data_dictionary(df)
    assert result["column_count"] == 2
    assert len(result["columns"]) == 2


# ============================================================
# 26. TARGET CANDIDATE DETECTION
# ============================================================

def test_target_candidate_detection():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0, 1, 0, 1, 1, 0],
    })

    result = build_data_dictionary(df)
    target_col = [c for c in result["columns"] if c["name"] == "target"][0]
    assert target_col["semantic_role"] == "target candidate"
    assert target_col["is_target_candidate"] is True
    assert result["summary"]["target_candidates"] >= 1


# ============================================================
# 27. BOOLEAN COLUMN
# ============================================================

def test_boolean_column():
    df = pd.DataFrame({
        "flag": [True, False, True, True, False],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["detected_type"] == "boolean"
    assert result["summary"]["boolean_columns"] == 1


# ============================================================
# 28. FREE-TEXT COLUMN
# ============================================================

def test_free_text_column():
    df = pd.DataFrame({
        "comments": [
            "Customer requested a full refund due to a delayed shipping confirmation notice.",
            "Account manager scheduled a follow-up consultation meeting for next Tuesday afternoon.",
            "System logged an unhandled timeout exception during batch processing transaction.",
        ],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert col["detected_type"] == "text"
    assert col["semantic_role"] == "text"


# ============================================================
# 29. OUTLIER QUALITY CONCERN
# ============================================================

def test_outlier_quality_concern():
    df = pd.DataFrame({
        "val": [10, 11, 12, 11, 10, 12, 11, 10, 999999],
    })

    result = build_data_dictionary(df)
    col = result["columns"][0]
    assert any("outlier" in c.lower() for c in col["quality_concerns"])
