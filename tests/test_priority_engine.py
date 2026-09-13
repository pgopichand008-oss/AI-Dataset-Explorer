"""
FILE: tests/test_priority_engine.py

Comprehensive unit test suite for Phase 3: Insight Priority Engine.
Tests all 20 required scenarios, deterministic scoring, rankings,
edge cases, and immutability.
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights


# ============================================================
# 1. NORMAL HEALTHY DATASET
# ============================================================

def test_normal_healthy_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=100),
        "income": np.random.normal(50000, 10000, size=100),
        "department": np.random.choice(["Sales", "HR", "Eng", "Finance"], size=100),
        "status": np.random.choice(["Active", "Inactive"], size=100),
    })

    result = build_prioritized_insights(df, dataset_name="healthy_corp")

    assert "insights" in result
    assert "summary" in result
    assert result["dataset_name"] == "healthy_corp"
    assert result["generated_from"] == "evidence"

    # For a healthy dataset, findings should have low severity or be info
    crit_count = result["summary"]["critical_count"]
    assert crit_count == 0

    # Ensure all insights conform to schema
    for ins in result["insights"]:
        assert "title" in ins
        assert "category" in ins
        assert ins["severity"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        assert isinstance(ins["evidence"], dict)
        assert isinstance(ins["priority_score"], float)
        assert isinstance(ins["priority_rank"], int)
        assert "interpretation" in ins
        assert "impact" in ins
        assert "recommended_action" in ins


# ============================================================
# 2. MISSING-VALUE DATASET
# ============================================================

def test_missing_value_dataset():
    df = pd.DataFrame({
        "feature_low": [1.0, 2.0, 3.0, None] * 25,  # 25% missing -> HIGH
        "feature_crit": [1.0, None, None, None] * 25,  # 75% missing -> CRITICAL
        "clean_col": list(range(100)),
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    missing_insights = [x for x in insights if x["category"] == "missingness"]
    assert len(missing_insights) > 0

    # The 75% missing column should be CRITICAL
    crit_missing = [x for x in missing_insights if "feature_crit" in x["title"]]
    assert len(crit_missing) == 1
    assert crit_missing[0]["severity"] == "CRITICAL"
    assert crit_missing[0]["evidence"]["missing_percentage"] == 75.0
    assert crit_missing[0]["evidence"]["missing_count"] == 75
    assert crit_missing[0]["priority_score"] >= 90.0

    # The 25% missing column should be HIGH
    high_missing = [x for x in missing_insights if "feature_low" in x["title"]]
    assert len(high_missing) == 1
    assert high_missing[0]["severity"] == "HIGH"
    assert high_missing[0]["evidence"]["missing_percentage"] == 25.0


# ============================================================
# 3. DUPLICATE-ROW DATASET
# ============================================================

def test_duplicate_row_dataset():
    # 10 unique rows, repeated 4 times -> 40 total rows, 30 duplicates (75% duplicates)
    base = pd.DataFrame({
        "id": list(range(10)),
        "val": [f"val_{i}" for i in range(10)],
    })
    df = pd.concat([base] * 4, ignore_index=True)

    result = build_prioritized_insights(df)
    insights = result["insights"]

    dup_insights = [x for x in insights if x["category"] == "duplicates"]
    assert len(dup_insights) == 1
    dup_ins = dup_insights[0]

    assert dup_ins["severity"] == "CRITICAL"
    assert dup_ins["evidence"]["duplicate_rows"] == 30
    assert dup_ins["evidence"]["duplicate_row_percentage"] == 75.0
    assert dup_ins["priority_score"] >= 90.0


# ============================================================
# 4. CONSTANT-COLUMN DATASET
# ============================================================

def test_constant_column_dataset():
    df = pd.DataFrame({
        "constant_num": [42] * 50,
        "constant_str": ["FIXED"] * 50,
        "varying": list(range(50)),
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    const_insights = [x for x in insights if "Constant Column" in x["title"]]
    assert len(const_insights) == 2

    for ins in const_insights:
        assert ins["category"] == "quality"
        assert ins["severity"] == "MEDIUM"
        assert ins["evidence"]["unique_count"] == 1


# ============================================================
# 5. ALL-MISSING / EMPTY COLUMN
# ============================================================

def test_all_missing_column():
    df = pd.DataFrame({
        "empty_col": [None] * 50,
        "regular_col": list(range(50)),
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    empty_insights = [x for x in insights if "Empty Column" in x["title"]]
    assert len(empty_insights) == 1
    assert empty_insights[0]["severity"] == "HIGH"
    assert empty_insights[0]["evidence"]["missing_percentage"] == 100.0

    # Ensure deduplication: empty column should NOT also be flagged as a regular missingness finding
    col_missing_findings = [x for x in insights if x["title"] == "Missing Values in Column 'empty_col' (100.0%)"]
    assert len(col_missing_findings) == 0


# ============================================================
# 6. INVALID-VALUE DATASET
# ============================================================

def test_invalid_value_dataset():
    # Numeric column corrupted with invalid text strings
    df = pd.DataFrame({
        "numeric_corrupted": [10, 20, 30, "corrupt_str", 50, "N/A_err", 70, 80, 90, 100],
        "clean": list(range(10)),
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    invalid_insights = [x for x in insights if x["category"] == "validity"]
    assert len(invalid_insights) >= 1
    inv_ins = invalid_insights[0]

    assert "numeric_corrupted" in inv_ins["title"]
    assert inv_ins["evidence"]["invalid_count"] == 2
    assert inv_ins["evidence"]["invalid_percentage"] == 20.0


# ============================================================
# 7. OUTLIER DATASET
# ============================================================

def test_outlier_dataset():
    # Normal distribution with extreme injected outliers
    vals = list(range(10, 110)) + [99999, -99999, 88888, -88888]
    df = pd.DataFrame({
        "outlier_feature": vals,
        "clean": list(range(len(vals))),
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    outlier_insights = [x for x in insights if x["category"] == "outliers"]
    assert len(outlier_insights) == 1
    out_ins = outlier_insights[0]

    assert "outlier_feature" in out_ins["title"]
    assert out_ins["evidence"]["outlier_count"] >= 4
    assert out_ins["evidence"]["outlier_percentage"] > 0


# ============================================================
# 8. STRONG-CORRELATION DATASET
# ============================================================

def test_strong_correlation_dataset():
    np.random.seed(123)
    x = np.linspace(1, 100, 100)
    y = 2.5 * x + np.random.normal(0, 0.5, 100)  # r ~ 0.999
    z = np.random.normal(0, 1, 100)

    df = pd.DataFrame({"feat_x": x, "feat_y": y, "feat_z": z})

    result = build_prioritized_insights(df)
    insights = result["insights"]

    corr_insights = [x for x in insights if x["category"] == "correlation"]
    assert len(corr_insights) >= 1

    corr_ins = corr_insights[0]
    assert corr_ins["severity"] in ["HIGH", "MEDIUM"]
    assert "feat_x" in corr_ins["title"] and "feat_y" in corr_ins["title"]
    assert abs(corr_ins["evidence"]["correlation"]) > 0.95


# ============================================================
# 9. ML-READINESS ISSUE
# ============================================================

def test_ml_readiness_issue():
    # Dataset unsuitable for ML: only 5 rows, severe missingness, no clear target
    df = pd.DataFrame({
        "a": [1, None, None, 4, 5],
        "b": [None, 2, None, None, 5],
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    ml_insights = [x for x in insights if x["category"] == "ml_readiness"]
    assert len(ml_insights) >= 1
    assert ml_insights[0]["severity"] in ["CRITICAL", "HIGH"]
    assert "ml_score" in ml_insights[0]["evidence"]


# ============================================================
# 10. SMALL DATASET
# ============================================================

def test_small_dataset():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": ["a", "b", "c", "d", "e"],
    })

    result = build_prioritized_insights(df)
    insights = result["insights"]

    sample_size_insights = [x for x in insights if x["category"] == "sample_size"]
    assert len(sample_size_insights) == 1
    assert sample_size_insights[0]["severity"] == "HIGH"
    assert sample_size_insights[0]["evidence"]["row_count"] == 5


# ============================================================
# 11. EMPTY DATAFRAME (0x0 AND 0xN)
# ============================================================

def test_empty_dataframe():
    # 0 rows, 0 columns
    df_empty = pd.DataFrame()
    result1 = build_prioritized_insights(df_empty)
    assert len(result1["insights"]) == 1
    assert result1["insights"][0]["title"] == "Empty Dataset (0 Usable Records or Columns)"
    assert result1["insights"][0]["severity"] == "CRITICAL"
    assert result1["insights"][0]["priority_score"] == 100.0

    # 0 rows, with columns
    df_empty_cols = pd.DataFrame(columns=["a", "b", "c"])
    result2 = build_prioritized_insights(df_empty_cols)
    assert len(result2["insights"]) == 1
    assert result2["insights"][0]["severity"] == "CRITICAL"
    assert result2["insights"][0]["priority_score"] == 100.0


# ============================================================
# 12. NONE INPUT
# ============================================================

def test_none_input():
    result = build_prioritized_insights(None)
    assert len(result["insights"]) == 1
    assert result["insights"][0]["severity"] == "CRITICAL"
    assert result["insights"][0]["priority_score"] == 100.0
    assert result["summary"]["total_findings"] == 1


# ============================================================
# 13. MULTIPLE FINDINGS ARE CORRECTLY RANKED
# ============================================================

def test_multiple_findings_ranking():
    # Construct a dataset with:
    # 1. 80% missing in col1 -> CRITICAL (score ~90+)
    # 2. 50% duplicate rows -> CRITICAL (score ~90+)
    # 3. Constant col2 -> MEDIUM (score ~55)
    # 4. Outliers in col3 -> MEDIUM/LOW (score ~40-55)
    base = pd.DataFrame({
        "col1": [1.0, None, None, None, None] * 20,
        "col2": [99] * 100,
        "col3": list(range(96)) + [10000, 20000, 30000, 40000],
    })
    # Double the dataset to create duplicates
    df = pd.concat([base, base], ignore_index=True)

    result = build_prioritized_insights(df)
    insights = result["insights"]

    assert len(insights) >= 3

    # Verify descending sort order by priority_score
    scores = [x["priority_score"] for x in insights]
    assert scores == sorted(scores, reverse=True)

    # Top findings should be CRITICAL
    assert insights[0]["severity"] == "CRITICAL"


# ============================================================
# 14. PRIORITY SCORES ARE DETERMINISTIC
# ============================================================

def test_deterministic_scoring():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        "b": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "c": ["x", "y", "z", "w", "a", "b", "c", "d", "e", "f"],
    })

    result1 = build_prioritized_insights(df)
    result2 = build_prioritized_insights(df)

    assert len(result1["insights"]) == len(result2["insights"])

    for ins1, ins2 in zip(result1["insights"], result2["insights"]):
        assert ins1["title"] == ins2["title"]
        assert ins1["priority_score"] == ins2["priority_score"]
        assert ins1["priority_rank"] == ins2["priority_rank"]
        assert ins1["severity"] == ins2["severity"]


# ============================================================
# 15. PRIORITY_RANK IS SEQUENTIAL (1, 2, 3...)
# ============================================================

def test_priority_rank_sequential():
    df = pd.DataFrame({
        "missing_high": [None] * 50 + list(range(50)),
        "const_col": ["A"] * 100,
        "dup_col": list(range(50)) + list(range(50)),
    })

    result = build_prioritized_insights(df)
    ranks = [ins["priority_rank"] for ins in result["insights"]]

    expected_ranks = list(range(1, len(result["insights"]) + 1))
    assert ranks == expected_ranks


# ============================================================
# 16. MAX_INSIGHTS WORKS
# ============================================================

def test_max_insights_truncation():
    # Many issues across multiple columns
    data = {}
    for i in range(15):
        data[f"const_{i}"] = [i] * 50
    df = pd.DataFrame(data)

    # Call with max_insights=5
    result = build_prioritized_insights(df, max_insights=5)

    assert len(result["insights"]) == 5
    assert result["summary"]["total_findings"] >= 15
    assert result["insights"][0]["priority_rank"] == 1
    assert result["insights"][4]["priority_rank"] == 5


# ============================================================
# 17. NO DUPLICATE INSIGHTS FOR SAME UNDERLYING ISSUE
# ============================================================

def test_no_duplicate_insights():
    df = pd.DataFrame({
        "all_null": [None] * 50,
        "valid": list(range(50)),
    })

    result = build_prioritized_insights(df)
    titles = [x["title"] for x in result["insights"]]

    # Ensure each title is completely unique
    assert len(titles) == len(set(titles))

    # Column 'all_null' should be flagged once as Empty Column, not also as Constant Column
    empty_findings = [t for t in titles if "all_null" in t]
    assert len(empty_findings) == 1
    assert "Empty Column: 'all_null'" in empty_findings[0]


# ============================================================
# 18. JSON SERIALIZATION WORKS
# ============================================================

def test_json_serialization():
    df = pd.DataFrame({
        "num": [1.0, 2.5, np.nan, 4.0, 1000.0],
        "cat": ["A", "B", "A", None, "A"],
    })

    result = build_prioritized_insights(df, dataset_name="serialization_test")

    # Must serialize without TypeError
    serialized = json.dumps(result, indent=2)
    assert isinstance(serialized, str)

    # Deserialize and verify roundtrip integrity
    parsed = json.loads(serialized)
    assert parsed["dataset_name"] == "serialization_test"
    assert len(parsed["insights"]) == len(result["insights"])


# ============================================================
# 19. ORIGINAL DATAFRAME IS NOT MODIFIED
# ============================================================

def test_dataframe_immutability():
    df = pd.DataFrame({
        "col_a": [1, 2, None, 4],
        "col_b": ["x", "y", "y", "z"],
    })
    df_copy = df.copy(deep=True)

    _ = build_prioritized_insights(df)

    pd.testing.assert_frame_equal(df, df_copy)


# ============================================================
# 20. EVIDENCE INPUT IS NOT MODIFIED
# ============================================================

def test_evidence_immutability():
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5] * 10,
        "col_b": ["a", "b", "c", "d", "e"] * 10,
    })
    evidence = build_evidence(df)
    evidence_copy = copy.deepcopy(evidence)

    _ = build_prioritized_insights(evidence)

    assert evidence == evidence_copy
