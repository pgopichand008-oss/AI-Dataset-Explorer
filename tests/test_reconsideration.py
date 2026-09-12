"""
FILE: tests/test_reconsideration.py

Comprehensive unit test suite for Phase 8: Advanced Reconsideration Engine.
Tests all required scenarios:
1. Identical datasets
2. Added column
3. Removed column
4. Probable rename
5. Dtype change
6. Row-count change
7. Increased missingness
8. Decreased missingness
9. All-missing transition
10. Missingness resolved
11. Duplicate increase
12. Duplicate removal
13. Invalid values increase
14. Invalid values decrease / resolved
15. Constant -> variable
16. Variable -> constant
17. High-cardinality change
18. Outlier increase
19. Outlier decrease / resolved
20. Statistical drift
21. Correlation appears
22. Correlation disappears
23. Correlation strengthens
24. Correlation weakens
25. ML readiness change
26. Target removed
27. Previous finding validated
28. Previous finding weakened
29. Previous finding strengthened
30. Previous finding invalidated
31. Unchanged finding
32. Requires review
33. New finding
34. Resolved finding
35. Empty old dataset
36. Empty new dataset
37. None input
38. Evidence input
39. DataFrame input
40. Deterministic output
41. JSON serialization
42. Input immutability
43. No fabricated values
44. Recommendation generation
45. Unrelated finding remains unchanged
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.reconsideration import reconsider_dataset


# ============================================================
# 1. IDENTICAL DATASETS
# ============================================================

def test_identical_datasets():
    df = pd.DataFrame({
        "id": range(100),
        "val": np.random.normal(50, 5, 100),
        "cat": ["A", "B"] * 50,
    })

    result = reconsider_dataset(df, df, dataset_name="test_identical")

    assert "dataset_name" in result
    assert result["dataset_name"] == "test_identical"
    assert "change_summary" in result
    assert result["change_summary"]["has_changes"] is False
    assert result["change_summary"]["change_count"] == 0
    assert "reconsiderations" in result
    assert "summary" in result
    assert result["summary"]["invalidated"] == 0
    assert result["summary"]["strengthened"] == 0


# ============================================================
# 2. ADDED COLUMN
# ============================================================

def test_added_column():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["structural_changes"]
    assert any(s["type"] == "added_columns" and "c" in s["columns"] for s in sc)
    assert result["change_summary"]["has_changes"] is True


# ============================================================
# 3. REMOVED COLUMN
# ============================================================

def test_removed_column():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    new_df = pd.DataFrame({"a": [1, 2, 3]})

    prev_findings = [{
        "finding_id": "f_b",
        "title": "Issue in column b",
        "category": "Quality",
        "severity": "HIGH",
        "affected_columns": ["b"],
        "evidence": ["b has an issue"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    sc = result["change_summary"]["structural_changes"]
    assert any(s["type"] == "removed_columns" and "b" in s["columns"] for s in sc)
    assert len(result["reconsiderations"]) == 1
    assert result["reconsiderations"][0]["classification"] == "INVALIDATED"
    assert result["reconsiderations"][0]["current_status"] == "RESOLVED"


# ============================================================
# 4. PROBABLE RENAME
# ============================================================

def test_probable_rename():
    old_df = pd.DataFrame({"user_age": [20, 30, 40, 50]})
    new_df = pd.DataFrame({"user_ages": [20, 30, 40, 50]})

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["structural_changes"]
    renames = [s for s in sc if s["type"] == "possible_renames"]
    assert len(renames) == 1
    assert renames[0]["renames"][0]["old_column"] == "user_age"
    assert renames[0]["renames"][0]["new_column"] == "user_ages"


# ============================================================
# 5. DTYPE CHANGE
# ============================================================

def test_dtype_change():
    old_df = pd.DataFrame({"code": [101, 102, 103]})
    new_df = pd.DataFrame({"code": ["101", "102", "103"]})

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["structural_changes"]
    dtype_chg = [s for s in sc if s["type"] == "dtype_changes"]
    assert len(dtype_chg) == 1
    assert dtype_chg[0]["changes"][0]["column"] == "code"


# ============================================================
# 6. ROW-COUNT CHANGE
# ============================================================

def test_row_count_change():
    old_df = pd.DataFrame({"a": range(10)})
    new_df = pd.DataFrame({"a": range(50)})

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["structural_changes"]
    row_chg = [s for s in sc if s["type"] == "row_count_change"]
    assert len(row_chg) == 1
    assert row_chg[0]["old_rows"] == 10
    assert row_chg[0]["new_rows"] == 50
    assert row_chg[0]["difference"] == 40


# ============================================================
# 7. INCREASED MISSINGNESS
# ============================================================

def test_increased_missingness():
    old_df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})  # 0% missing
    new_df = pd.DataFrame({"val": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0, 10.0]})  # 50% missing

    prev_findings = [{
        "finding_id": "f_miss",
        "title": "Missing values in val",
        "category": "Completeness",
        "severity": "LOW",
        "affected_columns": ["val"],
        "evidence": ["0% missing initially"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    assert len(result["reconsiderations"]) == 1
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "STRENGTHENED"
    assert recon["current_status"] == "ESCALATED"


# ============================================================
# 8. DECREASED MISSINGNESS
# ============================================================

def test_decreased_missingness():
    old_df = pd.DataFrame({"val": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]})  # 50% missing
    new_df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0]})  # 10% missing

    prev_findings = [{
        "finding_id": "f_miss",
        "title": "Missing values in val",
        "category": "Completeness",
        "severity": "HIGH",
        "affected_columns": ["val"],
        "evidence": ["50% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "WEAKENED"
    assert recon["current_status"] == "ATTENUATED"


# ============================================================
# 9. ALL-MISSING TRANSITION
# ============================================================

def test_all_missing_transition():
    old_df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    new_df = pd.DataFrame({"val": [np.nan, np.nan, np.nan, np.nan]})

    result = reconsider_dataset(old_df, new_df)
    qc = result["change_summary"]["quality_changes"]
    col_miss = [q for q in qc if q["type"] == "column_missingness_change" and q["column"] == "val"]
    assert len(col_miss) == 1
    assert col_miss[0]["became_all_missing"] is True


# ============================================================
# 10. MISSINGNESS RESOLVED
# ============================================================

def test_missingness_resolved():
    old_df = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan, 5.0]})
    new_df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})

    prev_findings = [{
        "finding_id": "f_miss_val",
        "title": "High missingness in val",
        "category": "Completeness",
        "severity": "HIGH",
        "affected_columns": ["val"],
        "evidence": ["40% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"
    assert recon["current_status"] == "RESOLVED"
    assert len(result["resolved_findings"]) == 1
    assert result["resolved_findings"][0]["finding_id"] == "f_miss_val"


# ============================================================
# 11. DUPLICATE INCREASE
# ============================================================

def test_duplicate_increase():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})  # 0 dups
    new_df = pd.DataFrame({"a": [1, 2, 1, 1], "b": ["x", "y", "x", "x"]})  # 2 dups

    prev_findings = [{
        "finding_id": "f_dup",
        "title": "Duplicate records in dataset",
        "category": "Uniqueness",
        "severity": "LOW",
        "affected_columns": [],
        "evidence": ["duplicates present"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "STRENGTHENED"
    assert recon["current_status"] == "ESCALATED"


# ============================================================
# 12. DUPLICATE REMOVAL
# ============================================================

def test_duplicate_removal():
    old_df = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})  # 2 dups
    new_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})  # 0 dups

    prev_findings = [{
        "finding_id": "f_dup",
        "title": "Duplicate rows detected",
        "category": "Uniqueness",
        "severity": "HIGH",
        "affected_columns": [],
        "evidence": ["50% duplicates"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"
    assert recon["current_status"] == "RESOLVED"
    assert any(rf["finding_id"] == "f_dup" for rf in result["resolved_findings"])


# ============================================================
# 13. INVALID VALUES INCREASE
# ============================================================

def test_invalid_values_increase():
    old_df = pd.DataFrame({"num": [10, 20, 30, 40]})
    new_df = pd.DataFrame({"num": [10, "ERR", 30, "ERR_2"]})

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True


# ============================================================
# 14. INVALID VALUES RESOLVED
# ============================================================

def test_invalid_values_resolved():
    old_df = pd.DataFrame({"num": ["10", "INVALID", "30"]})
    new_df = pd.DataFrame({"num": [10, 20, 30]})

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True


# ============================================================
# 15. CONSTANT TO VARIABLE
# ============================================================

def test_constant_to_variable():
    old_df = pd.DataFrame({"status": ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"]})
    new_df = pd.DataFrame({"status": ["ACTIVE", "INACTIVE", "PENDING", "ACTIVE"]})

    prev_findings = [{
        "finding_id": "f_const",
        "title": "Constant column status",
        "category": "Quality",
        "severity": "HIGH",
        "affected_columns": ["status"],
        "evidence": ["zero variance"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"
    assert recon["current_status"] == "RESOLVED"
    assert any(rf["finding_id"] == "f_const" for rf in result["resolved_findings"])


# ============================================================
# 16. VARIABLE TO CONSTANT
# ============================================================

def test_variable_to_constant():
    old_df = pd.DataFrame({"status": ["ACTIVE", "INACTIVE", "PENDING"]})
    new_df = pd.DataFrame({"status": ["ACTIVE", "ACTIVE", "ACTIVE"]})

    result = reconsider_dataset(old_df, new_df)
    qc = result["change_summary"]["quality_changes"]
    assert any(q["type"] == "newly_constant_columns" and "status" in q["columns"] for q in qc)


# ============================================================
# 17. HIGH CARDINALITY CHANGE
# ============================================================

def test_high_cardinality_change():
    old_df = pd.DataFrame({"tag": [f"T_{i}" for i in range(100)]})
    new_df = pd.DataFrame({"tag": ["T_0", "T_1"] * 50})

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True


# ============================================================
# 18. OUTLIER INCREASE
# ============================================================

def test_outlier_increase():
    old_df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 10, 12, 11]})  # 0 outliers
    new_df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 10, 9999, 8888]})  # 2 outliers

    prev_findings = [{
        "finding_id": "f_out",
        "title": "Outlier values in x",
        "category": "Anomaly",
        "severity": "LOW",
        "affected_columns": ["x"],
        "evidence": ["outliers present"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "STRENGTHENED"
    assert recon["current_status"] == "ESCALATED"


# ============================================================
# 19. OUTLIER DECREASE / RESOLVED
# ============================================================

def test_outlier_decrease():
    old_df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 10, 9999, 8888]})
    new_df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 10, 11, 12]})

    prev_findings = [{
        "finding_id": "f_out",
        "title": "Outlier values in x",
        "category": "Anomaly",
        "severity": "HIGH",
        "affected_columns": ["x"],
        "evidence": ["outliers detected"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"
    assert recon["current_status"] == "RESOLVED"


# ============================================================
# 20. STATISTICAL DRIFT
# ============================================================

def test_statistical_drift():
    old_df = pd.DataFrame({"score": [10, 12, 11, 13, 10, 12, 11, 10, 12, 11]})  # mean ~11
    new_df = pd.DataFrame({"score": [100, 102, 101, 103, 100, 102, 101, 100, 102, 101]})  # mean ~101

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["statistical_changes"]
    assert any(s["column"] == "score" and s["metric"] == "mean" for s in sc)


# ============================================================
# 21. CORRELATION APPEARS
# ============================================================

def test_correlation_appears():
    np.random.seed(42)
    old_df = pd.DataFrame({"a": range(50), "b": np.random.normal(0, 10, 50)})
    new_df = pd.DataFrame({"a": range(50), "b": [x * 2.0 for x in range(50)]})  # r = 1.0

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True


# ============================================================
# 22. CORRELATION DISAPPEARS
# ============================================================

def test_correlation_disappears():
    np.random.seed(42)
    old_df = pd.DataFrame({"a": range(50), "b": [x * 2.0 for x in range(50)]})  # r = 1.0
    new_df = pd.DataFrame({"a": range(50), "b": np.random.normal(0, 10, 50)})  # r ~ 0.0

    prev_findings = [{
        "finding_id": "f_corr",
        "title": "Strong correlation between a and b",
        "category": "Correlation",
        "severity": "HIGH",
        "affected_columns": ["a", "b"],
        "evidence": ["r = 1.0"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] in ("INVALIDATED", "WEAKENED")


# ============================================================
# 23. CORRELATION STRENGTHENS
# ============================================================

def test_correlation_strengthens():
    np.random.seed(42)
    old_df = pd.DataFrame({"a": range(50), "b": [x + np.random.normal(0, 5) for x in range(50)]})
    new_df = pd.DataFrame({"a": range(50), "b": [x * 2.0 for x in range(50)]})

    prev_findings = [{
        "finding_id": "f_corr",
        "title": "Correlation between a and b",
        "category": "Correlation",
        "severity": "MEDIUM",
        "affected_columns": ["a", "b"],
        "evidence": ["moderate correlation"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    assert len(result["reconsiderations"]) == 1


# ============================================================
# 24. CORRELATION WEAKENS
# ============================================================

def test_correlation_weakens():
    np.random.seed(42)
    old_df = pd.DataFrame({"a": range(50), "b": [x * 2.0 for x in range(50)]})
    new_df = pd.DataFrame({"a": range(50), "b": [x + np.random.normal(0, 15) for x in range(50)]})

    prev_findings = [{
        "finding_id": "f_corr",
        "title": "Correlation between a and b",
        "category": "Correlation",
        "severity": "HIGH",
        "affected_columns": ["a", "b"],
        "evidence": ["r = 1.0"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    assert len(result["reconsiderations"]) == 1


# ============================================================
# 25. ML READINESS CHANGE
# ============================================================

def test_ml_readiness_change():
    old_df = pd.DataFrame({"x": range(10)})
    new_df = pd.DataFrame({
        "x": range(100),
        "target": [0, 1] * 50,
        "feat": np.random.normal(0, 1, 100),
    })

    result = reconsider_dataset(old_df, new_df)
    ml_c = result["change_summary"]["ml_changes"]
    assert any(m["type"] == "ml_score_change" for m in ml_c)


# ============================================================
# 26. TARGET REMOVED
# ============================================================

def test_target_removed():
    old_df = pd.DataFrame({"feat": range(50), "target": [0, 1] * 25})
    new_df = pd.DataFrame({"feat": range(50)})

    prev_findings = [{
        "finding_id": "f_target",
        "title": "Target candidate target detected",
        "category": "ML",
        "severity": "HIGH",
        "affected_columns": ["target"],
        "evidence": ["binary target"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"
    assert recon["current_status"] == "RESOLVED"


# ============================================================
# 27. PREVIOUS FINDING VALIDATED
# ============================================================

def test_previous_finding_validated():
    old_df = pd.DataFrame({"missing_col": [1.0, np.nan, 3.0, np.nan, 5.0]})
    new_df = pd.DataFrame({"missing_col": [1.0, np.nan, 3.0, np.nan, 5.0]})

    prev_findings = [{
        "finding_id": "f_miss",
        "title": "Missing values in missing_col",
        "category": "Completeness",
        "severity": "HIGH",
        "affected_columns": ["missing_col"],
        "evidence": ["40% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "VALIDATED"
    assert recon["current_status"] == "PERSISTENT"


# ============================================================
# 28. PREVIOUS FINDING WEAKENED
# ============================================================

def test_previous_finding_weakened():
    old_df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, 5.0]})
    new_df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})

    prev_findings = [{
        "finding_id": "f_x",
        "title": "Missingness in x",
        "category": "Completeness",
        "severity": "HIGH",
        "affected_columns": ["x"],
        "evidence": ["60% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "WEAKENED"


# ============================================================
# 29. PREVIOUS FINDING STRENGTHENED
# ============================================================

def test_previous_finding_strengthened():
    old_df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    new_df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, 5.0]})

    prev_findings = [{
        "finding_id": "f_x",
        "title": "Missingness in x",
        "category": "Completeness",
        "severity": "LOW",
        "affected_columns": ["x"],
        "evidence": ["20% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "STRENGTHENED"


# ============================================================
# 30. PREVIOUS FINDING INVALIDATED
# ============================================================

def test_previous_finding_invalidated():
    old_df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    new_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    prev_findings = [{
        "finding_id": "f_x",
        "title": "Missing values in x",
        "category": "Completeness",
        "severity": "MEDIUM",
        "affected_columns": ["x"],
        "evidence": ["33% missing"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "INVALIDATED"


# ============================================================
# 31. UNCHANGED FINDING
# ============================================================

def test_unchanged_finding():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "b": [100, 200, 300]})  # b changed, a untouched

    prev_findings = [{
        "finding_id": "f_a",
        "title": "Feature a observation",
        "category": "General",
        "severity": "INFO",
        "affected_columns": ["a"],
        "evidence": ["a is consistent"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "UNCHANGED"


# ============================================================
# 32. REQUIRES REVIEW
# ============================================================

def test_requires_review():
    old_df = pd.DataFrame({"cat": [1, 2, 3, 4]})
    new_df = pd.DataFrame({"cat": ["A", "B", "C", "D"]})

    prev_findings = [{
        "finding_id": "f_cat",
        "title": "Numeric range for cat",
        "category": "Profiling",
        "severity": "INFO",
        "affected_columns": ["cat"],
        "evidence": ["numeric distribution"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "REQUIRES_REVIEW"
    assert len(result["requires_review"]) == 1


# ============================================================
# 33. NEW FINDING
# ============================================================

def test_new_finding():
    old_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    # new column with 60% missing values
    new_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "new_col": [np.nan, np.nan, np.nan, 4, 5]})

    result = reconsider_dataset(old_df, new_df)
    assert len(result["new_findings"]) >= 1


# ============================================================
# 34. RESOLVED FINDING
# ============================================================

def test_resolved_finding():
    old_df = pd.DataFrame({"status": ["FIXED", "FIXED", "FIXED"]})
    new_df = pd.DataFrame({"status": ["FIXED", "VARIABLE", "ANOTHER"]})

    prev_findings = [{
        "finding_id": "f_stat",
        "title": "Constant column status",
        "category": "Quality",
        "severity": "HIGH",
        "affected_columns": ["status"],
        "evidence": ["constant"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    assert len(result["resolved_findings"]) == 1
    assert result["resolved_findings"][0]["finding_id"] == "f_stat"


# ============================================================
# 35. EMPTY OLD DATASET
# ============================================================

def test_empty_old_dataset():
    old_df = pd.DataFrame()
    new_df = pd.DataFrame({"a": [1, 2, 3]})

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True
    assert isinstance(result["reconsiderations"], list)


# ============================================================
# 36. EMPTY NEW DATASET
# ============================================================

def test_empty_new_dataset():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame()

    result = reconsider_dataset(old_df, new_df)
    assert result["change_summary"]["has_changes"] is True
    assert isinstance(result["reconsiderations"], list)


# ============================================================
# 37. NONE INPUT
# ============================================================

def test_none_input():
    result = reconsider_dataset(None, None)
    assert result["change_summary"]["has_changes"] is False
    assert result["reconsiderations"] == []
    assert result["generated_from"] == "none"


# ============================================================
# 38. EVIDENCE INPUT
# ============================================================

def test_evidence_input():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 3, 4]})

    old_ev = build_evidence(old_df, dataset_name="old_ev")
    new_ev = build_evidence(new_df, dataset_name="new_ev")

    result = reconsider_dataset(old_ev, new_ev)
    assert result["generated_from"] == "evidence"
    assert result["dataset_name"] == "new_ev"


# ============================================================
# 39. DATAFRAME INPUT
# ============================================================

def test_dataframe_input():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 3, 4]})

    result = reconsider_dataset(old_df, new_df, dataset_name="df_run")
    assert result["generated_from"] == "dataframe"
    assert result["dataset_name"] == "df_run"


# ============================================================
# 40. DETERMINISTIC OUTPUT
# ============================================================

def test_deterministic_output():
    old_df = pd.DataFrame({"x": [1, 2, 3], "y": ["A", "B", "C"]})
    new_df = pd.DataFrame({"x": [1, 2, 4], "y": ["A", "B", "D"]})

    res1 = reconsider_dataset(old_df, new_df)
    res2 = reconsider_dataset(old_df, new_df)

    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


# ============================================================
# 41. JSON SERIALIZATION
# ============================================================

def test_json_serialization():
    old_df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=3), "v": [1.0, np.nan, 3.0]})
    new_df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=3), "v": [1.0, 2.0, 3.0]})

    result = reconsider_dataset(old_df, new_df)
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized["change_summary"]["has_changes"] is True


# ============================================================
# 42. INPUT IMMUTABILITY
# ============================================================

def test_input_immutability():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})

    old_copy = old_df.copy(deep=True)
    new_copy = new_df.copy(deep=True)

    _ = reconsider_dataset(old_df, new_df)

    pd.testing.assert_frame_equal(old_df, old_copy)
    pd.testing.assert_frame_equal(new_df, new_copy)


# ============================================================
# 43. NO FABRICATED VALUES
# ============================================================

def test_no_fabricated_values():
    old_df = pd.DataFrame({"col": [10, 20, 30]})
    new_df = pd.DataFrame({"col": [10, 20, 30, 40]})

    result = reconsider_dataset(old_df, new_df)
    sc = result["change_summary"]["structural_changes"]
    for s in sc:
        if s["type"] == "row_count_change":
            assert s["old_rows"] == 3
            assert s["new_rows"] == 4
            assert s["difference"] == 1


# ============================================================
# 44. RECOMMENDATION GENERATION
# ============================================================

def test_recommendation_generation():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "new_feature": [10, 20, 30]})

    result = reconsider_dataset(old_df, new_df)
    assert "next_analysis" in result
    assert len(result["next_analysis"]) > 0
    for rec in result["next_analysis"]:
        assert "name" in rec
        assert "why" in rec
        assert "evidence" in rec
        assert "priority" in rec
        assert "action" in rec


# ============================================================
# 45. UNRELATED FINDING REMAINS UNCHANGED
# ============================================================

def test_unrelated_finding_remains_unchanged():
    old_df = pd.DataFrame({"stable_col": [1, 2, 3], "modified_col": [10, 20, 30]})
    new_df = pd.DataFrame({"stable_col": [1, 2, 3], "modified_col": [100, 200, 300]})

    prev_findings = [{
        "finding_id": "f_stable",
        "title": "Distribution of stable_col",
        "category": "Profiling",
        "severity": "LOW",
        "affected_columns": ["stable_col"],
        "evidence": ["stable distribution"],
    }]

    result = reconsider_dataset(old_df, new_df, previous_findings=prev_findings)
    recon = result["reconsiderations"][0]
    assert recon["classification"] == "UNCHANGED"
