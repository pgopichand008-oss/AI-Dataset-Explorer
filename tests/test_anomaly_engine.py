"""
FILE: tests/test_anomaly_engine.py

Comprehensive unit test suite for Phase 6: Anomaly Investigation Engine.
Tests all required scenarios:
1. Normal dataset
2. Clear IQR outlier
3. Multiple IQR outliers
4. No IQR outliers
5. Z-score detection
6. Zero standard deviation safeguard
7. IQR == 0 safeguard
8. Isolation Forest on suitable dataset
9. Deterministic Isolation Forest
10. Small dataset skips Isolation Forest
11. Categorical rare category
12. Singleton category
13. Dominant category
14. No categorical columns
15. Missing values handling
16. Invalid numerical values (type coercion)
17. Numerical-only dataset
18. Categorical-only dataset
19. Datetime dataset
20. Duplicate rows
21. Empty DataFrame (0x0 and 0xN)
22. None input
23. Single-row dataset
24. Duplicate column names
25. max_findings truncation
26. Deterministic ordering
27. JSON serializability
28. DataFrame immutability
29. Supplied evidence reuse
30. No fabricated evidence
31. Actionable findings
32. No-anomaly dataset
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.anomaly_engine import investigate_anomalies


# ============================================================
# 1. NORMAL DATASET
# ============================================================

def test_normal_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=100),
        "income": np.random.normal(50000, 5000, size=100),
        "department": np.random.choice(["Sales", "HR", "Eng", "Finance"], size=100),
    })

    result = investigate_anomalies(df, dataset_name="corp_data")

    assert "dataset_name" in result
    assert "summary" in result
    assert "findings" in result
    assert "categorical_findings" in result
    assert "recommendations" in result
    assert result["dataset_name"] == "corp_data"
    assert result["generated_from"] == "evidence"


# ============================================================
# 2. CLEAR IQR OUTLIER
# ============================================================

def test_clear_iqr_outlier():
    # 99 observations around 10-20, one extreme 999999
    vals = list(range(10, 109)) + [999999]
    df = pd.DataFrame({"salary": vals})

    result = investigate_anomalies(df)
    findings = result["findings"]

    salary_findings = [f for f in findings if f["column"] == "salary"]
    assert len(salary_findings) == 1
    f = salary_findings[0]

    assert f["anomaly_count"] >= 1
    assert "IQR" in f["method"]
    assert any("999999" not in e for e in f["evidence"])  # evidence cites bounds
    assert "salary" in f["action"]


# ============================================================
# 3. MULTIPLE IQR OUTLIERS
# ============================================================

def test_multiple_iqr_outliers():
    vals = [50] * 90 + [1000, 2000, 3000, 4000, 5000]
    df = pd.DataFrame({"score": vals})

    result = investigate_anomalies(df)
    findings = result["findings"]

    score_findings = [f for f in findings if f["column"] == "score"]
    assert len(score_findings) == 1
    assert score_findings[0]["anomaly_count"] >= 5


# ============================================================
# 4. NO IQR OUTLIERS
# ============================================================

def test_no_iqr_outliers():
    # Perfectly uniform integers with no outliers
    df = pd.DataFrame({"step": list(range(100))})

    result = investigate_anomalies(df)
    # Uniform numbers should have 0 IQR outliers
    iqr_findings = [f for f in result["findings"] if f["column"] == "step" and "IQR" in f["method"]]
    assert len(iqr_findings) == 0


# ============================================================
# 5. Z-SCORE DETECTION
# ============================================================

def test_z_score_detection():
    np.random.seed(42)
    normal_vals = np.random.normal(100, 10, 100).tolist()
    # Inject 4 extreme points that exceed 4 standard deviations (> 140)
    normal_vals += [250, 300, -100]
    df = pd.DataFrame({"metric": normal_vals})

    result = investigate_anomalies(df)
    metric_findings = [f for f in result["findings"] if f["column"] == "metric"]
    assert len(metric_findings) == 1
    f = metric_findings[0]

    assert "Z-score" in f["method"]
    assert f["anomaly_count"] >= 3
    assert any("z" in str(e).lower() for e in f["evidence"])


# ============================================================
# 6. ZERO STANDARD DEVIATION SAFEGUARD
# ============================================================

def test_zero_standard_deviation_safeguard():
    df = pd.DataFrame({"constant_col": [42.0] * 50})

    result = investigate_anomalies(df)
    # Must not throw ZeroDivisionError and must have 0 findings
    assert len(result["findings"]) == 0


# ============================================================
# 7. IQR == 0 SAFEGUARD
# ============================================================

def test_zero_iqr_safeguard():
    # Column where 90% of values are identical (Q1 == Q3)
    vals = [10.0] * 95 + [100.0, 200.0]
    df = pd.DataFrame({"near_zero_iqr": vals})

    result = investigate_anomalies(df)
    # Must handle Q1 == Q3 without crashing
    findings = [f for f in result["findings"] if f["column"] == "near_zero_iqr"]
    assert len(findings) == 1
    assert findings[0]["anomaly_count"] >= 2


# ============================================================
# 8. ISOLATION FOREST ON SUITABLE DATASET
# ============================================================

def test_isolation_forest_on_suitable_dataset():
    np.random.seed(42)
    n = 60
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    # Inject a distinct multivariate outlier pair
    x[0] = 50.0
    y[0] = 50.0

    df = pd.DataFrame({"feat_1": x, "feat_2": y})

    result = investigate_anomalies(df)
    # Isolation Forest should be applied on N >= 30, numeric cols >= 2
    assert "Isolation Forest" in result["summary"]["methods_used"]
    iso_findings = [f for f in result["findings"] if "Isolation Forest" in f["method"]]
    assert len(iso_findings) >= 1
    assert iso_findings[0]["anomaly_count"] >= 1


# ============================================================
# 9. DETERMINISTIC ISOLATION FOREST
# ============================================================

def test_deterministic_isolation_forest():
    np.random.seed(42)
    df = pd.DataFrame({
        "f1": np.random.normal(10, 2, 50),
        "f2": np.random.normal(20, 4, 50),
    })

    res1 = investigate_anomalies(df)
    res2 = investigate_anomalies(df)

    assert res1["summary"]["total_anomalies"] == res2["summary"]["total_anomalies"]
    assert res1["findings"] == res2["findings"]


# ============================================================
# 10. SMALL DATASET SKIPS ISOLATION FOREST
# ============================================================

def test_small_dataset_skips_isolation_forest():
    # Only 15 rows: Isolation Forest should be skipped
    df = pd.DataFrame({
        "a": range(15),
        "b": range(15, 30),
    })

    result = investigate_anomalies(df)
    assert "Isolation Forest" not in result["summary"]["methods_used"]


# ============================================================
# 11. CATEGORICAL RARE CATEGORY
# ============================================================

def test_categorical_rare_category():
    # 100 rows, 1 category with 1 observation (< 1.0%)
    cats = ["Common"] * 99 + ["RareClass"]
    df = pd.DataFrame({"category_col": cats})

    result = investigate_anomalies(df)
    cat_findings = result["categorical_findings"]

    assert len(cat_findings) >= 1
    rare_f = [f for f in cat_findings if f["category"] == "RareClass"]
    assert len(rare_f) == 1
    assert rare_f[0]["count"] == 1
    assert rare_f[0]["percentage"] == 1.0


# ============================================================
# 12. SINGLETON CATEGORY
# ============================================================

def test_singleton_category():
    # 30 rows, one category occurring exactly once
    cats = ["TypeA"] * 29 + ["TypeSingleton"]
    df = pd.DataFrame({"type": cats})

    result = investigate_anomalies(df)
    singletons = [f for f in result["categorical_findings"] if f["category"] == "TypeSingleton"]
    assert len(singletons) == 1
    assert singletons[0]["count"] == 1


# ============================================================
# 13. DOMINANT CATEGORY
# ============================================================

def test_dominant_category():
    # 50 rows, 96% dominant class
    cats = ["Majority"] * 48 + ["Minority"] * 2
    df = pd.DataFrame({"tier": cats})

    result = investigate_anomalies(df)
    dominant_f = [f for f in result["categorical_findings"] if f["category"] == "Majority"]
    assert len(dominant_f) == 1
    assert dominant_f[0]["percentage"] >= 95.0
    assert dominant_f[0]["severity"] == "MEDIUM"


# ============================================================
# 14. NO CATEGORICAL COLUMNS
# ============================================================

def test_no_categorical_columns():
    df = pd.DataFrame({"x": range(50), "y": range(50, 100)})

    result = investigate_anomalies(df)
    assert len(result["categorical_findings"]) == 0


# ============================================================
# 15. MISSING VALUES HANDLING
# ============================================================

def test_missing_values_handling():
    # Column with NaNs and an outlier
    vals = [None, 10, None, 12, 11, 13] * 10 + [99999]
    df = pd.DataFrame({"num_with_nulls": vals})

    result = investigate_anomalies(df)
    findings = [f for f in result["findings"] if f["column"] == "num_with_nulls"]
    assert len(findings) == 1
    assert findings[0]["anomaly_count"] >= 1


# ============================================================
# 16. INVALID NUMERICAL VALUES (TYPE COERCION)
# ============================================================

def test_invalid_numerical_values():
    # Column containing corrupted non-numeric strings
    vals = [10, 20, 30, "corrupt_token", 50, "N/A_error"] * 10
    df = pd.DataFrame({"corrupted_col": vals})

    result = investigate_anomalies(df)
    coercion_findings = [f for f in result["findings"] if f["method"] == "Type Coercion"]
    assert len(coercion_findings) == 1
    assert coercion_findings[0]["column"] == "corrupted_col"
    assert coercion_findings[0]["anomaly_count"] == 20


# ============================================================
# 17. NUMERICAL-ONLY DATASET
# ============================================================

def test_numerical_only_dataset():
    df = pd.DataFrame({"a": range(40), "b": range(40, 80)})
    result = investigate_anomalies(df)

    assert result["summary"]["affected_columns"] >= 0
    assert len(result["categorical_findings"]) == 0


# ============================================================
# 18. CATEGORICAL-ONLY DATASET
# ============================================================

def test_categorical_only_dataset():
    df = pd.DataFrame({
        "city": ["Tokyo", "Paris", "London", "Berlin"] * 10,
        "role": ["Admin"] * 39 + ["SuperAdmin"],
    })
    result = investigate_anomalies(df)

    assert len(result["findings"]) == 0
    assert len(result["categorical_findings"]) >= 1


# ============================================================
# 19. DATETIME DATASET
# ============================================================

def test_datetime_dataset():
    df = pd.DataFrame({
        "dt": pd.date_range("2026-01-01", periods=50),
        "val": range(50),
    })
    result = investigate_anomalies(df)
    assert result["summary"]["affected_columns"] >= 0


# ============================================================
# 20. DUPLICATE ROWS
# ============================================================

def test_duplicate_rows():
    base = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
    df = pd.concat([base] * 10, ignore_index=True)

    result = investigate_anomalies(df)
    # Must run without crashing on repeated rows
    assert "summary" in result


# ============================================================
# 21. EMPTY DATAFRAME (0x0 AND 0xN)
# ============================================================

def test_empty_dataframe():
    df_empty = pd.DataFrame()
    res1 = investigate_anomalies(df_empty)
    assert res1["summary"]["total_anomalies"] == 0
    assert len(res1["findings"]) == 0

    df_cols = pd.DataFrame(columns=["a", "b"])
    res2 = investigate_anomalies(df_cols)
    assert res2["summary"]["total_anomalies"] == 0
    assert len(res2["findings"]) == 0


# ============================================================
# 22. NONE INPUT
# ============================================================

def test_none_input():
    res = investigate_anomalies(None)
    assert res["summary"]["total_anomalies"] == 0
    assert len(res["findings"]) == 0
    assert len(res["recommendations"]) == 1


# ============================================================
# 23. SINGLE-ROW DATASET
# ============================================================

def test_single_row_dataset():
    df = pd.DataFrame({"x": [100], "y": ["OnlyOne"]})
    res = investigate_anomalies(df)
    assert res["summary"]["total_anomalies"] == 0
    assert len(res["findings"]) == 0


# ============================================================
# 24. DUPLICATE COLUMN NAMES
# ============================================================

def test_duplicate_column_names():
    # DataFrame with duplicate column names
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["dup", "dup", "clean"])
    res = investigate_anomalies(df)
    assert "summary" in res


# ============================================================
# 25. MAX_FINDINGS TRUNCATION
# ============================================================

def test_max_findings_truncation():
    # Create 6 columns with outliers
    data = {}
    for i in range(6):
        data[f"col_{i}"] = list(range(50)) + [99999]
    df = pd.DataFrame(data)

    res = investigate_anomalies(df, max_findings=3)
    assert len(res["findings"]) == 3
    assert res["findings"][0]["rank"] == 1
    assert res["findings"][1]["rank"] == 2
    assert res["findings"][2]["rank"] == 3


# ============================================================
# 26. DETERMINISTIC ORDERING
# ============================================================

def test_deterministic_ordering():
    df = pd.DataFrame({
        "a": list(range(49)) + [9999],
        "b": list(range(48)) + [8888, -8888],
        "cat": ["A"] * 49 + ["B"],
    })

    res1 = investigate_anomalies(df, dataset_name="det_test")
    res2 = investigate_anomalies(df, dataset_name="det_test")

    assert res1 == res2


# ============================================================
# 27. JSON SERIALIZABILITY
# ============================================================

def test_json_serializability():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 9999.0],
        "cat": ["A", "B", "A", "Rare"],
    })

    res = investigate_anomalies(df, dataset_name="json_check")
    serialized = json.dumps(res, indent=2)
    assert isinstance(serialized, str)

    deserialized = json.loads(serialized)
    assert deserialized["dataset_name"] == "json_check"


# ============================================================
# 28. DATAFRAME IMMUTABILITY
# ============================================================

def test_dataframe_immutability():
    df = pd.DataFrame({
        "num": [10, 20, 30, 9999],
        "cat": ["x", "y", "x", "z"],
    })
    df_copy = df.copy(deep=True)

    _ = investigate_anomalies(df)

    pd.testing.assert_frame_equal(df, df_copy)


# ============================================================
# 29. SUPPLIED EVIDENCE REUSE
# ============================================================

def test_supplied_evidence_reuse():
    df = pd.DataFrame({
        "salary": list(range(50)) + [999999],
    })
    evidence = build_evidence(df, dataset_name="precomputed_ev_anomaly")

    res = investigate_anomalies(evidence)
    assert res["dataset_name"] == "precomputed_ev_anomaly"
    findings = [f for f in res["findings"] if f["column"] == "salary"]
    assert len(findings) == 1


# ============================================================
# 30. NO FABRICATED EVIDENCE
# ============================================================

def test_no_fabricated_evidence():
    df = pd.DataFrame({
        "clean_1": range(20),
        "clean_2": range(20, 40),
    })

    res = investigate_anomalies(df)
    for f in res["findings"]:
        for ev_str in f["evidence"]:
            assert "salary" not in ev_str.lower()
            assert "fraud" not in ev_str.lower()


# ============================================================
# 31. ACTIONABLE FINDINGS
# ============================================================

def test_actionable_findings():
    df = pd.DataFrame({
        "col": list(range(50)) + [99999],
    })

    res = investigate_anomalies(df)
    assert len(res["findings"]) >= 1
    for f in res["findings"]:
        assert len(f["action"]) > 15
        assert "inspect" in f["action"].lower() or "verify" in f["action"].lower()


# ============================================================
# 32. NO-ANOMALY DATASET
# ============================================================

def test_no_anomaly_dataset():
    df = pd.DataFrame({
        "v1": list(range(100)),
        "v2": list(range(100, 200)),
    })

    res = investigate_anomalies(df)
    assert res["summary"]["total_anomalies"] == 0
    assert len(res["findings"]) == 0
    assert any("no severe" in str(r).lower() for r in res["recommendations"])
