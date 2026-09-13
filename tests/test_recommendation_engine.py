"""
FILE: tests/test_recommendation_engine.py

Comprehensive unit test suite for Phase 5: Next Best Analysis Agent.
Tests all required scenarios:
1. Normal dataset
2. Missingness recommendation
3. Anomaly recommendation
4. Correlation recommendation
5. Categorical distribution recommendation
6. Target relationship recommendation
7. ML readiness recommendation (blockers)
8. ML readiness recommendation (predictive modeling when ready)
9. Duplicate rows recommendation
10. Constant-column recommendation
11. Invalid-values recommendation
12. No-target dataset (verifies NO target recommendation is created)
13. No-correlation dataset (verifies NO correlation recommendation is created)
14. Empty DataFrame (0x0 and 0xN)
15. None input
16. Single-row dataset
17. max_recommendations truncation
18. Deterministic ordering
19. JSON serializability
20. Input DataFrame immutability
21. Supplied evidence reuse
22. Supplied health reuse
23. Supplied prioritized insights reuse
24. Factual evidence grounding in recommendations
25. No fabricated values
26. Actionability of recommendations
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.recommendation_engine import recommend_next_analysis


# ============================================================
# 1. NORMAL DATASET
# ============================================================

def test_normal_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=100),
        "income": np.random.normal(50000, 10000, size=100),
        "department": np.random.choice(["Sales", "HR", "Eng", "Finance"], size=100),
    })

    result = recommend_next_analysis(df, dataset_name="healthy_corp")

    assert "dataset_name" in result
    assert "recommendations" in result
    assert "summary" in result
    assert result["dataset_name"] == "healthy_corp"
    assert result["generated_from"] == "evidence"

    # All recommendations must conform to contract
    for rec in result["recommendations"]:
        assert "rank" in rec
        assert "name" in rec
        assert "category" in rec
        assert "why" in rec
        assert isinstance(rec["evidence"], list)
        assert len(rec["evidence"]) > 0
        assert "expected_value" in rec
        assert rec["priority"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        assert "action" in rec


# ============================================================
# 2. MISSINGNESS RECOMMENDATION
# ============================================================

def test_missingness_recommendation():
    df = pd.DataFrame({
        "salary": [None, 50000, None, 70000] * 25,  # 50% missing
        "tenure": list(range(100)),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    miss_recs = [r for r in recs if r["category"] == "missingness"]
    assert len(miss_recs) == 1
    rec = miss_recs[0]

    assert "Missingness" in rec["name"]
    assert any("salary" in str(e).lower() for e in rec["evidence"])
    assert any("50.0%" in str(e) for e in rec["evidence"])
    assert rec["priority"] in ["CRITICAL", "HIGH"]


# ============================================================
# 3. ANOMALY RECOMMENDATION
# ============================================================

def test_anomaly_recommendation():
    vals = list(range(10, 100)) + [999999, -999999, 888888]
    df = pd.DataFrame({
        "feature_outlier": vals,
        "clean": list(range(len(vals))),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    anom_recs = [r for r in recs if r["category"] == "anomaly"]
    assert len(anom_recs) == 1
    rec = anom_recs[0]

    assert "Anomalies" in rec["name"] or "Anomaly" in rec["name"]
    assert any("feature_outlier" in str(e) for e in rec["evidence"])


# ============================================================
# 4. CORRELATION RECOMMENDATION
# ============================================================

def test_correlation_recommendation():
    np.random.seed(42)
    x = np.linspace(1, 100, 100)
    y = 2.5 * x + np.random.normal(0, 0.1, 100)  # r ~ 0.999
    df = pd.DataFrame({"x_var": x, "y_var": y})

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    corr_recs = [r for r in recs if r["category"] == "correlation"]
    assert len(corr_recs) == 1
    rec = corr_recs[0]

    assert "Correlation" in rec["name"] or "Relationship" in rec["name"]
    assert any("x_var" in str(e) and "y_var" in str(e) for e in rec["evidence"])
    # Non-causal verification
    assert "cause" not in rec["why"].lower()


# ============================================================
# 5. CATEGORICAL DISTRIBUTION RECOMMENDATION
# ============================================================

def test_categorical_distribution_recommendation():
    df = pd.DataFrame({
        "channel": ["Organic"] * 85 + ["Paid"] * 15,
        "amount": range(100),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    cat_recs = [r for r in recs if r["category"] == "category_distribution"]
    assert len(cat_recs) == 1
    rec = cat_recs[0]

    assert "Categorical" in rec["name"]
    assert any("channel" in str(e) for e in rec["evidence"])
    assert any("85.0%" in str(e) for e in rec["evidence"])


# ============================================================
# 6. TARGET RELATIONSHIP RECOMMENDATION
# ============================================================

def test_target_relationship_recommendation():
    # Pass an explicit target column via evidence
    df = pd.DataFrame({
        "churn": [0, 1] * 50,
        "feature_1": range(100),
        "feature_2": range(100, 200),
    })
    evidence = build_evidence(df, target_column="churn")

    result = recommend_next_analysis(evidence)
    recs = result["recommendations"]

    target_recs = [r for r in recs if r["category"] == "target_relationship"]
    assert len(target_recs) == 1
    rec = target_recs[0]

    assert "churn" in rec["name"]
    assert any("churn" in str(e) for e in rec["evidence"])


# ============================================================
# 7. ML READINESS RECOMMENDATION (BLOCKERS)
# ============================================================

def test_ml_readiness_blockers_recommendation():
    # Dataset unsuitable for ML: 5 rows only, severe missingness
    df = pd.DataFrame({
        "a": [1, None, None, 4, 5],
        "b": [None, 2, None, None, 5],
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    ml_recs = [r for r in recs if r["category"] == "ml_readiness"]
    assert len(ml_recs) == 1
    rec = ml_recs[0]

    assert "Resolve Machine Learning Readiness Blockers" in rec["name"]
    assert rec["priority"] in ["HIGH", "CRITICAL"]


# ============================================================
# 8. ML READINESS RECOMMENDATION (PREDICTIVE MODELING WHEN READY)
# ============================================================

def test_ml_predictive_modeling_when_ready():
    # Well-formed, clean tabular dataset with target
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, 100),
        "feat_2": np.random.normal(0, 1, 100),
        "target": np.random.choice([0, 1], 100),
    })
    evidence = build_evidence(df, target_column="target")

    result = recommend_next_analysis(evidence)
    recs = result["recommendations"]

    ml_recs = [r for r in recs if r["category"] in ["machine_learning", "ml_readiness"]]
    assert len(ml_recs) >= 1
    # Check if predictive modeling is recommended when clean
    assert any("predictive" in r["name"].lower() or "modeling" in r["name"].lower() or "readiness" in r["name"].lower() for r in ml_recs)


# ============================================================
# 9. DUPLICATE ROWS RECOMMENDATION
# ============================================================

def test_duplicate_rows_recommendation():
    base = pd.DataFrame({
        "id": list(range(10)),
        "val": [f"v_{i}" for i in range(10)],
    })
    df = pd.concat([base] * 6, ignore_index=True)  # 60 rows, 50 duplicates

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    struct_recs = [r for r in recs if r["category"] == "data_quality_structure"]
    assert len(struct_recs) == 1
    assert any("duplicate" in str(e).lower() for e in struct_recs[0]["evidence"])


# ============================================================
# 10. CONSTANT COLUMN RECOMMENDATION
# ============================================================

def test_constant_column_recommendation():
    df = pd.DataFrame({
        "fixed_flag": ["DEFAULT"] * 60,
        "varying": list(range(60)),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    struct_recs = [r for r in recs if r["category"] == "data_quality_structure"]
    assert len(struct_recs) == 1
    assert any("fixed_flag" in str(e) for e in struct_recs[0]["evidence"])


# ============================================================
# 11. INVALID VALUES RECOMMENDATION
# ============================================================

def test_invalid_values_recommendation():
    df = pd.DataFrame({
        "corrupted_num": [1, 2, "error_str", 4, "bad_val", 6] * 10,
        "clean": list(range(60)),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    anom_recs = [r for r in recs if r["category"] == "anomaly"]
    assert len(anom_recs) == 1
    assert any("corrupted_num" in str(e) for e in anom_recs[0]["evidence"])


# ============================================================
# 12. NO-TARGET DATASET
# ============================================================

def test_no_target_dataset():
    # If no target exists or detected, do NOT recommend target relationship analysis
    df = pd.DataFrame({
        "unrelated_a": range(50),
        "unrelated_b": range(50, 100),
    })
    # Target candidates forced to empty
    evidence = build_evidence(df)
    evidence["ml_readiness"]["evaluated_target"] = None
    evidence["ml_readiness"]["best_target_candidate"] = None
    evidence["ml_readiness"]["target_candidates"] = []

    result = recommend_next_analysis(evidence)
    recs = result["recommendations"]

    # MUST NOT have target_relationship recommendation
    assert not any(r["category"] == "target_relationship" for r in recs)


# ============================================================
# 13. NO-CORRELATION DATASET
# ============================================================

def test_no_correlation_dataset():
    # Independent random noise where |r| < 0.30
    np.random.seed(999)
    df = pd.DataFrame({
        "v1": np.random.normal(0, 1, 100),
        "v2": np.random.normal(0, 1, 100),
        "v3": np.random.normal(0, 1, 100),
    })

    result = recommend_next_analysis(df)
    recs = result["recommendations"]

    # MUST NOT recommend correlation investigation when no strong correlation exists
    assert not any(r["category"] == "correlation" for r in recs)


# ============================================================
# 14. EMPTY DATAFRAME (0x0 AND 0xN)
# ============================================================

def test_empty_dataframe():
    df_empty = pd.DataFrame()
    res1 = recommend_next_analysis(df_empty)
    assert len(res1["recommendations"]) >= 1
    assert res1["recommendations"][0]["priority"] == "CRITICAL"
    assert "empty" in res1["recommendations"][0]["why"].lower() or "0" in str(res1["recommendations"][0]["evidence"])

    df_cols = pd.DataFrame(columns=["col1", "col2"])
    res2 = recommend_next_analysis(df_cols)
    assert len(res2["recommendations"]) >= 1
    assert res2["recommendations"][0]["priority"] == "CRITICAL"


# ============================================================
# 15. NONE INPUT
# ============================================================

def test_none_input():
    res = recommend_next_analysis(None)
    assert len(res["recommendations"]) >= 1
    assert res["recommendations"][0]["priority"] == "CRITICAL"
    assert res["summary"]["recommendation_count"] >= 1


# ============================================================
# 16. SINGLE-ROW DATASET
# ============================================================

def test_single_row_dataset():
    df = pd.DataFrame({"x": [10], "y": ["Single"]})
    res = recommend_next_analysis(df)
    assert len(res["recommendations"]) >= 1
    assert res["recommendations"][0]["priority"] == "CRITICAL"
    assert any("single" in str(e).lower() or "1 row" in str(e).lower() for e in res["recommendations"][0]["evidence"])


# ============================================================
# 17. MAX_RECOMMENDATIONS TRUNCATION
# ============================================================

def test_max_recommendations_truncation():
    # Construct dataset triggering multiple recommendations
    np.random.seed(42)
    x = np.linspace(1, 100, 100)
    df = pd.DataFrame({
        "miss": [None, 1] * 50,
        "outlier": list(range(96)) + [9999, -9999, 8888, -8888],
        "const": [10] * 100,
        "corr_x": x,
        "corr_y": 2.0 * x,
        "cat": ["A"] * 80 + ["B"] * 20,
    })

    res = recommend_next_analysis(df, max_recommendations=3)
    assert len(res["recommendations"]) == 3
    assert res["recommendations"][0]["rank"] == 1
    assert res["recommendations"][1]["rank"] == 2
    assert res["recommendations"][2]["rank"] == 3
    assert res["summary"]["recommendation_count"] >= 3


# ============================================================
# 18. DETERMINISTIC ORDERING
# ============================================================

def test_deterministic_ordering():
    df = pd.DataFrame({
        "col_a": [1, None, 3, 4] * 20,
        "col_b": [10] * 80,
        "col_c": ["X"] * 60 + ["Y"] * 20,
    })

    res1 = recommend_next_analysis(df, dataset_name="det_test")
    res2 = recommend_next_analysis(df, dataset_name="det_test")

    assert res1 == res2


# ============================================================
# 19. JSON SERIALIZABILITY
# ============================================================

def test_json_serializability():
    df = pd.DataFrame({
        "num": [1.0, 2.5, np.nan, 4.0, 1000.0],
        "cat": ["A", "B", "A", None, "A"],
        "dt": pd.date_range("2026-01-01", periods=5),
    })

    res = recommend_next_analysis(df, dataset_name="serialization_check")
    serialized = json.dumps(res, indent=2)
    assert isinstance(serialized, str)

    deserialized = json.loads(serialized)
    assert deserialized["dataset_name"] == "serialization_check"
    assert len(deserialized["recommendations"]) == len(res["recommendations"])


# ============================================================
# 20. INPUT DATAFRAME IMMUTABILITY
# ============================================================

def test_input_dataframe_immutability():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "y", "z"],
    })
    df_copy = df.copy(deep=True)

    _ = recommend_next_analysis(df)

    pd.testing.assert_frame_equal(df, df_copy)


# ============================================================
# 21. SUPPLIED EVIDENCE REUSE
# ============================================================

def test_supplied_evidence_reuse():
    df = pd.DataFrame({"x": [1, 2, 3] * 10})
    evidence = build_evidence(df, dataset_name="custom_precomputed_ev")

    res = recommend_next_analysis(evidence)
    assert res["dataset_name"] == "custom_precomputed_ev"


# ============================================================
# 22. SUPPLIED HEALTH REUSE
# ============================================================

def test_supplied_health_reuse():
    df = pd.DataFrame({"x": [1, 2, 3] * 10})
    evidence = build_evidence(df)
    health = calculate_health(evidence)
    health["risks"] = ["MANUALLY INJECTED HEALTH RISK"]

    # Verify that function executes cleanly reusing supplied health
    res = recommend_next_analysis(evidence, health=health)
    assert "recommendations" in res


# ============================================================
# 23. SUPPLIED PRIORITIZED INSIGHTS REUSE
# ============================================================

def test_supplied_prioritized_insights_reuse():
    df = pd.DataFrame({"x": [1, 2, 3] * 10})
    evidence = build_evidence(df)
    health = calculate_health(evidence)
    prioritized = build_prioritized_insights(evidence, health=health)

    res = recommend_next_analysis(evidence, health=health, prioritized_insights=prioritized)
    assert "recommendations" in res


# ============================================================
# 24. FACTUAL EVIDENCE GROUNDING
# ============================================================

def test_factual_evidence_grounding():
    df = pd.DataFrame({
        "missing_test": [1, None, 3, None] * 25,  # 50%
        "clean": range(100),
    })

    res = recommend_next_analysis(df)
    miss_recs = [r for r in res["recommendations"] if r["category"] == "missingness"]
    assert len(miss_recs) == 1

    ev_strings = " ".join(miss_recs[0]["evidence"])
    # Must contain exact percentage and counts
    assert "50.0%" in ev_strings
    assert "missing_test" in ev_strings


# ============================================================
# 25. NO FABRICATED VALUES
# ============================================================

def test_no_fabricated_values():
    # Only 2 columns, perfectly clean
    df = pd.DataFrame({
        "val_1": [10, 20, 30, 40, 50],
        "val_2": [5, 15, 25, 35, 45],
    })

    res = recommend_next_analysis(df)
    for rec in res["recommendations"]:
        for ev_str in rec["evidence"]:
            # Never invent phantom columns
            assert "salary" not in ev_str.lower()
            assert "churn" not in ev_str.lower()
            assert "department" not in ev_str.lower()


# ============================================================
# 26. ACTIONABILITY OF RECOMMENDATIONS
# ============================================================

def test_actionability_of_recommendations():
    df = pd.DataFrame({
        "a": [1, None, None, 4] * 20,
        "b": [10] * 80,
    })

    res = recommend_next_analysis(df)
    for rec in res["recommendations"]:
        # Action must be non-empty, actionable, and not generic
        assert len(rec["action"]) > 15
        assert rec["action"] != "Perform more analysis."
        assert rec["action"] != "Investigate data."
