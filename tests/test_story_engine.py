"""
FILE: tests/test_story_engine.py

Comprehensive unit test suite for Phase 4: Data Story Mode Backend.
Tests all 26 required scenarios:
1. Normal healthy dataset
2. Missing-value dataset
3. Duplicate dataset
4. Constant-column dataset
5. All-missing column
6. Strong correlation dataset
7. Outlier dataset
8. Small dataset
9. Empty DataFrame
10. None input
11. Numeric-only dataset
12. Categorical-only dataset
13. Datetime dataset
14. Supplied Evidence is reused
15. Supplied Health is reused
16. Supplied prioritized insights are reused
17. Top findings preserve Phase 3 priority ordering
18. Health dimensions correctly summarized
19. Patterns are evidence-grounded
20. Analytical risks are evidence-grounded
21. Deterministic output
22. JSON serialization
23. DataFrame immutability
24. Evidence immutability
25. Health immutability
26. Priority input immutability
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story


# ============================================================
# 1. NORMAL HEALTHY DATASET
# ============================================================

def test_normal_healthy_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=100),
        "income": np.random.normal(50000, 10000, size=100),
        "department": np.random.choice(["Sales", "HR", "Eng", "Finance"], size=100),
        "active": np.random.choice([True, False], size=100),
    })

    story = build_data_story(df, dataset_name="healthy_corp")

    # Check top-level keys
    assert "dataset" in story
    assert "overview" in story
    assert "health" in story
    assert "important_findings" in story
    assert "patterns" in story
    assert "anomalies" in story
    assert "analytical_risks" in story
    assert "recommended_next_steps" in story
    assert "story_sections" in story
    assert story["generated_from"] == "evidence"

    # Dataset metadata
    assert story["dataset"]["name"] == "healthy_corp"
    assert story["dataset"]["rows"] == 100
    assert story["dataset"]["columns"] == 4

    # 7 story sections
    assert len(story["story_sections"]) == 7


# ============================================================
# 2. MISSING-VALUE DATASET
# ============================================================

def test_missing_value_dataset():
    df = pd.DataFrame({
        "sparse": [None, 2.0, None, 4.0] * 25,  # 50% missing
        "clean": list(range(100)),
    })

    story = build_data_story(df)

    # Overview notes missingness
    assert any("missing" in s.lower() for s in story["overview"]["key_characteristics"])

    # Important findings contain missingness
    findings = story["important_findings"]
    assert any(f["category"] == "missingness" for f in findings)

    # Analytical risks mention missing values
    risks = story["analytical_risks"]
    assert any("missing" in r.lower() for r in risks)


# ============================================================
# 3. DUPLICATE DATASET
# ============================================================

def test_duplicate_dataset():
    base = pd.DataFrame({
        "id": list(range(10)),
        "label": [f"item_{i}" for i in range(10)],
    })
    df = pd.concat([base] * 5, ignore_index=True)  # 80% duplicates

    story = build_data_story(df)

    # Duplicates flagged in findings
    assert any(f["category"] == "duplicates" for f in story["important_findings"])

    # Duplicates flagged in risks
    assert any("duplicate" in r.lower() for r in story["analytical_risks"])


# ============================================================
# 4. CONSTANT-COLUMN DATASET
# ============================================================

def test_constant_column_dataset():
    df = pd.DataFrame({
        "static_code": [999] * 50,
        "clean": list(range(50)),
    })

    story = build_data_story(df)

    # Check that constant column is flagged in risks or findings
    findings = story["important_findings"]
    assert any("constant" in f["title"].lower() or "zero-variance" in f["title"].lower() for f in findings)
    assert any("constant" in r.lower() for r in story["analytical_risks"])


# ============================================================
# 5. ALL-MISSING COLUMN
# ============================================================

def test_all_missing_column():
    df = pd.DataFrame({
        "empty_col": [None] * 40,
        "valid_col": list(range(40)),
    })

    story = build_data_story(df)

    # Must be in important findings as empty column
    findings = story["important_findings"]
    empty_findings = [f for f in findings if "empty" in f["title"].lower() or "100%" in f["title"]]
    assert len(empty_findings) >= 1
    assert any("empty" in r.lower() for r in story["analytical_risks"])


# ============================================================
# 6. STRONG CORRELATION DATASET
# ============================================================

def test_strong_correlation_dataset():
    np.random.seed(42)
    x = np.linspace(1, 100, 100)
    y = 3.0 * x + np.random.normal(0, 0.1, 100)  # r ~ 0.999
    df = pd.DataFrame({"feat_x": x, "feat_y": y})

    story = build_data_story(df)

    # Patterns should report the correlation
    patterns = story["patterns"]
    assert any("feat_x" in p and "feat_y" in p for p in patterns)


# ============================================================
# 7. OUTLIER DATASET
# ============================================================

def test_outlier_dataset():
    vals = list(range(10, 100)) + [99999, -99999, 88888, -88888]
    df = pd.DataFrame({
        "outlier_col": vals,
        "normal_col": list(range(len(vals))),
    })

    story = build_data_story(df)

    # Anomalies should report outlier_col
    anomalies = story["anomalies"]
    assert any(a.get("column") == "outlier_col" for a in anomalies)


# ============================================================
# 8. SMALL DATASET
# ============================================================

def test_small_dataset():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": ["a", "b", "c", "d", "e"],
    })

    story = build_data_story(df)

    # Analytical risks should mention small sample size
    risks = story["analytical_risks"]
    assert any("small sample" in r.lower() for r in risks)


# ============================================================
# 9. EMPTY DATAFRAME (0x0 AND 0xN)
# ============================================================

def test_empty_dataframe():
    df_empty = pd.DataFrame()
    story1 = build_data_story(df_empty)
    assert story1["dataset"]["rows"] == 0
    assert story1["health"]["status"] == "NOT SUITABLE"
    assert "empty" in story1["overview"]["summary"].lower()

    df_empty_cols = pd.DataFrame(columns=["a", "b", "c"])
    story2 = build_data_story(df_empty_cols)
    assert story2["dataset"]["rows"] == 0
    assert story2["health"]["status"] == "NOT SUITABLE"


# ============================================================
# 10. NONE INPUT
# ============================================================

def test_none_input():
    story = build_data_story(None)
    assert story["dataset"]["rows"] == 0
    assert story["health"]["status"] == "NOT SUITABLE"
    assert len(story["story_sections"]) == 7


# ============================================================
# 11. NUMERIC-ONLY DATASET
# ============================================================

def test_numeric_only_dataset():
    df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "col2": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
    })

    story = build_data_story(df)
    assert len(story["dataset"]["numerical_columns"]) == 2
    assert len(story["dataset"]["categorical_columns"]) == 0


# ============================================================
# 12. CATEGORICAL-ONLY DATASET
# ============================================================

def test_categorical_only_dataset():
    df = pd.DataFrame({
        "city": ["Tokyo", "Paris", "London", "Berlin"] * 10,
        "tier": ["Gold", "Silver", "Bronze", "Platinum"] * 10,
    })

    story = build_data_story(df)
    assert len(story["dataset"]["categorical_columns"]) == 2
    assert len(story["dataset"]["numerical_columns"]) == 0


# ============================================================
# 13. DATETIME DATASET
# ============================================================

def test_datetime_dataset():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="D"),
        "value": range(50),
    })

    story = build_data_story(df)
    assert "timestamp" in story["dataset"]["datetime_columns"]
    assert "value" in story["dataset"]["numerical_columns"]


# ============================================================
# 14. SUPPLIED EVIDENCE IS REUSED
# ============================================================

def test_supplied_evidence_is_reused():
    df = pd.DataFrame({"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10})
    evidence = build_evidence(df, dataset_name="custom_precomputed_evidence")

    # Pass the pre-computed evidence dictionary directly
    story = build_data_story(evidence)
    assert story["dataset"]["name"] == "custom_precomputed_evidence"
    assert story["dataset"]["rows"] == 30


# ============================================================
# 15. SUPPLIED HEALTH IS REUSED
# ============================================================

def test_supplied_health_is_reused():
    df = pd.DataFrame({"a": [1, 2, 3] * 10})
    evidence = build_evidence(df)
    health = calculate_health(evidence)
    health["overall"]["summary"] = "CUSTOM OVERRIDDEN HEALTH SUMMARY FOR TEST"

    story = build_data_story(evidence, health=health)
    assert story["health"]["summary"] == "CUSTOM OVERRIDDEN HEALTH SUMMARY FOR TEST"


# ============================================================
# 16. SUPPLIED PRIORITIZED INSIGHTS ARE REUSED
# ============================================================

def test_supplied_prioritized_insights_are_reused():
    df = pd.DataFrame({"a": [1, 2, 3] * 10})
    evidence = build_evidence(df)
    custom_insights = {
        "insights": [{
            "title": "CUSTOM PRIORITY FINDING TEST",
            "category": "custom",
            "severity": "CRITICAL",
            "evidence": {"marker": 42},
            "interpretation": "Mocked test finding.",
            "impact": "Test impact.",
            "priority_score": 99.9,
            "priority_rank": 1,
            "recommended_action": "Test action.",
        }],
        "summary": {"total_findings": 1},
        "generated_from": "evidence",
    }

    story = build_data_story(evidence, prioritized_insights=custom_insights)
    assert len(story["important_findings"]) == 1
    assert story["important_findings"][0]["title"] == "CUSTOM PRIORITY FINDING TEST"
    assert story["important_findings"][0]["priority_score"] == 99.9


# ============================================================
# 17. TOP FINDINGS PRESERVE PHASE 3 PRIORITY ORDERING
# ============================================================

def test_top_findings_preserve_priority_order():
    base = pd.DataFrame({
        "crit_missing": [None, None, None, 4] * 25,  # 75% missing -> CRITICAL
        "high_missing": [None, 2, 3, 4] * 25,        # 25% missing -> HIGH
        "constant": [10] * 100,                      # constant -> MEDIUM
    })
    # Duplicate base to create duplicate rows
    df = pd.concat([base, base], ignore_index=True)

    story = build_data_story(df, max_insights=4)
    findings = story["important_findings"]

    assert len(findings) <= 4
    # Ensure priority scores are in descending order
    scores = [f["priority_score"] for f in findings]
    assert scores == sorted(scores, reverse=True)

    # Ensure sequential priority_ranks
    ranks = [f["priority_rank"] for f in findings]
    assert ranks == list(range(1, len(findings) + 1))


# ============================================================
# 18. HEALTH DIMENSIONS CORRECTLY SUMMARIZED
# ============================================================

def test_health_dimensions_summary():
    df = pd.DataFrame({
        "clean": list(range(100)),
        "with_nulls": [None] * 30 + list(range(70)),
    })

    story = build_data_story(df)
    health = story["health"]

    assert "strongest_dimensions" in health
    assert "weakest_dimensions" in health
    assert len(health["strongest_dimensions"]) <= 3
    assert len(health["weakest_dimensions"]) <= 3

    # Strongest should have higher score than weakest
    if health["strongest_dimensions"] and health["weakest_dimensions"]:
        assert health["strongest_dimensions"][0]["score"] >= health["weakest_dimensions"][0]["score"]


# ============================================================
# 19. PATTERNS ARE EVIDENCE-GROUNDED
# ============================================================

def test_patterns_are_evidence_grounded():
    # Construct a dataset with heavy categorical dominance
    df = pd.DataFrame({
        "dominant_cat": ["Alpha"] * 80 + ["Beta"] * 20,
        "val": range(100),
    })

    story = build_data_story(df)
    patterns = story["patterns"]

    # Dominant category pattern must be detected
    assert any("Alpha" in p and "80.0%" in p for p in patterns)


# ============================================================
# 20. ANALYTICAL RISKS ARE EVIDENCE-GROUNDED
# ============================================================

def test_analytical_risks_are_grounded():
    df = pd.DataFrame({
        "col1": [1, None, 3, 4] * 25,
        "col2": [10] * 100,
    })

    story = build_data_story(df)
    risks = story["analytical_risks"]

    # Risks must cite actual evidence issues (missingness, constant)
    assert any("missing" in r.lower() for r in risks)
    assert any("constant" in r.lower() for r in risks)


# ============================================================
# 21. DETERMINISTIC OUTPUT
# ============================================================

def test_deterministic_output():
    df = pd.DataFrame({
        "a": [1, 2, 3, None, 5] * 20,
        "b": ["x", "y", "x", "y", "x"] * 20,
        "c": [100] * 100,
    })

    story1 = build_data_story(df, dataset_name="det_test")
    story2 = build_data_story(df, dataset_name="det_test")

    assert story1 == story2


# ============================================================
# 22. JSON SERIALIZATION WORKS
# ============================================================

def test_json_serialization():
    df = pd.DataFrame({
        "num": [1.0, 2.5, np.nan, 4.0, 1000.0],
        "cat": ["A", "B", "A", None, "A"],
        "dt": pd.date_range("2026-01-01", periods=5),
    })

    story = build_data_story(df, dataset_name="json_test")

    serialized = json.dumps(story, indent=2)
    assert isinstance(serialized, str)

    deserialized = json.loads(serialized)
    assert deserialized["dataset"]["name"] == "json_test"
    assert len(deserialized["story_sections"]) == 7


# ============================================================
# 23. DATAFRAME IMMUTABILITY
# ============================================================

def test_dataframe_immutability():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "y", "z"],
    })
    df_copy = df.copy(deep=True)

    _ = build_data_story(df)

    pd.testing.assert_frame_equal(df, df_copy)


# ============================================================
# 24. EVIDENCE IMMUTABILITY
# ============================================================

def test_evidence_immutability():
    df = pd.DataFrame({"col": [1, 2, 3, 4, 5] * 10})
    evidence = build_evidence(df)
    evidence_copy = copy.deepcopy(evidence)

    _ = build_data_story(evidence)

    assert evidence == evidence_copy


# ============================================================
# 25. HEALTH IMMUTABILITY
# ============================================================

def test_health_immutability():
    df = pd.DataFrame({"col": [1, 2, 3, 4, 5] * 10})
    evidence = build_evidence(df)
    health = calculate_health(evidence)
    health_copy = copy.deepcopy(health)

    _ = build_data_story(evidence, health=health)

    assert health == health_copy


# ============================================================
# 26. PRIORITY INPUT IMMUTABILITY
# ============================================================

def test_priority_input_immutability():
    df = pd.DataFrame({"col": [1, 2, 3, 4, 5] * 10})
    evidence = build_evidence(df)
    health = calculate_health(evidence)
    prioritized = build_prioritized_insights(evidence, health=health)
    prioritized_copy = copy.deepcopy(prioritized)

    _ = build_data_story(evidence, health=health, prioritized_insights=prioritized)

    assert prioritized == prioritized_copy
