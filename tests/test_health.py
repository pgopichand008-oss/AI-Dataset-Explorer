"""
tests/test_health.py

Comprehensive test suite for Phase 2 — Data Health / Readiness Score Engine (engine/health.py).

Tests:
1. Normal healthy dataset (high scores, READY status, strengths, valid summary)
2. Missing-heavy dataset (exact completeness threshold mapping, score penalty)
3. Duplicate-row dataset (exact duplicate row penalties, consistency status)
4. Invalid-value dataset (detecting non-numeric invalid values, validity penalty)
5. Empty dataset (both completely empty and with columns; NOT SUITABLE status, score 0)
6. Single-row dataset (zero variance safety, NOT SUITABLE status)
7. All-missing column (empty column penalty, completeness cap)
8. Constant column (constant column penalty in validity and consistency)
9. Numeric-only dataset (statistical health with variance and correlation)
10. Categorical-only dataset (unpunished statistical health with category diversity)
11. ML readiness passthrough (exact score and status preservation from evidence)
12. Weighted overall score (exact mathematical weights: 15%, 20%, 20%, 15%, 15%, 15%)
13. Status classification (READY >= 80, NEEDS ATTENTION 50-79.99, NOT SUITABLE < 50, safety caps)
14. JSON serialization (100% deep serialization safety)
15. No mutation of evidence (guaranteeing pure read-only execution)
"""

import copy
import json
import math
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health


# ============================================================
# 1. NORMAL HEALTHY DATASET
# ============================================================

def test_normal_healthy_dataset():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "salary": [50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 95000.0],
        "department": ["HR", "IT", "Finance", "IT", "HR", "Marketing", "IT", "Finance", "Marketing", "HR"],
    })

    health = calculate_health(df)

    # Overall assertions
    overall = health["overall"]
    assert overall["score"] >= 80.0
    assert overall["status"] == "READY"
    assert "healthy" in overall["summary"].lower()

    # Dimension assertions
    dims = health["dimensions"]
    assert dims["structural_health"]["score"] >= 80.0
    assert dims["structural_health"]["status"] == "GOOD"

    assert dims["completeness"]["score"] == 100.0
    assert dims["completeness"]["status"] == "GOOD"

    assert dims["validity"]["score"] == 100.0
    assert dims["validity"]["status"] == "GOOD"

    assert dims["consistency"]["score"] == 100.0
    assert dims["consistency"]["status"] == "GOOD"

    assert dims["statistical_health"]["score"] >= 80.0
    assert dims["statistical_health"]["status"] == "GOOD"

    # Evidence summary assertions
    summary = health["evidence_summary"]
    assert summary["row_count"] == 10
    assert summary["column_count"] == 3
    assert summary["missing_percentage"] == 0.0
    assert summary["duplicate_rows"] == 0

    # Strengths should capture zero missing and zero duplicates
    assert any("0% missing" in s for s in health["strengths"])
    assert any("No duplicate" in s for s in health["strengths"])


# ============================================================
# 2. MISSING-HEAVY DATASET
# ============================================================

def test_missing_heavy_dataset():
    # Exactly 10 rows, 2 columns -> 20 total cells.
    # Exactly 10 missing cells -> exactly 50.0% missing.
    # Rule: >40% to <=60% missing -> score = 25.0, status = POOR.
    df = pd.DataFrame({
        "a": [1, None, 3, None, 5, None, 7, None, 9, None],
        "b": [None, 2, None, 4, None, 6, None, 8, None, 10],
    })

    health = calculate_health(df)
    comp = health["dimensions"]["completeness"]

    assert comp["score"] == 25.0
    assert comp["status"] == "POOR"
    assert "50.00%" in comp["reason"]

    # Critical missingness should prevent READY status
    assert health["overall"]["status"] in ["NEEDS ATTENTION", "NOT SUITABLE"]
    assert any("missing" in w.lower() for w in health["weaknesses"])


# ============================================================
# 3. DUPLICATE-ROW DATASET
# ============================================================

def test_duplicate_row_dataset():
    # 10 rows: 5 rows repeated twice -> 5 duplicate rows (50.0%).
    # Rule: dup_pct > 30% -> penalty of 35.0 -> consistency score = 65.0 (WARNING).
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "y": ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"],
    })

    health = calculate_health(df)
    consistency = health["dimensions"]["consistency"]

    assert consistency["score"] == 65.0
    assert consistency["status"] == "WARNING"
    assert "50.00%" in consistency["reason"]

    assert any("duplicate row" in w.lower() for w in health["weaknesses"])
    assert any("Deduplicate" in a for a in health["recommended_actions"])


# ============================================================
# 4. INVALID-VALUE DATASET
# ============================================================

def test_invalid_value_dataset():
    # Numeric column containing invalid text entries
    df = pd.DataFrame({
        "salary": [50000, "Unknown", 70000, "N/A", 90000, 100000, 110000, 120000, 130000, 140000],
        "department": ["IT", "HR", "IT", "HR", "Finance", "IT", "HR", "Finance", "IT", "HR"],
    })

    health = calculate_health(df)
    validity = health["dimensions"]["validity"]

    # Validity should be penalized for 1 column containing 2 invalid non-numeric entries (penalty: 15.0)
    assert validity["score"] == 85.0
    assert "invalid non-numeric" in validity["reason"].lower()

    assert any("salary" in w and "invalid" in w.lower() for w in health["weaknesses"])
    assert any("clean non-numeric" in a.lower() for a in health["recommended_actions"])


# ============================================================
# 5. EMPTY DATASET
# ============================================================

def test_empty_dataset_completely_empty():
    df = pd.DataFrame()
    health = calculate_health(df)

    assert health["overall"]["score"] == 0.0
    assert health["overall"]["status"] == "NOT SUITABLE"
    assert "no usable records" in health["overall"]["summary"].lower()

    assert health["dimensions"]["structural_health"]["score"] == 0.0
    assert health["dimensions"]["completeness"]["score"] == 0.0
    assert health["dimensions"]["validity"]["score"] == 0.0


def test_empty_dataset_with_columns():
    df = pd.DataFrame(columns=["a", "b", "c"])
    health = calculate_health(df)

    assert health["overall"]["score"] == 0.0
    assert health["overall"]["status"] == "NOT SUITABLE"
    assert health["evidence_summary"]["row_count"] == 0
    assert health["evidence_summary"]["column_count"] == 3


# ============================================================
# 6. SINGLE-ROW DATASET
# ============================================================

def test_single_row_dataset():
    df = pd.DataFrame({
        "val_num": [42.0],
        "val_str": ["single"],
    })

    health = calculate_health(df)

    assert health["overall"]["status"] == "NOT SUITABLE"
    assert "only 1 record" in health["overall"]["summary"].lower()

    # Statistical health cannot compute variance on n=1
    assert health["dimensions"]["statistical_health"]["score"] == 40.0
    assert health["dimensions"]["statistical_health"]["status"] == "POOR"


# ============================================================
# 7. ALL-MISSING COLUMN
# ============================================================

def test_all_missing_column():
    df = pd.DataFrame({
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "valid": [1, 2, 3, 4, 5],
    })

    health = calculate_health(df)

    # Completeness is capped at 80 due to completely empty column
    comp = health["dimensions"]["completeness"]
    assert comp["score"] <= 80.0

    # Validity is penalized for completely empty column
    validity = health["dimensions"]["validity"]
    assert validity["score"] <= 85.0

    assert any("all_null" in w and "100%" in w for w in health["weaknesses"])


# ============================================================
# 8. CONSTANT COLUMN
# ============================================================

def test_constant_column():
    df = pd.DataFrame({
        "const": [7, 7, 7, 7, 7],
        "var": [10, 20, 30, 40, 50],
    })

    health = calculate_health(df)

    validity = health["dimensions"]["validity"]
    assert validity["score"] <= 90.0

    assert any("const" in w and "constant" in w.lower() for w in health["weaknesses"])
    assert any("drop or exclude constant column" in a.lower() for a in health["recommended_actions"])


# ============================================================
# 9. NUMERIC-ONLY DATASET
# ============================================================

def test_numeric_only_dataset():
    # 35 rows with active variance
    np.random.seed(42)
    df = pd.DataFrame({
        "f1": np.linspace(10, 50, 35),
        "f2": np.linspace(100, 200, 35) + np.random.normal(0, 1, 35),
        "f3": np.linspace(-5, 5, 35),
    })

    health = calculate_health(df)

    stat = health["dimensions"]["statistical_health"]
    assert stat["score"] >= 80.0
    assert stat["status"] == "GOOD"
    assert "active variance" in stat["reason"].lower()


# ============================================================
# 10. CATEGORICAL-ONLY DATASET
# ============================================================

def test_categorical_only_dataset():
    # 35 rows, pure categorical with healthy category distribution
    df = pd.DataFrame({
        "category": ["A", "B", "C", "D", "E"] * 7,
        "region": ["North", "South", "East", "West", "Central"] * 7,
    })

    health = calculate_health(df)

    # Must NOT be penalized simply for lacking numerical columns
    stat = health["dimensions"]["statistical_health"]
    assert stat["score"] >= 80.0
    assert stat["status"] == "GOOD"
    assert "categorical-only" in stat["reason"].lower()


# ============================================================
# 11. ML READINESS PASSTHROUGH
# ============================================================

def test_ml_readiness_passthrough():
    # Custom evidence dict with specific ML score
    evidence = {
        "structure": {"row_count": 50, "column_count": 3, "numerical_columns": ["x"], "categorical_columns": ["y"]},
        "completeness": {"total_cells": 150, "missing_cells": 0, "missing_percentage": 0.0, "columns_with_missing": []},
        "duplicates": {"duplicate_rows": 0, "duplicate_row_percentage": 0.0},
        "columns": [],
        "quality": {"empty_columns": [], "constant_columns": [], "outlier_columns": []},
        "numerical_statistics": {"x": {"std": 5.0}},
        "categorical_statistics": {"y": {"unique_count": 3}},
        "correlations": {"supported": False},
        "ml_readiness": {
            "ml_score": 75,
            "ml_status": "Mostly Ready",
            "can_train": True,
            "target_candidates": [{"column": "y"}],
            "warnings": [],
            "strengths": ["Balanced targets"],
        },
    }

    health = calculate_health(evidence)

    ml_dim = health["dimensions"]["ml_readiness"]
    assert ml_dim["score"] == 75.0
    assert ml_dim["status"] == "WARNING"
    assert "75" in ml_dim["reason"]
    assert health["evidence_summary"]["ml_score"] == 75


# ============================================================
# 12. WEIGHTED OVERALL SCORE
# ============================================================

def test_weighted_overall_score_math():
    # Mock evidence with known scores
    # Weights: structural (15%), completeness (20%), validity (20%), consistency (15%), statistical (15%), ml (15%)
    evidence = {
        "structure": {"row_count": 100, "column_count": 5, "numerical_columns": ["n"], "categorical_columns": ["c"]},
        "completeness": {"total_cells": 500, "missing_cells": 0, "missing_percentage": 0.0, "columns_with_missing": []},
        "duplicates": {"duplicate_rows": 0, "duplicate_row_percentage": 0.0},
        "columns": [],
        "quality": {"empty_columns": [], "constant_columns": [], "outlier_columns": []},
        "numerical_statistics": {"n": {"std": 10.0}},
        "categorical_statistics": {"c": {"unique_count": 4}},
        "correlations": {"supported": True, "strongest_pairs": [{"attribute_1": "n", "attribute_2": "n", "correlation": 1.0}]},
        "ml_readiness": {
            "ml_score": 80,
            "ml_status": "READY",
            "can_train": True,
            "target_candidates": [{"column": "c"}],
            "warnings": [],
            "strengths": [],
        },
    }

    health = calculate_health(evidence)
    dims = health["dimensions"]

    expected_overall = round(
        dims["structural_health"]["score"] * 0.15 +
        dims["completeness"]["score"] * 0.20 +
        dims["validity"]["score"] * 0.20 +
        dims["consistency"]["score"] * 0.15 +
        dims["statistical_health"]["score"] * 0.15 +
        dims["ml_readiness"]["score"] * 0.15,
        2
    )

    assert health["overall"]["score"] == expected_overall


# ============================================================
# 13. STATUS CLASSIFICATION THRESHOLDS & SAFETY OVERRIDES
# ============================================================

def test_status_classification_thresholds():
    # Test boundary classifications:
    # 80.0 -> READY
    # 79.99 -> NEEDS ATTENTION
    # 50.0 -> NEEDS ATTENTION
    # 49.99 -> NOT SUITABLE

    def make_evidence(comp_score, ml_score, empty_cols=None, dup_pct=0.0):
        return {
            "structure": {"row_count": 100, "column_count": 5, "numerical_columns": ["a", "b"], "categorical_columns": ["c"]},
            "completeness": {"total_cells": 500, "missing_cells": int(500 * (1 - comp_score / 100)), "missing_percentage": round(100 - comp_score, 2), "columns_with_missing": []},
            "duplicates": {"duplicate_rows": int(100 * dup_pct / 100), "duplicate_row_percentage": dup_pct},
            "columns": [],
            "quality": {"empty_columns": empty_cols or [], "constant_columns": [], "outlier_columns": []},
            "numerical_statistics": {"a": {"std": 1.0}, "b": {"std": 2.0}},
            "categorical_statistics": {"c": {"unique_count": 5}},
            "correlations": {"supported": True, "strongest_pairs": [{"attribute_1": "a", "attribute_2": "b", "correlation": 0.5}]},
            "ml_readiness": {"ml_score": ml_score, "ml_status": "READY" if ml_score >= 80 else ("CAUTION" if ml_score >= 50 else "NOT READY"), "can_train": ml_score >= 50, "target_candidates": [{"column": "c"}], "warnings": [], "strengths": []},
        }

    # Ready case (>= 80.0)
    h_ready = calculate_health(make_evidence(comp_score=100.0, ml_score=90))
    assert h_ready["overall"]["score"] >= 80.0
    assert h_ready["overall"]["status"] == "READY"

    # Needs attention case (50.0 <= score < 80.0)
    h_attn = calculate_health(make_evidence(comp_score=50.0, ml_score=30))
    assert 50.0 <= h_attn["overall"]["score"] < 80.0
    assert h_attn["overall"]["status"] == "NEEDS ATTENTION"

    # Not suitable case (< 50.0)
    poor_ev = {
        "structure": {"row_count": 100, "column_count": 5, "numerical_columns": ["a", "b"], "categorical_columns": ["c"]},
        "completeness": {"total_cells": 500, "missing_cells": 450, "missing_percentage": 90.0, "columns_with_missing": ["a", "b", "c"]},
        "duplicates": {"duplicate_rows": 80, "duplicate_row_percentage": 80.0},
        "columns": [],
        "quality": {"empty_columns": ["a", "b", "c", "d"], "constant_columns": [], "outlier_columns": []},
        "numerical_statistics": {},
        "categorical_statistics": {},
        "correlations": {"supported": False},
        "ml_readiness": {"ml_score": 0, "ml_status": "NOT READY", "can_train": False, "target_candidates": [], "warnings": [], "strengths": []},
    }
    h_poor = calculate_health(poor_ev)
    assert h_poor["overall"]["score"] < 50.0
    assert h_poor["overall"]["status"] == "NOT SUITABLE"


def test_structural_failure_safety_override():
    # If structural health is critically poor (< 50), overall status must NOT be READY even if score >= 80
    broken_structure_evidence = {
        "structure": {
            "row_count": 100,
            "column_count": 10,
            "has_duplicate_columns": True,
            "duplicate_column_names": ["a", "b", "c"],
            "numerical_columns": [],
            "categorical_columns": [],  # no valid types -> heavy penalty
        },
        "completeness": {"total_cells": 1000, "missing_cells": 0, "missing_percentage": 0.0, "columns_with_missing": []},
        "duplicates": {"duplicate_rows": 0, "duplicate_row_percentage": 0.0},
        "columns": [],
        "quality": {"empty_columns": ["col1", "col2"], "constant_columns": ["col3", "col4"], "outlier_columns": []},
        "numerical_statistics": {},
        "categorical_statistics": {},
        "correlations": {"supported": False},
        "ml_readiness": {"ml_score": 100, "ml_status": "READY", "can_train": True, "target_candidates": [], "warnings": [], "strengths": []},
    }

    health = calculate_health(broken_structure_evidence)
    assert health["dimensions"]["structural_health"]["score"] < 50.0
    assert health["overall"]["status"] in ["NEEDS ATTENTION", "NOT SUITABLE"]
    assert health["overall"]["status"] != "READY"


# ============================================================
# 14. JSON SERIALIZATION
# ============================================================

def test_json_serialization():
    dfs = [
        pd.DataFrame({"x": [1, 2, 3], "y": [4.0, np.nan, 6.0]}),
        pd.DataFrame(),
        pd.DataFrame({"d": pd.to_datetime(["2026-01-01", "2026-01-02"])}),
        None,
    ]

    for df in dfs:
        health = calculate_health(df)
        json_str = json.dumps(health)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "overall" in parsed
        assert "dimensions" in parsed


# ============================================================
# 15. NO MUTATION OF EVIDENCE
# ============================================================

def test_no_mutation_of_evidence():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    evidence = build_evidence(df)
    evidence_clone = copy.deepcopy(evidence)

    _ = calculate_health(evidence)

    assert evidence == evidence_clone
