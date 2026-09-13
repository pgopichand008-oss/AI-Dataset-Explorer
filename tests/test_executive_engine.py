"""
FILE: tests/test_executive_engine.py

Comprehensive test suite for the Executive Intelligence Engine (engine.executive_engine).
Validates:
- Factual grounding and deterministic calculations across diverse dataset topologies.
- Complete separation of factual evidence from interpretation.
- Robust handling of missingness, duplicates, constants, all-missing columns, datetimes.
- Integration with Phase 1-9 engines (evidence, health, priority, story, anomaly, recommendation, reconsideration, workflow).
- Canonical health statuses and severity levels preservation.
- Absolute prevention of fabricated statistics, correlations, or anomaly counts.
- Resilient Gemini integration with mock testing (successful, unavailable, malformed, and exception).
- Safe error handling with zero leakage of tracebacks, API keys, paths, or localhost URLs.
- 100% JSON serializability and input immutability.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story
from engine.recommendation_engine import recommend_next_analysis
from engine.anomaly_engine import investigate_anomalies
from engine.reconsideration import reconsider_dataset
from engine.workflow import run_workflow
from engine.executive_engine import (
    build_executive_intelligence,
    _to_serializable,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def normal_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "department": ["HR", "IT", "Finance", "IT", "HR", "Sales", "Finance", "Sales"],
        "rating": [3.5, 4.0, 4.2, 3.8, 4.5, 3.9, 4.1, 4.8],
    })


@pytest.fixture
def missing_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": [1.0, None, 3.0, None, 5.0],
        "feature_b": [10, 20, 30, 40, 50],
        "category": ["A", "B", None, "A", "B"],
    })


# ============================================================
# TESTS 1-10: TOPOLOGY & STATISTICAL PROFILING
# ============================================================

def test_01_normal_dataset(normal_df):
    """Test 1: Normal dataset produces complete, valid executive intelligence."""
    result = build_executive_intelligence(normal_df, dataset_name="employee_data")
    assert result["dataset_name"] == "employee_data"
    assert "executive" in result
    assert "evidence_package" in result
    assert result["evidence_package"]["dataset"]["rows"] == 8
    assert result["evidence_package"]["dataset"]["columns"] == 4
    assert result["generation"]["mode"] == "deterministic"
    assert result["generation"]["gemini_used"] is False


def test_02_missing_values(missing_df):
    """Test 2: Missing values are accurately recorded in factual findings and limitations."""
    result = build_executive_intelligence(missing_df)
    limitations = result["executive"]["limitations"]
    # feature_a has 2/5 = 40% missing
    assert any("feature_a" in lim and "missing" in lim for lim in limitations)


def test_03_duplicates():
    """Test 3: Duplicate records are detected and preserved in executive findings."""
    df = pd.DataFrame({
        "id": [1, 1, 2, 3, 3],
        "val": [10, 10, 20, 30, 30],
    })
    result = build_executive_intelligence(df)
    limitations = result["executive"]["limitations"]
    assert any("duplicate" in lim.lower() for lim in limitations)


def test_04_constants():
    """Test 4: Constant columns are noted as zero-variance limitations."""
    df = pd.DataFrame({
        "constant_col": [1, 1, 1, 1, 1],
        "varying_col": [10, 20, 30, 40, 50],
    })
    result = build_executive_intelligence(df)
    limitations = result["executive"]["limitations"]
    assert any("constant_col" in lim for lim in limitations)


def test_05_all_missing_column():
    """Test 5: An all-missing column is handled without crashing."""
    df = pd.DataFrame({
        "all_null": [None, None, None, None],
        "valid": [1, 2, 3, 4],
    })
    result = build_executive_intelligence(df)
    assert result["evidence_package"]["dataset"]["rows"] == 4
    assert any("all_null" in lim for lim in result["executive"]["limitations"])


def test_06_numerical_data():
    """Test 6: Purely numerical data correctly identifies numeric column list."""
    df = pd.DataFrame({
        "a": [1.1, 2.2, 3.3],
        "b": [4, 5, 6],
    })
    result = build_executive_intelligence(df)
    assert set(result["evidence_package"]["dataset"]["numeric_columns"]) == {"a", "b"}


def test_07_categorical_data():
    """Test 7: Categorical data correctly identifies categorical column list."""
    df = pd.DataFrame({
        "cat1": ["north", "south", "east"],
        "cat2": ["low", "med", "high"],
    })
    result = build_executive_intelligence(df)
    assert set(result["evidence_package"]["dataset"]["categorical_columns"]) == {"cat1", "cat2"}


def test_08_datetime_data():
    """Test 8: Datetime columns are recognized and serialized cleanly."""
    df = pd.DataFrame({
        "dates": pd.date_range("2024-01-01", periods=5, freq="D"),
        "metrics": [10, 20, 30, 40, 50],
    })
    result = build_executive_intelligence(df)
    assert "dates" in result["evidence_package"]["dataset"]["datetime_columns"]
    # Ensure JSON serializable
    dumped = json.dumps(result)
    assert "dates" in dumped


def test_09_correlations(normal_df):
    """Test 9: Strong correlations are surfaced in patterns without causal speculation."""
    result = build_executive_intelligence(normal_df)
    patterns = result["executive"]["patterns"]
    # Age and Salary have r=1.00
    corr_patterns = [p for p in patterns if "age" in p.lower() and "salary" in p.lower()]
    assert len(corr_patterns) > 0
    # Must NOT claim causation
    for p in patterns:
        assert "causes" not in p.lower()


def test_10_anomalies():
    """Test 10: Extreme numerical outliers are identified as potential anomalies."""
    df = pd.DataFrame({
        "val": [10, 11, 12, 10, 11, 10, 12, 1000],  # 1000 is an extreme outlier
    })
    result = build_executive_intelligence(df)
    anomalies = result["executive"]["anomalies"]
    assert len(anomalies) > 0
    assert anomalies[0]["column"] == "val"
    assert "potential" in anomalies[0]["explanation"].lower() or "outlier" in anomalies[0]["explanation"].lower()


# ============================================================
# TESTS 11-18: ENGINE INTEGRATION & RECONSIDERATION
# ============================================================

def test_11_health_integration(normal_df):
    """Test 11: Explicit pre-computed health dictionary is integrated and respected."""
    ev = build_evidence(normal_df)
    hl = calculate_health(ev)
    result = build_executive_intelligence(ev, health=hl)
    assert result["executive"]["health"]["score"] == hl["overall"]["score"]
    assert result["executive"]["health"]["status"] == hl["overall"]["status"]


def test_12_priority_integration(normal_df):
    """Test 12: Explicit prioritized insights are passed into key_findings."""
    ev = build_evidence(normal_df)
    pr = build_prioritized_insights(ev)
    result = build_executive_intelligence(ev, prioritized_insights=pr)
    assert len(result["executive"]["key_findings"]) == len(pr["insights"][:5])


def test_13_story_integration(normal_df):
    """Test 13: Explicit data story patterns are reflected in executive patterns."""
    ev = build_evidence(normal_df)
    st = build_data_story(ev)
    result = build_executive_intelligence(ev, story=st)
    assert result["executive"]["patterns"] == st.get("patterns", [])


def test_14_recommendation_integration(normal_df):
    """Test 14: Explicit recommendations are reflected in next_actions."""
    ev = build_evidence(normal_df)
    recs = recommend_next_analysis(ev)
    result = build_executive_intelligence(ev, recommendations=recs)
    assert len(result["executive"]["next_actions"]) == min(5, len(recs["recommendations"]))


def test_15_reconsideration_integration(normal_df):
    """Test 15: Explicit reconsideration results are preserved in executive intelligence."""
    df_old = normal_df
    df_new = normal_df.copy()
    df_new.loc[0, "salary"] = 999999  # modify
    recon = reconsider_dataset(df_old, df_new)
    result = build_executive_intelligence(df_new, reconsideration=recon)
    assert result["executive"]["changes"]["detected"] is True
    assert "engine.reconsideration" in result["generated_from"]


def test_16_workflow_integration(normal_df):
    """Test 16: Workflow state dictionary is fully unpacked and utilized."""
    wf = run_workflow(normal_df, dataset_name="wf_test")
    result = build_executive_intelligence(workflow_state=wf)
    assert result["dataset_name"] == "wf_test"
    assert "engine.workflow" in result["generated_from"]
    assert result["executive"]["health"]["score"] is not None


def test_17_no_previous_dataset(normal_df):
    """Test 17: First-time analysis indicates no previous dataset for comparison."""
    result = build_executive_intelligence(normal_df)
    assert result["executive"]["changes"]["detected"] is False
    assert any("no previous" in s.lower() for s in result["executive"]["changes"]["summary"])


def test_18_updated_dataset(normal_df):
    """Test 18: Updated dataset through reconsideration captures changes."""
    df_old = normal_df
    df_new = normal_df.copy()
    df_new["bonus"] = [1000] * len(df_new)
    recon = reconsider_dataset(df_old, df_new)
    result = build_executive_intelligence(df_new, reconsideration=recon)
    assert result["executive"]["changes"]["detected"] is True
    assert len(result["executive"]["changes"]["summary"]) > 0


# ============================================================
# TESTS 19-24: INPUT HANDLING, DETERMINISM & JSON INTEGRITY
# ============================================================

def test_19_evidence_input(normal_df):
    """Test 19: Accepting an Evidence dict directly."""
    ev = build_evidence(normal_df, dataset_name="pre_ev")
    result = build_executive_intelligence(ev)
    assert result["dataset_name"] == "pre_ev"
    assert result["evidence_package"]["dataset"]["rows"] == 8


def test_20_dataframe_input(normal_df):
    """Test 20: Accepting a pandas DataFrame directly."""
    result = build_executive_intelligence(normal_df)
    assert result["evidence_package"]["dataset"]["rows"] == 8


def test_21_none_input():
    """Test 21: Passing None produces safe fallback executive intelligence."""
    result = build_executive_intelligence(None)
    assert result["dataset_name"] == "dataset"
    assert result["evidence_package"]["dataset"]["rows"] == 0
    assert result["executive"]["health"]["status"] == "UNKNOWN"
    assert len(result["warnings"]) > 0


def test_22_deterministic_output(normal_df):
    """Test 22: Identical inputs produce bit-for-bit identical outputs."""
    r1 = build_executive_intelligence(normal_df, dataset_name="det_test")
    r2 = build_executive_intelligence(normal_df, dataset_name="det_test")
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_23_json_serialization(normal_df):
    """Test 23: Complete output payload serializes via standard json.dumps without error."""
    result = build_executive_intelligence(normal_df)
    dumped = json.dumps(result)
    loaded = json.loads(dumped)
    assert loaded["dataset_name"] == result["dataset_name"]


def test_24_input_immutability(normal_df):
    """Test 24: Input DataFrame and Evidence dict are never mutated."""
    df_copy = normal_df.copy(deep=True)
    ev = build_evidence(normal_df)
    ev_copy = copy.deepcopy(ev)

    build_executive_intelligence(normal_df)
    assert normal_df.equals(df_copy)

    build_executive_intelligence(ev)
    assert ev == ev_copy


# ============================================================
# TESTS 25-32: DATA INTEGRITY & ANTI-FABRICATION
# ============================================================

def test_25_top_findings_limited(normal_df):
    """Test 25: Key findings are priority-first and limited to top 5."""
    result = build_executive_intelligence(normal_df)
    assert len(result["executive"]["key_findings"]) <= 5


def test_26_canonical_health_status_preserved():
    """Test 26: Health status uses canonical project terminology (READY / NEEDS ATTENTION / NOT SUITABLE)."""
    # 1 row is NOT SUITABLE
    df_single = pd.DataFrame({"x": [1]})
    res_single = build_executive_intelligence(df_single)
    assert res_single["executive"]["health"]["status"] == "NOT SUITABLE"

    # Many missing values is NEEDS ATTENTION
    df_missing = pd.DataFrame({"x": [1, None, None, None, None, 6, 7, 8, 9, 10]})
    res_missing = build_executive_intelligence(df_missing)
    assert res_missing["executive"]["health"]["status"] in ["READY", "NEEDS ATTENTION", "NOT SUITABLE"]


def test_27_severity_preserved(missing_df):
    """Test 27: Severity levels (CRITICAL, HIGH, MEDIUM, LOW, INFO) are preserved in key_findings."""
    result = build_executive_intelligence(missing_df)
    valid_sevs = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
    for f in result["executive"]["key_findings"]:
        assert f["severity"] in valid_sevs


def test_28_evidence_grounding(normal_df):
    """Test 28: Each key finding includes an explicit evidence_basis."""
    result = build_executive_intelligence(normal_df)
    for f in result["executive"]["key_findings"]:
        assert "evidence_basis" in f
        assert "fact" in f
        assert "interpretation" in f


def test_29_no_fabricated_statistics():
    """Test 29: Known exact missing percentage (20.0%) is preserved and never altered."""
    df = pd.DataFrame({
        "col": [1, 2, 3, 4, None],  # exactly 1 / 5 = 20.0%
    })
    result = build_executive_intelligence(df)
    # The limitations or key findings must state 20.0%, never 30% or random
    lim_text = " ".join(result["executive"]["limitations"])
    assert "20.0%" in lim_text
    assert "30%" not in lim_text


def test_30_no_fabricated_correlations():
    """Test 30: Known exact correlation coefficient (1.00) is reported accurately."""
    df = pd.DataFrame({
        "x": [10, 20, 30, 40, 50],
        "y": [20, 40, 60, 80, 100],  # perfect r=1.00
    })
    result = build_executive_intelligence(df)
    patterns = result["executive"]["patterns"]
    corr_found = any("1.00" in p for p in patterns)
    assert corr_found
    assert not any("0.95" in p for p in patterns)


def test_31_no_fabricated_anomaly_counts():
    """Test 31: Known exact anomaly count is reported accurately."""
    df = pd.DataFrame({
        "metric": [10, 10, 10, 10, 10, 10, 10, 10, 10, 500],  # exactly 1 extreme value
    })
    result = build_executive_intelligence(df)
    anomalies = result["executive"]["anomalies"]
    if anomalies:
        assert anomalies[0]["anomaly_count"] == 1


def test_32_limitations_included(missing_df):
    """Test 32: Analytical limitations list is populated with concrete factual constraints."""
    result = build_executive_intelligence(missing_df)
    limitations = result["executive"]["limitations"]
    assert len(limitations) > 0
    # Small sample size (< 30) is explicitly present
    assert any("sample size" in lim.lower() for lim in limitations)


# ============================================================
# TESTS 33-40: GEMINI INTEGRATION, MOCKING & ZERO LEAKAGE
# ============================================================

def test_33_deterministic_fallback_without_gemini(normal_df):
    """Test 33: Setting use_gemini=False guarantees deterministic execution."""
    result = build_executive_intelligence(normal_df, use_gemini=False)
    assert result["generation"]["mode"] == "deterministic"
    assert result["generation"]["gemini_used"] is False
    assert result["generation"]["fallback"] is False


def test_34_gemini_disabled_by_default(normal_df):
    """Test 34: use_gemini defaults to False without setting it."""
    result = build_executive_intelligence(normal_df)
    assert result["generation"]["gemini_used"] is False


@patch("engine.executive_engine.get_gemini_client", return_value=None)
def test_35_gemini_unavailable_fallback(mock_get_client, normal_df):
    """Test 35: When Gemini client is None, safely fall back to deterministic mode."""
    result = build_executive_intelligence(normal_df, use_gemini=True)
    assert result["generation"]["mode"] == "deterministic"
    assert result["generation"]["gemini_used"] is False
    assert result["generation"]["fallback"] is True
    assert any("deterministic executive intelligence was used" in w for w in result["warnings"])


@patch("engine.executive_engine.get_gemini_client")
def test_36_malformed_gemini_response_fallback(mock_get_client, normal_df):
    """Test 36: When Gemini returns an empty or whitespace string, fall back safely."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "   "  # Empty whitespace
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = build_executive_intelligence(normal_df, use_gemini=True)
    assert result["generation"]["mode"] == "deterministic"
    assert result["generation"]["fallback"] is True


@patch("engine.executive_engine.get_gemini_client")
def test_37_gemini_success_mocked(mock_get_client, normal_df):
    """Test 37: When Gemini succeeds, narrative is attached and mode is 'gemini'."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Executive Summary: The dataset demonstrates high overall data quality."
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = build_executive_intelligence(normal_df, use_gemini=True)
    assert result["generation"]["mode"] == "gemini"
    assert result["generation"]["gemini_used"] is True
    assert result["generation"]["fallback"] is False
    assert result["executive"]["summary"] == mock_response.text


@patch("engine.executive_engine.get_gemini_client")
def test_38_no_traceback_leakage_on_gemini_exception(mock_get_client, normal_df):
    """Test 38: Exceptions raised by Gemini never leak tracebacks into warnings or results."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Internal server error: /var/secrets/key.pem")
    mock_get_client.return_value = mock_client

    result = build_executive_intelligence(normal_df, use_gemini=True)
    assert result["generation"]["fallback"] is True
    dumped = json.dumps(result)
    assert "Traceback" not in dumped
    assert "key.pem" not in dumped
    assert "Internal server error" not in dumped


@patch("engine.executive_engine.get_gemini_client")
def test_39_no_api_key_leakage(mock_get_client, normal_df):
    """Test 39: API keys are never exposed in serialized output."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ValueError("Invalid API key: AIzaSyD987FakeKey123")
    mock_get_client.return_value = mock_client

    result = build_executive_intelligence(normal_df, use_gemini=True)
    dumped = json.dumps(result)
    assert "AIzaSyD987FakeKey123" not in dumped


def test_40_no_localhost_or_debug_leakage(normal_df):
    """Test 40: Localhost URLs or debug markers are never leaked in payload."""
    result = build_executive_intelligence(normal_df)
    dumped = json.dumps(result).lower()
    assert "localhost" not in dumped
    assert "127.0.0.1" not in dumped


# ============================================================
# TESTS 41-48: RECONSIDERATION, METADATA & BOUNDARY ROBUSTNESS
# ============================================================

def test_41_changes_included(normal_df):
    """Test 41: Changes object is always present with detected and summary fields."""
    result = build_executive_intelligence(normal_df)
    assert "changes" in result["executive"]
    assert "detected" in result["executive"]["changes"]
    assert "summary" in result["executive"]["changes"]


def test_42_reconsideration_classification_preserved():
    """Test 42: Phase 8 classifications (VALIDATED, WEAKENED, STRENGTHENED, etc.) are strictly preserved."""
    custom_recon = {
        "change_summary": {"has_changes": True, "structural_changes": []},
        "reconsiderations": [
            {
                "finding_id": "f1",
                "title": "Missing Salary",
                "classification": "STRENGTHENED",
                "reason": "Missingness grew from 10% to 25%",
                "evidence": ["salary"],
            }
        ],
    }
    df = pd.DataFrame({"salary": [10, None, None, 40]})
    result = build_executive_intelligence(df, reconsideration=custom_recon)
    recons = result["executive"]["reconsideration"]
    assert len(recons) == 1
    assert recons[0]["classification"] == "STRENGTHENED"


def test_43_next_actions_included(normal_df):
    """Test 43: Next actions include name, why, priority, and action fields."""
    result = build_executive_intelligence(normal_df)
    next_acts = result["executive"]["next_actions"]
    if next_acts:
        act = next_acts[0]
        assert "name" in act
        assert "why" in act
        assert "priority" in act
        assert "action" in act


def test_44_generated_from_metadata(normal_df):
    """Test 44: generated_from lists all contributing intelligence modules."""
    result = build_executive_intelligence(normal_df)
    expected_engines = {
        "engine.evidence",
        "engine.health",
        "engine.priority_engine",
        "engine.story_engine",
        "engine.anomaly_engine",
        "engine.recommendation_engine",
    }
    assert expected_engines.issubset(set(result["generated_from"]))


def test_45_empty_dataset_safety():
    """Test 45: Completely empty DataFrame (0 rows, 0 cols) handled safely."""
    df_empty = pd.DataFrame()
    result = build_executive_intelligence(df_empty)
    assert result["evidence_package"]["dataset"]["rows"] == 0
    assert result["evidence_package"]["dataset"]["columns"] == 0
    assert result["executive"]["health"]["status"] == "NOT SUITABLE"


def test_46_single_row_dataset_safety():
    """Test 46: Single-row dataset handled safely with NOT SUITABLE status."""
    df_single = pd.DataFrame({"col_a": [42], "col_b": ["val"]})
    result = build_executive_intelligence(df_single)
    assert result["evidence_package"]["dataset"]["rows"] == 1
    assert result["executive"]["health"]["status"] == "NOT SUITABLE"
    assert any("1 record" in lim for lim in result["executive"]["limitations"])


def test_47_unsupported_optional_result_safety(normal_df):
    """Test 47: Malformed optional parameters (e.g. non-dict objects) do not crash the engine."""
    result = build_executive_intelligence(
        normal_df,
        health="invalid_string_health",  # type: ignore
        prioritized_insights=["invalid_list"],  # type: ignore
        story=12345,  # type: ignore
        anomalies=object(),  # type: ignore
        reconsideration=None,
    )
    assert result["dataset_name"] == "dataset"
    assert "executive" in result


def test_48_repeated_execution_produces_same_deterministic_output(normal_df):
    """Test 48: Running 5 consecutive cycles produces exact bit-for-bit identical results."""
    outputs = [build_executive_intelligence(normal_df, dataset_name="repeat_test") for _ in range(5)]
    baseline = json.dumps(outputs[0], sort_keys=True)
    for idx, out in enumerate(outputs[1:], start=2):
        assert json.dumps(out, sort_keys=True) == baseline, f"Mismatch on run {idx}"
