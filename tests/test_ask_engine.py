"""
FILE: tests/test_ask_engine.py

Comprehensive test suite for Ask-the-Dataset Backend Engine (engine.ask_engine).
Validates:
- Factual deterministic calculations across diverse questions and dataset topologies.
- Anti-fabrication and precision of statistical metrics.
- Handling of missingness, duplicates, outliers, health, ML readiness, quality, dictionary, changes, reconsideration.
- Robust column resolution (exact, case-insensitive, normalized, ambiguous, missing).
- Gemini mocking (success, unavailable, malformed, unknown column rejection, unsupported operation rejection).
- Zero leakage of API keys, tracebacks, localhost URLs, or internal file paths.
- Input immutability and complete JSON serializability.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.reconsideration import reconsider_dataset
from engine.ask_engine import ask_dataset, _to_serializable


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "department": ["HR", "IT", "IT", "HR", "Finance"],
        "rating": [3.5, 4.0, 4.2, 3.8, 4.5],
    })


@pytest.fixture
def missing_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": [1.0, None, 3.0, None, 5.0],
        "feature_b": [10, 20, 30, 40, 50],
        "category": ["A", "B", None, "A", "B"],
    })


# ============================================================
# TESTS 1-8: DATASET SIZE, MISSING VALUES & DUPLICATES
# ============================================================

def test_01_row_count(sample_df):
    """Test 1: How many rows are there?"""
    res = ask_dataset("How many rows?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 5
    assert "5 row(s)" in res["answer"]


def test_02_column_count(sample_df):
    """Test 2: How many columns are there?"""
    res = ask_dataset("How many columns?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 4
    assert "4 column(s)" in res["answer"]


def test_03_dataset_dimensions(sample_df):
    """Test 3: What are the dimensions of the dataset?"""
    res = ask_dataset("What are the dimensions of this dataset?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"]["rows"] == 5
    assert res["result"]["value"]["columns"] == 4


def test_04_missing_values(missing_df):
    """Test 4: Which columns have missing values?"""
    res = ask_dataset("Which columns have missing values?", dataset=missing_df)
    assert res["status"] == "answered"
    assert "feature_a" in res["entities"]["columns"]
    assert "category" in res["entities"]["columns"]


def test_05_missing_percentage(missing_df):
    """Test 5: What percentage of feature_a values are missing?"""
    res = ask_dataset("What percentage of feature_a values are missing?", dataset=missing_df)
    assert res["status"] == "answered"
    assert res["result"]["missing_percentage"] == 40.0
    assert "40.0%" in res["answer"]


def test_06_highest_missing_column(missing_df):
    """Test 6: Which column has the highest missing values?"""
    res = ask_dataset("Which column has the highest missingness?", dataset=missing_df)
    assert res["status"] == "answered"
    assert res["result"]["column"] == "feature_a"
    assert res["result"]["missing_percentage"] == 40.0


def test_07_duplicate_count():
    """Test 7: How many duplicate rows are there?"""
    df = pd.DataFrame({"id": [1, 1, 2, 3, 3], "val": [10, 10, 20, 30, 30]})
    res = ask_dataset("How many duplicate rows are there?", dataset=df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 2
    assert "2 duplicate row(s)" in res["answer"]


def test_08_duplicate_percentage():
    """Test 8: What is the duplicate percentage?"""
    df = pd.DataFrame({"id": [1, 1, 2, 3, 3], "val": [10, 10, 20, 30, 30]})
    res = ask_dataset("What is the duplicate percentage?", dataset=df)
    assert res["status"] == "answered"
    assert res["result"]["percentage"] == 40.0


# ============================================================
# TESTS 9-19: NUMERICAL & CATEGORICAL QUESTIONS
# ============================================================

def test_09_mean(sample_df):
    """Test 9: What is the average salary?"""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 70000.0
    assert "70,000" in res["answer"]


def test_10_median(sample_df):
    """Test 10: What is the median age?"""
    res = ask_dataset("What is the median age?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 35.0


def test_11_min(sample_df):
    """Test 11: What is the minimum age?"""
    res = ask_dataset("What is the minimum age?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 25.0


def test_12_max(sample_df):
    """Test 12: What is the maximum salary?"""
    res = ask_dataset("What is the maximum salary?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 90000.0


def test_13_standard_deviation(sample_df):
    """Test 13: What is the standard deviation of salary?"""
    res = ask_dataset("What is the standard deviation of salary?", dataset=sample_df)
    assert res["status"] == "answered"
    expected = float(sample_df["salary"].std())
    assert abs(res["result"]["value"] - expected) < 1.0


def test_14_variance(sample_df):
    """Test 14: What is the variance of age?"""
    res = ask_dataset("What is the variance of age?", dataset=sample_df)
    assert res["status"] == "answered"
    expected = float(sample_df["age"].var())
    assert abs(res["result"]["value"] - expected) < 0.1


def test_15_range(sample_df):
    """Test 15: What is the range of age?"""
    res = ask_dataset("What is the range of age?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 20.0  # 45 - 25


def test_16_unique_count(sample_df):
    """Test 16: How many unique departments are there?"""
    res = ask_dataset("How many unique departments are there?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["unique_count"] == 3


def test_17_most_common_category(sample_df):
    """Test 17: Which department is most common?"""
    res = ask_dataset("Which department is most common?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["top_category"] in ["HR", "IT"]
    assert res["result"]["count"] == 2


def test_18_category_distribution(sample_df):
    """Test 18: What is the distribution of department?"""
    res = ask_dataset("What is the distribution of department?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "distribution" in res["result"]
    assert len(res["result"]["distribution"]) == 3


def test_19_category_percentage(sample_df):
    """Test 19: Percentage of top category in department."""
    res = ask_dataset("What is the top category in department?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["percentage"] == 40.0


# ============================================================
# TESTS 20-24: GROUPED AGGREGATIONS & CORRELATIONS
# ============================================================

def test_20_grouped_mean(sample_df):
    """Test 20: Average salary by department."""
    res = ask_dataset("What is the average salary by department?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["intent"]["name"] == "GROUPED_AGGREGATION"
    assert res["calculation"]["group_by"] == "department"
    assert res["calculation"]["metric"] == "salary"
    assert res["result"]["top_group"] == "Finance"
    assert res["result"]["top_value"] == 90000.0


def test_21_grouped_sum(sample_df):
    """Test 21: Sum of salary by department."""
    res = ask_dataset("What is the total salary by department?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["top_group"] in ["IT", "HR"]
    assert res["result"]["top_value"] == 130000.0


def test_22_grouped_count(sample_df):
    """Test 22: Count of employees by department."""
    res = ask_dataset("Count of rating by department", dataset=sample_df)
    assert res["status"] == "answered"
    assert len(res["result"]["groups"]) == 3


def test_23_correlation(sample_df):
    """Test 23: What is the correlation between age and salary?"""
    res = ask_dataset("What is the correlation between age and salary?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["correlation"] == 1.0
    assert "1.00" in res["answer"]


def test_24_insufficient_correlation_data():
    """Test 24: Correlation with fewer than 2 valid points returns safe status."""
    df = pd.DataFrame({"x": [10], "y": [20]})
    res = ask_dataset("What is the correlation between x and y?", dataset=df)
    assert res["status"] == "not_available"
    assert "Insufficient" in res["answer"]


# ============================================================
# TESTS 25-35: ANOMALIES, HEALTH, QUALITY, CHANGES, RECONSIDERATION
# ============================================================

def test_25_outlier_question():
    """Test 25: Which columns contain outliers?"""
    df = pd.DataFrame({"metric": [10, 10, 10, 10, 10, 10, 10, 10, 10, 500]})
    res = ask_dataset("Which columns have outliers?", dataset=df)
    assert res["status"] == "answered"
    assert "metric" in res["entities"]["columns"]


def test_26_anomaly_question():
    """Test 26: Are there anomalies in the dataset?"""
    df = pd.DataFrame({"metric": [10, 10, 10, 10, 10, 10, 10, 10, 10, 500]})
    res = ask_dataset("Are there anomalies in this dataset?", dataset=df)
    assert res["status"] == "answered"
    assert "potential anomalies" in res["answer"].lower()


def test_27_health_question(sample_df):
    """Test 27: Is the dataset healthy?"""
    res = ask_dataset("Is the dataset healthy?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["status"] in ["READY", "NEEDS ATTENTION", "NOT SUITABLE"]
    assert res["result"]["score"] is not None


def test_28_ml_readiness(sample_df):
    """Test 28: Is this dataset ready for machine learning?"""
    res = ask_dataset("Is this dataset ready for machine learning?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "status" in res["result"]


def test_29_data_quality(sample_df):
    """Test 29: What are the biggest data quality problems?"""
    res = ask_dataset("What are the biggest data quality problems?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "issues" in res["result"]


def test_30_column_intelligence(sample_df):
    """Test 30: What do you know about salary?"""
    res = ask_dataset("What do you know about salary?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "salary" in res["answer"]
    assert "Numerical" in res["answer"] or "Numerical" in str(res["result"])


def test_31_pattern_question(sample_df):
    """Test 31: What patterns exist in this dataset?"""
    res = ask_dataset("What patterns exist in this dataset?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "patterns" in res["result"]


def test_32_change_question_no_previous_data(sample_df):
    """Test 32: What changed between the old and new dataset when no previous data exists?"""
    res = ask_dataset("What changed between the old and new dataset?", dataset=sample_df)
    assert res["status"] == "not_available"
    assert "No previous dataset" in res["answer"]


def test_33_change_question_with_previous_data(sample_df):
    """Test 33: What changed when previous dataset is provided via reconsideration."""
    df_old = sample_df
    df_new = sample_df.copy()
    df_new.loc[0, "salary"] = 999999
    recon = reconsider_dataset(df_old, df_new)
    res = ask_dataset("What changed?", dataset=df_new, reconsideration=recon)
    assert res["status"] == "answered"
    assert res["result"]["has_changes"] is True


def test_34_reconsideration_question(sample_df):
    """Test 34: Do previous findings still hold?"""
    df_old = sample_df
    df_new = sample_df.copy()
    recon = reconsider_dataset(df_old, df_new)
    res = ask_dataset("Do previous findings still hold?", dataset=df_new, reconsideration=recon)
    assert res["status"] == "answered"
    assert "reconsiderations" in res["result"]


def test_35_next_analysis(sample_df):
    """Test 35: What should I analyze next?"""
    res = ask_dataset("What should I analyze next?", dataset=sample_df)
    assert res["status"] == "answered"
    assert "recommendations" in res["result"]


# ============================================================
# TESTS 36-48: INPUT RESOLUTION, MATCHING & DETERMINISM
# ============================================================

def test_36_evidence_input(sample_df):
    """Test 36: Querying with pre-built Evidence dict."""
    ev = build_evidence(sample_df)
    res = ask_dataset("How many rows?", evidence=ev)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 5


def test_37_dataframe_input(sample_df):
    """Test 37: Querying with DataFrame."""
    res = ask_dataset("How many rows?", dataset=sample_df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 5


def test_38_none_input():
    """Test 38: Querying with None dataset returns safe not_available status."""
    res = ask_dataset("How many rows?", dataset=None, evidence=None)
    assert res["status"] == "not_available"


def test_39_exact_column_matching(sample_df):
    """Test 39: Exact column match."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    assert res["entities"]["columns"] == ["salary"]


def test_40_case_insensitive_column_matching(sample_df):
    """Test 40: Case-insensitive column match."""
    res = ask_dataset("What is the average SALARY?", dataset=sample_df)
    assert res["entities"]["columns"] == ["salary"]


def test_41_normalized_column_matching():
    """Test 41: Normalized column matching (spaces vs underscores)."""
    df = pd.DataFrame({"Annual_Salary": [50000, 60000, 70000]})
    res = ask_dataset("What is the average annual salary?", dataset=df)
    assert res["status"] == "answered"
    assert res["entities"]["columns"] == ["Annual_Salary"]
    assert res["result"]["value"] == 60000.0


def test_42_ambiguous_column(sample_df):
    """Test 42: Ambiguous question with no column specified when multiple exist."""
    res = ask_dataset("What is the average?", dataset=sample_df)
    assert res["status"] == "needs_clarification"
    assert len(res["candidates"]) > 1


def test_43_ambiguous_question(sample_df):
    """Test 43: Vague question asks for clarification."""
    res = ask_dataset("Can you check it?", dataset=sample_df)
    assert res["status"] in ["unsupported", "needs_clarification"]


def test_44_unsupported_question(sample_df):
    """Test 44: Completely unrelated question returns unsupported status."""
    res = ask_dataset("Who is the prime minister of Canada?", dataset=sample_df)
    assert res["status"] == "unsupported"


def test_45_empty_question(sample_df):
    """Test 45: Empty or whitespace question returns invalid_question status."""
    res = ask_dataset("   ", dataset=sample_df)
    assert res["status"] == "invalid_question"


def test_46_non_string_question(sample_df):
    """Test 46: Non-string question returns invalid_question status."""
    res = ask_dataset(12345, dataset=sample_df)
    assert res["status"] == "invalid_question"


def test_47_deterministic_output(sample_df):
    """Test 47: Repeated queries produce bit-for-bit identical outputs."""
    r1 = ask_dataset("What is the average salary?", dataset=sample_df)
    r2 = ask_dataset("What is the average salary?", dataset=sample_df)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_48_json_serialization(sample_df):
    """Test 48: Output serializes cleanly via json.dumps without error."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert loaded["answer"] == res["answer"]


# ============================================================
# TESTS 49-55: IMMUTABILITY, TRUTH & METRICS GROUNDING
# ============================================================

def test_49_input_immutability(sample_df):
    """Test 49: Original DataFrame and Evidence are never mutated."""
    df_copy = sample_df.copy(deep=True)
    ev = build_evidence(sample_df)
    ev_copy = copy.deepcopy(ev)

    ask_dataset("What is the average salary by department?", dataset=sample_df)
    assert sample_df.equals(df_copy)

    ask_dataset("What is the average salary by department?", evidence=ev)
    assert ev == ev_copy


def test_50_no_fabricated_numeric_values(sample_df):
    """Test 50: The calculated mean (70,000) matches Pandas exactly."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    assert res["result"]["value"] == sample_df["salary"].mean()
    assert "70,000" in res["answer"]
    assert "75,000" not in res["answer"]


def test_51_no_fabricated_correlation(sample_df):
    """Test 51: Correlation matches exact Pearson r (1.00)."""
    res = ask_dataset("What is the correlation between age and salary?", dataset=sample_df)
    assert res["result"]["correlation"] == 1.0


def test_52_no_fabricated_anomaly_count():
    """Test 52: Anomaly count matches exact detection."""
    df = pd.DataFrame({"metric": [10, 10, 10, 10, 10, 10, 10, 10, 10, 500]})
    res = ask_dataset("Which columns have outliers?", dataset=df)
    if res["result"].get("findings"):
        assert res["result"]["findings"][0]["anomaly_count"] == 1


def test_53_canonical_health_status(sample_df):
    """Test 53: Health status preserves canonical label."""
    res = ask_dataset("Is the dataset healthy?", dataset=sample_df)
    assert res["result"]["status"] in ["READY", "NEEDS ATTENTION", "NOT SUITABLE"]


def test_54_severity_preserved():
    """Test 54: Quality findings preserve severity."""
    df = pd.DataFrame({"x": [1]})
    res = ask_dataset("What are the data quality problems?", dataset=df)
    assert res["status"] == "answered"
    assert any("CRITICAL" in iss or "HIGH" in iss for iss in res["result"]["issues"])


def test_55_evidence_grounding(sample_df):
    """Test 55: Every answered question includes evidence basis."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    assert "evidence" in res
    assert res["evidence"]["valid_count"] == 5


# ============================================================
# TESTS 56-64: GEMINI INTEGRATION, MOCKING & ZERO LEAKAGE
# ============================================================

def test_56_gemini_disabled_by_default(sample_df):
    """Test 56: Gemini is disabled by default."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    assert res["generation"]["gemini_used"] is False
    assert res["generation"]["mode"] == "deterministic"


@patch("engine.ask_engine.get_gemini_client")
def test_57_gemini_successful_mock(mock_get_client, sample_df):
    """Test 57: Successful Gemini intent classification when enabled."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "intent": "NUMERIC_SUMMARY",
        "columns": ["salary"],
        "group_by": None,
        "aggregation": "mean",
        "confidence": 0.95,
    })
    mock_client.models.generate_content.return_value = mock_resp
    mock_get_client.return_value = mock_client

    # Query with slightly informal wording
    res = ask_dataset("tell me how much people earn on average", dataset=sample_df, use_gemini=True)
    assert res["status"] == "answered"
    assert res["generation"]["gemini_used"] is True
    assert res["result"]["value"] == 70000.0


@patch("engine.ask_engine.get_gemini_client", return_value=None)
def test_58_gemini_unavailable_mock(mock_get_client, sample_df):
    """Test 58: When Gemini is unavailable, deterministic engine answers."""
    res = ask_dataset("What is the average salary?", dataset=sample_df, use_gemini=True)
    assert res["status"] == "answered"
    assert res["generation"]["mode"] == "deterministic"


@patch("engine.ask_engine.get_gemini_client")
def test_59_malformed_gemini_response(mock_get_client, sample_df):
    """Test 59: Malformed Gemini response falls back safely."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "NOT JSON"
    mock_client.models.generate_content.return_value = mock_resp
    mock_get_client.return_value = mock_client

    res = ask_dataset("random question", dataset=sample_df, use_gemini=True)
    assert res["status"] in ["unsupported", "needs_clarification"]


@patch("engine.ask_engine.get_gemini_client")
def test_60_gemini_unknown_column_rejection(mock_get_client, sample_df):
    """Test 60: Gemini hallucinated column is strictly rejected."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "intent": "NUMERIC_SUMMARY",
        "columns": ["hallucinated_column_123"],
        "group_by": None,
        "aggregation": "mean",
    })
    mock_client.models.generate_content.return_value = mock_resp
    mock_get_client.return_value = mock_client

    res = ask_dataset("random query", dataset=sample_df, use_gemini=True)
    # The hallucinated column MUST NOT be used!
    assert res["entities"]["columns"] != ["hallucinated_column_123"]


@patch("engine.ask_engine.get_gemini_client")
def test_61_gemini_unsupported_operation_rejection(mock_get_client, sample_df):
    """Test 61: Gemini unsupported operation is rejected."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "intent": "UNSUPPORTED_OP",
        "columns": ["salary"],
        "group_by": None,
        "aggregation": "eigenvector",
    })
    mock_client.models.generate_content.return_value = mock_resp
    mock_get_client.return_value = mock_client

    res = ask_dataset("complex math", dataset=sample_df, use_gemini=True)
    assert res["status"] in ["unsupported", "needs_clarification"]


@patch("engine.ask_engine.get_gemini_client")
def test_62_no_api_key_leakage(mock_get_client, sample_df):
    """Test 62: Zero leakage of API key on exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Bad key AIzaSyD987FakeKey123")
    mock_get_client.return_value = mock_client

    res = ask_dataset("average salary", dataset=sample_df, use_gemini=True)
    dumped = json.dumps(res)
    assert "AIzaSyD987FakeKey123" not in dumped


@patch("engine.ask_engine.get_gemini_client")
def test_63_no_traceback_leakage(mock_get_client, sample_df):
    """Test 63: Zero leakage of traceback on exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Crash in /var/secrets/app.py line 42")
    mock_get_client.return_value = mock_client

    res = ask_dataset("average salary", dataset=sample_df, use_gemini=True)
    dumped = json.dumps(res)
    assert "Traceback" not in dumped
    assert "/var/secrets" not in dumped


def test_64_no_localhost_leakage(sample_df):
    """Test 64: Zero leakage of localhost or internal ports."""
    res = ask_dataset("What is the average salary?", dataset=sample_df)
    dumped = json.dumps(res).lower()
    assert "localhost" not in dumped
    assert "127.0.0.1" not in dumped


# ============================================================
# TESTS 65-75: EDGE CASES, DATA TYPES & METADATA
# ============================================================

def test_65_missing_column(sample_df):
    """Test 65: Question asking for nonexistent column returns unsupported with candidate list."""
    res = ask_dataset("What is the average height?", dataset=sample_df)
    assert res["status"] == "unsupported"
    assert "height" in res["answer"]


def test_66_non_numeric_aggregation_request(sample_df):
    """Test 66: Requesting mean on categorical column returns unsupported."""
    res = ask_dataset("What is the average department?", dataset=sample_df)
    assert res["status"] == "unsupported"
    assert "not numerical" in res["answer"]


def test_67_empty_dataset():
    """Test 67: Querying an empty DataFrame (0 rows)."""
    df = pd.DataFrame({"salary": []})
    res = ask_dataset("What is the average salary?", dataset=df)
    assert res["status"] == "not_available"


def test_68_single_row_dataset():
    """Test 68: Querying a single-row DataFrame."""
    df = pd.DataFrame({"salary": [50000]})
    res = ask_dataset("What is the average salary?", dataset=df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 50000.0


def test_69_all_missing_column():
    """Test 69: Querying an all-missing column."""
    df = pd.DataFrame({"salary": [None, None, None]})
    res = ask_dataset("What is the average salary?", dataset=df)
    assert res["status"] == "not_available"


def test_70_categorical_only_dataset():
    """Test 70: Querying a categorical-only dataset."""
    df = pd.DataFrame({"dept": ["HR", "IT", "Finance"]})
    res = ask_dataset("Which department occurs most often?", dataset=df)
    assert res["status"] == "answered"
    assert res["result"]["top_category"] in ["HR", "IT", "Finance"]


def test_71_numeric_only_dataset():
    """Test 71: Querying a numerical-only dataset."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res = ask_dataset("What is the average a?", dataset=df)
    assert res["status"] == "answered"
    assert res["result"]["value"] == 2.0


def test_72_datetime_question():
    """Test 72: Querying date ranges in datetime columns."""
    df = pd.DataFrame({"created_at": pd.date_range("2024-01-01", periods=3, freq="D")})
    res = ask_dataset("What is the date range of created_at?", dataset=df)
    assert res["status"] == "answered"
    assert "2024" in res["answer"]


def test_73_repeated_execution_equality(sample_df):
    """Test 73: 5 repeated calls produce identical JSON strings."""
    runs = [ask_dataset("What is the average salary?", dataset=sample_df) for _ in range(5)]
    base = json.dumps(runs[0], sort_keys=True)
    for r in runs[1:]:
        assert json.dumps(r, sort_keys=True) == base


def test_74_recommendation_grounding(sample_df):
    """Test 74: Recommendation question grounds in actual priority recommendations."""
    res = ask_dataset("What should I analyze next?", dataset=sample_df)
    assert res["status"] == "answered"
    assert len(res["result"]["recommendations"]) > 0
    assert "name" in res["result"]["recommendations"][0]


def test_75_reconsideration_classification_preservation(sample_df):
    """Test 75: Reconsideration preserves exact classifications."""
    recon = {
        "reconsiderations": [
            {"title": "Check A", "classification": "STRENGTHENED"},
            {"title": "Check B", "classification": "WEAKENED"},
        ]
    }
    res = ask_dataset("Do previous findings still hold?", dataset=sample_df, reconsideration=recon)
    assert res["status"] == "answered"
    assert "STRENGTHENED" in res["answer"]
    assert "WEAKENED" in res["answer"]
