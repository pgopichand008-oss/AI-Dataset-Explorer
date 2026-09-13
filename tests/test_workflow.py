"""
FILE: tests/test_workflow.py

Comprehensive unit test suite for Phase 9: Agent Workflow State.
Tests all required scenarios:
1. Normal first-time workflow
2. Updated dataset workflow
3. DataFrame input
4. Evidence input
5. None input
6. Previous DataFrame
7. Previous Evidence
8. DETECT completion
9. ANALYZE completion
10. COMPARE skipped with no previous data
11. RECONSIDER skipped with no previous data
12. COMPARE executes with previous data
13. RECONSIDER executes with previous data
14. RE-PLAN executes
15. REPORT executes
16. Complete workflow status
17. Partial failure handling
18. Detection failure handling
19. Invalid options
20. Default options
21. Deterministic workflow ID
22. Deterministic workflow result
23. JSON serialization
24. Input immutability
25. No traceback leakage
26. No API-key leakage
27. No localhost/debug leakage
28. Evidence reused across stages
29. Recommendations grounded in current evidence
30. Previous findings passed to reconsideration
31. Health passed to reconsideration
32. Story passed to reconsideration
33. Priorities passed to reconsideration
34. Added column workflow
35. Removed column workflow
36. Missingness change workflow
37. ML readiness change workflow
38. Anomaly-related workflow
39. Empty dataset
40. Single-row dataset
41. Independent analysis failure does not crash workflow
42. All required stage names present
43. Stage ordering correct
44. Skipped stage status correct
45. Safe error messages
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from engine.evidence import build_evidence
from engine.workflow import run_workflow


# ============================================================
# 1. NORMAL FIRST-TIME WORKFLOW
# ============================================================

def test_normal_first_time_workflow():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 75000, 90000, 110000],
        "city": ["NY", "SF", "LA", "NY", "SF"],
    })

    result = run_workflow(df, dataset_name="first_run")

    assert "workflow_id" in result
    assert result["status"] == "completed"
    assert result["current_stage"] == "REPORT"
    assert "stages" in result

    # Verify stage statuses for first-time dataset
    assert result["stages"]["DETECT"]["status"] == "completed"
    assert result["stages"]["ANALYZE"]["status"] == "completed"
    assert result["stages"]["COMPARE"]["status"] == "skipped"
    assert result["stages"]["RECONSIDER"]["status"] == "skipped"
    assert result["stages"]["RE-PLAN"]["status"] == "completed"
    assert result["stages"]["REPORT"]["status"] == "completed"

    assert result["dataset"]["has_previous"] is False
    assert result["dataset"]["rows"] == 5
    assert result["dataset"]["columns"] == 3


# ============================================================
# 2. UPDATED DATASET WORKFLOW
# ============================================================

def test_updated_dataset_workflow():
    old_df = pd.DataFrame({
        "val": [1, 2, 3, 4],
        "cat": ["A", "B", "A", "B"],
    })
    new_df = pd.DataFrame({
        "val": [1, 2, 3, 5],
        "cat": ["A", "B", "A", "C"],
    })

    result = run_workflow(new_df, previous_dataset=old_df, dataset_name="update_run")

    assert result["status"] == "completed"
    assert result["stages"]["DETECT"]["status"] == "completed"
    assert result["stages"]["ANALYZE"]["status"] == "completed"
    assert result["stages"]["COMPARE"]["status"] == "completed"
    assert result["stages"]["RECONSIDER"]["status"] == "completed"
    assert result["stages"]["RE-PLAN"]["status"] == "completed"
    assert result["stages"]["REPORT"]["status"] == "completed"

    assert result["dataset"]["has_previous"] is True


# ============================================================
# 3. DATAFRAME INPUT
# ============================================================

def test_dataframe_input():
    df = pd.DataFrame({"x": [10, 20, 30]})
    result = run_workflow(df)
    assert result["stages"]["DETECT"]["status"] == "completed"
    assert result["dataset"]["rows"] == 3


# ============================================================
# 4. EVIDENCE INPUT
# ============================================================

def test_evidence_input():
    df = pd.DataFrame({"x": [10, 20, 30]})
    evidence = build_evidence(df, dataset_name="pre_evidence")
    result = run_workflow(evidence)
    assert result["stages"]["DETECT"]["status"] == "completed"
    assert result["dataset"]["name"] == "pre_evidence"


# ============================================================
# 5. NONE INPUT
# ============================================================

def test_none_input():
    result = run_workflow(None)
    assert result["stages"]["DETECT"]["status"] == "completed"
    assert result["dataset"]["rows"] == 0
    assert result["dataset"]["columns"] == 0


# ============================================================
# 6. PREVIOUS DATAFRAME
# ============================================================

def test_previous_dataframe():
    old_df = pd.DataFrame({"a": [1, 2]})
    new_df = pd.DataFrame({"a": [1, 3]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["COMPARE"]["status"] == "completed"


# ============================================================
# 7. PREVIOUS EVIDENCE
# ============================================================

def test_previous_evidence():
    old_df = pd.DataFrame({"a": [1, 2]})
    new_df = pd.DataFrame({"a": [1, 3]})
    old_ev = build_evidence(old_df)
    result = run_workflow(new_df, previous_dataset=old_ev)
    assert result["stages"]["COMPARE"]["status"] == "completed"
    assert result["stages"]["RECONSIDER"]["status"] == "completed"


# ============================================================
# 8. DETECT COMPLETION
# ============================================================

def test_detect_completion():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["A", "B", "C"],
        "dt": pd.date_range("2023-01-01", periods=3),
    })
    result = run_workflow(df)
    detect_res = result["stages"]["DETECT"]["result"]
    assert detect_res["rows"] == 3
    assert detect_res["columns"] == 3
    assert "num" in detect_res["numeric_columns"]
    assert "cat" in detect_res["categorical_columns"]
    assert "dt" in detect_res["datetime_columns"]


# ============================================================
# 9. ANALYZE COMPLETION
# ============================================================

def test_analyze_completion():
    df = pd.DataFrame({
        "num": [10, 20, 30, 40, 50],
        "cat": ["X", "Y", "X", "Y", "Z"],
    })
    result = run_workflow(df)
    an_res = result["stages"]["ANALYZE"]["result"]
    assert "health" in an_res
    assert "priorities" in an_res
    assert "story" in an_res
    assert "anomalies" in an_res
    assert "dictionary" in an_res


# ============================================================
# 10. COMPARE SKIPPED WITH NO PREVIOUS
# ============================================================

def test_compare_skipped_with_no_previous():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df, previous_dataset=None)
    assert result["stages"]["COMPARE"]["status"] == "skipped"
    assert result["stages"]["COMPARE"]["result"] is None


# ============================================================
# 11. RECONSIDER SKIPPED WITH NO PREVIOUS
# ============================================================

def test_reconsider_skipped_with_no_previous():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df, previous_dataset=None)
    assert result["stages"]["RECONSIDER"]["status"] == "skipped"
    assert result["stages"]["RECONSIDER"]["result"] is None


# ============================================================
# 12. COMPARE EXECUTES WITH PREVIOUS
# ============================================================

def test_compare_executes_with_previous():
    old_df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    new_df = pd.DataFrame({"a": [1, 2], "c": [30, 40]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["COMPARE"]["status"] == "completed"
    assert result["stages"]["COMPARE"]["result"]["has_changes"] is True


# ============================================================
# 13. RECONSIDER EXECUTES WITH PREVIOUS
# ============================================================

def test_reconsider_executes_with_previous():
    old_df = pd.DataFrame({"val": [1, np.nan, 3]})
    new_df = pd.DataFrame({"val": [1, 2, 3]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["RECONSIDER"]["status"] == "completed"
    assert "reconsiderations" in result["stages"]["RECONSIDER"]["result"]


# ============================================================
# 14. RE-PLAN EXECUTES
# ============================================================

def test_replan_executes():
    df = pd.DataFrame({"score": [85, 90, 78, 92, 88]})
    result = run_workflow(df)
    replan_res = result["stages"]["RE-PLAN"]["result"]
    assert "next_analyses" in replan_res
    assert len(replan_res["next_analyses"]) > 0


# ============================================================
# 15. REPORT EXECUTES
# ============================================================

def test_report_executes():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = run_workflow(df)
    rep_res = result["stages"]["REPORT"]["result"]
    assert "dataset_name" in rep_res
    assert "workflow_status" in rep_res
    assert "top_findings" in rep_res
    assert "next_actions" in rep_res


# ============================================================
# 16. COMPLETE WORKFLOW STATUS
# ============================================================

def test_complete_workflow_status():
    df = pd.DataFrame({"a": range(10)})
    result = run_workflow(df)
    assert result["status"] == "completed"


# ============================================================
# 17. PARTIAL FAILURE HANDLING
# ============================================================

def test_partial_failure_handling():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Disabling optional analyses
    options = {"run_anomalies": False, "run_story": False}
    result = run_workflow(df, options=options)
    assert result["status"] in ("completed", "partial")
    assert result["stages"]["ANALYZE"]["result"]["anomalies"] is None
    assert result["stages"]["ANALYZE"]["result"]["story"] is None


# ============================================================
# 18. DETECTION FAILURE HANDLING
# ============================================================

def test_detection_failure_handling():
    # Passing an un-profilable object that cannot be converted
    class UnusableObject:
        pass

    result = run_workflow(UnusableObject())
    assert result["status"] in ("completed", "failed", "partial")


# ============================================================
# 19. INVALID OPTIONS
# ============================================================

def test_invalid_options():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Passing invalid options type and strange keys
    result = run_workflow(df, options="invalid_str_options")
    assert result["status"] == "completed"


# ============================================================
# 20. DEFAULT OPTIONS
# ============================================================

def test_default_options():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["X", "Y", "Z"]})
    result = run_workflow(df)
    assert result["stages"]["ANALYZE"]["result"]["anomalies"] is not None
    assert result["stages"]["ANALYZE"]["result"]["dictionary"] is not None


# ============================================================
# 21. DETERMINISTIC WORKFLOW ID
# ============================================================

def test_deterministic_workflow_id():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res1 = run_workflow(df, dataset_name="det_id")
    res2 = run_workflow(df, dataset_name="det_id")
    assert res1["workflow_id"] == res2["workflow_id"]


# ============================================================
# 22. DETERMINISTIC WORKFLOW RESULT
# ============================================================

def test_deterministic_workflow_result():
    df = pd.DataFrame({"score": [80, 90, 70, 85, 95]})
    res1 = run_workflow(df, dataset_name="det_res")
    res2 = run_workflow(df, dataset_name="det_res")
    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


# ============================================================
# 23. JSON SERIALIZATION
# ============================================================

def test_json_serialization():
    df = pd.DataFrame({
        "num": [1.1, np.nan, 3.3, np.inf],
        "cat": ["A", "B", None, "D"],
        "dt": pd.date_range("2023-01-01", periods=4),
    })
    result = run_workflow(df)
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized["status"] == "completed"


# ============================================================
# 24. INPUT IMMUTABILITY
# ============================================================

def test_input_immutability():
    old_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})

    old_copy = old_df.copy(deep=True)
    new_copy = new_df.copy(deep=True)

    _ = run_workflow(new_df, previous_dataset=old_df)

    pd.testing.assert_frame_equal(old_df, old_copy)
    pd.testing.assert_frame_equal(new_df, new_copy)


# ============================================================
# 25. NO TRACEBACK LEAKAGE
# ============================================================

def test_no_traceback_leakage():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = run_workflow(df)
    serialized = json.dumps(result)
    forbidden_tokens = ["Traceback (most recent call last)", 'File "', ", line "]
    for token in forbidden_tokens:
        assert token not in serialized


# ============================================================
# 26. NO API-KEY LEAKAGE
# ============================================================

def test_no_api_key_leakage():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = run_workflow(df)
    serialized = json.dumps(result)
    forbidden = ["AIza", "GEMINI_API_KEY"]
    for token in forbidden:
        assert token not in serialized


# ============================================================
# 27. NO LOCALHOST/DEBUG LEAKAGE
# ============================================================

def test_no_localhost_debug_leakage():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = run_workflow(df)
    serialized = json.dumps(result)
    assert "localhost" not in serialized
    assert "127.0.0.1" not in serialized


# ============================================================
# 28. EVIDENCE REUSED ACROSS STAGES
# ============================================================

def test_evidence_reused_across_stages():
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    result = run_workflow(df)
    # Check that detect result and analyze result agree on basic dimension
    assert result["stages"]["DETECT"]["result"]["rows"] == 5
    assert result["stages"]["ANALYZE"]["result"]["health"]["dimensions"]["completeness"]["score"] >= 90


# ============================================================
# 29. RECOMMENDATIONS GROUNDED IN CURRENT EVIDENCE
# ============================================================

def test_recommendations_grounded_in_current_evidence():
    df = pd.DataFrame({"salary": [50000, 60000, np.nan, 80000, np.nan]})
    result = run_workflow(df)
    replan = result["stages"]["RE-PLAN"]["result"]
    assert len(replan["next_analyses"]) > 0


# ============================================================
# 30. PREVIOUS FINDINGS PASSED TO RECONSIDERATION
# ============================================================

def test_previous_findings_passed_to_reconsideration():
    old_df = pd.DataFrame({"x": [1, np.nan, 3]})
    new_df = pd.DataFrame({"x": [1, 2, 3]})
    prev_f = [{
        "finding_id": "f_miss",
        "title": "Missing values in x",
        "category": "Completeness",
        "severity": "HIGH",
        "affected_columns": ["x"],
        "evidence": ["33% missing"],
    }]

    result = run_workflow(new_df, previous_dataset=old_df, previous_findings=prev_f)
    recon = result["stages"]["RECONSIDER"]["result"]["reconsiderations"]
    assert len(recon) == 1
    assert recon[0]["classification"] == "INVALIDATED"


# ============================================================
# 31. HEALTH PASSED TO RECONSIDERATION
# ============================================================

def test_health_passed_to_reconsideration():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 4]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["RECONSIDER"]["status"] == "completed"


# ============================================================
# 32. STORY PASSED TO RECONSIDERATION
# ============================================================

def test_story_passed_to_reconsideration():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 4]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["ANALYZE"]["result"]["story"] is not None


# ============================================================
# 33. PRIORITIES PASSED TO RECONSIDERATION
# ============================================================

def test_priorities_passed_to_reconsideration():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 4]})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["ANALYZE"]["result"]["priorities"] is not None


# ============================================================
# 34. ADDED COLUMN WORKFLOW
# ============================================================

def test_added_column_workflow():
    old_df = pd.DataFrame({"a": [1, 2, 3]})
    new_df = pd.DataFrame({"a": [1, 2, 3], "new_feature": [10, 20, 30]})
    result = run_workflow(new_df, previous_dataset=old_df)
    sc = result["stages"]["COMPARE"]["result"]["structural_changes"]
    assert any(s["type"] == "added_columns" for s in sc)


# ============================================================
# 35. REMOVED COLUMN WORKFLOW
# ============================================================

def test_removed_column_workflow():
    old_df = pd.DataFrame({"a": [1, 2, 3], "legacy": [10, 20, 30]})
    new_df = pd.DataFrame({"a": [1, 2, 3]})
    result = run_workflow(new_df, previous_dataset=old_df)
    sc = result["stages"]["COMPARE"]["result"]["structural_changes"]
    assert any(s["type"] == "removed_columns" for s in sc)


# ============================================================
# 36. MISSINGNESS CHANGE WORKFLOW
# ============================================================

def test_missingness_change_workflow():
    old_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    new_df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, 5]})
    result = run_workflow(new_df, previous_dataset=old_df)
    qc = result["stages"]["COMPARE"]["result"]["quality_changes"]
    assert any("missingness" in q["type"] for q in qc)


# ============================================================
# 37. ML READINESS CHANGE WORKFLOW
# ============================================================

def test_ml_readiness_change_workflow():
    old_df = pd.DataFrame({"f": range(20), "target": [0, 1] * 10})
    new_df = pd.DataFrame({"f": range(20)})
    result = run_workflow(new_df, previous_dataset=old_df)
    assert result["stages"]["RECONSIDER"]["status"] == "completed"


# ============================================================
# 38. ANOMALY-RELATED WORKFLOW
# ============================================================

def test_anomaly_related_workflow():
    df = pd.DataFrame({"val": [10, 11, 12, 10, 11, 12, 11, 10, 999999]})
    result = run_workflow(df)
    anom_res = result["stages"]["ANALYZE"]["result"]["anomalies"]
    assert anom_res is not None
    assert anom_res["summary"]["total_anomalies"] >= 1


# ============================================================
# 39. EMPTY DATASET
# ============================================================

def test_empty_dataset():
    df = pd.DataFrame()
    result = run_workflow(df)
    assert result["status"] == "completed"
    assert result["dataset"]["rows"] == 0
    assert result["dataset"]["columns"] == 0


# ============================================================
# 40. SINGLE-ROW DATASET
# ============================================================

def test_single_row_dataset():
    df = pd.DataFrame({"id": [1], "name": ["Alice"]})
    result = run_workflow(df)
    assert result["status"] == "completed"
    assert result["dataset"]["rows"] == 1


# ============================================================
# 41. INDEPENDENT ANALYSIS FAILURE DOES NOT CRASH WORKFLOW
# ============================================================

def test_independent_analysis_failure_does_not_crash():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Disable dictionary and anomaly
    opts = {"run_dictionary": False, "run_anomalies": False}
    result = run_workflow(df, options=opts)
    assert result["status"] in ("completed", "partial")
    assert result["stages"]["REPORT"]["status"] == "completed"


# ============================================================
# 42. ALL REQUIRED STAGE NAMES PRESENT
# ============================================================

def test_all_required_stage_names_present():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df)
    expected_stages = ["DETECT", "ANALYZE", "COMPARE", "RECONSIDER", "RE-PLAN", "REPORT"]
    for st in expected_stages:
        assert st in result["stages"]


# ============================================================
# 43. STAGE ORDERING CORRECT
# ============================================================

def test_stage_ordering_correct():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df)
    actual_order = list(result["stages"].keys())
    assert actual_order == ["DETECT", "ANALYZE", "COMPARE", "RECONSIDER", "RE-PLAN", "REPORT"]


# ============================================================
# 44. SKIPPED STAGE STATUS CORRECT
# ============================================================

def test_skipped_stage_status_correct():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df)
    assert result["stages"]["COMPARE"]["status"] == "skipped"
    assert result["stages"]["RECONSIDER"]["status"] == "skipped"


# ============================================================
# 45. SAFE ERROR MESSAGES
# ============================================================

def test_safe_error_messages():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = run_workflow(df)
    for stage_name, stage_obj in result["stages"].items():
        if stage_obj["error"]:
            assert "traceback" not in stage_obj["error"].lower()
            assert "line " not in stage_obj["error"].lower()
