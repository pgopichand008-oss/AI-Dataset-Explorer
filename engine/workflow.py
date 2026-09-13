"""
FILE: engine/workflow.py

Agent Workflow State Engine.
Orchestrates the end-to-end dataset intelligence lifecycle:
DETECT -> ANALYZE -> COMPARE -> RECONSIDER -> RE-PLAN -> REPORT

Responsibilities:
1. Orchestrate Phase 1–8 intelligence modules without duplicating their logic.
2. Maintain a deterministic workflow state machine with explicit stage transitions.
3. Establish a single factual Evidence package in DETECT and share it downstream.
4. Safely handle first-time datasets (COMPARE & RECONSIDER skipped) and updated datasets.
5. Provide resilient partial-failure handling without leaking tracebacks, paths, or keys.
6. Return 100% JSON-serializable, deterministic workflow state.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story
from engine.recommendation_engine import recommend_next_analysis
from engine.anomaly_engine import investigate_anomalies
from engine.dictionary_engine import build_data_dictionary
from engine.reconsideration import reconsider_dataset


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def _to_serializable(val: Any) -> Any:
    """
    Recursively sanitize values to ensure complete JSON serializability.
    Converts NumPy / Pandas scalars, NaNs, Infs, and Timestamps to native Python.
    """
    if val is None:
        return None

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or np.isnan(val) or math.isinf(val) or np.isinf(val):
            return None
        return round(float(val), 4)

    if hasattr(val, "isoformat"):
        try:
            return val.isoformat()
        except Exception:
            pass

    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}

    if isinstance(val, (list, tuple, set)):
        return [_to_serializable(v) for v in val]

    if isinstance(val, pd.Series):
        return [_to_serializable(v) for v in val.tolist()]

    if isinstance(val, pd.DataFrame):
        return [_to_serializable(v) for v in val.to_dict(orient="records")]

    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    return str(val)


def _generate_workflow_id(
    dataset_name: str,
    rows: int,
    cols: int,
    has_previous: bool,
    options_sig: str = "",
) -> str:
    """
    Deterministically generate a stable, readable workflow ID.
    Avoids random UUIDs so identical inputs produce identical workflow IDs.
    """
    seed = f"{dataset_name}_{rows}_{cols}_{has_previous}_{options_sig}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"wf_{digest}"


def _sanitize_error_message(err: Exception, fallback: str) -> str:
    """
    Sanitize internal exceptions so no Python tracebacks, file paths,
    localhost URLs, or API keys are ever surfaced to users.
    """
    # Strictly return safe, clean message without leaking exception details
    return fallback


# ============================================================
# STAGE CREATION HELPER
# ============================================================

def _create_stage_object(
    status: str = "pending",
    result: Any = None,
    evidence: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized stage object."""
    return {
        "status": status,
        "result": result,
        "evidence": evidence or [],
        "warnings": warnings or [],
        "error": error,
    }


# ============================================================
# MAIN WORKFLOW ORCHESTRATOR
# ============================================================

def run_workflow(
    dataset: Union[pd.DataFrame, Dict[str, Any], None],
    previous_dataset: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
    previous_findings: Optional[List[Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the end-to-end dataset intelligence workflow.

    Workflow Stages:
    1. DETECT: Factual evidence foundation (engine.evidence)
    2. ANALYZE: Multi-engine intelligence (health, priorities, story, anomaly, dictionary)
    3. COMPARE: Dataset change detection (change_engine / reconsideration)
    4. RECONSIDER: Finding re-evaluation against new evidence (engine.reconsideration)
    5. RE-PLAN: Adaptive recommendation planning (engine.recommendation_engine)
    6. REPORT: Structured workflow synthesis

    Parameters:
    -----------
    dataset : Union[pd.DataFrame, Dict[str, Any], None]
        Current dataset as DataFrame, pre-built Evidence dict, or None.
    previous_dataset : Optional[Union[pd.DataFrame, Dict[str, Any]]]
        Baseline dataset for change comparison and reconsideration.
    previous_findings : Optional[List[Dict[str, Any]]]
        Historical findings to re-evaluate. If None, derived deterministically.
    dataset_name : Optional[str]
        Optional label for the dataset.
    options : Optional[Dict[str, Any]]
        Workflow control options (e.g. run_anomalies, run_dictionary, run_story).

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable workflow state payload.
    """
    # ------------------------------------------------------------
    # 0. RESOLVE OPTIONS & INPUT IDENTIFIERS
    # ------------------------------------------------------------
    safe_options = {
        "run_anomalies": True,
        "run_dictionary": True,
        "run_story": True,
        "run_recommendations": True,
        "max_insights": 5,
        "max_recommendations": 5,
    }
    if isinstance(options, dict):
        for k, v in options.items():
            safe_options[k] = v

    options_sig = "_".join(f"{k}:{safe_options[k]}" for k in sorted(safe_options.keys()))

    # Resolve name safely
    if dataset_name:
        resolved_name = dataset_name
    elif isinstance(dataset, dict) and "metadata" in dataset:
        resolved_name = dataset.get("metadata", {}).get("dataset_name", "dataset")
    else:
        resolved_name = "dataset"

    # Initialize workflow stages
    stages: Dict[str, Dict[str, Any]] = {
        "DETECT": _create_stage_object("pending"),
        "ANALYZE": _create_stage_object("pending"),
        "COMPARE": _create_stage_object("pending"),
        "RECONSIDER": _create_stage_object("pending"),
        "RE-PLAN": _create_stage_object("pending"),
        "REPORT": _create_stage_object("pending"),
    }

    workflow_warnings: List[str] = []
    workflow_errors: List[str] = []
    current_stage = "DETECT"
    overall_status = "completed"

    has_previous = previous_dataset is not None

    # ------------------------------------------------------------
    # STAGE 1: DETECT
    # ------------------------------------------------------------
    current_stage = "DETECT"
    stages["DETECT"]["status"] = "running"
    current_evidence: Optional[Dict[str, Any]] = None

    try:
        if isinstance(dataset, dict) and "structure" in dataset:
            current_evidence = dataset
        elif isinstance(dataset, pd.DataFrame):
            current_evidence = build_evidence(dataset, dataset_name=resolved_name)
        elif dataset is None:
            current_evidence = build_evidence(None, dataset_name=resolved_name)
        else:
            current_evidence = build_evidence(None, dataset_name=resolved_name)

        struct = current_evidence.get("structure", {})
        comp = current_evidence.get("completeness", {})
        dup = current_evidence.get("duplicates", {})

        row_count = int(struct.get("row_count", 0))
        col_count = int(struct.get("column_count", 0))

        detect_result = {
            "dataset_name": resolved_name,
            "rows": row_count,
            "columns": col_count,
            "column_names": list(struct.get("column_names", [])),
            "numeric_columns": list(struct.get("numerical_columns", [])),
            "categorical_columns": list(struct.get("categorical_columns", [])),
            "datetime_columns": list(struct.get("datetime_columns", [])),
            "boolean_columns": list(struct.get("boolean_columns", [])),
            "missing_cells": int(comp.get("missing_cells", 0)),
            "missing_percentage": float(comp.get("missing_percentage", 0.0)),
            "duplicate_rows": int(dup.get("duplicate_rows", 0)),
            "duplicate_row_percentage": float(dup.get("duplicate_row_percentage", 0.0)),
            "memory_bytes": int(struct.get("memory_bytes", 0)),
        }

        detect_evidence = [
            f"Dataset dimensions: {row_count:,} rows × {col_count:,} columns.",
            f"Data types: {len(struct.get('numerical_columns', []))} numeric, {len(struct.get('categorical_columns', []))} categorical, {len(struct.get('datetime_columns', []))} datetime.",
            f"Completeness: {comp.get('complete_percentage', 100.0):.1f}% complete ({comp.get('missing_cells', 0):,} missing cells).",
        ]
        detect_warnings = list(current_evidence.get("limitations", []))

        stages["DETECT"]["status"] = "completed"
        stages["DETECT"]["result"] = detect_result
        stages["DETECT"]["evidence"] = detect_evidence
        stages["DETECT"]["warnings"] = detect_warnings
        workflow_warnings.extend(detect_warnings)

    except Exception as e:
        stages["DETECT"]["status"] = "failed"
        stages["DETECT"]["error"] = _sanitize_error_message(e, "Dataset detection could not be completed.")
        workflow_errors.append("Dataset detection failed.")
        overall_status = "failed"

    # If DETECT failed completely, skip all dependent stages
    if stages["DETECT"]["status"] == "failed" or current_evidence is None:
        for st in ["ANALYZE", "COMPARE", "RECONSIDER", "RE-PLAN", "REPORT"]:
            stages[st]["status"] = "skipped"
            stages[st]["error"] = "Skipped due to upstream detection failure."

        wf_id = _generate_workflow_id(resolved_name, 0, 0, has_previous, options_sig)
        return _to_serializable({
            "workflow_id": wf_id,
            "status": "failed",
            "current_stage": current_stage,
            "stages": stages,
            "dataset": {"name": resolved_name, "rows": 0, "columns": 0, "has_previous": has_previous},
            "summary": {"workflow_status": "failed"},
            "warnings": workflow_warnings,
            "errors": workflow_errors,
        })

    # ------------------------------------------------------------
    # STAGE 2: ANALYZE
    # ------------------------------------------------------------
    current_stage = "ANALYZE"
    stages["ANALYZE"]["status"] = "running"
    analyze_result: Dict[str, Any] = {}
    analyze_evidence: List[str] = []
    analyze_warnings: List[str] = []

    health_data: Optional[Dict[str, Any]] = None
    priorities_data: Optional[Dict[str, Any]] = None
    story_data: Optional[Dict[str, Any]] = None
    anomalies_data: Optional[Dict[str, Any]] = None
    dictionary_data: Optional[Dict[str, Any]] = None

    partial_failure_occurred = False

    # 2a. Health Engine
    try:
        health_data = calculate_health(current_evidence, dataset_name=resolved_name)
        analyze_result["health"] = health_data
        analyze_evidence.append(
            f"Health score: {health_data.get('health_score')}/100 ({health_data.get('health_status')})."
        )
    except Exception as e:
        partial_failure_occurred = True
        msg = _sanitize_error_message(e, "Health assessment could not be completed.")
        analyze_warnings.append(msg)
        analyze_result["health"] = None

    # 2b. Priority Insights Engine
    try:
        priorities_data = build_prioritized_insights(
            current_evidence,
            health=health_data,
            dataset_name=resolved_name,
            max_insights=safe_options.get("max_insights", 5),
        )
        analyze_result["priorities"] = priorities_data
        analyze_evidence.append(
            f"Prioritized insights: {priorities_data.get('summary', {}).get('total_findings', 0)} finding(s) discovered."
        )
    except Exception as e:
        partial_failure_occurred = True
        msg = _sanitize_error_message(e, "Insight prioritization could not be completed.")
        analyze_warnings.append(msg)
        analyze_result["priorities"] = None

    # 2c. Data Story Engine
    if safe_options.get("run_story", True):
        try:
            story_data = build_data_story(
                current_evidence,
                health=health_data,
                prioritized_insights=priorities_data,
                dataset_name=resolved_name,
                max_insights=safe_options.get("max_insights", 5),
            )
            analyze_result["story"] = story_data
        except Exception as e:
            partial_failure_occurred = True
            msg = _sanitize_error_message(e, "Data story generation could not be completed.")
            analyze_warnings.append(msg)
            analyze_result["story"] = None
    else:
        analyze_result["story"] = None

    # 2d. Anomaly Engine
    if safe_options.get("run_anomalies", True):
        try:
            anomalies_data = investigate_anomalies(
                current_evidence,
                dataset_name=resolved_name,
            )
            analyze_result["anomalies"] = anomalies_data
            tot_anom = anomalies_data.get("summary", {}).get("total_anomalies", 0)
            analyze_evidence.append(f"Anomalies detected: {tot_anom} observation(s).")
        except Exception as e:
            partial_failure_occurred = True
            msg = _sanitize_error_message(e, "Anomaly investigation could not be completed.")
            analyze_warnings.append(msg)
            analyze_result["anomalies"] = None
    else:
        analyze_result["anomalies"] = None

    # 2e. Data Dictionary Engine
    if safe_options.get("run_dictionary", True):
        try:
            dictionary_data = build_data_dictionary(
                current_evidence,
                dataset_name=resolved_name,
            )
            analyze_result["dictionary"] = dictionary_data
            analyze_evidence.append(
                f"Data dictionary compiled: {dictionary_data.get('column_count', 0)} column entries."
            )
        except Exception as e:
            partial_failure_occurred = True
            msg = _sanitize_error_message(e, "Data dictionary compilation could not be completed.")
            analyze_warnings.append(msg)
            analyze_result["dictionary"] = None
    else:
        analyze_result["dictionary"] = None

    if partial_failure_occurred:
        stages["ANALYZE"]["status"] = "completed"
        overall_status = "partial"
    else:
        stages["ANALYZE"]["status"] = "completed"

    stages["ANALYZE"]["result"] = analyze_result
    stages["ANALYZE"]["evidence"] = analyze_evidence
    stages["ANALYZE"]["warnings"] = analyze_warnings
    workflow_warnings.extend(analyze_warnings)

    # ------------------------------------------------------------
    # STAGE 3: COMPARE
    # ------------------------------------------------------------
    current_stage = "COMPARE"
    reconsideration_output: Optional[Dict[str, Any]] = None

    if not has_previous:
        stages["COMPARE"]["status"] = "skipped"
        stages["COMPARE"]["result"] = None
        stages["COMPARE"]["evidence"] = ["No previous dataset was supplied for comparison."]
    else:
        stages["COMPARE"]["status"] = "running"
        try:
            # We run reconsideration_dataset which encapsulates change_engine comparison
            reconsideration_output = reconsider_dataset(
                old_evidence_or_df=previous_dataset,
                new_evidence_or_df=current_evidence,
                previous_findings=previous_findings,
                new_health=health_data,
                new_story=story_data,
                new_priorities=priorities_data,
                dataset_name=resolved_name,
            )

            change_summary = reconsideration_output.get("change_summary", {})
            has_chg = change_summary.get("has_changes", False)
            chg_count = change_summary.get("change_count", 0)

            stages["COMPARE"]["status"] = "completed"
            stages["COMPARE"]["result"] = change_summary
            stages["COMPARE"]["evidence"] = [
                f"Dataset comparison completed: {chg_count} change(s) detected.",
                f"Has changes: {has_chg}.",
            ]
        except Exception as e:
            stages["COMPARE"]["status"] = "failed"
            msg = _sanitize_error_message(e, "Dataset comparison could not be completed.")
            stages["COMPARE"]["error"] = msg
            workflow_warnings.append(msg)
            overall_status = "partial"

    # ------------------------------------------------------------
    # STAGE 4: RECONSIDER
    # ------------------------------------------------------------
    current_stage = "RECONSIDER"

    if not has_previous:
        stages["RECONSIDER"]["status"] = "skipped"
        stages["RECONSIDER"]["result"] = None
        stages["RECONSIDER"]["evidence"] = ["Reconsideration skipped because no previous dataset was supplied."]
    else:
        stages["RECONSIDER"]["status"] = "running"
        if reconsideration_output is not None:
            recons = reconsideration_output.get("reconsiderations", [])
            recon_summary = reconsideration_output.get("summary", {})
            resolved_f = reconsideration_output.get("resolved_findings", [])
            new_f = reconsideration_output.get("new_findings", [])
            req_rev = reconsideration_output.get("requires_review", [])

            stages["RECONSIDER"]["status"] = "completed"
            stages["RECONSIDER"]["result"] = {
                "reconsiderations": recons,
                "summary": recon_summary,
                "new_findings": new_f,
                "resolved_findings": resolved_f,
                "requires_review": req_rev,
            }
            stages["RECONSIDER"]["evidence"] = [
                f"Evaluated {len(recons)} previous finding(s): {recon_summary.get('validated', 0)} validated, "
                f"{recon_summary.get('invalidated', 0)} invalidated, {recon_summary.get('strengthened', 0)} strengthened, "
                f"{recon_summary.get('weakened', 0)} weakened, {recon_summary.get('requires_review', 0)} requires review.",
                f"Resolved findings: {len(resolved_f)}.",
                f"New findings: {len(new_f)}.",
            ]
        else:
            stages["RECONSIDER"]["status"] = "skipped"
            stages["RECONSIDER"]["error"] = "Reconsideration skipped due to upstream comparison failure."

    # ------------------------------------------------------------
    # STAGE 5: RE-PLAN
    # ------------------------------------------------------------
    current_stage = "RE-PLAN"
    stages["RE-PLAN"]["status"] = "running"

    try:
        # Base recommendations from recommendation engine
        rec_result = recommend_next_analysis(
            current_evidence,
            health=health_data,
            prioritized_insights=priorities_data,
            dataset_name=resolved_name,
            max_recommendations=safe_options.get("max_recommendations", 5),
        )
        base_recs = rec_result.get("recommendations", [])

        # If reconsideration produced targeted next analyses, prioritize them
        merged_next: List[Dict[str, Any]] = []
        if reconsideration_output and reconsideration_output.get("next_analysis"):
            recon_next = reconsideration_output.get("next_analysis", [])
            for rn in recon_next:
                merged_next.append(rn)

        for br in base_recs:
            if not any(m.get("name") == br.get("name") for m in merged_next):
                merged_next.append(br)

        final_recs = merged_next[:safe_options.get("max_recommendations", 5)]

        stages["RE-PLAN"]["status"] = "completed"
        stages["RE-PLAN"]["result"] = {
            "next_analyses": final_recs,
            "recommendation_count": len(final_recs),
            "top_priority": final_recs[0]["name"] if final_recs else None,
        }
        stages["RE-PLAN"]["evidence"] = [
            f"Recommended {len(final_recs)} next analytical action(s).",
            f"Top priority: {final_recs[0]['name'] if final_recs else 'None'}.",
        ]
    except Exception as e:
        stages["RE-PLAN"]["status"] = "failed"
        msg = _sanitize_error_message(e, "Next analysis planning could not be completed.")
        stages["RE-PLAN"]["error"] = msg
        workflow_warnings.append(msg)
        overall_status = "partial"

    # ------------------------------------------------------------
    # STAGE 6: REPORT
    # ------------------------------------------------------------
    current_stage = "REPORT"
    stages["REPORT"]["status"] = "running"

    try:
        top_insights_titles = [
            ins.get("title", "") for ins in (priorities_data.get("insights", []) if priorities_data else [])[:3]
        ]
        next_action_names = [
            rec.get("name", "") for rec in stages["RE-PLAN"].get("result", {}).get("next_analyses", [])
        ]

        report_result = {
            "dataset_name": resolved_name,
            "workflow_status": overall_status,
            "health_score": health_data.get("health_score") if health_data else None,
            "health_status": health_data.get("health_status") if health_data else None,
            "top_findings": top_insights_titles,
            "changes_detected": (
                stages["COMPARE"].get("result", {}).get("has_changes", False)
                if stages["COMPARE"]["status"] == "completed"
                else False
            ),
            "reconsideration_summary": (
                stages["RECONSIDER"].get("result", {}).get("summary")
                if stages["RECONSIDER"]["status"] == "completed"
                else None
            ),
            "next_actions": next_action_names,
            "warnings": workflow_warnings[:5],
        }

        stages["REPORT"]["status"] = "completed"
        stages["REPORT"]["result"] = report_result
        stages["REPORT"]["evidence"] = [
            f"Synthesized workflow report with status '{overall_status}'.",
        ]
    except Exception as e:
        stages["REPORT"]["status"] = "failed"
        msg = _sanitize_error_message(e, "Workflow report synthesis could not be completed.")
        stages["REPORT"]["error"] = msg
        workflow_warnings.append(msg)
        overall_status = "partial"

    # ------------------------------------------------------------
    # 7. ASSEMBLE FINAL WORKFLOW PAYLOAD
    # ------------------------------------------------------------
    row_count = int(current_evidence.get("structure", {}).get("row_count", 0))
    col_count = int(current_evidence.get("structure", {}).get("column_count", 0))

    workflow_id = _generate_workflow_id(resolved_name, row_count, col_count, has_previous, options_sig)

    top_findings = [
        ins.get("title", "") for ins in (priorities_data.get("insights", []) if priorities_data else [])[:3]
    ]
    next_analysis_list = stages["RE-PLAN"].get("result", {}).get("next_analyses", [])

    top_summary = {
        "overall_health": health_data.get("health_score") if health_data else None,
        "health_status": health_data.get("health_status") if health_data else None,
        "top_findings": top_findings,
        "changes_detected": (
            stages["COMPARE"].get("result", {}).get("has_changes", False)
            if stages["COMPARE"]["status"] == "completed"
            else False
        ),
        "reconsiderations": (
            stages["RECONSIDER"].get("result", {}).get("summary")
            if stages["RECONSIDER"]["status"] == "completed"
            else None
        ),
        "next_analysis": next_analysis_list,
        "workflow_status": overall_status,
    }

    output = {
        "workflow_id": workflow_id,
        "status": overall_status,
        "current_stage": current_stage,
        "stages": stages,
        "dataset": {
            "name": resolved_name,
            "rows": row_count,
            "columns": col_count,
            "has_previous": has_previous,
        },
        "summary": top_summary,
        "warnings": workflow_warnings,
        "errors": workflow_errors,
    }

    return _to_serializable(output)
