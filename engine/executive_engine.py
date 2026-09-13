"""
FILE: engine/executive_engine.py

Executive Intelligence Engine.
Transforms structured, factual evidence from Phase 1-9 intelligence modules
into a concise, high-level executive analytical summary.

Architecture:
DATASET -> EVIDENCE -> HEALTH -> PRIORITY -> STORY -> ANOMALIES -> RECONSIDERATION -> RECOMMENDATIONS -> EXECUTIVE INTELLIGENCE -> (OPTIONAL) GEMINI

Core Principles:
1. Deterministic Truth: All factual numbers, health scores, counts, correlations,
   anomalies, and changes come from backend intelligence engines.
2. Fact / Interpretation Separation: Factual observations are kept distinct from
   analytical interpretations.
3. Optional Gemini Layer: If use_gemini=True, Gemini provides executive narrative
   interpretation of the factual evidence package, but never invents numbers.
4. Resilient Fallback: If Gemini is disabled, unavailable, or encounters errors,
   the engine seamlessly returns 100% deterministic executive intelligence.
5. Zero Leakage: No tracebacks, internal file paths, localhost URLs, or API keys
   are ever returned or exposed in warnings/messages.
6. 100% JSON-serializable output and input immutability.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story
from engine.recommendation_engine import recommend_next_analysis
from engine.anomaly_engine import investigate_anomalies
from engine.reconsideration import reconsider_dataset

try:
    from config import get_gemini_client, get_gemini_model
except ImportError:
    def get_gemini_client():
        return None

    def get_gemini_model():
        return "gemini-3.6-flash"


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


# ============================================================
# FACT EXTRACTION HELPER
# ============================================================

def _format_fact(evidence_val: Any, fallback_title: str) -> str:
    """
    Format a concise, authoritative factual string from evidence basis.
    Never invents statistics or numbers.
    """
    if isinstance(evidence_val, dict):
        parts: List[str] = []
        # Missingness specific
        if "missing_percentage" in evidence_val:
            col = evidence_val.get("column", "column")
            pct = evidence_val.get("missing_percentage", 0.0)
            cnt = evidence_val.get("missing_count", 0)
            return f"{pct:.1f}% of values in '{col}' are missing ({cnt} observation(s))."

        # Duplicates specific
        if "duplicate_percentage" in evidence_val or "duplicate_row_percentage" in evidence_val:
            cnt = evidence_val.get("duplicate_rows", 0)
            pct = evidence_val.get("duplicate_percentage", evidence_val.get("duplicate_row_percentage", 0.0))
            return f"{cnt} duplicate record(s) detected ({pct:.1f}% of dataset)."

        # Sample size specific
        if "row_count" in evidence_val and "minimum_recommended" in evidence_val:
            rc = evidence_val.get("row_count", 0)
            min_rec = evidence_val.get("minimum_recommended", 30)
            return f"Dataset contains {rc} row(s), which is below standard empirical threshold of {min_rec}."

        # Outliers / Anomalies
        if "anomaly_count" in evidence_val or "outlier_count" in evidence_val:
            cnt = evidence_val.get("anomaly_count", evidence_val.get("outlier_count", 0))
            pct = evidence_val.get("affected_percentage", evidence_val.get("outlier_percentage", 0.0))
            col = evidence_val.get("column", "feature")
            return f"{cnt} potential anomaly observation(s) ({pct:.1f}%) identified in '{col}'."

        # Generic dict key-values
        for k, v in evidence_val.items():
            clean_k = str(k).replace("_", " ")
            if isinstance(v, float):
                parts.append(f"{clean_k}: {v:.2f}")
            else:
                parts.append(f"{clean_k}: {v}")
        if parts:
            return ", ".join(parts)

    elif isinstance(evidence_val, list):
        if evidence_val:
            return "; ".join(str(x) for x in evidence_val)
    elif evidence_val:
        return str(evidence_val)

    return fallback_title


# ============================================================
# DETERMINISTIC EXECUTIVE SUMMARY BUILDER
# ============================================================

def _build_deterministic_summary(
    dataset_name: str,
    row_count: int,
    col_count: int,
    num_cols: List[str],
    cat_cols: List[str],
    dt_cols: List[str],
    health_score: Optional[float],
    health_status: str,
    key_findings: List[Dict[str, Any]],
    patterns: List[str],
    anomalies: List[Dict[str, Any]],
    changes: Dict[str, Any],
    reconsiderations: List[Dict[str, Any]],
    next_actions: List[Dict[str, Any]],
    limitations: List[str],
) -> str:
    """
    Synthesize an authoritative, deterministic executive summary
    answering the 9 core analytical questions strictly grounded in evidence.
    """
    sections: List[str] = []

    # 1. Dataset Characteristics
    type_parts = []
    if num_cols:
        type_parts.append(f"{len(num_cols)} numerical")
    if cat_cols:
        type_parts.append(f"{len(cat_cols)} categorical")
    if dt_cols:
        type_parts.append(f"{len(dt_cols)} datetime")
    types_desc = f" ({', '.join(type_parts)})" if type_parts else ""

    if row_count == 0 or col_count == 0:
        sec1 = (
            f"Dataset '{dataset_name}' contains {row_count} rows and {col_count} columns. "
            f"No usable observational records are available for analysis."
        )
    else:
        sec1 = (
            f"Dataset '{dataset_name}' comprises {row_count} rows across {col_count} columns{types_desc}."
        )
    sections.append(f"1. Dataset Characteristics:\n{sec1}")

    # 2. Health & Readiness
    score_str = f"{health_score:.1f}/100" if health_score is not None else "N/A"
    sec2 = f"Overall data health is evaluated as {health_status} with a composite score of {score_str}."
    sections.append(f"2. Data Health & Readiness:\n{sec2}")

    # 3. Key Issues / Priority Findings
    if key_findings:
        f_lines = []
        for f in key_findings[:5]:
            sev = f.get("severity", "INFO")
            fact = f.get("fact", "")
            interp = f.get("interpretation", "")
            f_lines.append(f"- [{sev}] {fact} {interp}".strip())
        sec3 = "\n".join(f_lines)
    else:
        sec3 = "No critical data quality or structural issues were identified."
    sections.append(f"3. Most Important Issues:\n{sec3}")

    # 4. Patterns
    if patterns:
        sec4 = "\n".join(f"- {p}" for p in patterns[:5])
    else:
        sec4 = "No pronounced linear correlations or extreme categorical imbalances were observed."
    sections.append(f"4. Important Patterns:\n{sec4}")

    # 5. Anomalies
    if anomalies:
        a_lines = []
        for a in anomalies[:5]:
            col = a.get("column", "feature")
            cnt = a.get("anomaly_count", a.get("count", 0))
            pct = a.get("affected_percentage", a.get("percentage", 0.0))
            method = a.get("method", "statistical detection")
            a_lines.append(
                f"- Potential anomalies in '{col}': {cnt} observation(s) ({pct:.1f}%) identified via {method}."
            )
        sec5 = "\n".join(a_lines)
    else:
        sec5 = "No potential anomalies were detected across numerical or categorical columns."
    sections.append(f"5. Potential Anomalies:\n{sec5}")

    # 6. Changes
    has_chg = changes.get("detected", False)
    chg_summary = changes.get("summary", [])
    if has_chg and chg_summary:
        sec6 = "\n".join(f"- {s}" for s in chg_summary[:5])
    elif chg_summary:
        sec6 = str(chg_summary[0])
    else:
        sec6 = "No previous baseline dataset was supplied for change detection."
    sections.append(f"6. Dataset Changes:\n{sec6}")

    # 7. Reconsideration
    if reconsiderations:
        r_lines = []
        for r in reconsiderations[:5]:
            title = r.get("title", "")
            cls = r.get("classification", "UNCHANGED")
            reason = r.get("reason", "")
            r_lines.append(f"- [{cls}] {title}: {reason}".strip())
        sec7 = "\n".join(r_lines)
    else:
        sec7 = "No previous analytical conclusions required reconsideration."
    sections.append(f"7. Reconsideration of Previous Conclusions:\n{sec7}")

    # 8. Next Actions
    if next_actions:
        act_lines = []
        for idx, act in enumerate(next_actions[:3], start=1):
            name = act.get("name", "")
            why = act.get("why", "")
            act_lines.append(f"{idx}. {name}: {why}".strip())
        sec8 = "\n".join(act_lines)
    else:
        sec8 = "Proceed with initial exploratory profiling and data ingestion validation."
    sections.append(f"8. Recommended Next Analyses:\n{sec8}")

    # 9. Limitations
    if limitations:
        sec9 = "\n".join(f"- {lim}" for lim in limitations[:5])
    else:
        sec9 = "No primary analytical limitations were noted from automated checks."
    sections.append(f"9. Main Analytical Limitations:\n{sec9}")

    return "\n\n".join(sections)


# ============================================================
# GEMINI EXECUTIVE CALLER
# ============================================================

def _call_gemini_interpretation(
    evidence_package: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Gemini with the structured evidence package.
    Strict prompt constraints ensure Gemini only interprets supplied facts
    and never invents numbers or facts.
    Returns (summary_text, error_message).
    """
    client = get_gemini_client()
    if client is None:
        return None, "Gemini client is not configured or GEMINI_API_KEY is missing."

    model = get_gemini_model()

    package_json = json.dumps(evidence_package, indent=2)

    prompt = (
        "You are an executive data analyst.\n\n"
        "Use ONLY the supplied evidence.\n"
        "Do not invent or estimate any numerical value.\n"
        "Do not introduce facts not present in the evidence.\n"
        "Do not infer causality unless explicitly supported.\n"
        "If evidence is insufficient, say so.\n"
        "Separate observed facts from interpretation.\n"
        "Return concise executive intelligence covering:\n"
        "- Executive Summary\n"
        "- Key Findings\n"
        "- Data Health\n"
        "- Important Patterns\n"
        "- Anomalies\n"
        "- Changes / Reconsideration\n"
        "- Recommended Next Analyses\n"
        "- Limitations\n\n"
        f"EVIDENCE PACKAGE:\n{package_json}\n"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and str(text).strip():
            return str(text).strip(), None

        return None, "Gemini returned an empty response."

    except Exception:
        # Never leak exception text, tracebacks, paths, or keys
        return None, "Gemini call encountered an error."


# ============================================================
# MAIN PUBLIC API
# ============================================================

def build_executive_intelligence(
    evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None] = None,
    health: Optional[Dict[str, Any]] = None,
    prioritized_insights: Optional[Dict[str, Any]] = None,
    story: Optional[Dict[str, Any]] = None,
    anomalies: Optional[Dict[str, Any]] = None,
    reconsideration: Optional[Dict[str, Any]] = None,
    recommendations: Optional[Dict[str, Any]] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    use_gemini: bool = False,
) -> Dict[str, Any]:
    """
    Build executive intelligence summary and structured package.

    Transforms factual evidence from Phase 1-9 engines into an executive-level summary
    with strictly separated facts and interpretations, and optional Gemini narrative.

    Parameters:
    -----------
    evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Underlying DataFrame, pre-computed Evidence dict, or None.
    health : Optional[Dict[str, Any]]
        Health engine output (engine.health.calculate_health).
    prioritized_insights : Optional[Dict[str, Any]]
        Priority engine output (engine.priority_engine.build_prioritized_insights).
    story : Optional[Dict[str, Any]]
        Data story output (engine.story_engine.build_data_story).
    anomalies : Optional[Dict[str, Any]]
        Anomaly engine output (engine.anomaly_engine.investigate_anomalies).
    reconsideration : Optional[Dict[str, Any]]
        Reconsideration engine output (engine.reconsideration.reconsider_dataset).
    recommendations : Optional[Dict[str, Any]]
        Recommendation engine output (engine.recommendation_engine.recommend_next_analysis).
    workflow_state : Optional[Dict[str, Any]]
        Agent workflow state output (engine.workflow.run_workflow).
    dataset_name : Optional[str]
        Human-readable name for the dataset.
    use_gemini : bool
        Whether to invoke Gemini for narrative interpretation (default: False).

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable executive intelligence payload.
    """
    warnings: List[str] = []
    generated_from: List[str] = []

    # 1. Resolve from workflow_state if provided
    if isinstance(workflow_state, dict):
        generated_from.append("engine.workflow")
        det_res = workflow_state.get("stages", {}).get("DETECT", {}).get("result", {}) or {}
        if not isinstance(det_res, dict):
            det_res = {}
        ds_info = workflow_state.get("dataset", {}) or {}
        if not isinstance(ds_info, dict):
            ds_info = {}

        if dataset_name is None:
            dataset_name = (
                workflow_state.get("dataset_name")
                or det_res.get("dataset_name")
                or ds_info.get("name")
                or (workflow_state.get("stages", {}).get("REPORT", {}).get("result", {}) or {}).get("dataset_name")
            )

        if evidence_or_df is None:
            evidence_or_df = {
                "metadata": {"dataset_name": dataset_name},
                "structure": {
                    "dataset_name": dataset_name,
                    "row_count": det_res.get("rows", det_res.get("row_count", ds_info.get("rows", 0))),
                    "column_count": det_res.get("columns", det_res.get("column_count", ds_info.get("columns", 0))),
                    "numerical_columns": det_res.get("numeric_columns", det_res.get("numerical_columns", [])),
                    "numeric_columns": det_res.get("numeric_columns", det_res.get("numerical_columns", [])),
                    "categorical_columns": det_res.get("categorical_columns", []),
                    "datetime_columns": det_res.get("datetime_columns", []),
                },
                "completeness": {
                    "missing_cells": det_res.get("missing_cells", 0),
                    "column_missing_percentages": {},
                    "column_missing_counts": {},
                },
                "duplicates": {
                    "duplicate_rows": det_res.get("duplicate_rows", 0),
                    "duplicate_row_percentage": det_res.get("duplicate_row_percentage", 0.0),
                },
                "quality": {
                    "constant_columns": [],
                },
                "limitations": list(workflow_state.get("warnings", [])),
            }

        # Extract analyze stage outputs if not explicitly passed
        analyze_res = (
            workflow_state.get("stages", {}).get("ANALYZE", {}).get("result", {})
            if isinstance(workflow_state.get("stages"), dict)
            else {}
        )
        if not isinstance(analyze_res, dict):
            analyze_res = {}

        if health is None:
            health = workflow_state.get("health") or analyze_res.get("health")
        if prioritized_insights is None:
            prioritized_insights = workflow_state.get("priorities") or analyze_res.get("priorities")
        if story is None:
            story = workflow_state.get("story") or analyze_res.get("story")
        if anomalies is None:
            anomalies = workflow_state.get("anomalies") or analyze_res.get("anomalies")

        # Extract compare/reconsider stage outputs
        reconsider_res = (
            workflow_state.get("stages", {}).get("RECONSIDER", {}).get("result", {})
            if isinstance(workflow_state.get("stages"), dict)
            else {}
        )
        compare_res = (
            workflow_state.get("stages", {}).get("COMPARE", {}).get("result", {})
            if isinstance(workflow_state.get("stages"), dict)
            else {}
        )

        if reconsideration is None:
            if isinstance(reconsider_res, dict) and reconsider_res:
                reconsideration = reconsider_res
            elif isinstance(compare_res, dict) and compare_res:
                reconsideration = {"change_summary": compare_res, "reconsiderations": []}

        # Extract next_analysis / recommendations
        replan_res = (
            workflow_state.get("stages", {}).get("RE-PLAN", {}).get("result", {})
            if isinstance(workflow_state.get("stages"), dict)
            else {}
        )
        if recommendations is None:
            recommendations = (
                workflow_state.get("next_analysis")
                or (replan_res.get("next_analyses") if isinstance(replan_res, dict) else None)
            )

    # 2. Resolve Evidence & Raw DataFrame safely (never mutate input)
    raw_df: Optional[pd.DataFrame] = None
    evidence: Optional[Dict[str, Any]] = None

    if isinstance(evidence_or_df, pd.DataFrame):
        raw_df = evidence_or_df
        resolved_name = dataset_name or getattr(evidence_or_df, "name", None) or "dataset"
        evidence = build_evidence(evidence_or_df, dataset_name=resolved_name)
        generated_from.append("engine.evidence")
    elif isinstance(evidence_or_df, dict) and "structure" in evidence_or_df:
        evidence = copy.deepcopy(evidence_or_df)
        resolved_name = (
            dataset_name
            or (evidence.get("structure", {}) or {}).get("dataset_name")
            or (evidence.get("metadata", {}) or {}).get("dataset_name")
            or evidence.get("dataset_name")
            or "dataset"
        )
        generated_from.append("engine.evidence")
    else:
        resolved_name = dataset_name or "dataset"

    # Handle None or empty evidence safely
    if evidence is None:
        warnings.append("No valid dataset or evidence was provided for analysis.")
        empty_health = {
            "score": None,
            "status": "UNKNOWN",
            "strengths": [],
            "weaknesses": [],
            "risks": [],
        }
        empty_package = {
            "dataset": {
                "name": resolved_name,
                "rows": 0,
                "columns": 0,
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
            },
            "health": empty_health,
            "top_findings": [],
            "patterns": [],
            "anomalies": [],
            "changes": {"detected": False, "summary": ["No dataset available for comparison."]},
            "reconsideration": [],
            "next_actions": [],
            "limitations": ["No dataset was supplied for analytical profiling."],
        }
        return {
            "dataset_name": resolved_name,
            "executive": {
                "summary": "No dataset was provided for executive intelligence analysis.",
                "health": empty_health,
                "key_findings": [],
                "patterns": [],
                "anomalies": [],
                "changes": {"detected": False, "summary": ["No dataset available for comparison."]},
                "reconsideration": [],
                "next_actions": [],
                "limitations": ["No dataset was supplied for analytical profiling."],
            },
            "evidence_package": empty_package,
            "generation": {
                "mode": "deterministic",
                "gemini_used": False,
                "fallback": False,
            },
            "warnings": warnings,
            "generated_from": ["engine.executive_engine"],
        }

    # 3. Resolve Missing Intelligence Components Deterministically
    # 3a. Health
    if health is None:
        try:
            health = calculate_health(evidence)
            generated_from.append("engine.health")
        except Exception:
            health = None
    elif isinstance(health, dict):
        generated_from.append("engine.health")

    # 3b. Priorities
    if prioritized_insights is None:
        try:
            prioritized_insights = build_prioritized_insights(
                evidence, health=health, dataset_name=resolved_name
            )
            generated_from.append("engine.priority_engine")
        except Exception:
            prioritized_insights = None
    elif isinstance(prioritized_insights, dict):
        generated_from.append("engine.priority_engine")

    # 3c. Story
    if story is None:
        try:
            story = build_data_story(
                evidence,
                health=health,
                prioritized_insights=prioritized_insights,
                dataset_name=resolved_name,
            )
            generated_from.append("engine.story_engine")
        except Exception:
            story = None
    elif isinstance(story, dict):
        generated_from.append("engine.story_engine")

    # 3d. Anomalies
    if anomalies is None:
        try:
            if raw_df is not None:
                anomalies = investigate_anomalies(raw_df, dataset_name=resolved_name)
            else:
                anomalies = investigate_anomalies(evidence, dataset_name=resolved_name)
            generated_from.append("engine.anomaly_engine")
        except Exception:
            anomalies = None
    elif isinstance(anomalies, dict):
        generated_from.append("engine.anomaly_engine")

    # 3e. Recommendations
    if recommendations is None:
        try:
            recommendations = recommend_next_analysis(
                evidence,
                health=health,
                prioritized_insights=prioritized_insights,
                dataset_name=resolved_name,
            )
            generated_from.append("engine.recommendation_engine")
        except Exception:
            recommendations = None
    elif isinstance(recommendations, (dict, list)):
        generated_from.append("engine.recommendation_engine")

    # 3f. Reconsideration
    if reconsideration is not None:
        generated_from.append("engine.reconsideration")

    # 4. Extract and Structure Factual Findings
    structure = evidence.get("structure", {}) if isinstance(evidence.get("structure"), dict) else {}
    row_count = int(structure.get("row_count", 0))
    col_count = int(structure.get("column_count", 0))
    numeric_cols = list(
        structure.get("numerical_columns")
        or structure.get("numeric_columns")
        or []
    )
    cat_cols = list(structure.get("categorical_columns", []))
    dt_cols = list(structure.get("datetime_columns", []))

    # Health Data Extraction
    if isinstance(health, dict):
        overall = health.get("overall", {}) if isinstance(health.get("overall"), dict) else {}
        health_score = overall.get("score", health.get("health_score"))
        if health_score is not None:
            try:
                health_score = round(float(health_score), 2)
            except (ValueError, TypeError):
                pass
        health_status = overall.get("status", health.get("health_status", "UNKNOWN"))
        strengths = list(health.get("strengths", []))
        weaknesses = list(health.get("weaknesses", []))
        risks = list(health.get("risks", []))
    else:
        health_score = None
        health_status = "UNKNOWN"
        strengths = []
        weaknesses = []
        risks = []

    exec_health = {
        "score": health_score,
        "status": health_status,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risks": risks,
    }

    # Key Findings (Priority-First Design: top 3–5)
    key_findings: List[Dict[str, Any]] = []
    if isinstance(prioritized_insights, dict):
        raw_insights = prioritized_insights.get("insights", [])
        if isinstance(raw_insights, list):
            for idx, ins in enumerate(raw_insights[:5], start=1):
                if not isinstance(ins, dict):
                    continue
                title = str(ins.get("title", f"Finding {idx}"))
                ev_basis = ins.get("evidence", {})
                fact_str = _format_fact(ev_basis, title)
                interp = str(ins.get("interpretation", ""))
                sev = str(ins.get("severity", "INFO"))
                rank = ins.get("priority_rank", idx)
                impact = str(ins.get("impact", ""))
                action = str(ins.get("recommended_action", ""))
                src = str(ins.get("source", "priority_engine"))

                key_findings.append({
                    "title": title,
                    "finding": title,
                    "fact": fact_str,
                    "interpretation": interp,
                    "severity": sev,
                    "priority_rank": rank,
                    "impact": impact,
                    "action": action,
                    "source": src,
                    "evidence_basis": ev_basis,
                })

    # Patterns
    patterns_list: List[str] = []
    if isinstance(story, dict):
        raw_patterns = story.get("patterns", [])
        if isinstance(raw_patterns, list):
            for p in raw_patterns:
                if isinstance(p, str):
                    patterns_list.append(p)
                elif isinstance(p, dict) and "description" in p:
                    patterns_list.append(str(p["description"]))

    # Anomalies
    anomalies_list: List[Dict[str, Any]] = []
    if isinstance(anomalies, dict):
        raw_anom_findings = anomalies.get("findings", [])
        if isinstance(raw_anom_findings, list):
            for af in raw_anom_findings[:5]:
                if not isinstance(af, dict):
                    continue
                cnt = af.get("anomaly_count", af.get("count", 0))
                pct = af.get("affected_percentage", af.get("percentage", 0.0))
                col = af.get("column", "")
                method = af.get("method", "IQR")
                sev = af.get("severity", "MEDIUM")
                why = af.get("why_unusual", af.get("explanation", ""))
                explanation = (
                    f"Potential anomalies detected in '{col}': {why}"
                    if why
                    else f"Potential anomalies detected in '{col}'."
                )

                anomalies_list.append({
                    "column": col,
                    "anomaly_count": cnt,
                    "count": cnt,
                    "affected_percentage": pct,
                    "percentage": pct,
                    "method": method,
                    "severity": sev,
                    "explanation": explanation,
                    "evidence": af.get("evidence", []),
                })
    elif isinstance(story, dict) and story.get("anomalies"):
        raw_story_anom = story.get("anomalies", [])
        if isinstance(raw_story_anom, list):
            for sa in raw_story_anom[:5]:
                if not isinstance(sa, dict):
                    continue
                cnt = sa.get("outlier_count", 0)
                pct = sa.get("outlier_percentage", 0.0)
                col = sa.get("column", "")
                method = sa.get("method", "IQR")

                anomalies_list.append({
                    "column": col,
                    "anomaly_count": cnt,
                    "count": cnt,
                    "affected_percentage": pct,
                    "percentage": pct,
                    "method": method,
                    "severity": "MEDIUM",
                    "explanation": f"Potential outliers identified in '{col}' ({pct:.1f}%).",
                    "evidence": [f"{cnt} outliers detected via {method}"],
                })

    # Changes / Reconsideration
    has_changes = False
    changes_summary_list: List[str] = []
    reconsiderations_list: List[Dict[str, Any]] = []

    if isinstance(reconsideration, dict):
        cs = reconsideration.get("change_summary", {})
        if isinstance(cs, dict):
            has_changes = bool(cs.get("has_changes", False))
            for chg_type in ["structural_changes", "quality_changes", "statistical_changes", "ml_changes"]:
                items = cs.get(chg_type, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            desc = item.get("description") or item.get("message") or str(item)
                            changes_summary_list.append(desc)
                        elif isinstance(item, str):
                            changes_summary_list.append(item)

            if not has_changes and not changes_summary_list:
                changes_summary_list.append("No significant structural, quality, or statistical changes detected.")
        elif reconsideration.get("summary"):
            changes_summary_list.append(str(reconsideration.get("summary")))

        raw_recons = reconsideration.get("reconsiderations", [])
        if isinstance(raw_recons, list):
            for r in raw_recons:
                if not isinstance(r, dict):
                    continue
                reconsiderations_list.append({
                    "finding_id": r.get("finding_id", ""),
                    "title": r.get("title", ""),
                    "classification": r.get("classification", "UNCHANGED"),
                    "reason": r.get("reason", ""),
                    "evidence": r.get("evidence", []),
                })
    else:
        changes_summary_list.append("No previous dataset was supplied for comparison.")

    changes_obj = {
        "detected": has_changes,
        "summary": changes_summary_list,
    }

    # Next Actions
    next_actions_list: List[Dict[str, Any]] = []
    raw_recs: List[Any] = []
    if isinstance(recommendations, dict):
        raw_recs = recommendations.get("recommendations", [])
    elif isinstance(recommendations, list):
        raw_recs = recommendations

    if isinstance(raw_recs, list):
        for r in raw_recs[:5]:
            if not isinstance(r, dict):
                continue
            next_actions_list.append({
                "name": r.get("name", ""),
                "why": r.get("why", ""),
                "evidence": r.get("evidence", []),
                "expected_value": r.get("expected_value", ""),
                "priority": r.get("priority", "MEDIUM"),
                "action": r.get("action", ""),
            })

    # Limitations
    limitations_list: List[str] = []
    if row_count == 0:
        limitations_list.append("Dataset contains 0 rows; statistical and modeling analysis cannot be performed.")
    elif row_count == 1:
        limitations_list.append("Dataset contains exactly 1 record; variance and distributions cannot be computed.")
    elif 1 < row_count < 30:
        limitations_list.append(
            f"Small sample size ({row_count} rows) introduces high sampling variance and limits statistical confidence."
        )

    # Completeness limitations
    completeness = evidence.get("completeness", {}) if isinstance(evidence.get("completeness"), dict) else {}
    col_missing_pcts = completeness.get("column_missing_percentages", {})
    col_missing_counts = completeness.get("column_missing_counts", {})
    if isinstance(col_missing_pcts, dict):
        for col_name, pct in col_missing_pcts.items():
            if pct > 0:
                cnt = col_missing_counts.get(col_name, 0)
                limitations_list.append(
                    f"Column '{col_name}' contains {pct:.1f}% missing values ({cnt} observation(s))."
                )

    # Duplicates limitations
    dup_section = evidence.get("duplicates", {}) if isinstance(evidence.get("duplicates"), dict) else {}
    dup_rows = dup_section.get("duplicate_rows", 0)
    if dup_rows > 0:
        dup_pct = dup_section.get("duplicate_row_percentage", 0.0)
        limitations_list.append(f"{dup_rows} duplicate record(s) ({dup_pct:.1f}%) detected.")

    # Constant columns limitations
    quality_section = evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}
    const_cols = quality_section.get("constant_columns", [])
    if isinstance(const_cols, list):
        for cc in const_cols:
            limitations_list.append(f"Column '{cc}' has zero variance (constant values).")

    # Add health risks
    for rsk in risks:
        if isinstance(rsk, str) and rsk not in limitations_list:
            limitations_list.append(rsk)

    # De-duplicate limitations preserving order
    seen_lims: Set[str] = set()
    dedup_limitations: List[str] = []
    for lim in limitations_list:
        if lim not in seen_lims:
            seen_lims.add(lim)
            dedup_limitations.append(lim)

    # 5. Build Structured Factual Evidence Package
    evidence_package = {
        "dataset": {
            "name": resolved_name,
            "rows": row_count,
            "columns": col_count,
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": dt_cols,
        },
        "health": exec_health,
        "top_findings": key_findings,
        "patterns": patterns_list,
        "anomalies": anomalies_list,
        "changes": changes_obj,
        "reconsideration": reconsiderations_list,
        "next_actions": next_actions_list,
        "limitations": dedup_limitations,
    }

    # 6. Build Deterministic Executive Summary
    deterministic_summary = _build_deterministic_summary(
        dataset_name=resolved_name,
        row_count=row_count,
        col_count=col_count,
        num_cols=numeric_cols,
        cat_cols=cat_cols,
        dt_cols=dt_cols,
        health_score=health_score,
        health_status=health_status,
        key_findings=key_findings,
        patterns=patterns_list,
        anomalies=anomalies_list,
        changes=changes_obj,
        reconsiderations=reconsiderations_list,
        next_actions=next_actions_list,
        limitations=dedup_limitations,
    )

    # 7. Optional Gemini Narrative Interpretation with Strict Fallback
    final_summary = deterministic_summary
    mode = "deterministic"
    gemini_used = False
    fallback = False

    if use_gemini:
        gemini_text, err = _call_gemini_interpretation(evidence_package)
        if gemini_text:
            final_summary = gemini_text
            mode = "gemini"
            gemini_used = True
            fallback = False
        else:
            fallback = True
            mode = "deterministic"
            gemini_used = False
            warnings.append(
                "Generative interpretation was unavailable; deterministic executive intelligence was used."
            )

    # Deduplicate generated_from
    dedup_generated_from: List[str] = []
    seen_gen: Set[str] = set()
    for g in generated_from:
        if g not in seen_gen:
            seen_gen.add(g)
            dedup_generated_from.append(g)

    # 8. Assemble and Sanitize Final Payload
    payload = {
        "dataset_name": resolved_name,
        "executive": {
            "summary": final_summary,
            "health": exec_health,
            "key_findings": key_findings,
            "patterns": patterns_list,
            "anomalies": anomalies_list,
            "changes": changes_obj,
            "reconsideration": reconsiderations_list,
            "next_actions": next_actions_list,
            "limitations": dedup_limitations,
        },
        "evidence_package": evidence_package,
        "generation": {
            "mode": mode,
            "gemini_used": gemini_used,
            "fallback": fallback,
        },
        "warnings": warnings,
        "generated_from": dedup_generated_from,
    }

    return _to_serializable(payload)
