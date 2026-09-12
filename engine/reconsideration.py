"""
FILE: engine/reconsideration.py

Advanced Reconsideration Engine.
Answers the foundational analytical question:
"After the dataset changed, do the previous conclusions still hold?"

Responsibilities:
1. Compare old and new datasets using change_engine and Evidence Layer factual outputs.
2. Detect structural, quality, statistical, and ML readiness changes.
3. Deterministically re-evaluate previous findings against new evidence.
4. Classify each previous conclusion into:
   - VALIDATED: Still supported by materially consistent evidence.
   - WEAKENED: Still has some support, but evidence is less strong.
   - STRENGTHENED: Evidence has become materially stronger / more severe.
   - INVALIDATED: Contradicted or rendered obsolete by new evidence.
   - UNCHANGED: Underlying evidence is identical and untouched.
   - REQUIRES_REVIEW: Ambiguous shift or insufficient evidence to confirm.
5. Identify new findings and resolved findings.
6. Recommend next analytical steps using engine/recommendation_engine.py.
7. Ensure 100% JSON serializability and strict input immutability.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from change_engine import (
    compare_datasets,
    compare_statistics,
    _name_similarity,
    _similar_values,
    _type_family,
    _type_conflict_level,
    _invalid_value_summary,
)
from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.recommendation_engine import recommend_next_analysis


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

    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()

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


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        if math.isnan(f) or np.isnan(f) or math.isinf(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _generate_finding_id(title: str, category: str, columns: List[str]) -> str:
    """Deterministically generate a stable, readable finding ID without randomness."""
    prefix = category.lower().replace(" ", "_") if category else "finding"
    clean_cols = "_".join(sorted(columns)) if columns else "global"
    seed = f"{title}_{category}_{clean_cols}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


# ============================================================
# COMPREHENSIVE CHANGE DETECTION
# ============================================================

def _detect_structural_and_quality_changes(
    old_evidence: Dict[str, Any],
    new_evidence: Dict[str, Any],
    old_df: Optional[pd.DataFrame] = None,
    new_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compare old and new evidence / dataframes to construct change_summary.
    Reuses change_engine.compare_datasets and compare_statistics when dfs are available.
    """
    old_struct = old_evidence.get("structure", {})
    new_struct = new_evidence.get("structure", {})

    old_rows = _safe_int(old_struct.get("row_count", 0))
    new_rows = _safe_int(new_struct.get("row_count", 0))

    old_cols = [str(c) for c in old_struct.get("column_names", [])]
    new_cols = [str(c) for c in new_struct.get("column_names", [])]

    old_cols_set = set(old_cols)
    new_cols_set = set(new_cols)

    added_cols = sorted(list(new_cols_set - old_cols_set))
    removed_cols = sorted(list(old_cols_set - new_cols_set))
    common_cols = sorted(list(old_cols_set.intersection(new_cols_set)))

    structural_changes: List[Dict[str, Any]] = []
    quality_changes: List[Dict[str, Any]] = []
    statistical_changes: List[Dict[str, Any]] = []
    ml_changes: List[Dict[str, Any]] = []

    # 1. Row count change
    if old_rows != new_rows:
        row_diff = new_rows - old_rows
        row_pct = round((row_diff / old_rows) * 100, 2) if old_rows > 0 else 0.0
        structural_changes.append({
            "type": "row_count_change",
            "old_rows": old_rows,
            "new_rows": new_rows,
            "difference": row_diff,
            "percentage_change": row_pct,
            "description": f"Row count changed from {old_rows:,} to {new_rows:,} ({row_diff:+,} rows, {row_pct:+.1f}%).",
        })

    # 2. Added and Removed Columns
    if added_cols:
        structural_changes.append({
            "type": "added_columns",
            "columns": added_cols,
            "count": len(added_cols),
            "description": f"Added {len(added_cols)} column(s): {added_cols}.",
        })

    if removed_cols:
        structural_changes.append({
            "type": "removed_columns",
            "columns": removed_cols,
            "count": len(removed_cols),
            "description": f"Removed {len(removed_cols)} column(s): {removed_cols}.",
        })

    # 3. Possible Column Renames (reusing change_engine._name_similarity)
    possible_renames: List[Dict[str, Any]] = []
    for r_col in removed_cols:
        for a_col in added_cols:
            sim = _name_similarity(r_col, a_col)
            # If dataframes available, also check value similarity
            val_sim = 0.0
            if old_df is not None and new_df is not None and r_col in old_df.columns and a_col in new_df.columns:
                val_sim = _similar_values(old_df[r_col], new_df[a_col])
            conf = round((sim * 0.6 + val_sim * 0.4), 1) if val_sim > 0 else sim
            if conf >= 55.0:
                possible_renames.append({
                    "old_column": r_col,
                    "new_column": a_col,
                    "confidence": conf,
                    "description": f"Probable rename: '{r_col}' -> '{a_col}' (confidence: {conf}%).",
                })
    if possible_renames:
        structural_changes.append({
            "type": "possible_renames",
            "renames": possible_renames,
            "count": len(possible_renames),
            "description": f"Detected {len(possible_renames)} probable column rename(s).",
        })

    # 4. Dtype changes on common columns
    old_types = old_struct.get("data_types", {})
    new_types = new_struct.get("data_types", {})
    dtype_changes: List[Dict[str, Any]] = []
    for col in common_cols:
        o_t = str(old_types.get(col, ""))
        n_t = str(new_types.get(col, ""))
        if o_t and n_t and o_t != n_t:
            dtype_changes.append({
                "column": col,
                "old_type": o_t,
                "new_type": n_t,
                "description": f"Column '{col}' data type changed from '{o_t}' to '{n_t}'.",
            })
    if dtype_changes:
        structural_changes.append({
            "type": "dtype_changes",
            "changes": dtype_changes,
            "count": len(dtype_changes),
            "description": f"Detected {len(dtype_changes)} data type change(s).",
        })

    # 5. Missingness changes
    old_comp = old_evidence.get("completeness", {})
    new_comp = new_evidence.get("completeness", {})
    old_col_miss_pct = old_comp.get("column_missing_percentages", {})
    new_col_miss_pct = new_comp.get("column_missing_percentages", {})

    old_tot_miss_pct = _safe_float(old_comp.get("missing_percentage", 0.0))
    new_tot_miss_pct = _safe_float(new_comp.get("missing_percentage", 0.0))

    if abs(new_tot_miss_pct - old_tot_miss_pct) >= 0.1:
        quality_changes.append({
            "type": "total_missingness_change",
            "old_percentage": old_tot_miss_pct,
            "new_percentage": new_tot_miss_pct,
            "difference": round(new_tot_miss_pct - old_tot_miss_pct, 2),
            "description": f"Overall missingness shifted from {old_tot_miss_pct:.1f}% to {new_tot_miss_pct:.1f}%.",
        })

    for col in common_cols:
        o_p = _safe_float(old_col_miss_pct.get(col, 0.0))
        n_p = _safe_float(new_col_miss_pct.get(col, 0.0))
        diff = round(n_p - o_p, 2)
        if abs(diff) >= 1.0 or (o_p > 0 and n_p == 0) or (o_p < 100.0 and n_p == 100.0):
            quality_changes.append({
                "type": "column_missingness_change",
                "column": col,
                "old_percentage": o_p,
                "new_percentage": n_p,
                "difference": diff,
                "became_complete": bool(o_p > 0 and n_p == 0),
                "became_all_missing": bool(o_p < 100.0 and n_p == 100.0),
                "description": f"Column '{col}' missingness changed from {o_p:.1f}% to {n_p:.1f}% ({diff:+.1f}%).",
            })

    # 6. Duplicate Row Changes
    old_dup = old_evidence.get("duplicates", {})
    new_dup = new_evidence.get("duplicates", {})
    o_dup_cnt = _safe_int(old_dup.get("duplicate_rows", 0))
    n_dup_cnt = _safe_int(new_dup.get("duplicate_rows", 0))
    if o_dup_cnt != n_dup_cnt:
        quality_changes.append({
            "type": "duplicate_rows_change",
            "old_duplicates": o_dup_cnt,
            "new_duplicates": n_dup_cnt,
            "difference": n_dup_cnt - o_dup_cnt,
            "description": f"Duplicate rows changed from {o_dup_cnt:,} to {n_dup_cnt:,} ({n_dup_cnt - o_dup_cnt:+,}).",
        })

    # 7. Constant & All-Missing Column Transitions
    old_quality = old_evidence.get("quality", {})
    new_quality = new_evidence.get("quality", {})
    old_const = set(old_quality.get("constant_columns", []))
    new_const = set(new_quality.get("constant_columns", []))

    newly_constant = sorted(list(new_const - old_const))
    no_longer_constant = sorted(list(old_const - new_const))

    if newly_constant:
        quality_changes.append({
            "type": "newly_constant_columns",
            "columns": newly_constant,
            "description": f"Column(s) became constant (zero variance): {newly_constant}.",
        })
    if no_longer_constant:
        quality_changes.append({
            "type": "no_longer_constant_columns",
            "columns": no_longer_constant,
            "description": f"Column(s) gained variance and are no longer constant: {no_longer_constant}.",
        })

    # 8. Outliers / Anomalies Changes
    old_outliers = {o.get("attribute", o.get("column", "")): o for o in old_quality.get("outlier_findings", [])}
    new_outliers = {o.get("attribute", o.get("column", "")): o for o in new_quality.get("outlier_findings", [])}

    for col in common_cols:
        o_o = old_outliers.get(col)
        n_o = new_outliers.get(col)
        o_cnt = _safe_int(o_o.get("count", 0)) if o_o else 0
        n_cnt = _safe_int(n_o.get("count", 0)) if n_o else 0
        if o_cnt != n_cnt:
            quality_changes.append({
                "type": "outlier_count_change",
                "column": col,
                "old_outliers": o_cnt,
                "new_outliers": n_cnt,
                "difference": n_cnt - o_cnt,
                "description": f"Outlier count for '{col}' changed from {o_cnt:,} to {n_cnt:,}.",
            })

    # 9. Statistical Drift in Numerical Columns
    old_num_stats = old_evidence.get("numerical_statistics", {})
    new_num_stats = new_evidence.get("numerical_statistics", {})

    for col in common_cols:
        if col in old_num_stats and col in new_num_stats:
            o_s = old_num_stats[col]
            n_s = new_num_stats[col]

            o_mean = o_s.get("mean")
            n_mean = n_s.get("mean")
            o_std = o_s.get("std")
            n_std = n_s.get("std")

            if o_mean is not None and n_mean is not None:
                mean_diff = round(n_mean - o_mean, 4)
                # Significant mean shift if > 10% relative or > 0.5 standard deviation
                rel_shift = abs(mean_diff) / (abs(o_mean) + 1e-6)
                if rel_shift >= 0.15 or (o_std and o_std > 0 and abs(mean_diff) >= 0.5 * o_std):
                    statistical_changes.append({
                        "column": col,
                        "metric": "mean",
                        "old_value": o_mean,
                        "new_value": n_mean,
                        "difference": mean_diff,
                        "description": f"Statistical drift in '{col}': mean shifted from {o_mean} to {n_mean} ({mean_diff:+}).",
                    })

    # 10. ML Readiness & Target Changes
    old_ml = old_evidence.get("ml_readiness", {})
    new_ml = new_evidence.get("ml_readiness", {})

    old_ml_score = _safe_int(old_ml.get("ml_score", 0))
    new_ml_score = _safe_int(new_ml.get("ml_score", 0))

    old_target = old_ml.get("evaluated_target") or old_ml.get("best_target_candidate")
    new_target = new_ml.get("evaluated_target") or new_ml.get("best_target_candidate")

    if old_ml_score != new_ml_score:
        ml_changes.append({
            "type": "ml_score_change",
            "old_score": old_ml_score,
            "new_score": new_ml_score,
            "difference": new_ml_score - old_ml_score,
            "description": f"ML readiness score changed from {old_ml_score} to {new_ml_score} ({new_ml_score - old_ml_score:+}).",
        })

    if old_target != new_target:
        ml_changes.append({
            "type": "target_change",
            "old_target": old_target,
            "new_target": new_target,
            "description": f"Primary target candidate shifted from '{old_target}' to '{new_target}'.",
        })

    # Total changes count
    total_changes = (
        len(structural_changes)
        + len(quality_changes)
        + len(statistical_changes)
        + len(ml_changes)
    )

    return {
        "has_changes": bool(total_changes > 0),
        "change_count": total_changes,
        "structural_changes": structural_changes,
        "quality_changes": quality_changes,
        "statistical_changes": statistical_changes,
        "ml_changes": ml_changes,
    }


# ============================================================
# RECONSIDERATION ENGINE CORE
# ============================================================

def _reconsider_single_finding(
    finding: Dict[str, Any],
    old_evidence: Dict[str, Any],
    new_evidence: Dict[str, Any],
    changes: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministically re-evaluate a previous finding against the updated evidence.
    Classifications: VALIDATED, WEAKENED, STRENGTHENED, INVALIDATED, UNCHANGED, REQUIRES_REVIEW.
    """
    title = str(finding.get("title", "Untitled Finding"))
    category = str(finding.get("category", "")).lower()
    prev_status = str(finding.get("severity", finding.get("priority", "ACTIVE")))
    aff_cols = [str(c) for c in finding.get("affected_columns", [])]
    orig_evidence = finding.get("evidence", [])
    if isinstance(orig_evidence, str):
        orig_evidence = [orig_evidence]

    f_id = str(finding.get("finding_id", ""))
    if not f_id:
        f_id = _generate_finding_id(title, category, aff_cols)

    title_lower = title.lower()

    # Evidence shortcuts
    old_struct = old_evidence.get("structure", {})
    new_struct = new_evidence.get("structure", {})
    old_cols = set(old_struct.get("column_names", []))
    new_cols = set(new_struct.get("column_names", []))

    removed_cols = old_cols - new_cols
    old_comp = old_evidence.get("completeness", {})
    new_comp = new_evidence.get("completeness", {})
    old_col_miss = old_comp.get("column_missing_percentages", {})
    new_col_miss = new_comp.get("column_missing_percentages", {})

    old_dup = _safe_int(old_evidence.get("duplicates", {}).get("duplicate_rows", 0))
    new_dup = _safe_int(new_evidence.get("duplicates", {}).get("duplicate_rows", 0))

    old_quality = old_evidence.get("quality", {})
    new_quality = new_evidence.get("quality", {})

    # Default fallback
    classification = "UNCHANGED"
    reason = "No relevant changes observed in the underlying evidence for this finding."
    curr_status = "PERSISTENT"
    current_evidence = list(orig_evidence)
    impact = str(finding.get("impact", "Previous impact remains unchanged."))
    action = str(finding.get("action", finding.get("recommended_action", "Continue standard monitoring.")))

    # ------------------------------------------------------------
    # 1. Check if affected columns were removed
    # ------------------------------------------------------------
    if aff_cols and any(c in removed_cols for c in aff_cols):
        missing_c = [c for c in aff_cols if c in removed_cols]
        classification = "INVALIDATED"
        curr_status = "RESOLVED"
        reason = f"Supporting column(s) {missing_c} were removed in the updated dataset; this finding is no longer applicable."
        current_evidence = [f"Column(s) {missing_c} are absent from updated dataset."]
        impact = "Issue has been retired as the affected data is no longer present."
        action = "No further remediation required on the removed column(s)."
        return {
            "finding_id": f_id,
            "title": title,
            "previous_status": prev_status,
            "current_status": curr_status,
            "classification": classification,
            "reason": reason,
            "evidence": current_evidence,
            "impact": impact,
            "recommended_action": action,
        }

    # ------------------------------------------------------------
    # 2. Missingness / Completeness Findings
    # ------------------------------------------------------------
    if "missing" in title_lower or category in ("completeness", "missing_data"):
        if aff_cols:
            col = aff_cols[0]
            o_m = _safe_float(old_col_miss.get(col, 0.0))
            n_m = _safe_float(new_col_miss.get(col, 0.0))

            if n_m == 0.0 and o_m > 0.0:
                classification = "INVALIDATED"
                curr_status = "RESOLVED"
                reason = f"Missing values in '{col}' were completely eliminated (dropped from {o_m:.1f}% to 0.0%)."
                current_evidence = [f"Missing percentage dropped from {o_m:.1f}% to 0.0%."]
                impact = f"Data completeness for '{col}' is fully restored."
                action = "Verify imputation or data ingestion pipeline integrity."
            elif n_m < o_m - 3.0:
                classification = "WEAKENED"
                curr_status = "ATTENUATED"
                reason = f"Missingness in '{col}' decreased significantly from {o_m:.1f}% to {n_m:.1f}%."
                current_evidence = [f"Missing percentage fell from {o_m:.1f}% to {n_m:.1f}%."]
                impact = "Reduced severity of missingness allows for greater usable sample size."
                action = "Assess whether remaining missing values require further imputation."
            elif n_m > o_m + 3.0:
                classification = "STRENGTHENED"
                curr_status = "ESCALATED"
                reason = f"Missingness in '{col}' increased from {o_m:.1f}% to {n_m:.1f}%, escalating data quality risk."
                current_evidence = [f"Missing percentage rose from {o_m:.1f}% to {n_m:.1f}%."]
                impact = "Elevated data loss threatens statistical validity and model training viability."
                action = "Investigate upstream data collection source for failures."
            elif abs(n_m - o_m) <= 3.0 and o_m > 0:
                classification = "VALIDATED"
                curr_status = "PERSISTENT"
                reason = f"Missingness in '{col}' remains materially consistent at {n_m:.1f}% (previously {o_m:.1f}%)."
                current_evidence = [f"Current missingness is {n_m:.1f}% vs baseline {o_m:.1f}%."]
            else:
                classification = "UNCHANGED"
        else:
            # Dataset-wide missingness
            o_tot = _safe_float(old_comp.get("missing_percentage", 0.0))
            n_tot = _safe_float(new_comp.get("missing_percentage", 0.0))
            if n_tot == 0.0 and o_tot > 0:
                classification = "INVALIDATED"
                curr_status = "RESOLVED"
                reason = f"Dataset-wide missing cells completely resolved (from {o_tot:.1f}% to 0.0%)."
            elif n_tot < o_tot - 2.0:
                classification = "WEAKENED"
                curr_status = "ATTENUATED"
                reason = f"Overall dataset missingness fell from {o_tot:.1f}% to {n_tot:.1f}%."
            elif n_tot > o_tot + 2.0:
                classification = "STRENGTHENED"
                curr_status = "ESCALATED"
                reason = f"Overall dataset missingness rose from {o_tot:.1f}% to {n_tot:.1f}%."
            else:
                classification = "VALIDATED" if o_tot > 0 else "UNCHANGED"

    # ------------------------------------------------------------
    # 3. Duplicate Rows Findings
    # ------------------------------------------------------------
    elif "duplicate" in title_lower or category == "uniqueness":
        if old_dup > 0 and new_dup == 0:
            classification = "INVALIDATED"
            curr_status = "RESOLVED"
            reason = f"Duplicate rows were completely eliminated (dropped from {old_dup:,} to 0)."
            current_evidence = [f"Zero duplicate rows detected in updated dataset (was {old_dup:,})."]
            impact = "Row duplication bias has been resolved."
            action = "Confirm deduplication logic in data preparation pipeline."
        elif new_dup > old_dup:
            classification = "STRENGTHENED"
            curr_status = "ESCALATED"
            reason = f"Duplicate rows increased from {old_dup:,} to {new_dup:,} (+{new_dup - old_dup:,} rows)."
            current_evidence = [f"Duplicates increased to {new_dup:,} rows."]
            impact = "Increased risk of inflated metrics, false variance, and train/test data leakage."
            action = "Apply explicit primary key deduplication before modeling."
        elif new_dup < old_dup and new_dup > 0:
            classification = "WEAKENED"
            curr_status = "ATTENUATED"
            reason = f"Duplicate rows decreased from {old_dup:,} to {new_dup:,} (-{old_dup - new_dup:,} rows)."
            current_evidence = [f"Duplicates fell to {new_dup:,} rows."]
        elif new_dup == old_dup and old_dup > 0:
            classification = "VALIDATED"
            curr_status = "PERSISTENT"
            reason = f"Duplicate rows remain identical at {new_dup:,} rows."

    # ------------------------------------------------------------
    # 4. Constant Column Findings
    # ------------------------------------------------------------
    elif "constant" in title_lower:
        if aff_cols:
            col = aff_cols[0]
            was_const = col in old_quality.get("constant_columns", [])
            is_const = col in new_quality.get("constant_columns", [])
            if was_const and not is_const:
                classification = "INVALIDATED"
                curr_status = "RESOLVED"
                reason = f"Column '{col}' gained variance and is no longer constant."
                current_evidence = ["Column now exhibits multiple distinct values."]
                impact = f"'{col}' is now usable as a predictive or analytical feature."
                action = "Re-include column in distribution profiling and feature set."
            elif is_const:
                classification = "VALIDATED"
                curr_status = "PERSISTENT"
                reason = f"Column '{col}' remains constant with zero variance."

    # ------------------------------------------------------------
    # 5. Outlier / Anomaly Findings
    # ------------------------------------------------------------
    elif "outlier" in title_lower or "anomaly" in title_lower:
        if aff_cols:
            col = aff_cols[0]
            o_out = [o for o in old_quality.get("outlier_findings", []) if o.get("attribute") == col or o.get("column") == col]
            n_out = [o for o in new_quality.get("outlier_findings", []) if o.get("attribute") == col or o.get("column") == col]
            o_c = _safe_int(o_out[0].get("count", 0)) if o_out else 0
            n_c = _safe_int(n_out[0].get("count", 0)) if n_out else 0

            if o_c > 0 and n_c == 0:
                classification = "INVALIDATED"
                curr_status = "RESOLVED"
                reason = f"Outliers in '{col}' were completely removed (dropped from {o_c} to 0)."
                current_evidence = [f"Outlier count dropped from {o_c} to 0."]
                impact = "Extreme value distortion has been eliminated."
                action = "Proceed with parametric modeling without outlier clipping."
            elif n_c > o_c:
                classification = "STRENGTHENED"
                curr_status = "ESCALATED"
                reason = f"Outliers in '{col}' increased from {o_c} to {n_c}."
                current_evidence = [f"Outliers increased to {n_c} values."]
            elif n_c < o_c:
                classification = "WEAKENED"
                curr_status = "ATTENUATED"
                reason = f"Outliers in '{col}' decreased from {o_c} to {n_c}."
                current_evidence = [f"Outliers fell to {n_c} values."]
            elif o_c > 0 and n_c == o_c:
                classification = "VALIDATED"
                curr_status = "PERSISTENT"
                reason = f"Outliers in '{col}' remain unchanged at {n_c} values."

    # ------------------------------------------------------------
    # 6. Correlation Findings
    # ------------------------------------------------------------
    elif "correlation" in title_lower:
        if len(aff_cols) >= 2:
            c1, c2 = aff_cols[0], aff_cols[1]
            o_pairs = {f"{p.get('attribute_1')}_{p.get('attribute_2')}": p for p in old_evidence.get("correlations", {}).get("strongest_pairs", [])}
            n_pairs = {f"{p.get('attribute_1')}_{p.get('attribute_2')}": p for p in new_evidence.get("correlations", {}).get("strongest_pairs", [])}

            k1 = f"{c1}_{c2}"
            k2 = f"{c2}_{c1}"
            o_p = o_pairs.get(k1) or o_pairs.get(k2)
            n_p = n_pairs.get(k1) or n_pairs.get(k2)

            o_val = abs(_safe_float(o_p.get("correlation", 0.0))) if o_p else 0.0
            n_val = abs(_safe_float(n_p.get("correlation", 0.0))) if n_p else 0.0

            if o_val >= 0.5 and n_val < 0.3:
                classification = "INVALIDATED"
                curr_status = "RESOLVED"
                reason = f"Correlation between '{c1}' and '{c2}' dropped from {o_val:.2f} to {n_val:.2f} below meaningful significance."
            elif n_val > o_val + 0.15:
                classification = "STRENGTHENED"
                curr_status = "ESCALATED"
                reason = f"Correlation between '{c1}' and '{c2}' strengthened from {o_val:.2f} to {n_val:.2f}."
            elif n_val < o_val - 0.15:
                classification = "WEAKENED"
                curr_status = "ATTENUATED"
                reason = f"Correlation between '{c1}' and '{c2}' weakened from {o_val:.2f} to {n_val:.2f}."
            elif o_val > 0.4:
                classification = "VALIDATED"
                reason = f"Correlation between '{c1}' and '{c2}' remains steady at {n_val:.2f}."

    # ------------------------------------------------------------
    # 7. ML Readiness / Target Findings
    # ------------------------------------------------------------
    elif "target" in title_lower or "ml readiness" in title_lower:
        old_ml = old_evidence.get("ml_readiness", {})
        new_ml = new_evidence.get("ml_readiness", {})
        o_score = _safe_int(old_ml.get("ml_score", 0))
        n_score = _safe_int(new_ml.get("ml_score", 0))

        if n_score >= o_score + 20:
            classification = "INVALIDATED" if "not ready" in title_lower else "STRENGTHENED"
            curr_status = "RESOLVED" if "not ready" in title_lower else "ESCALATED"
            reason = f"ML readiness significantly improved from {o_score} to {n_score}."
        elif n_score <= o_score - 20:
            classification = "STRENGTHENED" if "not ready" in title_lower else "WEAKENED"
            curr_status = "ESCALATED"
            reason = f"ML readiness score dropped from {o_score} to {n_score}."
        else:
            classification = "VALIDATED"
            reason = f"ML readiness remains consistent ({n_score} vs previous {o_score})."

    # ------------------------------------------------------------
    # 8. Check for Ambiguous / Requires Review condition
    # ------------------------------------------------------------
    if classification == "UNCHANGED":
        # Check if affected columns experienced type changes
        for c in aff_cols:
            o_t = old_struct.get("data_types", {}).get(c)
            n_t = new_struct.get("data_types", {}).get(c)
            if o_t and n_t and o_t != n_t:
                classification = "REQUIRES_REVIEW"
                curr_status = "UNKNOWN"
                reason = f"Column '{c}' underwent a data type change from '{o_t}' to '{n_t}', requiring manual review."
                break

    return {
        "finding_id": f_id,
        "title": title,
        "previous_status": prev_status,
        "current_status": curr_status,
        "classification": classification,
        "reason": reason,
        "evidence": current_evidence,
        "impact": impact,
        "recommended_action": action,
    }


# ============================================================
# NEW FINDINGS DETECTION
# ============================================================

def _detect_new_findings(
    old_evidence: Dict[str, Any],
    new_evidence: Dict[str, Any],
    previous_findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Identify meaningful findings that exist in the updated dataset
    which were NOT present in the previous dataset.
    """
    new_findings: List[Dict[str, Any]] = []
    prev_titles = {str(f.get("title", "")).lower() for f in previous_findings}
    prev_cols_affected = set()
    for f in previous_findings:
        for c in f.get("affected_columns", []):
            prev_cols_affected.add(c)

    curr_insights = build_prioritized_insights(new_evidence).get("insights", [])

    for ins in curr_insights:
        t_low = str(ins.get("title", "")).lower()
        cols = ins.get("affected_columns", [])

        # If title matches an existing finding exactly, it's not new
        if t_low in prev_titles:
            continue

        # Check if this represents a new column issue
        is_new = False
        if any(c not in old_evidence.get("structure", {}).get("column_names", []) for c in cols):
            is_new = True
        elif ins.get("severity") in ("CRITICAL", "HIGH") and t_low not in prev_titles:
            is_new = True

        if is_new:
            new_findings.append({
                "finding_id": ins.get("finding_id", _generate_finding_id(ins.get("title", ""), ins.get("category", ""), cols)),
                "title": ins.get("title"),
                "category": ins.get("category"),
                "severity": ins.get("severity"),
                "affected_columns": cols,
                "evidence": ins.get("evidence", []),
                "impact": ins.get("impact", ""),
                "action": ins.get("action", ""),
            })

    return new_findings


# ============================================================
# ADAPTIVE NEXT-ANALYSIS RECOMMENDATIONS
# ============================================================

def _generate_adaptive_next_analysis(
    new_evidence: Dict[str, Any],
    changes: Dict[str, Any],
    reconsiderations: List[Dict[str, Any]],
    new_findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate targeted recommendations for the adaptive agent loop.
    Reuses recommendations from recommendation_engine and adds change-specific actions.
    """
    next_actions: List[Dict[str, Any]] = []

    # 1. Structural schema change recommendation
    if changes.get("structural_changes"):
        added = [c for sc in changes["structural_changes"] if sc.get("type") == "added_columns" for c in sc.get("columns", [])]
        removed = [c for sc in changes["structural_changes"] if sc.get("type") == "removed_columns" for c in sc.get("columns", [])]
        renames = [r for sc in changes["structural_changes"] if sc.get("type") == "possible_renames" for r in sc.get("renames", [])]

        why_parts = []
        if added:
            why_parts.append(f"{len(added)} added column(s)")
        if removed:
            why_parts.append(f"{len(removed)} removed column(s)")
        if renames:
            why_parts.append(f"{len(renames)} renamed column(s)")

        next_actions.append({
            "name": "Schema Ingestion & Pipeline Validation",
            "why": f"Dataset structure altered with {', '.join(why_parts)}.",
            "evidence": [sc.get("description", "") for sc in changes["structural_changes"][:3]],
            "priority": "HIGH",
            "action": "Update feature store schemas and downstream transformation pipelines to align with altered columns.",
        })

    # 2. Statistical drift recommendation
    if changes.get("statistical_changes"):
        drift_cols = [sc["column"] for sc in changes["statistical_changes"]]
        next_actions.append({
            "name": "Distribution Drift & Covariate Shift Analysis",
            "why": f"Significant statistical drift detected across {len(drift_cols)} numerical column(s).",
            "evidence": [sc.get("description", "") for sc in changes["statistical_changes"][:3]],
            "priority": "HIGH",
            "action": "Perform two-sample Kolmogorov-Smirnov or Population Stability Index (PSI) tests on drifted features.",
        })

    # 3. New severe findings recommendation
    if new_findings:
        crit_high = [nf for nf in new_findings if nf.get("severity") in ("CRITICAL", "HIGH")]
        if crit_high:
            next_actions.append({
                "name": "Investigate Newly Emerged Data Quality Issues",
                "why": f"Detected {len(crit_high)} new high-severity issue(s) in the updated dataset.",
                "evidence": [f"{nf['title']}: {nf.get('affected_columns')}" for nf in crit_high[:2]],
                "priority": "CRITICAL",
                "action": "Address newly introduced data quality defects before proceeding to model training or reporting.",
            })

    # 4. Integrate base recommendations from recommendation_engine
    base_rec_result = recommend_next_analysis(new_evidence, max_recommendations=3)
    base_recs = base_rec_result.get("recommendations", [])
    for br in base_recs:
        # Avoid duplicate recommendation names
        if not any(na["name"] == br.get("name") for na in next_actions):
            next_actions.append({
                "name": br.get("name"),
                "why": br.get("why"),
                "evidence": br.get("evidence", []),
                "priority": br.get("priority", "MEDIUM"),
                "action": br.get("action"),
            })

    return next_actions[:5]


# ============================================================
# MAIN PUBLIC API
# ============================================================

def reconsider_dataset(
    old_evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    new_evidence_or_df: Union[Dict[str, Any], pd.DataFrame, None],
    previous_findings: Optional[List[Dict[str, Any]]] = None,
    old_health: Optional[Dict[str, Any]] = None,
    new_health: Optional[Dict[str, Any]] = None,
    old_story: Optional[Dict[str, Any]] = None,
    new_story: Optional[Dict[str, Any]] = None,
    old_priorities: Optional[Dict[str, Any]] = None,
    new_priorities: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Advanced Reconsideration Engine Public API.

    Compares old and new datasets, detects structural/statistical/quality shifts,
    re-evaluates previous findings, classifies each previous conclusion, and
    recommends adaptive next analysis.

    Parameters:
    -----------
    old_evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Previous Evidence dict or DataFrame.
    new_evidence_or_df : Union[Dict[str, Any], pd.DataFrame, None]
        Updated Evidence dict or DataFrame.
    previous_findings : Optional[List[Dict[str, Any]]]
        List of previously established conclusions/findings. If None, derived from old evidence.
    old_health, new_health : Optional health dictionaries.
    old_story, new_story : Optional data story dictionaries.
    old_priorities, new_priorities : Optional prioritized insight dictionaries.
    dataset_name : Optional dataset label.

    Returns:
    --------
    Dict[str, Any]
        Standardized, JSON-serializable reconsideration payload.
    """
    # 1. Resolve Inputs and Sources
    old_df: Optional[pd.DataFrame] = None
    new_df: Optional[pd.DataFrame] = None

    if isinstance(old_evidence_or_df, dict) and "structure" in old_evidence_or_df:
        old_evidence = old_evidence_or_df
        old_source = "evidence"
    elif isinstance(old_evidence_or_df, pd.DataFrame):
        old_df = old_evidence_or_df
        old_evidence = build_evidence(old_df, dataset_name=dataset_name)
        old_source = "dataframe"
    else:
        old_evidence = build_evidence(None, dataset_name=dataset_name)
        old_source = "none"

    if isinstance(new_evidence_or_df, dict) and "structure" in new_evidence_or_df:
        new_evidence = new_evidence_or_df
        new_source = "evidence"
    elif isinstance(new_evidence_or_df, pd.DataFrame):
        new_df = new_evidence_or_df
        new_evidence = build_evidence(new_df, dataset_name=dataset_name)
        new_source = "dataframe"
    else:
        new_evidence = build_evidence(None, dataset_name=dataset_name)
        new_source = "none"

    resolved_name = (
        dataset_name
        or new_evidence.get("metadata", {}).get("dataset_name")
        or old_evidence.get("metadata", {}).get("dataset_name")
        or "reconsidered_dataset"
    )

    generated_from = "dataframe" if (old_source == "dataframe" and new_source == "dataframe") else "evidence"
    if old_source == "none" and new_source == "none":
        generated_from = "none"

    # 2. Extract Previous Findings
    if previous_findings is not None:
        raw_prev = list(previous_findings)
    elif old_priorities and "insights" in old_priorities:
        raw_prev = list(old_priorities["insights"])
    else:
        old_row_cnt = old_evidence.get("structure", {}).get("row_count", 0)
        old_col_cnt = old_evidence.get("structure", {}).get("column_count", 0)
        if old_row_cnt > 0 or old_col_cnt > 0:
            raw_prev = build_prioritized_insights(old_evidence).get("insights", [])
        else:
            raw_prev = []

    # 3. Detect Changes
    changes = _detect_structural_and_quality_changes(
        old_evidence=old_evidence,
        new_evidence=new_evidence,
        old_df=old_df,
        new_df=new_df,
    )

    # 4. Reconsider Each Previous Finding
    reconsiderations: List[Dict[str, Any]] = []
    for f in raw_prev:
        recon = _reconsider_single_finding(f, old_evidence, new_evidence, changes)
        reconsiderations.append(recon)

    # Deterministic sort for reconsiderations by finding_id
    reconsiderations.sort(key=lambda x: str(x.get("finding_id", "")))

    # 5. Segment Resolved and Requires Review Findings
    resolved_findings: List[Dict[str, Any]] = []
    requires_review: List[Dict[str, Any]] = []

    for r in reconsiderations:
        if r["classification"] == "INVALIDATED" and r["current_status"] == "RESOLVED":
            resolved_findings.append({
                "finding_id": r["finding_id"],
                "title": r["title"],
                "reason": r["reason"],
                "evidence": r["evidence"],
            })
        elif r["classification"] == "REQUIRES_REVIEW":
            requires_review.append({
                "finding_id": r["finding_id"],
                "title": r["title"],
                "reason": r["reason"],
                "evidence": r["evidence"],
            })

    # 6. Detect New Findings
    new_findings = _detect_new_findings(old_evidence, new_evidence, raw_prev)
    new_findings.sort(key=lambda x: str(x.get("finding_id", "")))

    # 7. Adaptive Next Analysis Recommendations
    next_analysis = _generate_adaptive_next_analysis(
        new_evidence=new_evidence,
        changes=changes,
        reconsiderations=reconsiderations,
        new_findings=new_findings,
    )

    # 8. Summary Counts
    summary = {
        "validated": sum(1 for r in reconsiderations if r["classification"] == "VALIDATED"),
        "weakened": sum(1 for r in reconsiderations if r["classification"] == "WEAKENED"),
        "strengthened": sum(1 for r in reconsiderations if r["classification"] == "STRENGTHENED"),
        "invalidated": sum(1 for r in reconsiderations if r["classification"] == "INVALIDATED"),
        "unchanged": sum(1 for r in reconsiderations if r["classification"] == "UNCHANGED"),
        "requires_review": sum(1 for r in reconsiderations if r["classification"] == "REQUIRES_REVIEW"),
    }

    output = {
        "dataset_name": resolved_name,
        "change_summary": changes,
        "reconsiderations": reconsiderations,
        "new_findings": new_findings,
        "resolved_findings": resolved_findings,
        "requires_review": requires_review,
        "next_analysis": next_analysis,
        "summary": summary,
        "generated_from": generated_from,
    }

    return _to_serializable(output)
