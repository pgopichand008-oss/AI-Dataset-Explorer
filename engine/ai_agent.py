"""
Dataset Intelligence Agent

Coordinates dataset comparison, quality analysis, statistical analysis,
impact assessment, ML-readiness assessment, previous-finding reconsideration,
and final recommendation generation.
"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

READY = "READY FOR FURTHER ANALYSIS"
ATTENTION = "NEEDS ATTENTION BEFORE ANALYSIS"
UNSUITABLE = "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

RECONSIDER = "RECONSIDER"
STILL_VALID = "STILL_VALID"
NO_PREVIOUS_FINDING = "NO_PREVIOUS_FINDING"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _contains_column(text, column):
    """Return True when a column name is mentioned in a finding."""
    if not text or not column:
        return False

    return column.lower() in text.lower()


def _is_identifier(column):
    """
    Determine whether a column looks like an identifier.

    Identifier-like columns are excluded from statistical impact checks
    because percentage changes in IDs are usually not meaningful.
    """
    name = str(column).strip().lower()

    return (
        name == "id"
        or name.endswith("_id")
        or name.endswith("id")
    )


def _safe_percentage_change(old_value, change):
    """Calculate percentage change safely."""
    if old_value in (None, 0):
        return None

    try:
        return abs(float(change) / float(old_value) * 100)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _type_changes_by_severity(changes, severity):
    """Return type changes matching a requested conflict severity."""
    return [
        change
        for change in changes.get("type_changes", [])
        if change.get("conflict_level") == severity
    ]


def _invalid_value_entries(changes):
    """Return columns containing invalid values."""
    invalid_values = changes.get("invalid_values", {})

    return [
        (column, summary)
        for column, summary in invalid_values.items()
        if summary.get("invalid_count", 0) > 0
    ]


def _increased_missing_values(quality):
    """Return columns where missing-value rate increased."""
    return [
        item
        for item in quality.get("missing_changes", [])
        if item.get("change", 0) > 0
    ]


def _significant_missing_values(quality, threshold=20):
    """Return columns with missing-value increases at or above a threshold."""
    return [
        item
        for item in quality.get("missing_changes", [])
        if item.get("change", 0) >= threshold
    ]


def _statistical_shifts(statistics, threshold=10):
    """
    Return meaningful statistical shifts.

    Mean percentage change is used as the primary signal while the
    underlying statistics still contain mean, median, std, min and max.
    """
    shifts = []

    for stat in statistics or []:
        column = stat.get("column")

        if _is_identifier(column):
            continue

        percent_change = _safe_percentage_change(
            stat.get("old_mean"),
            stat.get("mean_change")
        )

        if percent_change is not None and percent_change >= threshold:
            shifts.append({
                "column": column,
                "percentage_change": percent_change,
                "statistics": stat
            })

    return shifts


# ---------------------------------------------------------------------------
# Previous finding reconsideration
# ---------------------------------------------------------------------------

def reconsider_previous_finding(finding, changes, quality):
    """
    Check whether a previous analysis finding remains valid.

    A finding is reconsidered when it refers to a column affected by:
    - removal
    - type changes
    - invalid values
    - increased missing values
    """
    finding_text = str(finding or "").lower()
    warnings = []

    if not finding_text:
        return {
            "status": NO_PREVIOUS_FINDING,
            "warnings": []
        }

    # Removed columns
    for column in changes.get("removed_columns", []):
        if _contains_column(finding_text, column):
            warnings.append(
                f"'{column}' was removed from the updated dataset."
            )

    # Type changes
    for change in changes.get("type_changes", []):
        column = change.get("column")

        if _contains_column(finding_text, column):
            warnings.append(
                f"'{column}' changed type from "
                f"{change.get('old_type')} to {change.get('new_type')}."
            )

    # Invalid values
    for column, summary in _invalid_value_entries(changes):
        if _contains_column(finding_text, column):
            invalid_count = summary.get("invalid_count", 0)

            warnings.append(
                f"'{column}' contains {invalid_count} invalid value(s) "
                "in the updated dataset."
            )

    # Missing values
    for change in _increased_missing_values(quality):
        column = change.get("column")

        if _contains_column(finding_text, column):
            warnings.append(
                f"Missing values in '{column}' increased by "
                f"{change.get('change', 0)} percentage points."
            )

    if warnings:
        return {
            "status": RECONSIDER,
            "warnings": warnings
        }

    return {
        "status": STILL_VALID,
        "warnings": []
    }


# ---------------------------------------------------------------------------
# Impact assessment
# ---------------------------------------------------------------------------

def assess_impact(changes, quality, statistics=None):
    """
    Assess structural, quality, type, duplicate, and statistical impact.
    """
    impacts = []

    # Structural changes
    added = changes.get("added_columns", [])
    removed = changes.get("removed_columns", [])

    if added:
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "MEDIUM",
            "message": f"{len(added)} column(s) were added."
        })

    if removed:
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "HIGH",
            "message": f"{len(removed)} column(s) were removed."
        })

    # Type changes
    high_type_changes = _type_changes_by_severity(changes, "HIGH")
    low_type_changes = _type_changes_by_severity(changes, "LOW")

    if high_type_changes:
        impacts.append({
            "area": "DATA TYPE",
            "severity": "HIGH",
            "message": (
                f"{len(high_type_changes)} high-severity "
                "data-type change(s) detected."
            )
        })

    if low_type_changes:
        impacts.append({
            "area": "DATA TYPE",
            "severity": "LOW",
            "message": (
                f"{len(low_type_changes)} low-severity "
                "data-type change(s) detected."
            )
        })

    # Invalid values
    invalid_entries = _invalid_value_entries(changes)

    if invalid_entries:
        invalid_count = sum(
            summary.get("invalid_count", 0)
            for _, summary in invalid_entries
        )

        impacts.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": (
                f"{invalid_count} invalid value(s) detected "
                "in the updated dataset."
            )
        })

    # Missing values
    increased_missing = _increased_missing_values(quality)

    if increased_missing:
        impacts.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": (
                f"Missing values increased in "
                f"{len(increased_missing)} column(s)."
            )
        })

    # Duplicate rows
    duplicate_change = quality.get("duplicate_change", 0)

    if duplicate_change > 0:
        impacts.append({
            "area": "DUPLICATES",
            "severity": "MEDIUM",
            "message": "Duplicate rows increased in the updated dataset."
        })

    # Statistical changes
    for shift in _statistical_shifts(statistics, threshold=10):
        severity = (
            "HIGH"
            if shift["percentage_change"] >= 20
            else "MEDIUM"
        )

        impacts.append({
            "area": "STATISTICAL",
            "severity": severity,
            "message": (
                f"'{shift['column']}' mean changed by "
                f"{round(shift['percentage_change'], 1)}%."
            )
        })

    return impacts


# ---------------------------------------------------------------------------
# ML readiness
# ---------------------------------------------------------------------------

def assess_ml_readiness(changes, quality, statistics=None):
    """
    Assess whether dataset changes may affect ML readiness.
    """
    issues = []

    # Removed features
    if changes.get("removed_columns"):
        issues.append(
            "Removed columns may affect previously selected ML features."
        )

    # High-severity type conflicts
    if _type_changes_by_severity(changes, "HIGH"):
        issues.append(
            "High-severity data-type conflicts may prevent reliable "
            "machine-learning preprocessing."
        )

    # Low-severity type conflicts
    if _type_changes_by_severity(changes, "LOW"):
        issues.append(
            "Some data-type changes may require preprocessing "
            "before ML analysis."
        )

    # Invalid values
    invalid_entries = _invalid_value_entries(changes)

    if invalid_entries:
        issues.append(
            "Invalid values were detected and should be handled "
            "before machine-learning analysis."
        )

    # Significant missing values
    for change in _significant_missing_values(quality, threshold=10):
        issues.append(
            f"Missing values increased in '{change.get('column')}'."
        )

    # Significant statistical shifts
    for shift in _statistical_shifts(statistics, threshold=20):
        issues.append(
            f"'{shift['column']}' shows a significant "
            "statistical change."
        )

    if not issues:
        return {
            "status": "GOOD",
            "issues": []
        }

    return {
        "status": "REVIEW_REQUIRED",
        "issues": issues
    }


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

def identify_limitations(changes, quality, statistics=None):
    """
    Identify limitations that should be communicated to the user.
    """
    limitations = []

    # Possible renames
    if changes.get("possible_renames"):
        limitations.append(
            "Some column changes may represent renaming, but rename "
            "detection is probabilistic and should be verified."
        )

    # Removed columns
    if changes.get("removed_columns"):
        limitations.append(
            "Previous analyses involving removed columns may no "
            "longer apply."
        )

    # Type changes
    if changes.get("type_changes"):
        limitations.append(
            "Changed data types may affect statistical and "
            "machine-learning analysis."
        )

    # Invalid values
    if _invalid_value_entries(changes):
        limitations.append(
            "Invalid values were detected; affected entries should "
            "be reviewed before relying on the affected analysis."
        )

    # Missing values
    for change in _increased_missing_values(quality):
        limitations.append(
            f"Missing values changed in '{change.get('column')}'."
        )

    # Statistical shifts
    for shift in _statistical_shifts(statistics, threshold=10):
        limitations.append(
            f"'{shift['column']}' shows a notable "
            "statistical shift."
        )

    return limitations


# ---------------------------------------------------------------------------
# Final recommendation
# ---------------------------------------------------------------------------

def generate_recommendation(changes, quality):
    """
    Generate exactly one final recommendation.

    Priority:
    1. HIGH type conflict
    2. Completely invalid affected column
    3. Mixed valid/invalid values
    4. Genuine removed columns
    5. Significant missing-value increase
    6. Increased duplicates
    7. Ready
    """

    # 1. HIGH type conflicts
    if _type_changes_by_severity(changes, "HIGH"):
        return UNSUITABLE

    # 2. Invalid values
    for _, summary in _invalid_value_entries(changes):
        invalid_count = summary.get("invalid_count", 0)
        valid_count = summary.get("valid_count", 0)

        # No usable values remain
        if invalid_count > 0 and valid_count == 0:
            return UNSUITABLE

        # Some valid values remain, but invalid values need attention
        if invalid_count > 0 and valid_count > 0:
            return ATTENTION

    # 3. Genuine removed columns
    possible_rename_old = {
        item.get("old_column")
        for item in changes.get("possible_renames", [])
    }

    truly_removed = [
        column
        for column in changes.get("removed_columns", [])
        if column not in possible_rename_old
    ]

    if truly_removed:
        return ATTENTION

    # 4. Significant missing-value increase
    if _significant_missing_values(quality, threshold=20):
        return ATTENTION

    # 5. Increased duplicates
    if quality.get("duplicate_change", 0) > 0:
        return ATTENTION

    # 6. No critical issues
    return READY


# ---------------------------------------------------------------------------
# Main intelligence pipeline
# ---------------------------------------------------------------------------

def run_intelligence_analysis(old_df, new_df, previous_finding=""):
    """
    Run the complete dataset intelligence workflow.

    Workflow:
        Detect → Analyze → Compare → Reconsider → Re-plan → Report
    """
    from change_engine import compare_datasets, compare_statistics
    from quality_engine import compare_quality

    # ------------------------------------------------------------------
    # 1. DETECT
    # ------------------------------------------------------------------
    changes = compare_datasets(old_df, new_df)

    # ------------------------------------------------------------------
    # 2. ANALYZE
    # ------------------------------------------------------------------
    statistics = compare_statistics(old_df, new_df)

    # ------------------------------------------------------------------
    # 3. COMPARE
    # ------------------------------------------------------------------
    quality = compare_quality(old_df, new_df)

    impact = assess_impact(
        changes,
        quality,
        statistics
    )

    ml_readiness = assess_ml_readiness(
        changes,
        quality,
        statistics
    )

    limitations = identify_limitations(
        changes,
        quality,
        statistics
    )

    # ------------------------------------------------------------------
    # 4. RECONSIDER
    # ------------------------------------------------------------------
    reconsideration = reconsider_previous_finding(
        previous_finding,
        changes,
        quality
    )

    # ------------------------------------------------------------------
    # 5. RE-PLAN
    # ------------------------------------------------------------------
    recommendation = generate_recommendation(
        changes,
        quality
    )

    # ------------------------------------------------------------------
    # 6. REPORT
    # ------------------------------------------------------------------
    return {
        "workflow": [
            "Detect",
            "Analyze",
            "Compare",
            "Reconsider",
            "Re-plan",
            "Report"
        ],
        "changes": changes,
        "quality": quality,
        "statistics": statistics,
        "impact": impact,
        "ml_readiness": ml_readiness,
        "limitations": limitations,
        "reconsideration": reconsideration,
        "recommendation": recommendation
    }