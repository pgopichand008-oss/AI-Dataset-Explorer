import pandas as pd

from change_engine import compare_datasets, compare_statistics


READY = "READY FOR FURTHER ANALYSIS"
ATTENTION = "NEEDS ATTENTION BEFORE ANALYSIS"
UNSUITABLE = "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

RECONSIDER = "RECONSIDER"
STILL_VALID = "STILL VALID"
NO_PREVIOUS_FINDING = "NO PREVIOUS FINDING"


def _contains_column(text, column):
    """Check whether a finding/message refers to a column name."""
    if not text:
        return False

    text = str(text).lower()
    column = str(column).lower()

    return column in text


def _is_identifier(column):
    """Detect columns that are likely identifiers."""
    name = str(column).lower()

    identifier_terms = [
        "id",
        "identifier",
        "uuid",
        "code",
        "key",
    ]

    return (
        name in identifier_terms
        or name.endswith("_id")
        or name.endswith("id")
        or name.endswith("_code")
        or name.endswith("_key")
    )


def _safe_percentage_change(old_value, new_value):
    """Calculate percentage change safely."""
    if old_value in (None, 0) or new_value is None:
        return None

    try:
        return abs((new_value - old_value) / old_value) * 100
    except (TypeError, ZeroDivisionError):
        return None


def _type_changes_by_severity(changes, severity):
    """Return type changes matching a severity."""
    return [
        change
        for change in changes.get("type_changes", [])
        if change.get("severity") == severity
    ]


def _invalid_value_entries(quality):
    """Return invalid-value entries from the quality comparison."""
    return quality.get("invalid_values", {})


def _increased_missing_values(quality):
    """Return columns where missing values increased."""
    return quality.get("missing_values", {}).get("increased", [])


def _significant_missing_values(quality):
    """Return missing-value changes that are large enough to matter."""
    significant = []

    for item in _increased_missing_values(quality):
        old_pct = item.get("old_percentage", 0)
        new_pct = item.get("new_percentage", 0)

        if new_pct - old_pct >= 10:
            significant.append(item)

    return significant


def _statistical_shifts(statistics):
    """Return statistically meaningful shifts."""
    shifts = []

    for item in statistics:
        mean_change = item.get("mean_change")

        if mean_change is None:
            continue

        shifts.append(item)

    return shifts


def reconsider_previous_finding(finding, changes, quality):
    """
    Reconsider a previous analysis finding against the updated dataset.
    """

    if not finding:
        return {
            "status": NO_PREVIOUS_FINDING,
            "warnings": [],
        }

    warnings = []

    # Check removed columns.
    for column in changes.get("removed_columns", []):
        if _contains_column(finding, column):
            warnings.append(
                f"Previous finding refers to removed column '{column}'."
            )

    # Check type changes.
    for change in changes.get("type_changes", []):
        column = change.get("column")

        if _contains_column(finding, column):
            warnings.append(
                f"'{column}' changed type from "
                f"{change.get('old_type')} to {change.get('new_type')}."
            )

    # Check invalid values.
    for column, summary in _invalid_value_entries(quality).items():
        invalid_count = summary.get("invalid_count", 0)

        if invalid_count > 0 and _contains_column(finding, column):
            warnings.append(
                f"'{column}' contains {invalid_count} invalid value(s) "
                "in the updated dataset."
            )

    # Check increased missing values.
    for item in _increased_missing_values(quality):
        column = item.get("column")

        if _contains_column(finding, column):
            warnings.append(
                f"Missing values increased for '{column}' "
                "in the updated dataset."
            )

    if warnings:
        return {
            "status": RECONSIDER,
            "warnings": warnings,
        }

    return {
        "status": STILL_VALID,
        "warnings": [],
    }


def assess_impact(changes, quality, statistics):
    """Assess the impact of detected dataset changes."""

    impact = []

    added = changes.get("added_columns", [])
    removed = changes.get("removed_columns", [])

    if added:
        impact.append({
            "area": "STRUCTURAL",
            "severity": "MEDIUM",
            "message": f"{len(added)} column(s) were added.",
        })

    if removed:
        impact.append({
            "area": "STRUCTURAL",
            "severity": "HIGH",
            "message": f"{len(removed)} column(s) were removed.",
        })

    for change in _type_changes_by_severity(changes, "HIGH"):
        impact.append({
            "area": "DATA TYPE",
            "severity": "HIGH",
            "message": (
                f"'{change.get('column')}' changed from "
                f"{change.get('old_type')} to {change.get('new_type')}."
            ),
        })

    for change in _type_changes_by_severity(changes, "LOW"):
        impact.append({
            "area": "DATA TYPE",
            "severity": "LOW",
            "message": (
                f"'{change.get('column')}' has a low-risk type change "
                f"from {change.get('old_type')} to "
                f"{change.get('new_type')}."
            ),
        })

    invalid_values = _invalid_value_entries(quality)

    if invalid_values:
        total_invalid = sum(
            item.get("invalid_count", 0)
            for item in invalid_values.values()
        )

        impact.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": (
                f"{total_invalid} invalid value(s) detected "
                "in the updated dataset."
            ),
        })

    missing_changes = _significant_missing_values(quality)

    if missing_changes:
        impact.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": (
                f"{len(missing_changes)} column(s) have a significant "
                "increase in missing values."
            ),
        })

    duplicate_change = quality.get("duplicate_change", 0)

    if duplicate_change > 0:
        impact.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": "Duplicate rows increased in the updated dataset.",
        })

    for item in _statistical_shifts(statistics):
        column = item.get("column")

        if _is_identifier(column):
            continue

        mean_change = _safe_percentage_change(
            item.get("old_mean"),
            item.get("new_mean"),
        )

        if mean_change is None:
            continue

        if mean_change >= 20:
            severity = "HIGH"
        elif mean_change >= 10:
            severity = "MEDIUM"
        else:
            continue

        impact.append({
            "area": "STATISTICAL",
            "severity": severity,
            "message": (
                f"Mean of '{column}' changed by "
                f"{round(mean_change, 2)}%."
            ),
        })

    return impact


def assess_ml_readiness(changes, quality, statistics):
    """Assess whether the updated dataset is ready for ML analysis."""

    issues = []

    if changes.get("removed_columns"):
        issues.append(
            "Columns were removed and previous analysis may no longer apply."
        )

    high_type_changes = _type_changes_by_severity(changes, "HIGH")

    if high_type_changes:
        issues.append(
            "High-risk data type conflicts were detected."
        )

    low_type_changes = _type_changes_by_severity(changes, "LOW")

    if low_type_changes:
        issues.append(
            "Low-risk data type changes should be reviewed."
        )

    if _invalid_value_entries(quality):
        issues.append(
            "Invalid values were detected and should be handled "
            "before machine-learning analysis."
        )

    for item in _significant_missing_values(quality):
        old_pct = item.get("old_percentage", 0)
        new_pct = item.get("new_percentage", 0)

        if new_pct - old_pct >= 10:
            issues.append(
                f"Missing values increased significantly for "
                f"'{item.get('column')}'."
            )

    for item in _statistical_shifts(statistics):
        column = item.get("column")

        if _is_identifier(column):
            continue

        change = _safe_percentage_change(
            item.get("old_mean"),
            item.get("new_mean"),
        )

        if change is not None and change >= 20:
            issues.append(
                f"Mean of '{column}' changed substantially."
            )

    if issues:
        return {
            "status": "REVIEW_REQUIRED",
            "issues": issues,
        }

    return {
        "status": "GOOD",
        "issues": [],
    }


def identify_limitations(changes, quality, statistics):
    """Identify limitations of the automated comparison."""

    limitations = []

    if changes.get("possible_renames"):
        limitations.append(
            "Possible renamed columns are probabilistic matches and "
            "should be manually verified."
        )

    if changes.get("removed_columns"):
        limitations.append(
            "Removed columns may invalidate parts of previous analysis."
        )

    if changes.get("type_changes"):
        limitations.append(
            "Data type changes may affect downstream analysis and models."
        )

    if _invalid_value_entries(quality):
        limitations.append(
            "Invalid values may require correction or preprocessing."
        )

    if _increased_missing_values(quality):
        limitations.append(
            "Changes in missing-value patterns may affect analysis."
        )

    for item in _statistical_shifts(statistics):
        column = item.get("column")

        if _is_identifier(column):
            continue

        change = _safe_percentage_change(
            item.get("old_mean"),
            item.get("new_mean"),
        )

        if change is not None and change >= 10:
            limitations.append(
                f"Statistical distribution for '{column}' changed noticeably."
            )

    return limitations


def generate_recommendation(changes, quality, statistics):
    """
    Generate exactly one final recommendation.
    """

    # Highest priority: severe type conflicts.
    high_type_changes = _type_changes_by_severity(changes, "HIGH")

    if high_type_changes:
        return UNSUITABLE

    # Completely invalid numeric data.
    invalid_values = _invalid_value_entries(quality)

    for summary in invalid_values.values():
        invalid_count = summary.get("invalid_count", 0)
        valid_count = summary.get("valid_count", 0)

        if invalid_count > 0 and valid_count == 0:
            return UNSUITABLE

    # Partially invalid data.
    for summary in invalid_values.values():
        if summary.get("invalid_count", 0) > 0:
            return ATTENTION

    # Removed columns are important unless they are likely renames.
    removed_columns = set(changes.get("removed_columns", []))
    possible_renames = changes.get("possible_renames", [])

    renamed_old_columns = {
        item.get("old_column")
        for item in possible_renames
    }

    genuine_removed = removed_columns - renamed_old_columns

    if genuine_removed:
        return ATTENTION

    # Large missing-value increases.
    for item in _increased_missing_values(quality):
        old_pct = item.get("old_percentage", 0)
        new_pct = item.get("new_percentage", 0)

        if new_pct - old_pct >= 20:
            return ATTENTION

    # Duplicate increase.
    if quality.get("duplicate_change", 0) > 0:
        return ATTENTION

    return READY


def run_intelligence_analysis(
    old_df,
    new_df,
    previous_finding=None,
):
    """
    Run the complete dataset intelligence workflow.

    Workflow:
    Detect → Analyze → Compare → Reconsider → Re-plan → Report
    """

    workflow = [
        "Detect",
        "Analyze",
        "Compare",
        "Reconsider",
        "Re-plan",
        "Report",
    ]

    # Detect and compare structural/data-quality changes.
    changes = compare_datasets(old_df, new_df)

    # Compare numerical statistics.
    statistics = compare_statistics(old_df, new_df)

    # Quality information is stored inside compare_datasets.
    quality = {
        "missing_values": changes.get("missing_values", {}),
        "duplicate_change": changes.get("duplicate_change", 0),
        "invalid_values": changes.get("invalid_values", {}),
    }

    # Reconsider previous analysis.
    reconsideration = reconsider_previous_finding(
        previous_finding,
        changes,
        quality,
    )

    # Assess downstream impact.
    impact = assess_impact(
        changes,
        quality,
        statistics,
    )

    # Assess ML readiness.
    ml_readiness = assess_ml_readiness(
        changes,
        quality,
        statistics,
    )

    # Identify limitations.
    limitations = identify_limitations(
        changes,
        quality,
        statistics,
    )

    # Generate exactly one recommendation.
    recommendation = generate_recommendation(
        changes,
        quality,
        statistics,
    )

    return {
        "workflow": workflow,
        "changes": changes,
        "quality": quality,
        "statistics": statistics,
        "impact": impact,
        "ml_readiness": ml_readiness,
        "limitations": limitations,
        "reconsideration": reconsideration,
        "recommendation": recommendation,
    }