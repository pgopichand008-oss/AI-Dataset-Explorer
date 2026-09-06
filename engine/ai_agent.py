def reconsider_previous_finding(finding, changes, quality):
    """
    Check whether a previous analysis finding is still valid
    after the dataset has been updated.
    """

    finding_text = finding.lower()
    warnings = []

    # Removed columns
    for column in changes.get("removed_columns", []):
        if column.lower() in finding_text:
            warnings.append(
                f"'{column}' was removed from the updated dataset."
            )

    # Data type changes
    for change in changes.get("type_changes", []):
        column = change["column"]

        if column.lower() in finding_text:
            warnings.append(
                f"'{column}' changed type from "
                f"{change['old_type']} to {change['new_type']}."
            )

    # Increased missing values
    for change in quality.get("missing_changes", []):
        column = change["column"]

        if column.lower() in finding_text and change["change"] > 0:
            warnings.append(
                f"Missing values in '{column}' increased by "
                f"{change['change']} percentage points."
            )

    if warnings:
        return {
            "status": "RECONSIDER",
            "warnings": warnings
        }

    return {
        "status": "STILL_VALID",
        "warnings": []
    }


def assess_impact(changes, quality, statistics=None):
    """
    Assess how significant the dataset changes are.
    """

    impacts = []

    # Added columns
    if changes.get("added_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "MEDIUM",
            "message": (
                f"{len(changes['added_columns'])} column(s) were added."
            )
        })

    # Removed columns
    if changes.get("removed_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "HIGH",
            "message": (
                f"{len(changes['removed_columns'])} column(s) were removed."
            )
        })

    # Data type changes
    if changes.get("type_changes"):
        impacts.append({
            "area": "DATA TYPE",
            "severity": "HIGH",
            "message": (
                f"{len(changes['type_changes'])} column data type "
                f"change(s) detected."
            )
        })

    # Missing values
    increased_missing = [
        item
        for item in quality.get("missing_changes", [])
        if item["change"] > 0
    ]

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
    if quality.get("duplicate_change", 0) > 0:
        impacts.append({
            "area": "DUPLICATES",
            "severity": "MEDIUM",
            "message": (
                "Duplicate rows increased in the updated dataset."
            )
        })

    # Statistical changes
    if statistics:
        for stat in statistics:

            column_name = stat["column"].lower()

            # Ignore identifier-like columns
            if (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
            ):
                continue

            if stat["old_mean"] == 0:
                continue

            mean_change_percent = abs(
                stat["mean_change"] / stat["old_mean"] * 100
            )

            if mean_change_percent >= 10:
                impacts.append({
                    "area": "STATISTICAL",
                    "severity": "MEDIUM",
                    "message": (
                        f"'{stat['column']}' mean changed by "
                        f"{round(mean_change_percent, 1)}%."
                    )
                })

    return impacts


def assess_ml_readiness(changes, quality, statistics=None):
    """
    Assess how dataset changes may affect machine-learning readiness.
    """

    issues = []

    # Removed columns
    if changes.get("removed_columns"):
        issues.append(
            "Removed columns may affect previously selected ML features."
        )

    # Data type changes
    if changes.get("type_changes"):
        issues.append(
            "Data-type changes may require preprocessing before ML analysis."
        )

    # Missing values
    for change in quality.get("missing_changes", []):
        if change["change"] >= 10:
            issues.append(
                f"Missing values increased in '{change['column']}'."
            )

    # Statistical changes
    if statistics:
        for stat in statistics:

            column_name = stat["column"].lower()

            # Ignore identifier-like columns
            if (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
            ):
                continue

            if stat["old_mean"] == 0:
                continue

            change_percent = abs(
                stat["mean_change"] / stat["old_mean"] * 100
            )

            if change_percent >= 20:
                issues.append(
                    f"'{stat['column']}' shows a significant "
                    f"change in its mean."
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


def identify_limitations(changes, quality, statistics=None):
    """
    Identify limitations that should be communicated before analysis.
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

    # Data type changes
    if changes.get("type_changes"):
        limitations.append(
            "Changed data types may affect statistical and "
            "machine-learning analysis."
        )

    # Missing values
    for change in quality.get("missing_changes", []):
        if change["change"] > 0:
            limitations.append(
                f"Missing values changed in '{change['column']}'."
            )

    # Statistical changes
    if statistics:
        for stat in statistics:

            column_name = stat["column"].lower()

            # Ignore identifier-like columns
            if (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
            ):
                continue

            if stat["old_mean"] == 0:
                continue

            change_percent = abs(
                stat["mean_change"] / stat["old_mean"] * 100
            )

            if change_percent >= 10:
                limitations.append(
                    f"'{stat['column']}' shows a notable "
                    f"statistical shift."
                )

    return limitations


def generate_recommendation(changes, quality):
    """
    Generate exactly one final recommendation.
    """

    # Critical data-type problems
    if changes.get("type_changes"):
        return "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

    # Determine whether removed columns are actually possible renames
    possible_rename_old = {
        item["old_column"]
        for item in changes.get("possible_renames", [])
    }

    truly_removed = [
        column
        for column in changes.get("removed_columns", [])
        if column not in possible_rename_old
    ]

    # Genuine removed columns require attention
    if truly_removed:
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    # Significant missing-value increase
    for change in quality.get("missing_changes", []):
        if change["change"] >= 20:
            return "NEEDS ATTENTION BEFORE ANALYSIS"

    # Increased duplicates
    if quality.get("duplicate_change", 0) > 0:
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    return "READY FOR FURTHER ANALYSIS"


def run_intelligence_analysis(old_df, new_df, previous_finding=""):
    """
    Run the complete dataset intelligence workflow.
    """

    from change_engine import compare_datasets, compare_statistics
    from quality_engine import compare_quality

    # 1. Detect changes
    changes = compare_datasets(old_df, new_df)

    # 2. Analyze statistics
    statistics = compare_statistics(old_df, new_df)

    # 3. Compare data quality
    quality = compare_quality(old_df, new_df)

    # 4. Assess impact
    impact = assess_impact(
        changes,
        quality,
        statistics
    )

    # 5. Assess ML readiness
    ml_readiness = assess_ml_readiness(
        changes,
        quality,
        statistics
    )

    # 6. Identify limitations
    limitations = identify_limitations(
        changes,
        quality,
        statistics
    )

    # 7. Reconsider previous findings
    if previous_finding:
        reconsideration = reconsider_previous_finding(
            previous_finding,
            changes,
            quality
        )
    else:
        reconsideration = {
            "status": "NO_PREVIOUS_FINDING",
            "warnings": []
        }

    # 8. Generate final recommendation
    recommendation = generate_recommendation(
        changes,
        quality
    )

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