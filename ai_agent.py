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


def assess_impact(changes, quality):
    """
    Assess how significant the dataset changes are.
    """

    impacts = []

    if changes.get("added_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "MEDIUM",
            "message": (
                f"{len(changes['added_columns'])} column(s) were added."
            )
        })

    if changes.get("removed_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "HIGH",
            "message": (
                f"{len(changes['removed_columns'])} column(s) were removed."
            )
        })

    if changes.get("type_changes"):
        impacts.append({
            "area": "DATA TYPE",
            "severity": "HIGH",
            "message": (
                f"{len(changes['type_changes'])} column data type change(s) detected."
            )
        })

    increased_missing = [
        x for x in quality.get("missing_changes", [])
        if x["change"] > 0
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

    if quality.get("new_duplicates", 0) > quality.get("old_duplicates", 0):
        impacts.append({
            "area": "DUPLICATES",
            "severity": "MEDIUM",
            "message": "Duplicate rows increased in the updated dataset."
        })

    return impacts


def generate_recommendation(changes, quality):
    """
    Generate exactly one final recommendation.
    """

    # Critical structural/type problems
    if changes.get("type_changes"):
        return "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

    if changes.get("removed_columns"):
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    # Significant quality degradation
    for change in quality.get("missing_changes", []):
        if change["change"] >= 20:
            return "NEEDS ATTENTION BEFORE ANALYSIS"

    if quality.get("duplicate_change", 0) > 0:
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    return "READY FOR FURTHER ANALYSIS"

def run_intelligence_analysis(old_df, new_df, previous_finding=""):
    """
    Run the complete dataset intelligence workflow.
    """

    from change_engine import compare_datasets
    from quality_engine import compare_quality

    changes = compare_datasets(old_df, new_df)

    quality = compare_quality(old_df, new_df)

    impact = assess_impact(changes, quality)

    reconsideration = reconsider_previous_finding(
        previous_finding,
        changes,
        quality
    ) if previous_finding else {
        "status": "NO_PREVIOUS_FINDING",
        "warnings": []
    }

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
        "impact": impact,
        "reconsideration": reconsideration,
        "recommendation": recommendation
    }