import pandas as pd


def _missing_summary(df):
    """Calculate missing-value counts and rates for each column."""

    total_rows = len(df)

    summary = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        missing_rate = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        summary[column] = {
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 2)
        }

    return summary


def compare_quality(old_df, new_df):
    """
    Compare data quality between previous and updated datasets.
    """

    old_missing = _missing_summary(old_df)
    new_missing = _missing_summary(new_df)

    missing_changes = []

    common_columns = set(old_df.columns).intersection(new_df.columns)

    for column in sorted(common_columns):

        old_rate = old_missing[column]["missing_rate"]
        new_rate = new_missing[column]["missing_rate"]

        if old_rate != new_rate:
            missing_changes.append({
                "column": column,
                "old_missing_rate": old_rate,
                "new_missing_rate": new_rate,
                "change": round(new_rate - old_rate, 2)
            })

    old_duplicates = int(old_df.duplicated().sum())
    new_duplicates = int(new_df.duplicated().sum())

    duplicate_change = new_duplicates - old_duplicates

    return {
        "old_missing": old_missing,
        "new_missing": new_missing,
        "missing_changes": missing_changes,
        "old_duplicates": old_duplicates,
        "new_duplicates": new_duplicates,
        "duplicate_change": duplicate_change
    }