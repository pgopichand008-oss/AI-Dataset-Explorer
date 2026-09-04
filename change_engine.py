import pandas as pd


def _similar_values(old_series, new_series):
    """Check whether two columns contain mostly similar values."""

    old_values = set(old_series.dropna().astype(str).head(1000))
    new_values = set(new_series.dropna().astype(str).head(1000))

    if not old_values or not new_values:
        return 0.0

    intersection = len(old_values.intersection(new_values))
    union = len(old_values.union(new_values))

    return intersection / union if union else 0.0


def compare_datasets(old_df, new_df):
    """
    Compare the previous and updated datasets
    and detect structural changes.
    """

    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)

    added_columns = sorted(new_columns - old_columns)
    removed_columns = sorted(old_columns - new_columns)

    # Detect data-type changes
    type_changes = []

    for column in old_columns.intersection(new_columns):
        old_type = str(old_df[column].dtype)
        new_type = str(new_df[column].dtype)

        if old_type != new_type:
            type_changes.append({
                "column": column,
                "old_type": old_type,
                "new_type": new_type
            })

    # Detect possible renamed columns
    possible_renames = []

    for old_column in removed_columns:
        for new_column in added_columns:

            old_type = str(old_df[old_column].dtype)
            new_type = str(new_df[new_column].dtype)

            # Only compare columns with the same data type
            if old_type != new_type:
                continue

            similarity = _similar_values(
                old_df[old_column],
                new_df[new_column]
            )

            if similarity >= 0.5:
                possible_renames.append({
                    "old_column": old_column,
                    "new_column": new_column,
                    "confidence": round(similarity * 100, 1)
                })

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "possible_renames": possible_renames
    }

def compare_statistics(old_df, new_df):
    """
    Compare basic statistics for common numerical columns.
    """

    results = []

    common_columns = set(old_df.columns).intersection(new_df.columns)

    for column in sorted(common_columns):

        if not (
            pd.api.types.is_numeric_dtype(old_df[column])
            and pd.api.types.is_numeric_dtype(new_df[column])
        ):
            continue

        old_mean = old_df[column].mean()
        new_mean = new_df[column].mean()

        old_median = old_df[column].median()
        new_median = new_df[column].median()

        old_std = old_df[column].std()
        new_std = new_df[column].std()

        results.append({
            "column": column,
            "old_mean": round(old_mean, 2),
            "new_mean": round(new_mean, 2),
            "mean_change": round(new_mean - old_mean, 2),
            "old_median": round(old_median, 2),
            "new_median": round(new_median, 2),
            "median_change": round(new_median - old_median, 2),
            "old_std": round(old_std, 2),
            "new_std": round(new_std, 2)
        })

    return results