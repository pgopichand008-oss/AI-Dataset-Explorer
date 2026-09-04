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