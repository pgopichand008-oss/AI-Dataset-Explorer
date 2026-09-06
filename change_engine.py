import re

import pandas as pd


def _normalize_column_name(name):
    """Normalize a column name for similarity comparison."""

    name = str(name).strip().lower()
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return " ".join(name.split())


def _name_similarity(old_name, new_name):
    """Calculate token-based similarity between two column names."""

    old_tokens = set(_normalize_column_name(old_name).split())
    new_tokens = set(_normalize_column_name(new_name).split())

    if not old_tokens or not new_tokens:
        return 0.0

    intersection = len(old_tokens.intersection(new_tokens))
    union = len(old_tokens.union(new_tokens))

    return intersection / union if union else 0.0


def _similar_values(old_series, new_series):
    """Check whether two columns contain mostly similar values."""

    old_values = set(old_series.dropna().astype(str).head(1000))
    new_values = set(new_series.dropna().astype(str).head(1000))

    if not old_values or not new_values:
        return 0.0

    intersection = len(old_values.intersection(new_values))
    union = len(old_values.union(new_values))

    return intersection / union if union else 0.0


def _type_family(series):
    """Classify a pandas Series into a broader data-type family."""

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    return "text"


def _invalid_value_summary(series):
    """
    Identify values that cannot be interpreted as numeric.

    Valid numeric values are preserved; invalid values are only reported.
    """

    if pd.api.types.is_numeric_dtype(series):
        return {
            "invalid_count": 0,
            "valid_count": int(series.notna().sum()),
            "invalid_examples": []
        }

    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna()

    invalid_mask = non_missing & converted.isna()

    invalid_values = (
        series.loc[invalid_mask]
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    return {
        "invalid_count": int(invalid_mask.sum()),
        "valid_count": int(converted.notna().sum()),
        "invalid_examples": invalid_values
    }


def compare_datasets(old_df, new_df):
    """
    Compare previous and updated datasets.

    Detects:
    - added columns
    - removed columns
    - possible renamed columns
    - data-type conflicts
    - mixed/invalid values
    """

    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)

    added_columns = sorted(new_columns - old_columns)
    removed_columns = sorted(old_columns - new_columns)

    common_columns = old_columns.intersection(new_columns)

    type_changes = []

    for column in sorted(common_columns):
        old_type = str(old_df[column].dtype)
        new_type = str(new_df[column].dtype)

        old_family = _type_family(old_df[column])
        new_family = _type_family(new_df[column])

        if old_type != new_type or old_family != new_family:
            type_changes.append({
                "column": column,
                "old_type": old_type,
                "new_type": new_type,
                "old_type_family": old_family,
                "new_type_family": new_family
            })

    possible_renames = []

    for old_column in removed_columns:
        for new_column in added_columns:

            name_score = _name_similarity(
                old_column,
                new_column
            )

            value_score = _similar_values(
                old_df[old_column],
                new_df[new_column]
            )

            old_family = _type_family(old_df[old_column])
            new_family = _type_family(new_df[new_column])

            if old_family != new_family:
                continue

            combined_score = (
                name_score * 0.6
                + value_score * 0.4
            )

            if combined_score >= 0.55:
                possible_renames.append({
                    "old_column": old_column,
                    "new_column": new_column,
                    "name_similarity": round(name_score * 100, 1),
                    "value_similarity": round(value_score * 100, 1),
                    "confidence": round(combined_score * 100, 1)
                })

    invalid_values = {}

    for column in sorted(new_df.columns):
        summary = _invalid_value_summary(new_df[column])

        if summary["invalid_count"] > 0:
            invalid_values[column] = summary

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "possible_renames": possible_renames,
        "invalid_values": invalid_values
    }


def compare_statistics(old_df, new_df):
    """
    Compare numerical statistics between previous and updated datasets.

    Calculates:
    - mean
    - median
    - standard deviation
    - minimum
    - maximum

    Invalid numerical values are coerced to NaN for calculation,
    so valid values remain usable.
    """

    common_columns = [
        column
        for column in old_df.columns
        if column in new_df.columns
    ]

    statistical_changes = []

    for column in common_columns:

        old_series = pd.to_numeric(
            old_df[column],
            errors="coerce"
        ).dropna()

        new_series = pd.to_numeric(
            new_df[column],
            errors="coerce"
        ).dropna()

        if old_series.empty or new_series.empty:
            continue

        old_mean = float(old_series.mean())
        new_mean = float(new_series.mean())

        old_median = float(old_series.median())
        new_median = float(new_series.median())

        old_std = (
            float(old_series.std())
            if len(old_series) > 1
            else 0.0
        )

        new_std = (
            float(new_series.std())
            if len(new_series) > 1
            else 0.0
        )

        old_min = float(old_series.min())
        new_min = float(new_series.min())

        old_max = float(old_series.max())
        new_max = float(new_series.max())

        statistical_changes.append({
            "column": column,
            "old_mean": round(old_mean, 4),
            "new_mean": round(new_mean, 4),
            "mean_change": round(new_mean - old_mean, 4),
            "old_median": round(old_median, 4),
            "new_median": round(new_median, 4),
            "median_change": round(
                new_median - old_median,
                4
            ),
            "old_std": round(old_std, 4),
            "new_std": round(new_std, 4),
            "std_change": round(new_std - old_std, 4),
            "old_min": round(old_min, 4),
            "new_min": round(new_min, 4),
            "min_change": round(new_min - old_min, 4),
            "old_max": round(old_max, 4),
            "new_max": round(new_max, 4),
            "max_change": round(new_max - old_max, 4)
        })

    return statistical_changes