import re
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# COLUMN NAME SIMILARITY
# ============================================================

def _normalize_column_name(name):
    """
    Normalize a column name for similarity comparison.
    """

    text = str(name).strip().lower()

    # Split camelCase names.
    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text
    )

    # Replace separators with spaces.
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def _name_similarity(old_name, new_name):
    """
    Calculate column-name similarity.

    Uses:
    - token overlap
    - sequence similarity

    Returns a percentage from 0 to 100.
    """

    old_normalized = _normalize_column_name(
        old_name
    )

    new_normalized = _normalize_column_name(
        new_name
    )

    if not old_normalized or not new_normalized:
        return 0.0

    old_tokens = set(
        old_normalized.split()
    )

    new_tokens = set(
        new_normalized.split()
    )

    union = old_tokens | new_tokens

    if union:
        token_similarity = (
            len(old_tokens & new_tokens)
            / len(union)
        )
    else:
        token_similarity = 0.0

    sequence_similarity = SequenceMatcher(
        None,
        old_normalized,
        new_normalized
    ).ratio()

    return round(
        (
            token_similarity * 0.5
            + sequence_similarity * 0.5
        ) * 100,
        1
    )


# ============================================================
# VALUE SIMILARITY
# ============================================================

def _similar_values(old_series, new_series):
    """
    Compare values between two columns.

    The comparison is position-based when both columns
    contain the same number of non-null values.

    Returns a percentage from 0 to 100.
    """

    old_values = (
        old_series.dropna()
        .astype(str)
        .tolist()
    )

    new_values = (
        new_series.dropna()
        .astype(str)
        .tolist()
    )

    if not old_values or not new_values:
        return 0.0

    if len(old_values) != len(new_values):
        return 0.0

    matches = sum(
        old == new
        for old, new in zip(
            old_values,
            new_values
        )
    )

    return round(
        (
            matches
            / len(old_values)
        ) * 100,
        1
    )


# ============================================================
# TYPE DETECTION
# ============================================================

def _type_family(series):
    """
    Classify a pandas Series into a broad type family.

    Families:
    - numeric
    - datetime
    - boolean
    - text
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    return "text"


def _type_conflict_level(
    old_series,
    new_series
):
    """
    Determine the severity of a type change.

    Returns:
    - None
    - LOW
    - HIGH
    """

    old_family = _type_family(
        old_series
    )

    new_family = _type_family(
        new_series
    )

    # Same broad family.
    if old_family == new_family:

        # Different numeric dtypes are generally
        # manageable and therefore LOW severity.
        if old_family == "numeric":

            if (
                str(old_series.dtype)
                != str(new_series.dtype)
            ):
                return "LOW"

        return None

    # --------------------------------------------------------
    # Numeric -> text
    # --------------------------------------------------------
    #
    # This is the most important mixed-data case.
    #
    # Example:
    # Previous: [21, 24, 30]
    # Updated:  [21, 24, "Unknown", 30]
    #
    # The updated pandas column becomes object/text.
    # Numeric-looking values remain usable while genuinely
    # non-numeric values are flagged separately.
    if (
        old_family == "numeric"
        and new_family == "text"
    ):

        converted = pd.to_numeric(
            new_series,
            errors="coerce"
        )

        non_missing = new_series.notna()

        invalid_count = int(
            (
                non_missing
                & converted.isna()
            ).sum()
        )

        valid_count = int(
            (
                non_missing
                & converted.notna()
            ).sum()
        )

        # All values are still numeric in practice.
        if invalid_count == 0:
            return "LOW"

        # Some numeric values remain usable.
        if valid_count > 0:
            return "LOW"

        # No usable numeric values remain.
        return "HIGH"

    # --------------------------------------------------------
    # Text -> numeric
    # --------------------------------------------------------
    #
    # If the previous text values are all numeric-looking,
    # converting them to numeric is normally harmless.
    if (
        old_family == "text"
        and new_family == "numeric"
    ):

        converted = pd.to_numeric(
            old_series,
            errors="coerce"
        )

        non_missing = old_series.notna()

        invalid_count = int(
            (
                non_missing
                & converted.isna()
            ).sum()
        )

        if invalid_count == 0:
            return "LOW"

        return "HIGH"

    # Any other broad-family change is potentially significant.
    return "HIGH"


# ============================================================
# INVALID VALUE DETECTION
# ============================================================

def _invalid_value_summary(series):
    """
    Identify values that cannot be interpreted as numeric.

    This function should ONLY be used when the column is
    expected to contain numeric data.

    Normal categorical/text columns such as:
        ["A", "B", "C"]

    are NOT considered invalid.
    """

    # A true numeric pandas Series cannot contain
    # non-numeric values.
    if pd.api.types.is_numeric_dtype(series):

        return {
            "invalid_count": 0,
            "valid_count": int(
                series.notna().sum()
            ),
            "invalid_examples": []
        }

    converted = pd.to_numeric(
        series,
        errors="coerce"
    )

    non_missing = series.notna()

    invalid_mask = (
        non_missing
        & converted.isna()
    )

    invalid_values = (
        series.loc[invalid_mask]
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    valid_count = int(
        (
            non_missing
            & converted.notna()
        ).sum()
    )

    return {
        "invalid_count": int(
            invalid_mask.sum()
        ),
        "valid_count": valid_count,
        "invalid_examples": invalid_values
    }


# ============================================================
# DATASET STRUCTURAL COMPARISON
# ============================================================

def compare_datasets(old_df, new_df):
    """
    Compare previous and updated datasets.

    Detects:
    - added columns
    - removed columns
    - possible renamed columns
    - data-type conflicts
    - invalid values in columns that were previously numeric
    """

    old_columns = set(
        old_df.columns
    )

    new_columns = set(
        new_df.columns
    )

    # --------------------------------------------------------
    # Added / removed columns
    # --------------------------------------------------------

    added_columns = sorted(
        new_columns - old_columns
    )

    removed_columns = sorted(
        old_columns - new_columns
    )

    common_columns = (
        old_columns.intersection(
            new_columns
        )
    )

    # --------------------------------------------------------
    # Type changes
    # --------------------------------------------------------

    type_changes = []

    for column in sorted(
        common_columns
    ):

        old_series = old_df[column]
        new_series = new_df[column]

        old_type = str(
            old_series.dtype
        )

        new_type = str(
            new_series.dtype
        )

        severity = _type_conflict_level(
            old_series,
            new_series
        )

        if severity:

            type_changes.append(
                {
                    "column": column,
                    "old_type": old_type,
                    "new_type": new_type,
                    "severity": severity
                }
            )

    # --------------------------------------------------------
    # Possible rename detection
    # --------------------------------------------------------

    possible_renames = []

    for old_column in removed_columns:

        for new_column in added_columns:

            old_series = old_df[
                old_column
            ]

            new_series = new_df[
                new_column
            ]

            old_family = _type_family(
                old_series
            )

            new_family = _type_family(
                new_series
            )

            # Only compare compatible broad families.
            if old_family != new_family:
                continue

            name_score = _name_similarity(
                old_column,
                new_column
            )

            value_score = _similar_values(
                old_series,
                new_series
            )

            # Name similarity is slightly more important
            # than value similarity because identical values
            # alone do not prove that two columns represent
            # the same concept.
            confidence = round(
                (
                    name_score * 0.6
                    + value_score * 0.4
                ),
                1
            )

            # Conservative threshold.
            if confidence >= 55:

                possible_renames.append(
                    {
                        "old_column": old_column,
                        "new_column": new_column,
                        "name_similarity": name_score,
                        "value_similarity": value_score,
                        "confidence": confidence
                    }
                )

    # --------------------------------------------------------
    # Invalid-value detection
    # --------------------------------------------------------

    invalid_values = {}

    for column in sorted(
        new_df.columns
    ):

        # A newly added column has no previous expectation,
        # so ordinary text values must not be called invalid.
        if column not in old_df.columns:
            continue

        old_series = old_df[column]
        new_series = new_df[column]

        # Only validate numeric expectations.
        #
        # Example:
        #
        # Old:
        # age = [21, 24, 30]
        #
        # New:
        # age = [21, 24, "Unknown", 30]
        #
        # "Unknown" should be reported.
        if not pd.api.types.is_numeric_dtype(
            old_series
        ):
            continue

        summary = _invalid_value_summary(
            new_series
        )

        if summary["invalid_count"] > 0:

            invalid_values[column] = summary

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "possible_renames": possible_renames,
        "invalid_values": invalid_values
    }


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def compare_statistics(old_df, new_df):
    """
    Compare numerical statistics between previous
    and updated datasets.

    Calculates:
    - mean
    - median
    - standard deviation
    - minimum
    - maximum

    Values that cannot be interpreted as numeric are
    coerced to NaN so valid observations remain usable.
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
        )

        new_series = pd.to_numeric(
            new_df[column],
            errors="coerce"
        )

        old_values = (
            old_series
            .dropna()
        )

        new_values = (
            new_series
            .dropna()
        )

        # Skip columns with no usable numeric data.
        if (
            old_values.empty
            and new_values.empty
        ):
            continue

        old_stats = {
            "mean": (
                float(old_values.mean())
                if not old_values.empty
                else None
            ),
            "median": (
                float(old_values.median())
                if not old_values.empty
                else None
            ),
            "std": (
                float(old_values.std())
                if len(old_values) > 1
                else 0.0
            ),
            "min": (
                float(old_values.min())
                if not old_values.empty
                else None
            ),
            "max": (
                float(old_values.max())
                if not old_values.empty
                else None
            )
        }

        new_stats = {
            "mean": (
                float(new_values.mean())
                if not new_values.empty
                else None
            ),
            "median": (
                float(new_values.median())
                if not new_values.empty
                else None
            ),
            "std": (
                float(new_values.std())
                if len(new_values) > 1
                else 0.0
            ),
            "min": (
                float(new_values.min())
                if not new_values.empty
                else None
            ),
            "max": (
                float(new_values.max())
                if not new_values.empty
                else None
            )
        }

        changes = {}

        for key in old_stats:

            old_value = old_stats[key]
            new_value = new_stats[key]

            if (
                old_value is None
                or new_value is None
            ):
                changes[key] = None

            else:
                changes[key] = round(
                    new_value - old_value,
                    4
                )

        statistical_changes.append(
            {
                "column": column,
                "old": old_stats,
                "new": new_stats,
                "changes": changes
            }
        )

    return statistical_changes