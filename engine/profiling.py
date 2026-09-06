import pandas as pd

from utils import (
    safe_percentage,
    infer_column_role
)


def build_basic_profile(df):

    rows = len(df)

    columns = len(
        df.columns
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns

    return {
        "rows": rows,
        "columns": columns,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "numeric_df": numeric_df,
        "categorical_columns": categorical_columns
    }


def build_column_intelligence(
    df,
    rows
):

    column_rows = []

    identifier_columns = []

    date_columns = []

    target_candidates = []

    for column in df.columns:

        (
            role,
            confidence,
            possible_target,
            reason
        ) = infer_column_role(
            df,
            column
        )

        if role == "Identifier":
            identifier_columns.append(
                column
            )

        if role == "Date / Time":
            date_columns.append(
                column
            )

        if possible_target:
            target_candidates.append(
                column
            )

        column_rows.append(
            {
                "Attribute": column,
                "Role": role,
                "Confidence": confidence,
                "Missing %": round(
                    safe_percentage(
                        df[column].isna().sum(),
                        rows
                    ),
                    2
                ),
                "Unique %": round(
                    safe_percentage(
                        df[column].nunique(
                            dropna=True
                        ),
                        rows
                    ),
                    2
                ),
                "Potential Target": (
                    "Yes"
                    if possible_target
                    else "No"
                ),
                "Reason": reason
            }
        )

    column_intelligence_df = pd.DataFrame(
        column_rows
    )

    return (
        column_intelligence_df,
        identifier_columns,
        date_columns,
        target_candidates
    )


def build_numerical_summary(
    numeric_df
):

    if numeric_df.empty:

        return pd.DataFrame()

    numerical_summary = (
        numeric_df
        .describe()
        .T
    )

    numerical_summary[
        "median"
    ] = numeric_df.median()

    numerical_summary[
        "range"
    ] = (
        numerical_summary["max"]
        -
        numerical_summary["min"]
    )

    numerical_summary[
        "skewness"
    ] = numeric_df.skew()

    return numerical_summary


def build_categorical_summary(
    df,
    categorical_columns,
    rows
):

    categorical_rows = []

    for column in categorical_columns:

        counts = (
            df[column]
            .value_counts(
                dropna=False
            )
        )

        if counts.empty:

            top_category = "N/A"

            top_frequency = 0

        else:

            top_category = str(
                counts.index[0]
            )

            top_frequency = int(
                counts.iloc[0]
            )

        categorical_rows.append(
            {
                "Attribute": column,
                "Unique Values": int(
                    df[column].nunique(
                        dropna=True
                    )
                ),
                "Top Category": top_category,
                "Frequency": top_frequency,
                "Dominance %": round(
                    safe_percentage(
                        top_frequency,
                        rows
                    ),
                    2
                )
            }
        )

    return pd.DataFrame(
        categorical_rows
    )