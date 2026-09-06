# FILE: visualization.py

import pandas as pd
import plotly.express as px


def create_numerical_histogram(df, column):
    return px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )


def create_numerical_boxplot(df, column):
    return px.box(
        df,
        y=column,
        title=f"Box Plot of {column}"
    )


def create_correlation_heatmap(df, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

    if len(numeric_columns) < 2:
        return None

    correlation = df[numeric_columns].corr()

    return px.imshow(
        correlation,
        text_auto=True,
        title="Correlation Heatmap",
        aspect="auto"
    )


def create_scatter_matrix(df, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

    if len(numeric_columns) < 2:
        return None

    return px.scatter_matrix(
        df,
        dimensions=numeric_columns,
        title="Multidimensional Explorer"
    )


def create_categorical_frequency(df, column):
    counts = (
        df[column]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Category",
        "Frequency"
    ]

    return px.bar(
        counts,
        x="Category",
        y="Frequency",
        title=f"Frequency of {column}"
    )


def create_temporal_trend(df, date_column, value_column):
    temp = df[[date_column, value_column]].copy()

    temp[date_column] = pd.to_datetime(
        temp[date_column],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=[date_column]
    ).sort_values(date_column)

    return px.line(
        temp,
        x=date_column,
        y=value_column,
        title=f"Temporal Trend: {value_column}"
    )


def create_missing_value_chart(df):
    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        return None

    chart_df = missing.reset_index()
    chart_df.columns = [
        "Attribute",
        "Missing Values"
    ]

    return px.bar(
        chart_df,
        x="Attribute",
        y="Missing Values",
        title="Missing Value Analysis"
    )


def create_binary_class_balance(df, column):
    counts = (
        df[column]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Class",
        "Count"
    ]

    return px.bar(
        counts,
        x="Class",
        y="Count",
        title=f"Class Balance: {column}"
    )