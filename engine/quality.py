# FILE: quality.py

import pandas as pd


def detect_outliers(df, numeric_df):
    outlier_columns = []
    total_outlier_values = 0
    anomaly_rows = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        count = len(outliers)

        if count > 0:
            outlier_columns.append(column)
            total_outlier_values += count

            anomaly_rows.append({
                "Attribute": column,
                "Anomaly Type": "Outliers",
                "Count": count,
                "Percentage": round(
                    (count / len(series)) * 100,
                    2
                ),
                "Lower Bound": round(lower_bound, 4),
                "Upper Bound": round(upper_bound, 4)
            })

    anomaly_df = pd.DataFrame(anomaly_rows)

    return (
        outlier_columns,
        total_outlier_values,
        anomaly_df
    )


def build_quality_findings(
    df,
    rows,
    columns,
    missing_cells,
    duplicate_rows,
    outlier_columns
):
    quality_findings = []

    if rows > 0:
        missing_ratio = missing_cells / (rows * columns)
    else:
        missing_ratio = 0

    if missing_cells > 0:
        severity = (
            "High"
            if missing_ratio >= 0.10
            else "Medium"
        )

        quality_findings.append({
            "Issue": "Missing Values",
            "Severity": severity,
            "Details": (
                f"{missing_cells:,} missing cells "
                f"({missing_ratio * 100:.2f}%)"
            )
        })

    if duplicate_rows > 0:
        duplicate_ratio = duplicate_rows / rows

        severity = (
            "High"
            if duplicate_ratio >= 0.10
            else "Medium"
        )

        quality_findings.append({
            "Issue": "Duplicate Rows",
            "Severity": severity,
            "Details": (
                f"{duplicate_rows:,} duplicate rows "
                f"({duplicate_ratio * 100:.2f}%)"
            )
        })

    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:
        quality_findings.append({
            "Issue": "Empty Columns",
            "Severity": "High",
            "Details": (
                f"{len(empty_columns)} completely empty "
                "column(s)"
            )
        })

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        quality_findings.append({
            "Issue": "Constant Columns",
            "Severity": "Medium",
            "Details": (
                f"{len(constant_columns)} column(s) "
                "contain only one unique value"
            )
        })

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    high_cardinality_columns = []

    for column in categorical_columns:
        unique_ratio = (
            df[column].nunique(dropna=True) / rows
            if rows > 0
            else 0
        )

        if unique_ratio >= 0.50:
            high_cardinality_columns.append(column)

    if high_cardinality_columns:
        quality_findings.append({
            "Issue": "High Cardinality",
            "Severity": "Medium",
            "Details": (
                f"{len(high_cardinality_columns)} categorical "
                "column(s) have very high uniqueness"
            )
        })

    if outlier_columns:
        quality_findings.append({
            "Issue": "Outliers",
            "Severity": "Medium",
            "Details": (
                f"{len(outlier_columns)} numerical "
                "column(s) contain outliers"
            )
        })

    return (
        quality_findings,
        empty_columns,
        constant_columns,
        high_cardinality_columns
    )


def calculate_quality_score(
    rows,
    columns,
    missing_cells,
    duplicate_rows,
    outlier_columns,
    constant_columns
):
    if rows > 0 and columns > 0:
        missing_ratio = (
            missing_cells / (rows * columns)
        )
    else:
        missing_ratio = 0

    if rows > 0:
        duplicate_ratio = duplicate_rows / rows
    else:
        duplicate_ratio = 0

    quality_score = 100

    quality_score -= min(
        missing_ratio * 60,
        30
    )

    quality_score -= min(
        duplicate_ratio * 40,
        15
    )

    quality_score -= min(
        len(outlier_columns) * 3,
        15
    )

    quality_score -= min(
        len(constant_columns) * 5,
        15
    )

    quality_score = int(
        max(
            0,
            min(
                100,
                round(quality_score)
            )
        )
    )

    if quality_score >= 90:
        quality_status = "Excellent"
    elif quality_score >= 75:
        quality_status = "Good"
    elif quality_score >= 50:
        quality_status = "Moderate"
    else:
        quality_status = "Poor"

    return quality_score, quality_status


def calculate_ml_readiness(
    rows,
    missing_cells,
    duplicate_rows,
    outlier_columns,
    constant_columns,
    identifier_columns,
    total_columns
):
    if rows > 0 and total_columns > 0:
        missing_ratio = (
            missing_cells /
            (rows * total_columns)
        )
    else:
        missing_ratio = 0

    if rows > 0:
        duplicate_ratio = duplicate_rows / rows
    else:
        duplicate_ratio = 0

    ml_score = 100

    if rows < 100:
        ml_score -= 15
    elif rows < 500:
        ml_score -= 8

    ml_score -= min(
        missing_ratio * 75,
        25
    )

    ml_score -= min(
        duplicate_ratio * 50,
        15
    )

    ml_score -= min(
        len(outlier_columns) * 2,
        10
    )

    ml_score -= min(
        len(constant_columns) * 5,
        15
    )

    ml_score -= min(
        len(identifier_columns) * 2,
        6
    )

    ml_score = int(
        max(
            0,
            min(
                100,
                round(ml_score)
            )
        )
    )

    if ml_score >= 90:
        ml_status = "Highly Ready"
    elif ml_score >= 75:
        ml_status = "Mostly Ready"
    elif ml_score >= 50:
        ml_status = "Needs Cleaning"
    else:
        ml_status = "Not Ready"

    return ml_score, ml_status
