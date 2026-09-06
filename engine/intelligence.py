# FILE: intelligence.py

def build_agent_decisions(
    df,
    missing_cells,
    rows,
    outlier_columns,
    identifier_columns,
    date_columns,
    categorical_columns
):
    decisions = []

    if rows > 0:
        missing_ratio = missing_cells / (rows * len(df.columns))
    else:
        missing_ratio = 0

    if missing_ratio >= 0.10:
        decisions.append({
            "Priority": "High",
            "Finding": "Significant missing data",
            "Action": "Investigate missing-value patterns and consider suitable imputation."
        })
    elif missing_ratio > 0:
        decisions.append({
            "Priority": "Medium",
            "Finding": "Some missing data detected",
            "Action": "Review missing values before advanced analysis."
        })

    if outlier_columns:
        decisions.append({
            "Priority": "Medium",
            "Finding": "Potential outliers detected",
            "Action": (
                f"Review outliers in {len(outlier_columns)} "
                "numerical column(s) before modeling."
            )
        })

    if identifier_columns:
        decisions.append({
            "Priority": "Medium",
            "Finding": "Identifier-like columns detected",
            "Action": (
                "Avoid using identifier columns directly as predictive "
                "features unless there is a clear reason."
            )
        })

    if date_columns:
        decisions.append({
            "Priority": "Low",
            "Finding": "Date/time columns detected",
            "Action": (
                "Consider temporal trends, seasonality, or date-based "
                "feature engineering."
            )
        })

    if len(categorical_columns) > 0:
        decisions.append({
            "Priority": "Low",
            "Finding": "Categorical attributes detected",
            "Action": (
                "Review category frequency and encoding requirements "
                "before machine learning."
            )
        })

    if not decisions:
        decisions.append({
            "Priority": "Low",
            "Finding": "No major issues detected",
            "Action": "Dataset can proceed to deeper analysis."
        })

    return decisions


def build_visualization_plan(
    df,
    numeric_columns,
    categorical_columns,
    date_columns
):
    plan = []

    for column in numeric_columns:
        plan.append({
            "Attribute": column,
            "Visualization": "Histogram + Box Plot",
            "Purpose": "Understand distribution and detect potential outliers."
        })

    for column in categorical_columns:
        unique_count = df[column].nunique(dropna=True)

        if unique_count == 2:
            visualization = "Class Balance"
            purpose = "Compare the distribution of the two classes."
        else:
            visualization = "Frequency Chart"
            purpose = "Understand category frequency."

        plan.append({
            "Attribute": column,
            "Visualization": visualization,
            "Purpose": purpose
        })

    for column in date_columns:
        plan.append({
            "Attribute": column,
            "Visualization": "Temporal Trend",
            "Purpose": "Explore changes over time."
        })

    if len(numeric_columns) >= 2:
        plan.append({
            "Attribute": "Numerical Variables",
            "Visualization": "Correlation Heatmap",
            "Purpose": "Identify relationships between numerical variables."
        })

        plan.append({
            "Attribute": "Numerical Variables",
            "Visualization": "Scatter Matrix",
            "Purpose": "Explore multidimensional relationships."
        })

    if df.isna().sum().sum() > 0:
        plan.append({
            "Attribute": "Missing Values",
            "Visualization": "Missing Value Chart",
            "Purpose": "Identify attributes with the most missing data."
        })

    return plan


def build_executive_summary(
    quality_score,
    ml_score,
    quality_findings,
    outlier_columns,
    identifier_columns,
    target_candidates
):
    if quality_score >= 75 and ml_score >= 75:
        risk_level = "Low"
    elif quality_score >= 50 and ml_score >= 50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    if quality_findings:
        most_important_finding = quality_findings[0]["Issue"]
    else:
        most_important_finding = "No major data-quality issue detected"

    if target_candidates:
        key_opportunity = (
            f"Potential target variable detected: "
            f"{target_candidates[0]}"
        )
    elif identifier_columns:
        key_opportunity = (
            "Review identifier columns before using the dataset "
            "for machine learning."
        )
    else:
        key_opportunity = (
            "Dataset can be explored further using statistical "
            "and visual analysis."
        )

    if quality_score >= 75 and ml_score >= 75:
        next_action = "Proceed with further analysis."
    elif quality_score >= 50 or ml_score >= 50:
        next_action = "Perform data cleaning and validation before modeling."
    else:
        next_action = "Correct major data-quality issues before analysis."

    return {
        "Risk Level": risk_level,
        "Quality Score": quality_score,
        "ML Readiness Score": ml_score,
        "Most Important Finding": most_important_finding,
        "Key Opportunity": key_opportunity,
        "Next Best Action": next_action,
        "Outlier Columns": outlier_columns,
        "Identifier Columns": identifier_columns
    }