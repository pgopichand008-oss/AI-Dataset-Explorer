import pandas as pd


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0
    return (numerator / denominator) * 100


def infer_column_role(df, column):
    series = df[column]

    # Date / Time
    if pd.api.types.is_datetime64_any_dtype(series):
        return (
            "Date / Time",
            0.95,
            False,
            "Detected datetime column"
        )

    # Try detecting date-like object columns
    if series.dtype == "object":
        try:
            converted = pd.to_datetime(
                series.dropna(),
                errors="coerce"
            )

            if len(converted) > 0:
                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.8:
                    return (
                        "Date / Time",
                        0.85,
                        False,
                        "Values appear to represent dates or timestamps"
                    )
        except Exception:
            pass

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):

        unique_count = series.nunique(dropna=True)
        total_count = len(series)

        if total_count > 0:
            unique_ratio = unique_count / total_count
        else:
            unique_ratio = 0

        # Possible target
        if unique_count == 2:
            return (
                "Numerical",
                0.90,
                True,
                "Binary numeric column; possible classification target"
            )

        if unique_count <= 10:
            return (
                "Numerical",
                0.85,
                True,
                "Low-cardinality numerical column; possible target"
            )

        return (
            "Numerical",
            0.90,
            False,
            "Continuous numerical feature"
        )

    # Categorical columns
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):

        unique_count = series.nunique(dropna=True)
        total_count = len(series)

        if total_count > 0:
            unique_ratio = unique_count / total_count
        else:
            unique_ratio = 0

        # Identifier-like column
        if unique_ratio >= 0.95 and unique_count > 10:
            return (
                "Identifier",
                0.90,
                False,
                "Very high uniqueness suggests an identifier column"
            )

        # Possible target
        if 2 <= unique_count <= 10:
            return (
                "Categorical",
                0.85,
                True,
                "Low-cardinality categorical column; possible target"
            )

        return (
            "Categorical",
            0.90,
            False,
            "Categorical feature"
        )

    # Fallback
    return (
        "Other",
        0.50,
        False,
        "Column type could not be confidently classified"
    )


def clean_ai_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("```markdown", "")
    text = text.replace("```text", "")
    text = text.replace("```", "")

    return text.strip()


def split_ai_sections(text):
    sections = {}

    if not text:
        return sections

    current_section = "General"
    sections[current_section] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()

            if heading:
                current_section = heading
                sections[current_section] = []

        else:
            sections[current_section].append(line)

    for key in sections:
        sections[key] = "\n".join(sections[key])

    return sections