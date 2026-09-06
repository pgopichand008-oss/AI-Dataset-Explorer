import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance


# =========================================================
# TARGET DETECTION
# =========================================================

def detect_target_candidates(df):
    """
    Detect possible target columns for machine learning.
    """

    candidates = []

    if df is None or df.empty:
        return candidates

    for column in df.columns:

        series = df[column]

        if series.isna().all():
            continue

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)

        confidence = 0
        reasons = []
        problem_type = None

        # -------------------------------------------------
        # NUMERICAL TARGET
        # -------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            if unique_count == 2:

                confidence = 85
                problem_type = "Classification"

                reasons.append(
                    "Binary numerical target"
                )

            elif 2 < unique_count <= 10:

                confidence = 75
                problem_type = "Classification"

                reasons.append(
                    "Low-cardinality numerical column"
                )

            elif unique_count > 10:

                confidence = 60
                problem_type = "Regression"

                reasons.append(
                    "Continuous numerical column"
                )

        # -------------------------------------------------
        # CATEGORICAL TARGET
        # -------------------------------------------------

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):

            if unique_ratio >= 0.95 and unique_count > 10:

                confidence = 0

                reasons.append(
                    "Likely identifier; excluded"
                )

            elif 2 <= unique_count <= 10:

                confidence = 80
                problem_type = "Classification"

                reasons.append(
                    "Low-cardinality categorical column"
                )

            else:

                confidence = 30
                problem_type = "Classification"

                reasons.append(
                    "Categorical column with many classes"
                )

        # -------------------------------------------------
        # TARGET KEYWORDS
        # -------------------------------------------------

        column_name = str(column).lower()

        target_keywords = [
            "target",
            "label",
            "class",
            "outcome",
            "result",
            "status",
            "category",
            "churn",
            "default",
            "prediction",
            "salary",
            "price",
            "sales",
            "score"
        ]

        if any(
            keyword in column_name
            for keyword in target_keywords
        ):

            confidence += 15

            reasons.append(
                "Column name suggests a target"
            )

        # -------------------------------------------------
        # IDENTIFIER DETECTION
        # -------------------------------------------------

        identifier_keywords = [
            "id",
            "identifier",
            "uuid",
            "code",
            "index"
        ]

        if any(
            keyword == column_name.strip()
            or column_name.endswith("_" + keyword)
            or column_name.startswith(keyword + "_")
            for keyword in identifier_keywords
        ):

            confidence -= 40

            reasons.append(
                "Column name suggests an identifier"
            )

        confidence = max(
            0,
            min(100, confidence)
        )

        if (
            confidence >= 40
            and problem_type is not None
        ):

            candidates.append({
                "column": column,
                "confidence": confidence,
                "problem_type": problem_type,
                "reason": "; ".join(reasons)
            })

    candidates.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return candidates


def get_best_target_candidate(df):
    """
    Return the strongest target candidate.
    """

    candidates = detect_target_candidates(df)

    if not candidates:
        return None

    return candidates[0]


def determine_problem_type(df, target_column):
    """
    Determine Classification or Regression.
    """

    if (
        df is None
        or target_column not in df.columns
    ):
        return None

    series = df[target_column].dropna()

    if series.empty:
        return None

    unique_count = series.nunique()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):

        return "Classification"

    if pd.api.types.is_numeric_dtype(series):

        if unique_count <= 10:
            return "Classification"

        return "Regression"

    return None


def get_ml_target_summary(df):
    """
    Generate target detection summary.
    """

    candidates = detect_target_candidates(df)

    best_candidate = (
        candidates[0]
        if candidates
        else None
    )

    if best_candidate:

        target_column = best_candidate["column"]

        problem_type = determine_problem_type(
            df,
            target_column
        )

        return {
            "target_found": True,
            "target_column": target_column,
            "confidence": best_candidate["confidence"],
            "problem_type": problem_type,
            "reason": best_candidate["reason"],
            "candidates": candidates
        }

    return {
        "target_found": False,
        "target_column": None,
        "confidence": 0,
        "problem_type": None,
        "reason": "No suitable target column was detected.",
        "candidates": []
    }


# =========================================================
# ML READINESS GATE
# =========================================================

def assess_ml_readiness(
    df,
    target_column=None,
    identifier_columns=None
):
    """
    Decide whether the dataset is suitable for
    experimental machine learning.

    This function does NOT train a model.
    """

    if df is None or df.empty:

        return {
            "score": 0,
            "status": "NOT READY",
            "can_train": False,
            "warnings": [
                "Dataset is empty."
            ],
            "strengths": [],
            "details": {}
        }

    if identifier_columns is None:
        identifier_columns = []

    warnings = []
    strengths = []

    rows = len(df)
    columns = len(df.columns)

    # -------------------------------------------------
    # TARGET
    # -------------------------------------------------

    if target_column is None:

        target_summary = get_ml_target_summary(df)

        if target_summary["target_found"]:

            target_column = (
                target_summary["target_column"]
            )

    if target_column is None:

        warnings.append(
            "No suitable target column was detected."
        )

    elif target_column not in df.columns:

        warnings.append(
            "Selected target column does not exist."
        )

        target_column = None

    else:

        strengths.append(
            "A valid target column is available."
        )

    # -------------------------------------------------
    # ROW COUNT
    # -------------------------------------------------

    if rows < 30:

        warnings.append(
            f"Very small dataset: only {rows} rows."
        )

    elif rows < 100:

        warnings.append(
            f"Small dataset: {rows} rows."
        )

    else:

        strengths.append(
            f"Dataset contains {rows} rows."
        )

    # -------------------------------------------------
    # FEATURES
    # -------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    usable_features = [
        column
        for column in feature_columns
        if column not in identifier_columns
        and not df[column].isna().all()
    ]

    if len(usable_features) == 0:

        warnings.append(
            "No usable feature columns were found."
        )

    else:

        strengths.append(
            f"{len(usable_features)} usable feature columns detected."
        )

    # -------------------------------------------------
    # MISSING VALUES
    # -------------------------------------------------

    total_cells = df.size

    missing_cells = int(
        df.isna().sum().sum()
    )

    if total_cells > 0:

        missing_percentage = (
            missing_cells / total_cells
        ) * 100

    else:

        missing_percentage = 100

    if missing_percentage >= 30:

        warnings.append(
            f"High missing-value rate: "
            f"{missing_percentage:.1f}%."
        )

    elif missing_percentage > 10:

        warnings.append(
            f"Moderate missing-value rate: "
            f"{missing_percentage:.1f}%."
        )

    else:

        strengths.append(
            f"Missing-value rate is low: "
            f"{missing_percentage:.1f}%."
        )

    # -------------------------------------------------
    # DUPLICATES
    # -------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows > 0:

        duplicate_percentage = (
            duplicate_rows / rows
        ) * 100

        warnings.append(
            f"{duplicate_rows} duplicate rows detected "
            f"({duplicate_percentage:.1f}%)."
        )

    else:

        strengths.append(
            "No duplicate rows detected."
        )

    # -------------------------------------------------
    # TARGET QUALITY
    # -------------------------------------------------

    target_unique_values = None
    target_missing_percentage = None
    target_class_balance = None
    target_problem_type = None

    if target_column is not None:

        target_series = df[target_column]

        target_missing = int(
            target_series.isna().sum()
        )

        target_missing_percentage = (
            target_missing / rows
        ) * 100

        target_unique_values = (
            target_series.dropna().nunique()
        )

        target_problem_type = determine_problem_type(
            df,
            target_column
        )

        if target_missing_percentage > 20:

            warnings.append(
                f"Target column has "
                f"{target_missing_percentage:.1f}% missing values."
            )

        elif target_missing_percentage > 0:

            warnings.append(
                f"Target column contains "
                f"{target_missing_percentage:.1f}% missing values."
            )

        else:

            strengths.append(
                "Target column has no missing values."
            )

        # Classification balance
        if target_problem_type == "Classification":

            value_counts = (
                target_series
                .dropna()
                .value_counts(normalize=True)
            )

            if not value_counts.empty:

                target_class_balance = (
                    value_counts.min() * 100
                )

                if target_class_balance < 5:

                    warnings.append(
                        "Target classes are highly imbalanced."
                    )

                elif target_class_balance < 20:

                    warnings.append(
                        "Target classes show some imbalance."
                    )

                else:

                    strengths.append(
                        "Target class distribution is reasonably balanced."
                    )

            if target_unique_values < 2:

                warnings.append(
                    "Target contains fewer than two classes."
                )

    # -------------------------------------------------
    # IDENTIFIERS
    # -------------------------------------------------

    if identifier_columns:

        strengths.append(
            f"{len(identifier_columns)} identifier "
            f"column(s) can be excluded from training."
        )

    # -------------------------------------------------
    # FEATURE / ROW RATIO
    # -------------------------------------------------

    if rows > 0 and len(usable_features) > 0:

        feature_row_ratio = (
            len(usable_features) / rows
        )

        if feature_row_ratio > 0.5:

            warnings.append(
                "High feature-to-row ratio may cause "
                "unstable model performance."
            )

    # -------------------------------------------------
    # SCORE
    # -------------------------------------------------

    score = 100

    if rows < 30:
        score -= 35

    elif rows < 100:
        score -= 20

    if target_column is None:

        score -= 40

    elif target_missing_percentage is not None:

        if target_missing_percentage >= 20:
            score -= 25

        elif target_missing_percentage > 0:
            score -= 10

    if missing_percentage >= 30:

        score -= 25

    elif missing_percentage > 10:

        score -= 10

    if duplicate_rows > 0:
        score -= 5

    if len(usable_features) == 0:
        score -= 30

    if target_problem_type == "Classification":

        if target_unique_values is not None:

            if target_unique_values < 2:

                score -= 30

            elif target_unique_values > 20:

                score -= 10

    if rows > 0 and len(usable_features) > 0:

        if len(usable_features) / rows > 0.5:

            score -= 10

    score = max(
        0,
        min(100, score)
    )

    # -------------------------------------------------
    # FINAL TRAINING DECISION
    # -------------------------------------------------

    if target_column is None:

        status = "NOT READY"
        can_train = False

    elif (
        target_unique_values is not None
        and target_unique_values < 2
    ):

        status = "NOT READY"
        can_train = False

    elif score >= 75:

        status = "READY"
        can_train = True

    elif score >= 50:

        status = "CAUTION"
        can_train = True

    else:

        status = "NOT READY"
        can_train = False

    return {
        "score": score,
        "status": status,
        "can_train": can_train,
        "warnings": warnings,
        "strengths": strengths,
        "details": {
            "rows": rows,
            "columns": columns,
            "usable_features": len(usable_features),
            "missing_cells": missing_cells,
            "missing_percentage": round(
                missing_percentage,
                2
            ),
            "duplicate_rows": duplicate_rows,
            "target_column": target_column,
            "target_problem_type": target_problem_type,
            "target_unique_values": target_unique_values,
            "target_missing_percentage": (
                round(
                    target_missing_percentage,
                    2
                )
                if target_missing_percentage is not None
                else None
            ),
            "target_class_balance": (
                round(
                    target_class_balance,
                    2
                )
                if target_class_balance is not None
                else None
            ),
            "identifier_columns": identifier_columns
        }
    }


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessing_pipeline(
    df,
    target_column,
    identifier_columns=None
):
    """
    Build preprocessing pipeline for numerical
    and categorical features.
    """

    if df is None or df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if identifier_columns is None:
        identifier_columns = []

    X = df.drop(
        columns=[target_column]
    ).copy()

    # Remove identifiers
    columns_to_remove = [
        column
        for column in identifier_columns
        if column in X.columns
    ]

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    # Remove empty columns
    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        X = X.drop(
            columns=empty_columns
        )

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:

        raise ValueError(
            "No supported feature columns found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def prepare_ml_data(
    df,
    target_column,
    identifier_columns=None,
    test_size=0.2,
    random_state=42
):
    """
    Prepare X/y and split into training and testing data.
    """

    if df is None or df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if identifier_columns is None:
        identifier_columns = []

    # Remove rows with missing target
    working_df = df.dropna(
        subset=[target_column]
    ).copy()

    if working_df.empty:

        raise ValueError(
            "No rows remain after removing missing target values."
        )

    y = working_df[target_column].copy()

    X = working_df.drop(
        columns=[target_column]
    ).copy()

    # Remove identifiers
    columns_to_remove = [
        column
        for column in identifier_columns
        if column in X.columns
    ]

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    # Remove completely empty columns
    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        X = X.drop(
            columns=empty_columns
        )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns remain."
        )

    problem_type = determine_problem_type(
        working_df,
        target_column
    )

    stratify_value = None

    if problem_type == "Classification":

        class_counts = y.value_counts()

        if (
            len(class_counts) >= 2
            and class_counts.min() >= 2
        ):

            stratify_value = y

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_value
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

    preprocessor = build_preprocessing_pipeline(
        working_df,
        target_column,
        identifier_columns
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "problem_type": problem_type,
        "target_column": target_column,
        "removed_identifier_columns": columns_to_remove,
        "removed_empty_columns": empty_columns
    }


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_classification_models():
    """
    Return lightweight classification models.
    """

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    }


def get_regression_models():
    """
    Return lightweight regression models.
    """

    return {

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    }


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_classification_model(
    model,
    X_test,
    y_test
):
    """
    Calculate classification metrics.
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {

        "accuracy": round(
            accuracy,
            4
        ),

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1": round(
            f1,
            4
        )

    }


def evaluate_regression_model(
    model,
    X_test,
    y_test
):
    """
    Calculate regression metrics.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "mae": round(
            mae,
            4
        ),

        "rmse": round(
            rmse,
            4
        ),

        "r2": round(
            r2,
            4
        )

    }


# =========================================================
# TRAIN + COMPARE MODELS
# =========================================================

def train_and_compare_models(
    df,
    target_column,
    identifier_columns=None,
    test_size=0.2,
    random_state=42
):
    """
    Train multiple suitable ML models and compare them.
    """

    if identifier_columns is None:
        identifier_columns = []

    readiness = assess_ml_readiness(
        df,
        target_column,
        identifier_columns
    )

    if not readiness["can_train"]:

        return {

            "success": False,

            "message": (
                "Dataset is not ready for machine learning."
            ),

            "readiness": readiness,

            "results": [],

            "best_model": None,

            "models": {}

        }

    prepared = prepare_ml_data(
        df,
        target_column,
        identifier_columns,
        test_size,
        random_state
    )

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]

    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    preprocessor = prepared["preprocessor"]

    problem_type = prepared["problem_type"]

    # -------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------

    if problem_type == "Classification":

        model_definitions = (
            get_classification_models()
        )

    elif problem_type == "Regression":

        model_definitions = (
            get_regression_models()
        )

    else:

        return {

            "success": False,

            "message": (
                "Unsupported ML problem type."
            ),

            "readiness": readiness,

            "results": [],

            "best_model": None,

            "models": {}

        }

    results = []

    trained_models = {}

    # -------------------------------------------------
    # TRAIN MODELS
    # -------------------------------------------------

    for model_name, model in model_definitions.items():

        try:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    )
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            # -----------------------------------------
            # CLASSIFICATION
            # -----------------------------------------

            if problem_type == "Classification":

                metrics = evaluate_classification_model(
                    pipeline,
                    X_test,
                    y_test
                )

                comparison_score = (
                    metrics["f1"]
                )

            # -----------------------------------------
            # REGRESSION
            # -----------------------------------------

            else:

                metrics = evaluate_regression_model(
                    pipeline,
                    X_test,
                    y_test
                )

                comparison_score = (
                    metrics["r2"]
                )

            results.append({

                "model": model_name,

                "problem_type": problem_type,

                "metrics": metrics,

                "comparison_score": round(
                    comparison_score,
                    4
                ),

                "status": "Success"

            })

            trained_models[
                model_name
            ] = pipeline

        except Exception as error:

            results.append({

                "model": model_name,

                "problem_type": problem_type,

                "metrics": {},

                "comparison_score": None,

                "status": "Failed",

                "error": str(error)

            })

    # -------------------------------------------------
    # FIND BEST MODEL
    # -------------------------------------------------

    successful_results = [

        result

        for result in results

        if (
            result["status"] == "Success"
            and result["comparison_score"] is not None
        )

    ]

    if not successful_results:

        return {

            "success": False,

            "message": (
                "All candidate models failed."
            ),

            "readiness": readiness,

            "results": results,

            "best_model": None,

            "models": trained_models

        }

    best_result = max(
        successful_results,
        key=lambda x: x["comparison_score"]
    )

    best_model_name = best_result[
        "model"
    ]

    return {

        "success": True,

        "message": (
            "Models trained and compared successfully."
        ),

        "readiness": readiness,

        "problem_type": problem_type,

        "target_column": target_column,

        "results": results,

        "best_model": best_result,

        "models": trained_models,

        "training_data": {

            "X_test": X_test,

            "y_test": y_test

        },

        "training_details": {

            "training_rows": len(X_train),

            "testing_rows": len(X_test),

            "feature_count": X_train.shape[1],

            "removed_identifier_columns": prepared[
                "removed_identifier_columns"
            ],

            "removed_empty_columns": prepared[
                "removed_empty_columns"
            ]

        }

    }


# =========================================================
# PERMUTATION FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(
    ml_result,
    top_n=10,
    n_repeats=5
):
    """
    Calculate permutation feature importance on the
    held-out test dataset.

    Importance is calculated at the original input-column
    level, making the result easier to explain.

    Positive importance:
        Feature helps the model.

    Near-zero importance:
        Feature has little measurable effect.

    Negative importance:
        Shuffling the feature unexpectedly improved the
        score on this test split.
    """

    if not ml_result:
        return []

    if not ml_result.get("success"):
        return []

    best_model_info = ml_result.get(
        "best_model"
    )

    if not best_model_info:
        return []

    best_model_name = best_model_info.get(
        "model"
    )

    trained_models = ml_result.get(
        "models",
        {}
    )

    pipeline = trained_models.get(
        best_model_name
    )

    if pipeline is None:
        return []

    training_data = ml_result.get(
        "training_data",
        {}
    )

    X_test = training_data.get(
        "X_test"
    )

    y_test = training_data.get(
        "y_test"
    )

    if X_test is None or y_test is None:
        return []

    if X_test.empty:
        return []

    try:

        problem_type = ml_result.get(
            "problem_type"
        )

        # -----------------------------------------
        # SELECT SCORING METHOD
        # -----------------------------------------

        if problem_type == "Classification":

            scoring = "f1_weighted"

        elif problem_type == "Regression":

            scoring = "r2"

        else:

            return []

        # -----------------------------------------
        # PERMUTATION IMPORTANCE
        # -----------------------------------------

        importance_result = permutation_importance(

            pipeline,

            X_test,

            y_test,

            scoring=scoring,

            n_repeats=n_repeats,

            random_state=42,

            n_jobs=-1

        )

        importance_df = pd.DataFrame({

            "feature": X_test.columns,

            "importance": (
                importance_result.importances_mean
            ),

            "std": (
                importance_result.importances_std
            )

        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(top_n)
        )

        importance_df["importance"] = (
            importance_df["importance"]
            .round(6)
        )

        importance_df["std"] = (
            importance_df["std"]
            .round(6)
        )

        # -----------------------------------------
        # INTERPRETATION
        # -----------------------------------------

        interpretation = []

        for _, row in importance_df.iterrows():

            importance_value = row[
                "importance"
            ]

            if importance_value > 0.01:

                interpretation_text = (
                    "Meaningful model contribution"
                )

            elif importance_value > 0:

                interpretation_text = (
                    "Small positive contribution"
                )

            elif importance_value < 0:

                interpretation_text = (
                    "Unstable or potentially non-useful "
                    "on this test split"
                )

            else:

                interpretation_text = (
                    "Little measurable contribution"
                )

            interpretation.append(
                interpretation_text
            )

        importance_df[
            "interpretation"
        ] = interpretation

        return importance_df.to_dict(
            orient="records"
        )

    except Exception:

        return []


# =========================================================
# PREDICTION ENGINE
# =========================================================

def generate_predictions(
    ml_result,
    X_input=None
):
    """
    Generate predictions using the selected best model.

    If X_input is None, predictions are generated for
    the held-out test data.

    Classification:
        Returns predicted class and probability when
        the model supports predict_proba().

    Regression:
        Returns predicted numerical values.
    """

    if not ml_result:
        return []

    if not ml_result.get("success"):
        return []

    best_model_info = ml_result.get(
        "best_model"
    )

    if not best_model_info:
        return []

    best_model_name = best_model_info.get(
        "model"
    )

    pipeline = ml_result.get(
        "models",
        {}
    ).get(
        best_model_name
    )

    if pipeline is None:
        return []

    if X_input is None:

        X_input = (
            ml_result
            .get("training_data", {})
            .get("X_test")
        )

    if X_input is None or X_input.empty:

        return []

    try:

        predictions = pipeline.predict(
            X_input
        )

        problem_type = ml_result.get(
            "problem_type"
        )

        output = []

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        if problem_type == "Classification":

            probabilities = None

            if hasattr(
                pipeline,
                "predict_proba"
            ):

                try:

                    probabilities = (
                        pipeline.predict_proba(
                            X_input
                        )
                    )

                except Exception:

                    probabilities = None

            classes = None

            if hasattr(
                pipeline,
                "classes_"
            ):

                classes = (
                    pipeline.classes_
                )

            for index, prediction in enumerate(
                predictions
            ):

                row = {
                    "prediction": prediction
                }

                if (
                    probabilities is not None
                    and classes is not None
                ):

                    max_probability = float(
                        np.max(
                            probabilities[index]
                        )
                    )

                    predicted_class_index = int(
                        np.argmax(
                            probabilities[index]
                        )
                    )

                    predicted_class = classes[
                        predicted_class_index
                    ]

                    row[
                        "prediction"
                    ] = predicted_class

                    row[
                        "probability"
                    ] = round(
                        max_probability,
                        4
                    )

                    row[
                        "confidence_level"
                    ] = classify_confidence(
                        max_probability
                    )

                output.append(row)

        # -------------------------------------------------
        # REGRESSION
        # -------------------------------------------------

        else:

            for prediction in predictions:

                output.append({

                    "prediction": round(
                        float(prediction),
                        4
                    )

                })

        return output

    except Exception:

        return []


# =========================================================
# CONFIDENCE INTERPRETATION
# =========================================================

def classify_confidence(probability):
    """
    Convert classification probability into a simple
    human-readable confidence level.

    This is a model-output interpretation, not a guarantee
    that the probability is perfectly calibrated.
    """

    if probability >= 0.85:

        return "High"

    if probability >= 0.65:

        return "Moderate"

    return "Low"


# =========================================================
# ML RELIABILITY ASSESSMENT
# =========================================================

def assess_ml_reliability(
    ml_result,
    feature_importance=None
):
    """
    Assess how cautiously the ML results should be treated.

    This does NOT claim statistical certainty.
    It provides an interpretable reliability warning layer.
    """

    if not ml_result:

        return {

            "level": "UNKNOWN",

            "score": 0,

            "warnings": [
                "No ML result is available."
            ]

        }

    if not ml_result.get("success"):

        return {

            "level": "NOT RELIABLE",

            "score": 0,

            "warnings": [
                ml_result.get(
                    "message",
                    "ML analysis failed."
                )
            ]

        }

    readiness = ml_result.get(
        "readiness",
        {}
    )

    readiness_score = readiness.get(
        "score",
        0
    )

    warnings = []

    score = 100

    # -------------------------------------------------
    # READINESS
    # -------------------------------------------------

    if readiness_score < 50:

        score -= 40

        warnings.append(
            "Dataset readiness is low."
        )

    elif readiness_score < 75:

        score -= 20

        warnings.append(
            "Dataset has moderate ML-readiness."
        )

    # -------------------------------------------------
    # DATASET SIZE
    # -------------------------------------------------

    training_details = ml_result.get(
        "training_details",
        {}
    )

    training_rows = training_details.get(
        "training_rows",
        0
    )

    testing_rows = training_details.get(
        "testing_rows",
        0
    )

    if training_rows < 30:

        score -= 30

        warnings.append(
            "Very few training rows may make "
            "performance estimates unstable."
        )

    elif training_rows < 100:

        score -= 15

        warnings.append(
            "Limited training data may reduce "
            "generalization reliability."
        )

    if testing_rows < 10:

        score -= 15

        warnings.append(
            "Very small test set makes evaluation "
            "less reliable."
        )

    # -------------------------------------------------
    # BEST MODEL SCORE
    # -------------------------------------------------

    best_model = ml_result.get(
        "best_model",
        {}
    )

    comparison_score = best_model.get(
        "comparison_score"
    )

    problem_type = ml_result.get(
        "problem_type"
    )

    if comparison_score is not None:

        if problem_type == "Classification":

            if comparison_score < 0.50:

                score -= 25

                warnings.append(
                    "Best model F1 score is low."
                )

            elif comparison_score < 0.70:

                score -= 10

                warnings.append(
                    "Best model performance is moderate."
                )

        elif problem_type == "Regression":

            if comparison_score < 0:

                score -= 25

                warnings.append(
                    "Best model R² is below zero."
                )

            elif comparison_score < 0.50:

                score -= 15

                warnings.append(
                    "Best model explains limited target variation."
                )

    # -------------------------------------------------
    # FEATURE IMPORTANCE STABILITY
    # -------------------------------------------------

    if feature_importance:

        unstable_features = [

            item

            for item in feature_importance

            if item.get("std", 0)
            > abs(
                item.get("importance", 0)
            )
            and item.get("importance", 0) != 0

        ]

        if unstable_features:

            score -= 10

            warnings.append(
                "Some feature importance estimates "
                "are unstable across permutations."
            )

    score = max(
        0,
        min(100, score)
    )

    # -------------------------------------------------
    # FINAL RELIABILITY
    # -------------------------------------------------

    if score >= 75:

        level = "HIGH"

    elif score >= 50:

        level = "MODERATE"

    else:

        level = "LOW"

    return {

        "level": level,

        "score": score,

        "warnings": warnings

    }


# =========================================================
# FINAL ML RECOMMENDATION
# =========================================================

def generate_ml_recommendation(
    ml_result,
    reliability=None
):
    """
    Produce a final recommendation for the ML result.

    Possible outcomes:

    🟢 ML RESULTS SUITABLE FOR FURTHER ANALYSIS
    🟡 ML RESULTS SHOULD BE USED WITH CAUTION
    🔴 ML RESULTS ARE NOT RELIABLE ENOUGH
    """

    if not ml_result:

        return {

            "status": (
                "🔴 ML RESULTS ARE NOT RELIABLE ENOUGH"
            ),

            "reason": (
                "No ML analysis result is available."
            )

        }

    if not ml_result.get("success"):

        return {

            "status": (
                "🔴 ML RESULTS ARE NOT RELIABLE ENOUGH"
            ),

            "reason": ml_result.get(
                "message",
                "ML analysis failed."
            )

        }

    if reliability is None:

        reliability = assess_ml_reliability(
            ml_result
        )

    reliability_level = reliability.get(
        "level",
        "LOW"
    )

    readiness = ml_result.get(
        "readiness",
        {}
    )

    readiness_score = readiness.get(
        "score",
        0
    )

    best_model = ml_result.get(
        "best_model",
        {}
    )

    comparison_score = best_model.get(
        "comparison_score"
    )

    problem_type = ml_result.get(
        "problem_type"
    )

    # -------------------------------------------------
    # LOW RELIABILITY
    # -------------------------------------------------

    if reliability_level == "LOW":

        return {

            "status": (
                "🔴 ML RESULTS ARE NOT RELIABLE ENOUGH"
            ),

            "reason": (
                "The dataset or evaluation conditions "
                "are too weak to confidently rely on "
                "the current model results."
            )

        }

    # -------------------------------------------------
    # MODERATE RELIABILITY
    # -------------------------------------------------

    if reliability_level == "MODERATE":

        return {

            "status": (
                "🟡 ML RESULTS SHOULD BE USED WITH CAUTION"
            ),

            "reason": (
                "The model produced usable results, "
                "but dataset size, quality, or evaluation "
                "limitations reduce confidence."
            )

        }

    # -------------------------------------------------
    # HIGH RELIABILITY
    # -------------------------------------------------

    if (
        readiness_score >= 75
        and reliability_level == "HIGH"
    ):

        return {

            "status": (
                "🟢 ML RESULTS SUITABLE FOR FURTHER ANALYSIS"
            ),

            "reason": (
                f"The dataset passed the ML-readiness "
                f"checks and the selected "
                f"{problem_type.lower() if problem_type else 'ML'} "
                f"model produced acceptable evaluation results."
            )

        }

    return {

        "status": (
            "🟡 ML RESULTS SHOULD BE USED WITH CAUTION"
        ),

        "reason": (
            "The model completed successfully, but "
            "additional validation is recommended before "
            "treating the results as dependable."
        )

    }


# =========================================================
# COMPLETE ML ANALYSIS
# =========================================================

def run_ml_analysis(
    df,
    target_column=None,
    identifier_columns=None,
    test_size=0.2,
    random_state=42
):
    """
    Complete ML intelligence pipeline.

    Flow:

    Target Detection
        ↓
    ML Readiness
        ↓
    Preprocessing
        ↓
    Model Training
        ↓
    Model Comparison
        ↓
    Best Model
        ↓
    Feature Importance
        ↓
    Predictions
        ↓
    Reliability
        ↓
    Final Recommendation
    """

    if df is None or df.empty:

        return {

            "success": False,

            "message": "Dataset is empty.",

            "target_summary": {},

            "readiness": {},

            "model_result": {},

            "feature_importance": [],

            "predictions": [],

            "reliability": {},

            "recommendation": {}

        }

    # -------------------------------------------------
    # TARGET DETECTION
    # -------------------------------------------------

    target_summary = get_ml_target_summary(
        df
    )

    if target_column is None:

        target_column = (
            target_summary.get(
                "target_column"
            )
        )

    if target_column is None:

        return {

            "success": False,

            "message": (
                "No suitable target column was detected."
            ),

            "target_summary": target_summary,

            "readiness": {},

            "model_result": {},

            "feature_importance": [],

            "predictions": [],

            "reliability": {},

            "recommendation": {}

        }

    # -------------------------------------------------
    # TRAIN + COMPARE
    # -------------------------------------------------

    model_result = train_and_compare_models(

        df,

        target_column,

        identifier_columns,

        test_size,

        random_state

    )

    if not model_result.get("success"):

        return {

            "success": False,

            "message": model_result.get(
                "message",
                "ML analysis failed."
            ),

            "target_summary": target_summary,

            "readiness": model_result.get(
                "readiness",
                {}
            ),

            "model_result": model_result,

            "feature_importance": [],

            "predictions": [],

            "reliability": {},

            "recommendation": {}

        }

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    feature_importance = get_feature_importance(

        model_result,

        top_n=10,

        n_repeats=5

    )

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------

    predictions = generate_predictions(
        model_result
    )

    # -------------------------------------------------
    # RELIABILITY
    # -------------------------------------------------

    reliability = assess_ml_reliability(

        model_result,

        feature_importance

    )

    # -------------------------------------------------
    # RECOMMENDATION
    # -------------------------------------------------

    recommendation = generate_ml_recommendation(

        model_result,

        reliability

    )

    return {

        "success": True,

        "message": (
            "Complete ML analysis finished successfully."
        ),

        "target_summary": target_summary,

        "readiness": model_result.get(
            "readiness",
            {}
        ),

        "model_result": model_result,

        "feature_importance": feature_importance,

        "predictions": predictions,

        "reliability": reliability,

        "recommendation": recommendation

    }


# =========================================================
# ML SUMMARY FOR UI / GEMINI
# =========================================================

def build_ml_summary(
    ml_result
):
    """
    Convert complete ML results into a compact,
    explainable structure for Streamlit and Gemini.
    """

    if not ml_result:

        return {

            "status": (
                "No ML result available."
            )

        }

    if not ml_result.get("success"):

        return {

            "status": (
                "ML analysis could not be completed."
            ),

            "message": ml_result.get(
                "message",
                "Unknown error."
            )

        }

    model_result = ml_result.get(
        "model_result",
        {}
    )

    best_model = model_result.get(
        "best_model",
        {}
    )

    readiness = ml_result.get(
        "readiness",
        {}
    )

    reliability = ml_result.get(
        "reliability",
        {}
    )

    recommendation = ml_result.get(
        "recommendation",
        {}
    )

    return {

        "status": (
            "ML analysis completed."
        ),

        "target": model_result.get(
            "target_column"
        ),

        "problem_type": model_result.get(
            "problem_type"
        ),

        "target_confidence": (
            ml_result
            .get("target_summary", {})
            .get("confidence")
        ),

        "best_model": best_model.get(
            "model"
        ),

        "metrics": best_model.get(
            "metrics",
            {}
        ),

        "comparison_score": best_model.get(
            "comparison_score"
        ),

        "all_models": model_result.get(
            "results",
            []
        ),

        "training_rows": (
            model_result
            .get("training_details", {})
            .get("training_rows")
        ),

        "testing_rows": (
            model_result
            .get("training_details", {})
            .get("testing_rows")
        ),

        "feature_importance": ml_result.get(
            "feature_importance",
            []
        ),

        "readiness_score": readiness.get(
            "score"
        ),

        "readiness_status": readiness.get(
            "status"
        ),

        "readiness_warnings": readiness.get(
            "warnings",
            []
        ),

        "reliability_score": reliability.get(
            "score"
        ),

        "reliability_level": reliability.get(
            "level"
        ),

        "reliability_warnings": reliability.get(
            "warnings",
            []
        ),

        "recommendation": recommendation.get(
            "status"
        ),

        "recommendation_reason": recommendation.get(
            "reason"
        ),

        "prediction_count": len(
            ml_result.get(
                "predictions",
                []
            )
        )

    }