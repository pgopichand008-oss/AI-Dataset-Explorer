"""views/ml_view.py — Machine learning workspace."""

from __future__ import annotations

import inspect
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from components import cards, state
from components.charts_theme import apply_chart_theme


def _find_kwargs(fn, df, target) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    kwargs = {}

    for name in params:
        lowered = name.lower()

        if lowered in ["df", "data", "dataset", "dataframe"]:
            kwargs[name] = df

        elif lowered in [
            "target",
            "target_column",
            "target_col",
            "target_name",
            "y_column",
            "label",
        ]:
            kwargs[name] = target

    return kwargs


def call_ml_engine(run_ml_analysis, df: pd.DataFrame, target: str) -> Any:
    kwargs = _find_kwargs(run_ml_analysis, df, target)

    try:
        return run_ml_analysis(**kwargs) if kwargs else run_ml_analysis(df, target)
    except TypeError:
        return run_ml_analysis(df, target)


def _extract(result: dict, keys: list[str], default=None):
    for key in keys:
        if key in result:
            return result[key]

    return default


def display_ml_result(
    result: Any,
    build_ml_summary_fn,
    theme: str,
) -> None:

    if result is None:
        return

    if not isinstance(result, dict):
        st.write(result)
        return

    try:
        summary = (
            build_ml_summary_fn(result)
            if build_ml_summary_fn
            else None
        )
    except Exception:
        summary = None

    # --------------------------------------------------------
    # ML SUMMARY
    # --------------------------------------------------------

    if summary:
        st.markdown("#### ML Summary")

        if isinstance(summary, str):
            cards.finding_card(
                "Summary",
                summary,
                "info",
            )

        elif isinstance(summary, dict):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Metric": k,
                            "Value": str(v),
                        }
                        for k, v in summary.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # ML READINESS
    # --------------------------------------------------------

    readiness = _extract(
        result,
        [
            "ml_readiness",
            "readiness",
            "readiness_score",
            "score",
        ],
    )

    if readiness is not None:
        try:
            val = max(
                0,
                min(100, float(readiness)),
            )

            st.markdown("#### ML Readiness")
            st.progress(val / 100)
            st.metric(
                "ML Readiness Score",
                f"{val:.0f}/100",
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # PROBLEM TYPE
    # --------------------------------------------------------

    problem_type = _extract(
        result,
        [
            "problem_type",
            "task_type",
            "task",
            "problem",
        ],
    )

    if problem_type:
        cards.finding_card(
            "Problem Type",
            str(problem_type),
            "success",
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warnings = _extract(
        result,
        [
            "warnings",
            "limitations",
            "issues",
        ],
        [],
    )

    if warnings:
        st.markdown("#### ML Warnings")

        items = (
            warnings
            if isinstance(warnings, list)
            else [warnings]
        )

        for warning in items:
            cards.finding_card(
                "Warning",
                str(warning),
                "warning",
            )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    model_comparison = _extract(
        result,
        [
            "model_comparison",
            "model_results",
            "models",
            "comparison",
        ],
    )

    if model_comparison is not None:
        st.markdown("#### Model Comparison")

        if isinstance(model_comparison, pd.DataFrame):
            comp_df = model_comparison

        elif isinstance(model_comparison, list):
            comp_df = pd.DataFrame(model_comparison)

        elif isinstance(model_comparison, dict):
            try:
                comp_df = pd.DataFrame(model_comparison)
            except Exception:
                comp_df = pd.DataFrame(
                    [
                        {
                            "Model": k,
                            "Result": v,
                        }
                        for k, v in model_comparison.items()
                    ]
                )

        else:
            comp_df = pd.DataFrame()

        if not comp_df.empty:
            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = _extract(
        result,
        [
            "best_model",
            "selected_model",
            "best_model_name",
        ],
    )

    if best_model:
        cards.finding_card(
            "Selected Best Model",
            str(best_model),
            "success",
        )

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    metrics = _extract(
        result,
        [
            "metrics",
            "best_metrics",
            "evaluation",
        ],
    )

    if metrics is not None:
        st.markdown("#### Evaluation Metrics")

        if isinstance(metrics, dict):
            cols = st.columns(
                min(4, max(1, len(metrics)))
            )

            for i, (key, value) in enumerate(
                metrics.items()
            ):
                with cols[i % len(cols)]:
                    if isinstance(value, float):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = str(value)

                    st.metric(
                        str(key),
                        display_value,
                    )

        elif isinstance(metrics, pd.DataFrame):
            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.write(metrics)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = _extract(
        result,
        [
            "feature_importance",
            "feature_importances",
            "importance",
        ],
    )

    if feature_importance is not None:
        st.markdown("#### Feature Importance")

        if isinstance(
            feature_importance,
            pd.DataFrame,
        ):
            imp_df = feature_importance.copy()

        elif isinstance(feature_importance, list):
            imp_df = pd.DataFrame(
                feature_importance
            )

        elif isinstance(feature_importance, dict):
            imp_df = pd.DataFrame(
                [
                    {
                        "Feature": key,
                        "Importance": value,
                    }
                    for key, value
                    in feature_importance.items()
                ]
            )

        else:
            imp_df = pd.DataFrame()

        if not imp_df.empty:
            st.dataframe(
                imp_df,
                use_container_width=True,
                hide_index=True,
            )

            numeric_cols = (
                imp_df
                .select_dtypes(include="number")
                .columns
                .tolist()
            )

            label_col = next(
                (
                    column
                    for column in [
                        "Feature",
                        "feature",
                        "Attribute",
                        "attribute",
                        "feature_name",
                    ]
                    if column in imp_df.columns
                ),
                None,
            )

            if numeric_cols and label_col:
                chart_df = (
                    imp_df[
                        [
                            label_col,
                            numeric_cols[-1],
                        ]
                    ]
                    .dropna()
                    .sort_values(
                        numeric_cols[-1]
                    )
                    .tail(15)
                )

                if not chart_df.empty:
                    fig = px.bar(
                        chart_df,
                        x=numeric_cols[-1],
                        y=label_col,
                        orientation="h",
                        title="Top Feature Importance",
                    )

                    st.plotly_chart(
                        apply_chart_theme(
                            fig,
                            theme,
                            height=500,
                        ),
                        use_container_width=True,
                    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = _extract(
        result,
        [
            "predictions",
            "prediction_samples",
            "sample_predictions",
        ],
    )

    if predictions is not None:
        st.markdown("#### Prediction Samples")

        if isinstance(
            predictions,
            pd.DataFrame,
        ):
            pred_df = predictions
        else:
            pred_df = pd.DataFrame(
                predictions or []
            )

        if not pred_df.empty:
            st.dataframe(
                pred_df.head(20),
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # MODEL RELIABILITY
    # --------------------------------------------------------

    reliability = _extract(
        result,
        [
            "reliability",
            "ml_reliability",
            "reliability_assessment",
        ],
    )

    if reliability:
        st.markdown("#### Model Reliability")

        if isinstance(
            reliability,
            dict,
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Assessment": key,
                            "Result": str(value),
                        }
                        for key, value
                        in reliability.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:
            cards.finding_card(
                "Reliability",
                str(reliability),
                "info",
            )

    # --------------------------------------------------------
    # ML RECOMMENDATION
    # --------------------------------------------------------

    recommendation = _extract(
        result,
        [
            "final_recommendation",
            "recommendation",
            "ml_recommendation",
        ],
    )

    if recommendation:
        st.markdown("#### ML Recommendation")

        text = str(recommendation)

        if "NOT SUITABLE" in text.upper():
            kind = "danger"

        elif "NEEDS ATTENTION" in text.upper():
            kind = "warning"

        else:
            kind = "success"

        cards.finding_card(
            "Recommendation",
            text,
            kind,
        )


def render(
    df: pd.DataFrame,
    uploaded_file,
    rows: int,
    columns: int,
    ml_score: int,
    target_candidates: list[str],
    ml_engine_available: bool,
    ml_engine_error: str,
    run_ml_analysis_fn,
    build_ml_summary_fn,
    theme: str,
) -> None:

    cards.section_header(
        "Machine Learning Intelligence",
        "Evaluate readiness, identify a target, train candidate models, and explain the result.",
    )

    # --------------------------------------------------------
    # ENGINE AVAILABILITY
    # --------------------------------------------------------

    if not ml_engine_available:
        cards.finding_card(
            "ML engine unavailable",
            "engine/ml_engine.py could not be loaded.",
            "danger",
        )

        st.code(
            ml_engine_error,
            language="text",
        )

        return

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    top_cols = st.columns(4)

    with top_cols[0]:
        st.metric(
            "ML Readiness",
            f"{ml_score}/100",
        )

    with top_cols[1]:
        st.metric(
            "Rows",
            f"{rows:,}",
        )

    with top_cols[2]:
        st.metric(
            "Features",
            f"{max(0, columns - 1):,}",
        )

    with top_cols[3]:
        st.metric(
            "Target Candidates",
            f"{len(target_candidates):,}",
        )

    st.divider()

    # --------------------------------------------------------
    # TARGET SELECTION
    # --------------------------------------------------------

    st.markdown("#### Target Selection")

    if target_candidates:
        cards.finding_card(
            "Auto-detected",
            "Possible target variables were detected. First candidate selected by default.",
            "info",
        )

    else:
        cards.finding_card(
            "No target detected",
            "Select a target column manually.",
            "warning",
        )

    target_mode = st.radio(
        "Target strategy",
        ["Automatic", "Manual"],
        horizontal=True,
        key="ml_target_mode",
    )

    selected_target: Optional[str] = None

    if target_mode == "Automatic":

        selected_target = (
            target_candidates[0]
            if target_candidates
            else None
        )

        if selected_target is None:
            cards.finding_card(
                "Cannot continue",
                "Automatic mode needs a detected target candidate.",
                "warning",
            )

    else:

        selected_target = st.selectbox(
            "Choose target column",
            list(df.columns),
            key="manual_ml_target",
        )

    if selected_target:
        cards.ml_card(
            "Selected Target",
            selected_target,
        )

    # --------------------------------------------------------
    # READINESS WARNINGS
    # --------------------------------------------------------

    if ml_score < 40:
        cards.finding_card(
            "Low ML readiness",
            "The agent can still run, but results may be unreliable.",
            "warning",
        )

    if rows < 10:
        cards.finding_card(
            "Very few rows",
            "Model evaluation may be unstable.",
            "warning",
        )

    # --------------------------------------------------------
    # RUN ML
    # --------------------------------------------------------

    run_ml = st.button(
        "Run Intelligent ML Analysis",
        type="primary",
        disabled=selected_target is None,
        key="run_ml_button",
    )

    if run_ml and selected_target:

        with st.spinner(
            "ML agent is preparing data, training, and evaluating models..."
        ):

            try:
                result = call_ml_engine(
                    run_ml_analysis_fn,
                    df,
                    selected_target,
                )

                state.set_ml_result(
                    result,
                    uploaded_file.name,
                    selected_target,
                )

                st.success(
                    "Machine learning analysis completed."
                )

            except Exception as e:

                state.set_ml_result(
                    None,
                    uploaded_file.name,
                    selected_target,
                )

                cards.finding_card(
                    "Analysis failed",
                    str(e),
                    "danger",
                )

    # --------------------------------------------------------
    # DISPLAY STORED RESULT
    # --------------------------------------------------------

    result = state.get_ml_result()

    if (
        result is not None
        and state.get("ml_file_name")
        == uploaded_file.name
    ):

        st.divider()

        st.markdown("### ML Agent Results")

        display_ml_result(
            result,
            build_ml_summary_fn,
            theme,
        )

        st.divider()

        cards.finding_card(
            "Original Dataset Protection",
            "The ML workflow analyzes the dataset in memory. The uploaded file is never overwritten.",
            "info",
        )
