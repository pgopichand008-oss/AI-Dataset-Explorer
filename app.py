"""
app.py

Main entry point / router for the AI Dataset Intelligence Platform.

Responsibilities:
- Streamlit page configuration
- Session-state initialization
- Global design system
- Sidebar dataset controls
- Dataset loading
- Core profiling and quality pipeline
- Correlation analysis
- Agent intelligence preparation
- ML engine availability
- Routing computed results into modular views
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from config import get_gemini_client
from data_loader import load_dataset


# ============================================================
# ENGINE IMPORTS
# ============================================================

from engine.profiling import (
    build_basic_profile,
    build_column_intelligence,
    build_numerical_summary,
    build_categorical_summary,
)

from engine.quality import (
    detect_outliers,
    build_quality_findings,
    calculate_quality_score,
    calculate_ml_readiness,
)

from engine.intelligence import (
    build_agent_decisions,
    build_visualization_plan,
    build_executive_summary,
)

from engine.ai_report import generate_ai_report
from engine.ai_agent import run_intelligence_analysis


# ============================================================
# LEGACY UI
# ============================================================
#
# ui.py is located in the project root.
#
# This is still required by the Executive AI view.
# ============================================================

import ui as legacy_ui


# ============================================================
# COMPONENTS
# ============================================================

from components import state
from components import layout
from components import cards
from components.design_system import inject_design_system


# ============================================================
# VIEWS
# ============================================================

from views import (
    overview_view,
    quality_view,
    visual_view,
    intelligence_view,
    ml_view,
    ai_view,
    change_view,
)


# ============================================================
# OPTIONAL ML ENGINE
# ============================================================

try:

    from engine.ml_engine import (
        run_ml_analysis,
        build_ml_summary,
    )

    ML_ENGINE_AVAILABLE = True
    ML_ENGINE_ERROR = ""

except Exception as exc:

    ML_ENGINE_AVAILABLE = False
    ML_ENGINE_ERROR = str(exc)

    run_ml_analysis = None
    build_ml_summary = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Dataset Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

state.init_state()


# ============================================================
# DESIGN SYSTEM
# ============================================================

theme = state.get_theme()

inject_design_system(theme)


# ============================================================
# SERVICES
# ============================================================

client = get_gemini_client()

api_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# GLOBAL HEADER
# ============================================================

layout.render_hero()


# ============================================================
# SIDEBAR
# ============================================================

uploaded_file, previous_file, updated_file = (
    layout.render_sidebar()
)


# ============================================================
# SIDEBAR SERVICE STATUS
# ============================================================

with st.sidebar:

    st.markdown("### System Status")

    if api_key:

        st.success(
            "🟢 Gemini AI connected"
        )

    else:

        st.warning(
            "🟡 Gemini API not connected"
        )

    if ML_ENGINE_AVAILABLE:

        st.success(
            "🟢 ML engine available"
        )

    else:

        st.warning(
            "🟡 ML engine unavailable"
        )


# ============================================================
# NO CURRENT DATASET
# ============================================================

if uploaded_file is None:

    cards.info_card(
        "Welcome",
        "Start with your dataset",
        (
            "Upload a CSV or Excel file from the sidebar to activate "
            "profiling, quality intelligence, visualization, machine "
            "learning, executive AI, and adaptive change intelligence."
        ),
    )

    if (
        previous_file is not None
        and updated_file is not None
    ):

        st.success(
            "Previous and updated datasets detected. "
            "Upload a current dataset to open the full workspace."
        )

    st.stop()


# ============================================================
# LOAD CURRENT DATASET
# ============================================================

df, error = load_dataset(
    uploaded_file
)


if error is not None:

    st.error(
        f"Unable to read the dataset: {error}"
    )

    st.stop()


if df is None or df.empty:

    st.error(
        "The uploaded dataset contains no usable records."
    )

    st.stop()


# ============================================================
# DATASET SESSION LIFECYCLE
# ============================================================

if (
    state.get("loaded_file_name")
    != uploaded_file.name
):

    state.reset_for_new_dataset(
        uploaded_file.name
    )


# ============================================================
# CHANGE-INTELLIGENCE SESSION LIFECYCLE
# ============================================================

current_previous_name = (
    previous_file.name
    if previous_file is not None
    else ""
)

current_updated_name = (
    updated_file.name
    if updated_file is not None
    else ""
)


if (
    state.get("previous_file_name")
    != current_previous_name
    or
    state.get("updated_file_name")
    != current_updated_name
):

    state.reset_change_analysis(
        current_previous_name,
        current_updated_name,
    )


# ============================================================
# CORE DATASET PROFILE
# ============================================================

profile = build_basic_profile(
    df
)

rows = profile["rows"]

columns = profile["columns"]

missing_cells = profile["missing_cells"]

duplicate_rows = profile["duplicate_rows"]

numeric_df = profile["numeric_df"]

categorical_columns = profile[
    "categorical_columns"
]


# ============================================================
# COLUMN INTELLIGENCE
# ============================================================

(
    column_intelligence_df,
    identifier_columns,
    date_columns,
    target_candidates,
) = build_column_intelligence(
    df,
    rows,
)


# ============================================================
# STATISTICAL SUMMARIES
# ============================================================

numerical_summary = build_numerical_summary(
    numeric_df
)

categorical_summary_df = (
    build_categorical_summary(
        df,
        categorical_columns,
        rows,
    )
)


# ============================================================
# OUTLIER / ANOMALY DETECTION
# ============================================================

(
    outlier_columns,
    total_outlier_values,
    anomaly_df,
) = detect_outliers(
    df,
    numeric_df,
)


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

(
    quality_findings,
    empty_columns,
    constant_columns,
    high_cardinality_columns,
) = build_quality_findings(
    df,
    rows,
    columns,
    missing_cells,
    duplicate_rows,
    outlier_columns,
)


quality_df = pd.DataFrame(
    quality_findings
)


# ============================================================
# QUALITY SCORE
# ============================================================

quality_score, quality_status = (
    calculate_quality_score(
        rows,
        columns,
        missing_cells,
        duplicate_rows,
        outlier_columns,
        constant_columns,
    )
)


# ============================================================
# ML READINESS SCORE
# ============================================================

ml_score, ml_status = (
    calculate_ml_readiness(
        rows,
        missing_cells,
        duplicate_rows,
        outlier_columns,
        constant_columns,
        identifier_columns,
        columns,
    )
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

correlation_matrix = pd.DataFrame()

correlation_pairs = pd.DataFrame()


if len(numeric_df.columns) >= 2:

    correlation_matrix = (
        numeric_df
        .corr()
        .round(3)
    )

    pair_rows = []

    cols_list = list(
        correlation_matrix.columns
    )

    for i in range(
        len(cols_list)
    ):

        for j in range(
            i + 1,
            len(cols_list),
        ):

            value = (
                correlation_matrix
                .iloc[i, j]
            )

            if pd.notna(value):

                pair_rows.append(
                    {
                        "Attribute 1": cols_list[i],
                        "Attribute 2": cols_list[j],
                        "Correlation": value,
                        "Absolute Correlation": abs(value),
                    }
                )

    if pair_rows:

        correlation_pairs = (
            pd.DataFrame(pair_rows)
            .sort_values(
                "Absolute Correlation",
                ascending=False,
            )
            .reset_index(drop=True)
        )


# ============================================================
# AGENT DECISIONS
# ============================================================

agent_decisions = build_agent_decisions(
    df,
    missing_cells,
    rows,
    outlier_columns,
    identifier_columns,
    date_columns,
    categorical_columns,
)


# ============================================================
# SMART VISUALIZATION PLAN
# ============================================================

visual_plan = build_visualization_plan(
    df,
    list(numeric_df.columns),
    list(categorical_columns),
    date_columns,
)

visual_plan_df = pd.DataFrame(
    visual_plan
)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

executive_summary = build_executive_summary(
    quality_score,
    ml_score,
    quality_findings,
    outlier_columns,
    identifier_columns,
    target_candidates,
)


risk_level = executive_summary[
    "Risk Level"
]

most_important_finding = (
    executive_summary[
        "Most Important Finding"
    ]
)

key_opportunity = executive_summary[
    "Key Opportunity"
]

next_best_action = executive_summary[
    "Next Best Action"
]


# ============================================================
# ANALYSIS STATUS
# ============================================================

cards.section_header(
    "Analysis Status"
)

layout.render_status_row(
    client,
    ML_ENGINE_AVAILABLE,
    previous_file,
    updated_file,
)


st.divider()


# ============================================================
# TOP METRICS
# ============================================================

layout.render_metric_row(
    rows,
    columns,
    missing_cells,
    duplicate_rows,
)


# ============================================================
# EXECUTIVE DATASET SUMMARY
# ============================================================

cards.section_header(
    "Executive Dataset Summary",
    "A high-level view of dataset health, risk, and opportunity.",
)


s1, s2, s3 = st.columns(3)


with s1:

    st.metric(
        "Data Quality",
        f"{quality_score}/100",
        quality_status,
    )


with s2:

    st.metric(
        "ML Readiness",
        f"{ml_score}/100",
        ml_status,
    )


with s3:

    st.metric(
        "Risk Level",
        risk_level,
    )


f1, f2 = st.columns(2)


with f1:

    cards.finding_card(
        "Most Important Finding",
        most_important_finding,
        "info",
    )


with f2:

    cards.finding_card(
        "Key Opportunity",
        key_opportunity,
        "success",
    )


cards.finding_card(
    "Next Best Action",
    next_best_action,
    "warning",
)


# ============================================================
# WORKSPACE TABS
# ============================================================

(
    overview_tab,
    quality_tab,
    visual_tab,
    intelligence_tab,
    ml_tab,
    ai_tab,
    change_tab,
) = st.tabs(
    [
        "Overview",
        "Quality",
        "Visuals",
        "Intelligence",
        "Machine Learning",
        "Executive AI",
        "Change Intelligence",
    ]
)


# ============================================================
# OVERVIEW VIEW
# ============================================================

with overview_tab:

    overview_view.render(
        df,
        uploaded_file,
        rows,
        columns,
        numeric_df,
        categorical_columns,
        identifier_columns,
        date_columns,
        target_candidates,
        quality_score,
        quality_status,
        ml_score,
        ml_status,
    )


# ============================================================
# QUALITY VIEW
# ============================================================

with quality_tab:

    quality_view.render(
        quality_df,
        anomaly_df,
        quality_score,
        rows,
        columns,
        missing_cells,
        duplicate_rows,
        total_outlier_values,
    )


# ============================================================
# VISUAL EXPLORATION VIEW
# ============================================================

with visual_tab:

    visual_view.render(
        df,
        numeric_df,
        categorical_columns,
        date_columns,
        correlation_matrix,
        correlation_pairs,
        missing_cells,
        theme,
    )


# ============================================================
# EXPLAINABLE INTELLIGENCE VIEW
# ============================================================
#
# IMPORTANT:
# Current intelligence_view.py accepts exactly 4 arguments.
# Therefore do NOT pass the two old legacy UI functions here.
# ============================================================

with intelligence_tab:

    intelligence_view.render(
        column_intelligence_df,
        agent_decisions,
        visual_plan_df,
        target_candidates,
    )


# ============================================================
# MACHINE LEARNING VIEW
# ============================================================

with ml_tab:

    ml_view.render(
        df,
        uploaded_file,
        rows,
        columns,
        ml_score,
        target_candidates,
        ML_ENGINE_AVAILABLE,
        ML_ENGINE_ERROR,
        (
            run_ml_analysis
            if ML_ENGINE_AVAILABLE
            else None
        ),
        (
            build_ml_summary
            if ML_ENGINE_AVAILABLE
            else None
        ),
        theme,
    )


# ============================================================
# EXECUTIVE AI VIEW
# ============================================================

with ai_tab:

    ai_view.render(
        client,
        uploaded_file,
        rows,
        columns,
        numeric_df,
        categorical_columns,
        identifier_columns,
        date_columns,
        target_candidates,
        quality_score,
        ml_score,
        quality_findings,
        agent_decisions,
        numerical_summary,
        categorical_summary_df,
        correlation_pairs,
        anomaly_df,
        column_intelligence_df,
        df,
        generate_ai_report,
        legacy_ui.show_ai_report,
    )


# ============================================================
# DATASET CHANGE INTELLIGENCE VIEW
# ============================================================

with change_tab:

    change_view.render(
        previous_file,
        updated_file,
        most_important_finding,
        load_dataset,
        run_intelligence_analysis,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>AI Dataset Intelligence</strong><br>
        Automated Profiling · Quality Intelligence ·
        Interactive Visualization · Machine Learning ·
        Executive AI · Adaptive Dataset Change Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)