"""
views/intelligence_view.py

Explainable Dataset Intelligence workspace.

Displays:
- Column intelligence
- Agent decisions
- Smart visualization plan
- Potential target columns
- Important findings
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import (
    section_header,
    info_card,
    finding_card,
)


# ============================================================
# HELPERS
# ============================================================

def _safe_dataframe(value):
    """Convert supported values into a displayable DataFrame."""

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _format_value(value):
    """Safely format values for display."""

    if value is None:
        return ""

    if isinstance(value, float):
        return round(value, 3)

    return value


def _find_column(
    df: pd.DataFrame,
    possible_names: list[str],
):
    """Find the first matching column name."""

    if df.empty:
        return None

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def _display_table(
    df: pd.DataFrame,
    height: int = 400,
):
    """Display a dataframe safely."""

    if df.empty:
        st.info(
            "No information is available for this section."
        )
        return

    display_df = df.copy()

    for column in display_df.columns:
        display_df[column] = display_df[column].apply(
            _format_value
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=height,
    )


# ============================================================
# COLUMN INTELLIGENCE
# ============================================================

def _render_column_intelligence(
    column_intelligence_df: pd.DataFrame,
):
    section_header(
        "Column Intelligence",
        "The agent interprets the role, type, quality, and analytical importance of each column.",
    )

    if column_intelligence_df.empty:
        info_card(
            "No Column Intelligence",
            "The analysis did not produce column-level intelligence.",
            "Try uploading a dataset with valid tabular data.",
        )
        return

    _display_table(
        column_intelligence_df,
        height=430,
    )


# ============================================================
# AGENT DECISIONS
# ============================================================

def _render_agent_decisions(
    agent_decisions,
):
    section_header(
        "Agent Decisions",
        "Prioritized decisions generated from the dataset profile and quality signals.",
    )

    decisions_df = _safe_dataframe(
        agent_decisions
    )

    if decisions_df.empty:
        info_card(
            "No Agent Decisions",
            "The intelligence engine did not generate decision records.",
            "The dataset may not contain enough signals for automated prioritization.",
        )
        return

    _display_table(
        decisions_df,
        height=430,
    )


# ============================================================
# SMART VISUALIZATION PLAN
# ============================================================

def _render_visualization_plan(
    visual_plan_df: pd.DataFrame,
):
    section_header(
        "Smart Visualization Plan",
        "Recommended visualizations selected from the structure and characteristics of the dataset.",
    )

    if visual_plan_df.empty:
        info_card(
            "No Visualization Recommendations",
            "The agent did not generate a visualization plan.",
            "More suitable numeric, categorical, or temporal columns may be required.",
        )
        return

    _display_table(
        visual_plan_df,
        height=380,
    )


# ============================================================
# POTENTIAL TARGETS
# ============================================================

def _render_target_candidates(
    target_candidates,
):
    section_header(
        "Potential ML Targets",
        "Columns that may be suitable as prediction targets.",
    )

    targets_df = _safe_dataframe(
        target_candidates
    )

    if targets_df.empty:
        info_card(
            "No Clear Target Candidate",
            "The agent could not identify a strong machine-learning target.",
            "You can still explore the dataset using the other intelligence tools.",
        )
        return

    _display_table(
        targets_df,
        height=320,
    )


# ============================================================
# IMPORTANT FINDINGS
# ============================================================

def _render_important_findings(
    column_intelligence_df: pd.DataFrame,
    agent_decisions,
):
    section_header(
        "Important Findings",
        "Critical and high-priority observations that deserve attention.",
    )

    findings = []

    # --------------------------------------------------------
    # COLUMN INTELLIGENCE FINDINGS
    # --------------------------------------------------------

    if not column_intelligence_df.empty:

        severity_column = _find_column(
            column_intelligence_df,
            [
                "Severity",
                "Risk",
                "Priority",
                "Level",
            ],
        )

        finding_column = _find_column(
            column_intelligence_df,
            [
                "Finding",
                "Issue",
                "Recommendation",
                "Insight",
                "Description",
            ],
        )

        column_column = _find_column(
            column_intelligence_df,
            [
                "Column",
                "column_name",
                "Name",
            ],
        )

        if (
            severity_column is not None
            and finding_column is not None
        ):

            for _, row in column_intelligence_df.iterrows():

                severity = str(
                    row[severity_column]
                ).strip().lower()

                if severity in {
                    "critical",
                    "high",
                    "critical risk",
                    "high risk",
                }:

                    column_name = ""

                    if column_column is not None:
                        column_name = str(
                            row[column_column]
                        )

                    finding_text = str(
                        row[finding_column]
                    )

                    if column_name:
                        finding_text = (
                            f"{column_name}: "
                            f"{finding_text}"
                        )

                    findings.append(
                        {
                            "severity": severity,
                            "text": finding_text,
                        }
                    )

    # --------------------------------------------------------
    # AGENT DECISION FINDINGS
    # --------------------------------------------------------

    decisions_df = _safe_dataframe(
        agent_decisions
    )

    if not decisions_df.empty:

        severity_column = _find_column(
            decisions_df,
            [
                "Severity",
                "Risk",
                "Priority",
                "Level",
            ],
        )

        finding_column = _find_column(
            decisions_df,
            [
                "Finding",
                "Issue",
                "Decision",
                "Recommendation",
                "Insight",
                "Description",
                "Action",
            ],
        )

        if (
            severity_column is not None
            and finding_column is not None
        ):

            for _, row in decisions_df.iterrows():

                severity = str(
                    row[severity_column]
                ).strip().lower()

                if severity in {
                    "critical",
                    "high",
                    "critical risk",
                    "high risk",
                }:

                    findings.append(
                        {
                            "severity": severity,
                            "text": str(
                                row[finding_column]
                            ),
                        }
                    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    if not findings:

        finding_card(
            "No Critical or High Findings",
            (
                "The current intelligence analysis did not "
                "identify any critical or high-priority findings."
            ),
            "success",
        )

        return

    # Remove duplicate findings

    unique_findings = []

    seen = set()

    for finding in findings:

        text = finding["text"].strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)

        unique_findings.append(
            finding
        )

    # Display maximum useful findings

    for finding in unique_findings[:10]:

        severity = finding["severity"]

        if "critical" in severity:

            finding_type = "error"
            title = "Critical Finding"

        else:

            finding_type = "warning"
            title = "High Priority Finding"

        finding_card(
            title,
            finding["text"],
            finding_type,
        )


# ============================================================
# MAIN VIEW
# ============================================================

def render(
    column_intelligence_df,
    agent_decisions,
    visual_plan_df,
    target_candidates,
):
    """
    Render the Dataset Intelligence workspace.

    Parameters
    ----------
    column_intelligence_df:
        DataFrame containing column-level intelligence.

    agent_decisions:
        Agent-generated decisions and recommendations.

    visual_plan_df:
        DataFrame containing recommended visualizations.

    target_candidates:
        Potential machine-learning target columns.
    """

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div class="hero-card">

            <div class="hero-kicker">
                EXPLAINABLE INTELLIGENCE
            </div>

            <h1>
                Dataset Intelligence
            </h1>

            <p>
                An explainable intelligence layer that analyzes
                dataset structure, quality, patterns, and
                machine-learning readiness to generate
                prioritized findings and actionable
                recommendations.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # QUICK SUMMARY
    # ========================================================

    column_df = _safe_dataframe(
        column_intelligence_df
    )

    decisions_df = _safe_dataframe(
        agent_decisions
    )

    visual_df = _safe_dataframe(
        visual_plan_df
    )

    targets_df = _safe_dataframe(
        target_candidates
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Columns Analyzed",
            len(column_df),
        )

    with c2:

        st.metric(
            "Agent Decisions",
            len(decisions_df),
        )

    with c3:

        st.metric(
            "Visual Recommendations",
            len(visual_df),
        )

    with c4:

        st.metric(
            "Potential Targets",
            len(targets_df),
        )

    st.write("")

    # ========================================================
    # COLUMN INTELLIGENCE
    # ========================================================

    _render_column_intelligence(
        column_df
    )

    st.divider()

    # ========================================================
    # AGENT DECISIONS
    # ========================================================

    _render_agent_decisions(
        agent_decisions
    )

    st.divider()

    # ========================================================
    # SMART VISUALIZATION PLAN
    # ========================================================

    _render_visualization_plan(
        visual_df
    )

    st.divider()

    # ========================================================
    # ML TARGET CANDIDATES
    # ========================================================

    _render_target_candidates(
        target_candidates
    )

    st.divider()

    # ========================================================
    # IMPORTANT FINDINGS
    # ========================================================

    _render_important_findings(
        column_df,
        agent_decisions,
    )