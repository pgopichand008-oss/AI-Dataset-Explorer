"""
components/layout.py

Global chrome: hero header, theme toggle, sidebar workspace controls,
pipeline status rail, and the top-line metric row.
"""

from __future__ import annotations
from typing import Optional
import streamlit as st

from components import cards, state


def render_theme_toggle() -> None:
    current = state.get_theme()
    label = "🌙 Dark mode" if current == "dark" else "☀️ Light mode"
    if st.sidebar.toggle(label, value=(current == "dark"), key="theme_toggle"):
        state.set_theme("dark")
    else:
        state.set_theme("light")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">AI · DATA · ML · INTELLIGENCE</div>
            <div class="hero-title">Dataset Intelligence Platform</div>
            <div class="hero-subtitle">
                From raw data to actionable intelligence — profiling, quality,
                visualization, machine learning, and adaptive change reasoning
                in one workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders the sidebar and returns (current_file, previous_file, updated_file)."""
    with st.sidebar:
        st.markdown("## Workspace")
        render_theme_toggle()
        st.divider()

        st.markdown("### Dataset")
        current_file = st.file_uploader(
            "Upload current dataset", type=["csv", "xlsx"], key="current_dataset"
        )
        st.caption("CSV or Excel — activates the full analysis pipeline.")

        st.divider()
        st.markdown("### Change Intelligence")
        previous_file = st.file_uploader("Previous dataset", type=["csv", "xlsx"], key="previous_dataset")
        updated_file = st.file_uploader("Updated dataset", type=["csv", "xlsx"], key="updated_dataset")
        st.caption("Compare two versions for structural and quality drift.")

        st.divider()
        st.markdown("### Pipeline")
        st.markdown(
            "".join(
                cards.status_pill(label, "success")
                for label in ["① Discovery", "② Profiling", "③ Quality", "④ Visuals", "⑤ AI", "⑥ ML", "⑦ Changes"]
            ),
            unsafe_allow_html=True,
        )

        st.divider()

    return current_file, previous_file, updated_file


def render_status_row(client, ml_engine_available: bool, previous_file, updated_file) -> None:
    section_labels = ["Dataset", "Profile", "Quality", "Visuals", "AI", "ML", "Changes"]
    completed_flags = [
        True, True, True, True,
        client is not None,
        ml_engine_available,
        previous_file is not None and updated_file is not None,
    ]
    cols = st.columns(7)
    for col, label, completed in zip(cols, section_labels, completed_flags):
        with col:
            st.success(f"✓ {label}") if completed else st.warning(f"○ {label}")


def render_metric_row(rows: int, columns: int, missing_cells: int, duplicate_rows: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records", f"{rows:,}")
    with c2:
        st.metric("Attributes", f"{columns:,}")
    with c3:
        missing_pct = (missing_cells / (rows * columns) * 100) if rows * columns else 0
        st.metric("Missing Cells", f"{missing_cells:,}", f"{missing_pct:.2f}%")
    with c4:
        dup_pct = (duplicate_rows / rows * 100) if rows else 0
        st.metric("Duplicate Rows", f"{duplicate_rows:,}", f"{dup_pct:.2f}%")
