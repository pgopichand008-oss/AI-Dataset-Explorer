"""views/quality_view.py — Quality & anomaly command center."""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards


def render(quality_df: pd.DataFrame, anomaly_df: pd.DataFrame, quality_score: int,
           rows: int, columns: int, missing_cells: int, duplicate_rows: int,
           total_outlier_values: int) -> None:

    cards.section_header(
        "Quality & Anomaly Command Center",
        "Detect missing data, duplicates, anomalies, and structural quality risks.",
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Quality Score", f"{quality_score}/100")
    with q2:
        pct = (missing_cells / (rows * columns) * 100) if rows * columns else 0
        st.metric("Missing Data", f"{pct:.2f}%")
    with q3:
        pct = (duplicate_rows / rows * 100) if rows else 0
        st.metric("Duplicate Data", f"{pct:.2f}%")
    with q4:
        st.metric("Potential Anomalies", f"{total_outlier_values:,}")

    st.divider()

    st.markdown("#### Quality Findings")
    if not quality_df.empty:
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
    else:
        cards.finding_card("All clear", "No major quality issues detected.", "success")

    st.markdown("#### Numerical Anomalies")
    if not anomaly_df.empty:
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    else:
        cards.finding_card("All clear", "No IQR-based numerical anomalies detected.", "success")
