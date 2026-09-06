"""views/overview_view.py — Overview workspace."""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards


def render(df, uploaded_file, rows, columns, numeric_df, categorical_columns,
           identifier_columns, date_columns, target_candidates,
           quality_score, quality_status, ml_score, ml_status) -> None:

    cards.section_header("Dataset Overview")

    c1, c2 = st.columns([1.15, 1])

    with c1:
        overview_df = pd.DataFrame({
            "Metric": ["Dataset", "Records", "Attributes", "Numerical Attributes",
                       "Categorical Attributes", "Identifier Candidates",
                       "Date/Time Fields", "Target Candidates"],
            "Value": [
                uploaded_file.name, f"{rows:,}", f"{columns:,}",
                f"{len(numeric_df.columns):,}", f"{len(categorical_columns):,}",
                f"{len(identifier_columns):,}", f"{len(date_columns):,}",
                f"{len(target_candidates):,}",
            ],
        })
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**ML Readiness**")
        st.progress(ml_score / 100)
        st.caption(f"{ml_score}/100 · {ml_status}")

        st.markdown("**Data Quality**")
        st.progress(quality_score / 100)
        st.caption(f"{quality_score}/100 · {quality_status}")

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)