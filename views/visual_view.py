"""views/visual_view.py — Deep visual exploration, themed uniformly."""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st

from components import cards
from components.charts_theme import apply_chart_theme
from engine.visualization import (
    create_numerical_histogram, create_numerical_boxplot,
    create_scatter_matrix, create_categorical_frequency,
    create_missing_value_chart,
)


def render(df, numeric_df, categorical_columns, date_columns,
           correlation_matrix, correlation_pairs, missing_cells, theme: str) -> None:

    cards.section_header(
        "Deep Visual Exploration",
        "Distributions, relationships, categories, time trends, and multidimensional patterns.",
    )

    # ---- Numerical ----
    if not numeric_df.empty:
        st.markdown("#### Numerical Distribution")
        selected = st.selectbox("Select numerical attribute", list(numeric_df.columns), key="visual_numeric")
        series = numeric_df[selected].dropna()

        c1, c2 = st.columns(2)
        with c1:
            fig = apply_chart_theme(create_numerical_histogram(df, selected), theme)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = apply_chart_theme(create_numerical_boxplot(df, selected), theme)
            st.plotly_chart(fig, use_container_width=True)

        if not series.empty:
            stat_cols = st.columns(5)
            stats = [("Mean", series.mean()), ("Median", series.median()),
                     ("Min", series.min()), ("Max", series.max()), ("Std Dev", series.std())]
            for col, (label, value) in zip(stat_cols, stats):
                with col:
                    if pd.notna(value):
                        st.metric(label, f"{value:.3f}")

    # ---- Correlation ----
    if len(numeric_df.columns) >= 2:
        st.divider()
        st.markdown("#### Correlation Explorer")
        threshold = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.50, 0.05, key="correlation_threshold")

        filtered = correlation_matrix.mask(correlation_matrix.abs() < threshold)
        fig = px.imshow(filtered, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                         title=f"Relationships with |correlation| ≥ {threshold:.2f}")
        st.plotly_chart(apply_chart_theme(fig, theme, height=600), use_container_width=True)

        if not correlation_pairs.empty:
            st.markdown("**Strongest Numerical Relationships**")
            st.dataframe(
                correlation_pairs[["Attribute 1", "Attribute 2", "Correlation"]].head(10),
                use_container_width=True, hide_index=True,
            )

        st.markdown("#### Multidimensional Explorer")
        default_cols = list(numeric_df.columns)[:min(4, len(numeric_df.columns))]
        matrix_cols = st.multiselect("Select 2–5 numerical attributes", list(numeric_df.columns),
                                      default=default_cols, max_selections=5, key="matrix_columns")
        if len(matrix_cols) >= 2:
            matrix_df = df[matrix_cols].dropna()
            if len(matrix_df) > 1000:
                matrix_df = matrix_df.sample(1000, random_state=42)
            fig = apply_chart_theme(create_scatter_matrix(matrix_df, matrix_cols), theme, height=720)
            st.plotly_chart(fig, use_container_width=True)

        if len(numeric_df.columns) >= 3:
            st.divider()
            st.markdown("#### 3D Data Space")
            three_d_cols = st.multiselect("Choose 3 numerical attributes", list(numeric_df.columns),
                                           default=list(numeric_df.columns)[:3], max_selections=3, key="three_d_columns")
            if len(three_d_cols) == 3:
                plot_df = df[three_d_cols].dropna()
                if len(plot_df) > 2000:
                    plot_df = plot_df.sample(2000, random_state=42)
                fig = px.scatter_3d(plot_df, x=three_d_cols[0], y=three_d_cols[1], z=three_d_cols[2],
                                     title="Interactive 3D Numerical Data Space", opacity=0.75)
                st.plotly_chart(apply_chart_theme(fig, theme, height=650), use_container_width=True)

    # ---- Categorical ----
    if len(categorical_columns) > 0:
        st.divider()
        st.markdown("#### Categorical Exploration")
        selected_cat = st.selectbox("Select categorical attribute", list(categorical_columns), key="visual_category")
        fig = apply_chart_theme(create_categorical_frequency(df, selected_cat), theme, height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Temporal ----
    if date_columns:
        st.divider()
        st.markdown("#### Temporal Exploration")
        selected_date = st.selectbox("Select date/time attribute", date_columns, key="visual_date")
        date_series = pd.to_datetime(df[selected_date], errors="coerce")
        temporal_df = pd.DataFrame({selected_date: date_series}).dropna()
        if not temporal_df.empty:
            temporal_df["Period"] = temporal_df[selected_date].dt.to_period("M").astype(str)
            trend_df = temporal_df.groupby("Period").size().reset_index(name="Records")
            if len(trend_df) > 1:
                fig = px.line(trend_df, x="Period", y="Records", markers=True, title=f"Record Trend — {selected_date}")
                st.plotly_chart(apply_chart_theme(fig, theme), use_container_width=True)

    # ---- Missing values ----
    if missing_cells > 0:
        st.divider()
        st.markdown("#### Missing Data Pattern")
        fig = create_missing_value_chart(df)
        if fig is not None:
            st.plotly_chart(apply_chart_theme(fig, theme), use_container_width=True)