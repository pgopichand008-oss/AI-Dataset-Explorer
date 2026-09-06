"""views/ai_view.py — Executive AI intelligence report."""

from __future__ import annotations
import streamlit as st

from components import cards, state


def render(client, uploaded_file, rows, columns, numeric_df, categorical_columns,
           identifier_columns, date_columns, target_candidates,
           quality_score, ml_score, quality_findings, agent_decisions,
           numerical_summary, categorical_summary_df, correlation_pairs,
           anomaly_df, column_intelligence_df, df, generate_ai_report_fn,
           show_ai_report_fn) -> None:

    cards.section_header(
        "Executive AI Intelligence",
        "Gemini interprets the analytical evidence and produces an executive-level briefing.",
    )

    if client is None:
        cards.finding_card("AI unavailable", "Gemini API key was not detected. All other analysis remains available.", "warning")
    else:
        if st.button("Generate Executive Intelligence", type="primary", key="generate_ai_report"):
            with st.spinner("AI Agent is investigating your dataset..."):
                profile_data = {
                    "dataset": uploaded_file.name, "rows": rows, "columns": columns,
                    "numerical_columns": list(numeric_df.columns),
                    "categorical_columns": list(categorical_columns),
                    "identifier_columns": identifier_columns,
                    "date_columns": date_columns,
                    "target_candidates": target_candidates,
                }

                numerical_evidence = (
                    numerical_summary.reset_index().rename(columns={"index": "Attribute"}).head(30)
                    if not numerical_summary.empty else None
                )
                categorical_evidence = categorical_summary_df.head(30) if not categorical_summary_df.empty else None
                correlation_evidence = (
                    correlation_pairs[["Attribute 1", "Attribute 2", "Correlation"]].head(15)
                    if not correlation_pairs.empty else None
                )
                anomaly_evidence = anomaly_df.head(30) if not anomaly_df.empty else None
                column_evidence = column_intelligence_df.head(50)
                missing_evidence = df.isna().sum().sort_values(ascending=False)
                missing_evidence = missing_evidence[missing_evidence > 0].head(30).to_dict()

                report = generate_ai_report_fn(
                    profile=profile_data, quality_score=quality_score, ml_score=ml_score,
                    quality_findings=quality_findings, agent_decisions=agent_decisions,
                    target_candidates=target_candidates, numerical_summary=numerical_evidence,
                    categorical_summary=categorical_evidence, correlation_pairs=correlation_evidence,
                    anomaly_data=anomaly_evidence, column_intelligence=column_evidence,
                    missing_data=missing_evidence,
                )

                if report:
                    state.set_ai_report(report)
                    st.success("Executive intelligence generated successfully.")
                else:
                    cards.finding_card("No report", "The AI service returned no report.", "warning")

    if state.get_ai_report():
        st.divider()
        show_ai_report_fn(state.get_ai_report())
        st.download_button(
            "Download Executive Report", data=state.get_ai_report(),
            file_name="AI_Dataset_Executive_Report.txt", mime="text/plain",
            key="download_ai_report",
        )