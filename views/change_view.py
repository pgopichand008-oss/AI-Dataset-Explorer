"""views/change_view.py — Adaptive dataset change intelligence."""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards, state


def render(previous_file, updated_file, most_important_finding, load_dataset_fn, run_intelligence_analysis_fn) -> None:

    cards.section_header(
        "Dataset Change Intelligence",
        "Detect structural changes, quality changes, and whether previous findings still hold.",
    )

    if previous_file is None or updated_file is None:
        cards.finding_card(
            "Upload both files",
            "Upload a Previous Dataset and an Updated Dataset from the sidebar to activate this workflow.",
            "info",
        )
        st.markdown(
            "**Workflow:** Detect → Analyze → Compare → Reconsider → Re-plan → Report\n\n"
            "The agent evaluates added/removed columns, renames, type conflicts, "
            "missing-value and duplicate changes, and impact on prior findings."
        )
        return

    old_df, old_error = load_dataset_fn(previous_file)
    new_df, new_error = load_dataset_fn(updated_file)

    if old_error is not None:
        cards.finding_card("Read error", f"Unable to read previous dataset: {old_error}", "danger")
        return
    if new_error is not None:
        cards.finding_card("Read error", f"Unable to read updated dataset: {new_error}", "danger")
        return
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        cards.finding_card("Empty dataset", "One of the datasets contains no usable records.", "danger")
        return

    cards.finding_card("Loaded", "Previous and updated datasets loaded successfully.", "success")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Previous Dataset**")
        st.caption(previous_file.name)
        st.metric("Rows", f"{len(old_df):,}")
        st.metric("Columns", f"{len(old_df.columns):,}")
    with c2:
        st.markdown("**Updated Dataset**")
        st.caption(updated_file.name)
        st.metric("Rows", f"{len(new_df):,}")
        st.metric("Columns", f"{len(new_df.columns):,}")

    st.divider()
    previous_finding = st.text_area(
        "Previous important finding", value=most_important_finding, height=100, key="previous_finding"
    )

    if st.button("Run Dataset Intelligence", type="primary", key="run_change_analysis"):
        with st.spinner("Agent is comparing datasets and reconsidering previous findings..."):
            try:
                result = run_intelligence_analysis_fn(old_df, new_df, previous_finding)
                state.set_change_analysis(result)
            except Exception as e:
                cards.finding_card("Change intelligence failed", f"{type(e).__name__}: {e}", "danger")

    result = state.get_change_analysis()
    if result is None:
        return

    st.divider()
    st.markdown("### Intelligence Workflow")
    steps = ["Detect", "Analyze", "Compare", "Reconsider", "Re-plan", "Report"]
    cols = st.columns(6)
    for col, step in zip(cols, steps):
        with col:
            st.success(step)

    changes = result.get("changes", {})
    st.divider()
    st.markdown("### Detected Changes")

    added = changes.get("added_columns", [])
    removed = changes.get("removed_columns", [])
    type_changes = changes.get("type_changes", [])
    renames = changes.get("possible_renames", [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Added Columns", len(added))
    m2.metric("Removed Columns", len(removed))
    m3.metric("Type Changes", len(type_changes))
    m4.metric("Possible Renames", len(renames))

    for col in added:
        cards.finding_card("Added", col, "success")
    for col in removed:
        cards.finding_card("Removed", col, "danger")
    if type_changes:
        st.markdown("**Data-Type Changes**")
        st.dataframe(pd.DataFrame(type_changes), use_container_width=True, hide_index=True)
    if renames:
        st.markdown("**Possible Renamed Columns**")
        st.dataframe(pd.DataFrame(renames), use_container_width=True, hide_index=True)
    if not (added or removed or type_changes or renames):
        cards.finding_card("No structural changes", "Nothing changed structurally.", "success")

    quality = result.get("quality", {})
    st.divider()
    st.markdown("### Quality Comparison")
    q1, q2, q3 = st.columns(3)
    q1.metric("Previous Duplicates", quality.get("old_duplicates", 0))
    q2.metric("Updated Duplicates", quality.get("new_duplicates", 0))
    q3.metric("Duplicate Change", quality.get("duplicate_change", 0))

    missing_changes = quality.get("missing_changes", [])
    if missing_changes:
        st.markdown("**Missing-Value Changes**")
        st.dataframe(pd.DataFrame(missing_changes), use_container_width=True, hide_index=True)
    else:
        cards.finding_card("No missing-value changes", "Missing-value rates are unchanged.", "success")

    impact = result.get("impact", [])
    st.divider()
    st.markdown("### Impact Assessment")
    if impact:
        for item in impact:
            severity = item.get("severity", "MEDIUM")
            kind = {"HIGH": "danger", "MEDIUM": "warning"}.get(severity, "info")
            cards.finding_card(f"{item.get('area', 'UNKNOWN')} — {severity}", item.get("message", ""), kind)
    else:
        cards.finding_card("No significant impact", "No significant impact detected.", "success")

    reconsideration = result.get("reconsideration", {})
    st.divider()
    st.markdown("### Reconsider Previous Finding")
    status = reconsideration.get("status", "UNKNOWN")
    if status == "RECONSIDER":
        cards.finding_card("Reconsider", "Previous finding should be reconsidered.", "warning")
        for w in reconsideration.get("warnings", []):
            cards.finding_card("Warning", w, "warning")
    elif status == "STILL_VALID":
        cards.finding_card("Still valid", "Previous finding is still valid.", "success")
    else:
        cards.finding_card("No prior finding", "No previous finding was supplied.", "info")

    recommendation = result.get("recommendation", "UNKNOWN")
    st.divider()
    st.markdown("### Final Recommendation")
    kind = {
        "READY FOR FURTHER ANALYSIS": "success",
        "NEEDS ATTENTION BEFORE ANALYSIS": "warning",
        "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION": "danger",
    }.get(recommendation, "info")
    cards.finding_card("Recommendation", recommendation, kind)