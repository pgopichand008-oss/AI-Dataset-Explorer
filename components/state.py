"""
components/state.py

Formal, typed session_state management. All reads/writes to
st.session_state should go through these functions so state shape
stays consistent across reruns and refactors.
"""

from __future__ import annotations
from typing import Any, Optional
import streamlit as st

DEFAULT_STATE: dict[str, Any] = {
    "theme": "dark",
    "ai_report": "",
    "loaded_file_name": "",
    "analysis_status": False,
    "previous_file_name": "",
    "updated_file_name": "",
    "change_analysis": None,
    "ml_analysis": None,
    "ml_file_name": "",
    "ml_target": "",
    "ml_run_target": "",
}


def init_state() -> None:
    """Populate st.session_state with defaults for any missing keys."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_(key: str, value: Any) -> None:
    st.session_state[key] = value


# ---- theme ----
def get_theme() -> str:
    return st.session_state.get("theme", "dark")


def set_theme(theme: str) -> None:
    st.session_state["theme"] = theme


def toggle_theme() -> None:
    st.session_state["theme"] = "light" if get_theme() == "dark" else "dark"


# ---- dataset lifecycle ----
def reset_for_new_dataset(file_name: str) -> None:
    st.session_state["loaded_file_name"] = file_name
    st.session_state["ai_report"] = ""
    st.session_state["analysis_status"] = False
    st.session_state["ml_analysis"] = None
    st.session_state["ml_file_name"] = ""
    st.session_state["ml_target"] = ""
    st.session_state["ml_run_target"] = ""


def reset_change_analysis(previous_name: str, updated_name: str) -> None:
    st.session_state["previous_file_name"] = previous_name
    st.session_state["updated_file_name"] = updated_name
    st.session_state["change_analysis"] = None


# ---- AI report ----
def set_ai_report(report: str) -> None:
    st.session_state["ai_report"] = report
    st.session_state["analysis_status"] = True


def get_ai_report() -> str:
    return st.session_state.get("ai_report", "")


# ---- ML ----
def set_ml_result(result: Optional[dict], file_name: str, target: str) -> None:
    st.session_state["ml_analysis"] = result
    st.session_state["ml_file_name"] = file_name
    st.session_state["ml_target"] = target
    st.session_state["ml_run_target"] = target


def get_ml_result() -> Optional[dict]:
    return st.session_state.get("ml_analysis")


# ---- Change intelligence ----
def set_change_analysis(result: dict) -> None:
    st.session_state["change_analysis"] = result


def get_change_analysis() -> Optional[dict]:
    return st.session_state.get("change_analysis")