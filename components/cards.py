"""
components/cards.py

Reusable, theme-aware UI building blocks used across every view.
"""

from __future__ import annotations
import streamlit as st


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def info_card(title: str, value: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-value">{value}</div>
            <div class="info-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ml_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">{title}</div>
            <div class="ml-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, kind: str = "success") -> str:
    """Returns pill HTML; kind is one of success|warning|error."""
    return f'<div class="status-pill status-{kind}">{label}</div>'


def finding_card(title: str, body: str, kind: str = "info") -> None:
    """kind: success | warning | danger | info"""
    st.markdown(
        f"""
        <div class="finding-card finding-{kind}">
            <strong>{title}</strong><br>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )