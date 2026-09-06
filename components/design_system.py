"""
components/design_system.py

Centralized design-token system for the AI Dataset Intelligence platform.
Implements an enterprise dark/light hybrid aesthetic inspired by
Vercel, Stripe, and Datadog dashboards.

All color, spacing, and typography decisions live here so the rest of
the codebase never hardcodes a hex value or px unit.
"""

from __future__ import annotations
import streamlit as st


# ============================================================
# DESIGN TOKENS
# ============================================================

LIGHT_TOKENS: dict[str, str] = {
    "bg-app": "#f7f8fa",
    "bg-surface": "#ffffff",
    "bg-surface-alt": "#f2f3f5",
    "bg-inverse": "#0b0d12",
    "border": "#e4e7ec",
    "border-strong": "#d0d5dd",
    "text-primary": "#0b0d12",
    "text-secondary": "#5b6472",
    "text-inverse": "#f5f6f8",
    "accent": "#5b5bf7",
    "accent-soft": "rgba(91, 91, 247, 0.10)",
    "accent-2": "#0ea5e9",
    "success": "#12855c",
    "success-soft": "#e3f8ef",
    "warning": "#a15c07",
    "warning-soft": "#fdf1dd",
    "danger": "#c0293c",
    "danger-soft": "#fdeaec",
    "shadow": "0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.06)",
}

DARK_TOKENS: dict[str, str] = {
    "bg-app": "#0a0b0f",
    "bg-surface": "#111319",
    "bg-surface-alt": "#161923",
    "bg-inverse": "#f5f6f8",
    "border": "#22242e",
    "border-strong": "#2e313d",
    "text-primary": "#f2f3f5",
    "text-secondary": "#9aa1b1",
    "text-inverse": "#0b0d12",
    "accent": "#7c7cfb",
    "accent-soft": "rgba(124, 124, 251, 0.14)",
    "accent-2": "#38bdf8",
    "success": "#3ddc97",
    "success-soft": "rgba(61, 220, 151, 0.10)",
    "warning": "#f6b93b",
    "warning-soft": "rgba(246, 185, 59, 0.10)",
    "danger": "#fb7185",
    "danger-soft": "rgba(251, 113, 133, 0.10)",
    "shadow": "0 1px 2px rgba(0,0,0,0.24), 0 10px 30px rgba(0,0,0,0.35)",
}


def get_tokens(theme: str) -> dict[str, str]:
    """Return the token dict for 'dark' or 'light'."""
    return DARK_TOKENS if theme == "dark" else LIGHT_TOKENS


# ============================================================
# CSS BUILDER
# ============================================================

def _build_css(t: dict[str, str]) -> str:
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'system-ui', sans-serif;
        letter-spacing: -0.011em;
    }}

    .stApp {{
        background: {t['bg-app']};
        color: {t['text-primary']};
    }}

    .main .block-container {{
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }}

    /* ---------------- HERO ---------------- */
    .hero-card {{
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        background: {t['bg-inverse']};
        border: 1px solid {t['border-strong']};
        box-shadow: {t['shadow']};
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        color: {t['text-inverse']};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }}
    .hero-title {{
        color: {t['text-inverse']};
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }}
    .hero-subtitle {{
        color: {t['text-inverse']};
        opacity: 0.65;
        font-size: 0.98rem;
        max-width: 640px;
    }}

    /* ---------------- CARDS ---------------- */
    .info-card {{
        padding: 18px 20px;
        border-radius: 14px;
        background: {t['bg-surface']};
        border: 1px solid {t['border']};
        box-shadow: {t['shadow']};
        margin-bottom: 14px;
    }}
    .info-card-title {{
        font-size: 0.72rem;
        font-weight: 700;
        color: {t['text-secondary']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .info-card-value {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {t['text-primary']};
        letter-spacing: -0.02em;
    }}
    .info-card-description {{
        font-size: 0.83rem;
        color: {t['text-secondary']};
        margin-top: 4px;
    }}

    .ml-card {{
        padding: 20px 22px;
        border-radius: 14px;
        background: linear-gradient(135deg, {t['bg-inverse']}, {t['bg-inverse']});
        border: 1px solid {t['border-strong']};
        color: {t['text-inverse']};
        box-shadow: {t['shadow']};
        margin-bottom: 16px;
    }}
    .ml-card-title {{
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .ml-card-text {{
        opacity: 0.75;
        font-size: 0.88rem;
    }}

    /* ---------------- FINDING CARDS ---------------- */
    .finding-card {{
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid {t['border']};
        font-size: 0.87rem;
        line-height: 1.55;
    }}
    .finding-success {{ background: {t['success-soft']}; border-color: {t['success']}33; color: {t['text-primary']}; }}
    .finding-warning {{ background: {t['warning-soft']}; border-color: {t['warning']}33; color: {t['text-primary']}; }}
    .finding-danger  {{ background: {t['danger-soft']};  border-color: {t['danger']}33;  color: {t['text-primary']}; }}
    .finding-info    {{ background: {t['accent-soft']};  border-color: {t['accent']}33;  color: {t['text-primary']}; }}

    /* ---------------- SECTION HEADERS ---------------- */
    .section-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {t['text-primary']};
        letter-spacing: -0.02em;
        margin-top: 6px;
        margin-bottom: 3px;
    }}
    .section-subtitle {{
        color: {t['text-secondary']};
        font-size: 0.88rem;
        margin-bottom: 16px;
    }}

    /* ---------------- STATUS PILLS ---------------- */
    .status-pill {{
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 700;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid transparent;
    }}
    .status-success {{ background: {t['success-soft']}; color: {t['success']}; border-color: {t['success']}33; }}
    .status-warning {{ background: {t['warning-soft']}; color: {t['warning']}; border-color: {t['warning']}33; }}
    .status-error   {{ background: {t['danger-soft']};  color: {t['danger']};  border-color: {t['danger']}33; }}

    /* ---------------- STREAMLIT NATIVE OVERRIDES ---------------- */
    div[data-testid="stMetric"] {{
        background: {t['bg-surface']};
        border: 1px solid {t['border']};
        padding: 12px 14px;
        border-radius: 12px;
        box-shadow: {t['shadow']};
    }}
    div[data-testid="stMetricLabel"] {{ color: {t['text-secondary']}; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: {t['text-primary']}; font-weight: 800; }}

    .stButton > button {{
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid {t['border-strong']};
        background: {t['bg-surface']};
        color: {t['text-primary']};
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {t['accent']};
        color: {t['accent']};
        transform: translateY(-1px);
    }}
    button[kind="primary"] {{
        background: {t['accent']} !important;
        border-color: {t['accent']} !important;
        color: white !important;
        box-shadow: 0 6px 18px {t['accent-soft']};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        padding: 4px;
        background: {t['bg-surface-alt']};
        border-radius: 12px;
        border: 1px solid {t['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 600;
        color: {t['text-secondary']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['bg-surface']};
        color: {t['text-primary']};
        box-shadow: {t['shadow']};
    }}

    div[data-testid="stFileUploader"] {{
        border-radius: 12px;
        background: {t['bg-surface']};
        border: 1px dashed {t['border-strong']};
    }}

    .stDataFrame {{ border-radius: 12px; overflow: hidden; border: 1px solid {t['border']}; }}

    section[data-testid="stSidebar"] {{
        background: {t['bg-inverse']};
        border-right: 1px solid {t['border-strong']};
    }}
    section[data-testid="stSidebar"] * {{ color: {t['text-inverse']}; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.08); }}

    .footer {{
        text-align: center;
        padding: 22px;
        color: {t['text-secondary']};
        font-size: 0.8rem;
    }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border-strong']}; border-radius: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    """


def inject_design_system(theme: str = "dark") -> None:
    """Injects the full CSS design system for the given theme ('dark' | 'light')."""
    tokens = get_tokens(theme)
    st.markdown(f"<style>{_build_css(tokens)}</style>", unsafe_allow_html=True)
