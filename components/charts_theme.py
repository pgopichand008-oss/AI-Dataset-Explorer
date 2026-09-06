"""
components/charts_theme.py

Applies a single, uniform visual theme to every Plotly figure in the
app so charts feel like one coherent product surface rather than
ad-hoc plots.
"""

from __future__ import annotations
import plotly.graph_objects as go
from components.design_system import get_tokens

DARK_COLORWAY = ["#7c7cfb", "#38bdf8", "#3ddc97", "#f6b93b", "#fb7185", "#a78bfa", "#22d3ee"]
LIGHT_COLORWAY = ["#5b5bf7", "#0ea5e9", "#12855c", "#a15c07", "#c0293c", "#7c3aed", "#0891b2"]


def apply_chart_theme(fig: go.Figure, theme: str = "dark", height: int = 430, title: str | None = None) -> go.Figure:
    """Apply the platform's unified chart theme to a Plotly figure in-place-safe manner."""
    t = get_tokens(theme)
    colorway = DARK_COLORWAY if theme == "dark" else LIGHT_COLORWAY

    fig.update_layout(
        height=height,
        title=title if title else fig.layout.title.text,
        colorway=colorway,
        paper_bgcolor=t["bg-surface"],
        plot_bgcolor=t["bg-surface"],
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12, color=t["text-primary"]),
        title_font=dict(size=15, color=t["text-primary"], family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["text-secondary"], size=11),
        ),
        hoverlabel=dict(
            bgcolor=t["bg-inverse"],
            font_color=t["text-inverse"],
            font_family="Inter, sans-serif",
            bordercolor=t["border-strong"],
        ),
    )
    fig.update_xaxes(
        gridcolor=t["border"], zerolinecolor=t["border"],
        tickfont=dict(color=t["text-secondary"], size=11),
        title_font=dict(color=t["text-secondary"], size=11),
    )
    fig.update_yaxes(
        gridcolor=t["border"], zerolinecolor=t["border"],
        tickfont=dict(color=t["text-secondary"], size=11),
        title_font=dict(color=t["text-secondary"], size=11),
    )
    return fig
