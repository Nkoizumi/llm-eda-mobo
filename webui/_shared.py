"""Shared constants and helpers for the auto-EDA Streamlit UI.

Imported by ``app.py`` and the per-tab modules in ``webui/``.
Grows as tabs are extracted; keep additions minimal and only promote
something here once it is genuinely reused across tabs.
"""

import streamlit as st

PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    font_family="JetBrains Mono",
)


def render_latency(val: float, model_name: str) -> None:
    """Display a colour-coded latency badge for a local-LLM call."""
    if val <= 0:
        st.warning(
            f"⚠️ **{model_name}** latency = `{val}ms` — "
            f"Model may not have responded. Fallback was used."
        )
    elif val < 500:
        st.success(f"⚡ **{model_name}** responded in `{val:.0f}ms` — Fast!")
    elif val < 3000:
        st.info(f"🕐 **{model_name}** responded in `{val:.0f}ms` — Normal")
    else:
        st.warning(f"🐢 **{model_name}** responded in `{val:.0f}ms` — Slow!")
