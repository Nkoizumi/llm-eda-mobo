"""Shared constants and helpers for the auto-EDA Streamlit UI.

Imported by ``app.py`` and the per-tab modules in ``webui/``.
Grows as tabs are extracted; keep additions minimal and only promote
something here once it is genuinely reused across tabs.
"""

PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    font_family="JetBrains Mono",
)
