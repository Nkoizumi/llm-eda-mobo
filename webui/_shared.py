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


def pipeline_expected_features(pipeline_obj) -> set[str]:
    """Names of columns the cached preprocessor expects to receive as features."""
    expected: set[str] = set()
    try:
        pre = pipeline_obj.pipeline_.named_steps["preprocessor"]
        for _name, _trans, cols in pre.transformers:
            if cols and cols != "drop":
                expected.update(cols)
    except Exception:
        pass
    return expected


def stale_pipeline_targets(target_cols) -> set[str]:
    """Targets that are also in the cached pipeline's feature set.

    Non-empty when the pipeline was built before ``target_cols`` was expanded —
    e.g. user clicked Run LLM Ensemble with one target, then changed the sidebar
    to three. Returns ``set()`` when no pipeline is cached or the cache matches
    the current target list.
    """
    pipeline_obj = st.session_state.get("pipeline")
    if pipeline_obj is None:
        return set()
    return set(target_cols) & pipeline_expected_features(pipeline_obj)


def warn_if_stale_pipeline(target_cols) -> bool:
    """Render an st.error when the cached pipeline is stale. Returns True if stale."""
    stale = stale_pipeline_targets(target_cols)
    if not stale:
        return False
    st.error(
        f"⚠️ The cached preprocessing pipeline was built when "
        f"`{', '.join(sorted(stale))}` were still treated as features, not "
        "targets — running it now would leak them into X under transformed "
        "names like `num__"
        + sorted(stale)[0] +
        "`.\n\n"
        "**Fix:** go to **LLM Decisions** and click "
        "*Run Local LLM Ensemble Analysis* again so the pipeline is rebuilt "
        "with the current target list."
    )
    return True
