"""Feedback Loop tab — surfaces iterative model improvement results from the AutoEDA controller."""

import streamlit as st


def render() -> None:
    st.markdown(
        '<p class="tab-header">🔄 Feedback Loop</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "The Feedback Loop tab will display iterative model improvement "
        "results driven by the AutoEDA controller.\n\n"
        "Run the pipeline from **LLM Decisions** first to populate results here."
    )

    if st.session_state.get("feedback_loop_results") is not None:
        st.json(st.session_state["feedback_loop_results"])
    else:
        st.warning("No feedback loop results yet. Run the LLM pipeline first.")
