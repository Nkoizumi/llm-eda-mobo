"""LLM Decisions tab — runs the Phi-4 + Mistral ensemble, visualizes agreement and conflicts."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.orchestrator     import AutoEDAPipeline
from pipeline.local_llm_engine import EnsembleDecision

from webui._shared import PLOT_THEME, render_latency


def render(
    df: pd.DataFrame,
    *,
    target_col_input: str,
    task_type: str,
    ollama_host: str,
    use_llm: bool,
    target_cols: list[str],
) -> None:
    st.markdown('<p class="tab-header">Phi-4 + Mistral Ensemble Decisions</p>',
                unsafe_allow_html=True)

    if st.button("Run Local LLM Ensemble Analysis", key="btn_run_llm"):
        with st.spinner("🤖 Querying Phi-4 + Mistral in parallel (local)..."):
            try:
                eda = AutoEDAPipeline(
                    target_col=target_col_input,
                    task=task_type,
                    ollama_host=ollama_host,
                    use_local_llm=use_llm
                )
                # Drop every selected target from the feature matrix so the
                # ensemble pipeline never sees one target as a feature
                # of another.
                X_only = df.drop(columns=target_cols, errors="ignore")
                eda.build_pipeline(X_only)
                st.session_state.pipeline  = eda
                st.session_state.decisions = eda.ensemble_result_
                st.success("✅ Ensemble analysis complete!")
            except Exception as e:
                st.error(f"LLM Error: {e}")

    if st.session_state.decisions is None:
        return

    ens: EnsembleDecision = st.session_state.decisions

    # ── Agreement gauge ───────────────────────────────────────────
    score = ens.agreement_score
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": "Model Agreement Score",
               "font": {"color": "#c9d1d9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#c9d1d9"},
            "bar":  {"color": "#58a6ff"},
            "steps": [
                {"range": [0,  50], "color": "#3d1c1c"},
                {"range": [50, 75], "color": "#2d2d0d"},
                {"range": [75, 100], "color": "#0d2d1c"},
            ],
            "threshold": {
                "line":      {"color": "#56d364", "width": 3},
                "thickness": 0.8,
                "value":     75,
            },
        },
        number={"suffix": "%", "font": {"color": "#58a6ff"}},
    ))
    fig_gauge.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        **PLOT_THEME
    )
    col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
    with col_g1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_g2:
        st.metric("Phi-4 Confidence",
                  f"{ens.phi4_decision.confidence:.0%}")
        render_latency(ens.phi4_decision.latency_ms, "Phi-4")
    with col_g3:
        st.metric("Mistral Confidence",
                  f"{ens.mistral_decision.confidence:.0%}")
        render_latency(ens.mistral_decision.latency_ms, "Mistral")

    if ens.tiebreak_used:
        n_conflicts       = len(ens.conflicts)
        gemma2_succeeded  = any(
            c.get("method") == "gemma2_tiebreak"
            for c in ens.conflicts
        )
        if gemma2_succeeded:
            st.warning(
                f"⚖️ **{n_conflicts} conflict(s) detected** — "
                f"Gemma2 acted as tiebreaker and resolved them."
            )
        else:
            st.warning(
                f"⚖️ **{n_conflicts} conflict(s) detected** — "
                f"Gemma2 tiebreaker failed. Phi-4 used as fallback."
            )

    # ── Side-by-side comparison ───────────────────────────────────
    st.markdown("#### Model Decision Comparison")
    compare_data = {
        "Decision": [
            "Imputation", "Power Transform",
            "Outlier Method", "Outlier Threshold",
            "Corr Threshold", "Scaler",
        ],
        "Phi-4": [
            ens.phi4_decision.imputation_strategy,
            ens.phi4_decision.power_transform,
            ens.phi4_decision.outlier_method,
            str(ens.phi4_decision.outlier_threshold),
            str(ens.phi4_decision.correlation_threshold),
            ens.phi4_decision.scaler,
        ],
        "Mistral": [
            ens.mistral_decision.imputation_strategy,
            ens.mistral_decision.power_transform,
            ens.mistral_decision.outlier_method,
            str(ens.mistral_decision.outlier_threshold),
            str(ens.mistral_decision.correlation_threshold),
            ens.mistral_decision.scaler,
        ],
        "Ensemble Final": [
            ens.final.imputation_strategy,
            ens.final.power_transform,
            ens.final.outlier_method,
            str(ens.final.outlier_threshold),
            str(ens.final.correlation_threshold),
            ens.final.scaler,
        ],
    }
    compare_df = pd.DataFrame(compare_data)

    def highlight_conflicts(row):
        if row["Phi-4"] != row["Mistral"]:
            return ["background-color:#3d2200;color:#ffa657"] * len(row)
        return [""] * len(row)

    st.dataframe(
        compare_df.style.apply(highlight_conflicts, axis=1),
        use_container_width=True
    )

    if ens.conflicts:
        st.markdown("#### Conflict Resolution Details")
        st.dataframe(
            pd.DataFrame(ens.conflicts), use_container_width=True
        )

    st.markdown("#### Model Reasoning")
    tab_phi4, tab_mistral, tab_json = st.tabs(
        ["Phi-4 Reasoning", "Mistral Reasoning", "Final JSON"]
    )
    with tab_phi4:
        st.info(
            ens.phi4_decision.reasoning_summary or "No reasoning returned."
        )
        with st.expander("Raw Phi-4 Response"):
            st.code(ens.phi4_decision.raw_response, language="json")
    with tab_mistral:
        st.info(
            ens.mistral_decision.reasoning_summary or "No reasoning returned."
        )
        with st.expander("Raw Mistral Response"):
            st.code(ens.mistral_decision.raw_response, language="json")
    with tab_json:
        from dataclasses import asdict
        st.json(asdict(ens.final))
