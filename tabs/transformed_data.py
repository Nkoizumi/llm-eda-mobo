"""Transformed Data tab — runs the LLM-chosen preprocessing pipeline, lets the user download the result."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from tabs._shared import PLOT_THEME, warn_if_stale_pipeline


def render(df: pd.DataFrame, *, target_cols: list[str]) -> None:
    st.markdown('<p class="tab-header">Transformed Data Inspection</p>',
                unsafe_allow_html=True)

    if st.button("Transform Dataset", key="btn_transform"):
        if st.session_state.pipeline is None:
            st.warning("Run LLM Analysis first (Tab: LLM Decisions).")
        elif not warn_if_stale_pipeline(target_cols):
            with st.spinner("Transforming data..."):
                try:
                    # Drop ALL selected targets, not just the pipeline's primary
                    # one. AutoEDAPipeline.get_transformed_df only strips its
                    # own self.target_col, so any secondary target left in the
                    # frame would be re-transformed into a feature (e.g.
                    # compressive_strength -> num__compressive_strength) and
                    # leak.
                    X_only = df.drop(columns=target_cols, errors="ignore")
                    t_df = st.session_state.pipeline.get_transformed_df(X_only)
                    st.session_state.transformed_df = t_df
                    st.success(
                        f"Transformed: "
                        f"{t_df.shape[0]} rows × {t_df.shape[1]} columns"
                    )
                except Exception as e:
                    st.error(f"Transform Error: {e}")

    if st.session_state.transformed_df is None:
        return

    t_df = st.session_state.transformed_df

    td1, td2, td3 = st.columns(3)
    td1.metric("Features Before", len(df.columns))
    td2.metric("Features After",  len(t_df.columns))
    td3.metric("Delta", f"{len(t_df.columns) - len(df.columns):+d}")

    # ── Preview table ─────────────────────────────────────────────
    st.dataframe(t_df.head(20).round(4), use_container_width=True)

    # ── Download Section ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Download Transformed Data")

    dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 1])

    with dl_col1:
        csv_features = t_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Transformed Features (CSV)",
            data=csv_features,
            file_name="transformed_features.csv",
            mime="text/csv",
            help="Downloads only the preprocessed feature columns (no target).",
            use_container_width=True,
        )

    t_df_with_target = None
    with dl_col2:
        try:
            t_df_with_target = t_df.copy().reset_index(drop=True)
            for t_col in target_cols:
                if t_col in df.columns:
                    t_df_with_target[t_col] = (
                        df[t_col].reset_index(drop=True)
                    )

            csv_with_target = t_df_with_target.to_csv(
                index=False
            ).encode("utf-8")
            label = (
                "⬇️ Download With Target (CSV)"
                if len(target_cols) == 1
                else "⬇️ Download With Targets (CSV)"
            )
            st.download_button(
                label=label,
                data=csv_with_target,
                file_name="transformed_with_target.csv",
                mime="text/csv",
                help="Downloads transformed features with the original target column(s) appended.",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not append target column(s): {e}")

    with dl_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Download Size</div>
                <div class="value">{t_df.shape[0]:,} × {t_df.shape[1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### Excel Format")
    try:
        from io import BytesIO
        import openpyxl  # noqa: F401  (engine for pandas.ExcelWriter)

        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            t_df.to_excel(
                writer, sheet_name="Transformed", index=False
            )
            if t_df_with_target is not None and any(t in df.columns for t in target_cols):
                t_df_with_target.to_excel(
                    writer, sheet_name="With Target", index=False
                )
        excel_buf.seek(0)

        st.download_button(
            label="⬇️ Download as Excel (.xlsx)",
            data=excel_buf.read(),
            file_name="transformed_data.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            help="Downloads both sheets: Transformed features + With Target.",
            use_container_width=False,
        )
    except ImportError:
        st.info(
            "💡 Install openpyxl for Excel export: "
            "`pip install openpyxl`"
        )
    except Exception as e:
        st.warning(f"Excel export failed: {e}")

    # ── Before vs After Distribution Comparison ───────────────────
    st.markdown("---")
    st.markdown("#### Before vs After: Distribution Comparison")
    shared_num = [
        c for c in df.select_dtypes(include=np.number).columns
        if c in t_df.columns and c not in target_cols
    ]
    if shared_num:
        compare_feat = st.selectbox("Feature to Compare", shared_num)
        fig_compare  = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Before Transform", "After Transform"]
        )
        fig_compare.add_trace(
            go.Histogram(
                x=df[compare_feat].dropna(),
                marker_color="#58a6ff", nbinsx=30, name="Before"
            ),
            row=1, col=1
        )
        fig_compare.add_trace(
            go.Histogram(
                x=t_df[compare_feat].dropna(),
                marker_color="#56d364", nbinsx=30, name="After"
            ),
            row=1, col=2
        )
        fig_compare.update_layout(
            **PLOT_THEME, height=350,
            title_text=f"Distribution: {compare_feat}"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
