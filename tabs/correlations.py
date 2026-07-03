"""Correlations tab — pearson/spearman/kendall heatmap, high-|corr| pairs, feature-vs-target scatter."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from tabs._shared import PLOT_THEME


def render(df: pd.DataFrame, *, target_cols: list[str]) -> None:
    st.markdown('<p class="tab-header">Correlation Analysis</p>',
                unsafe_allow_html=True)

    num_cols_corr = df.select_dtypes(include=np.number).columns.tolist()
    corr_method   = st.radio(
        "Correlation Method", ["pearson", "spearman", "kendall"],
        horizontal=True
    )
    corr_matrix = df[num_cols_corr].corr(method=corr_method)

    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        title=f"{corr_method.capitalize()} Correlation Heatmap",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(**PLOT_THEME, height=550)
    fig_corr.update_traces(textfont_size=9)
    st.plotly_chart(fig_corr, use_container_width=True)

    threshold = st.slider("Highlight Threshold", 0.5, 1.0, 0.80, 0.01)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_pairs = [
        {
            "Feature 1":   c1,
            "Feature 2":   c2,
            "Correlation": round(upper.loc[c2, c1], 4),
        }
        for c1 in upper.columns
        for c2 in upper.index
        if not pd.isna(upper.loc[c2, c1])
        and abs(upper.loc[c2, c1]) >= threshold
    ]
    if high_pairs:
        high_df = pd.DataFrame(high_pairs).sort_values(
            "Correlation", ascending=False, key=abs
        )
        st.markdown(f"#### Pairs with |corr| ≥ {threshold}")
        st.dataframe(
            high_df.style.background_gradient(
                subset=["Correlation"], cmap="Reds"
            ),
            use_container_width=True
        )
    else:
        st.info(f"No pairs with |correlation| ≥ {threshold}")

    targets_in_df = [t for t in target_cols if t in df.columns]
    if targets_in_df and num_cols_corr:
        feat_choices = [c for c in num_cols_corr if c not in target_cols]
        if feat_choices:
            for t_idx, t_col in enumerate(targets_in_df):
                st.markdown(f"#### Feature vs Target: `{t_col}`")
                sel_corr_feat = st.selectbox(
                    "Select Feature", feat_choices, key=f"corr_feat_{t_idx}"
                )
                fig_scatter = px.scatter(
                    df, x=sel_corr_feat, y=t_col, trendline="ols",
                    color_discrete_sequence=["#58a6ff"],
                    title=f"{sel_corr_feat} vs {t_col}"
                )
                fig_scatter.update_layout(**PLOT_THEME)
                st.plotly_chart(fig_scatter, use_container_width=True)
