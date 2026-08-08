"""Distributions tab — histogram + Q-Q plot for a selected feature, per-feature stats, violin overview."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs._shared import PLOT_THEME


def render(df: pd.DataFrame, *, target_cols: list[str]) -> None:
    st.markdown('<p class="tab-header">Feature Distributions</p>',
                unsafe_allow_html=True)

    num_cols_disp = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in target_cols
    ]
    if not num_cols_disp:
        st.warning("No numeric feature columns found (excluding target(s)).")
        return

    selected_feat = st.selectbox("Select Feature", num_cols_disp)

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df, x=selected_feat, nbins=40, marginal="box",
            title=f"Distribution: {selected_feat}",
            color_discrete_sequence=["#58a6ff"]
        )
        fig_hist.update_layout(**PLOT_THEME)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        from scipy import stats
        clean = df[selected_feat].dropna()
        (osm, osr), (slope, intercept, _) = stats.probplot(
            clean, dist="norm"
        )
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(color="#58a6ff", size=5), name="Data"
        ))
        fig_qq.add_trace(go.Scatter(
            x=osm,
            y=slope * np.array(osm) + intercept,
            mode="lines",
            line=dict(color="#f85149", width=2),
            name="Normal"
        ))
        fig_qq.update_layout(
            title=f"Q-Q Plot: {selected_feat}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            **PLOT_THEME
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    stats_data = pd.DataFrame({
        "Feature":         num_cols_disp,
        "Mean":            [df[c].mean()            for c in num_cols_disp],
        "Std":             [df[c].std()             for c in num_cols_disp],
        "Skewness":        [round(df[c].skew(),     3) for c in num_cols_disp],
        "Kurtosis":        [round(df[c].kurtosis(), 3) for c in num_cols_disp],
        "Needs Transform": [
            "Yes" if abs(df[c].skew()) > 0.5 or abs(df[c].kurtosis()) > 3
            else "No"
            for c in num_cols_disp
        ],
    }).round(4)

    def highlight_transform(val):
        return (
            "background-color:#3d1c1c;color:#f85149;"
            if val == "Yes" else ""
        )

    # `Styler.map`, not `Styler.applymap`. The latter was deprecated in pandas
    # 2.1 and REMOVED in 3.0, and `pandas` is unpinned in requirements.txt — so
    # a fresh install picked up 3.x and this line raised AttributeError. Because
    # app.py renders this tab unconditionally, that crashed the whole app at
    # startup for anyone installing the project today. It stayed invisible in
    # development only because the author's environment still had pandas 2.3.
    st.dataframe(
        stats_data.style.map(
            highlight_transform, subset=["Needs Transform"]
        ),
        use_container_width=True
    )

    st.markdown("#### All Feature Distributions (Violin)")
    fig_violin = go.Figure()
    for col in num_cols_disp[:8]:
        fig_violin.add_trace(go.Violin(
            y=df[col].dropna(), name=col,
            box_visible=True, meanline_visible=True
        ))
    fig_violin.update_layout(
        title="Feature Distribution Violin Plots",
        **PLOT_THEME, height=450
    )
    st.plotly_chart(fig_violin, use_container_width=True)
