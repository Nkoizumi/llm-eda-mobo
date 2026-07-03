"""Outliers tab — IQR / Z-Score / Isolation Forest counts + per-feature box plot."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from tabs._shared import PLOT_THEME


def render(df: pd.DataFrame, *, target_cols: list[str]) -> None:
    st.markdown('<p class="tab-header">Outlier Detection</p>',
                unsafe_allow_html=True)

    out_method = st.radio(
        "Method", ["IQR", "Z-Score", "Isolation Forest"], horizontal=True
    )
    iqr_mult = st.slider("IQR / Z-Score Multiplier", 1.0, 3.0, 1.5, 0.1)

    num_cols_out = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in target_cols
    ]
    outlier_counts = {}

    if out_method in ("IQR", "Z-Score"):
        for col in num_cols_out:
            series = df[col].dropna()
            if out_method == "IQR":
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR    = Q3 - Q1
                mask   = (
                    (series < Q1 - iqr_mult * IQR) |
                    (series > Q3 + iqr_mult * IQR)
                )
            else:
                mean, std = series.mean(), series.std()
                mask = (
                    (series < mean - iqr_mult * std) |
                    (series > mean + iqr_mult * std)
                )
            outlier_counts[col] = int(mask.sum())

        out_df = pd.DataFrame({
            "Feature":       list(outlier_counts.keys()),
            "Outlier Count": list(outlier_counts.values()),
            "Outlier %":     [
                round(100 * v / len(df), 2) for v in outlier_counts.values()
            ],
        }).sort_values("Outlier Count", ascending=False)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(out_df, use_container_width=True)
        with col2:
            fig_out = px.bar(
                out_df.query("`Outlier Count` > 0"),
                x="Feature", y="Outlier %",
                title=f"Outliers Detected ({out_method})",
                color="Outlier %",
                color_continuous_scale="Oranges",
                text="Outlier Count"
            )
            fig_out.update_layout(**PLOT_THEME)
            st.plotly_chart(fig_out, use_container_width=True)

    if num_cols_out:
        sel_out_feat = st.selectbox("Inspect Feature Box Plot", num_cols_out)
        fig_box = px.box(
            df, y=sel_out_feat,
            title=f"Box Plot: {sel_out_feat}",
            color_discrete_sequence=["#58a6ff"],
            points="outliers"
        )
        fig_box.update_layout(**PLOT_THEME)
        st.plotly_chart(fig_box, use_container_width=True)
