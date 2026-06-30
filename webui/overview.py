"""Overview tab — dataset summary metrics, preview, descriptive stats, dtypes, skewness."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from webui._shared import PLOT_THEME


def render(df: pd.DataFrame) -> None:
    st.markdown('<p class="tab-header">Dataset Overview</p>',
                unsafe_allow_html=True)

    num_cols      = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
    total_missing = df.isna().sum().sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col_widget, label, value in [
        (c1, "Rows",        f"{len(df):,}"),
        (c2, "Columns",     str(len(df.columns))),
        (c3, "Numeric",     str(len(num_cols))),
        (c4, "Categorical", str(len(cat_cols))),
        (c5, "Missing %",   f"{100 * total_missing / df.size:.1f}%"),
    ]:
        with col_widget:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Data Preview")
    st.dataframe(
        df.head(20).style.background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)
    with col_b:
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            "Column":   df.columns,
            "DType":    df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Null":     df.isna().sum().values,
            "Unique":   df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True)

    if num_cols:
        skew_vals = df[num_cols].skew().sort_values()
        fig_skew  = px.bar(
            x=skew_vals.values, y=skew_vals.index,
            orientation="h",
            title="Skewness per Numeric Feature",
            labels={"x": "Skewness", "y": "Feature"},
            color=skew_vals.values,
            color_continuous_scale="RdBu_r"
        )
        fig_skew.update_layout(**PLOT_THEME)
        fig_skew.add_vline(x= 0.5, line_dash="dash", line_color="#f85149")
        fig_skew.add_vline(x=-0.5, line_dash="dash", line_color="#f85149")
        st.plotly_chart(fig_skew, use_container_width=True)
