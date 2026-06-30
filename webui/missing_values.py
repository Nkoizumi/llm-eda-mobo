"""Missing Values tab — per-column missing counts + missingness heatmap."""

import pandas as pd
import plotly.express as px
import streamlit as st

from webui._shared import PLOT_THEME


def render(df: pd.DataFrame) -> None:
    st.markdown('<p class="tab-header">Missing Value Analysis</p>',
                unsafe_allow_html=True)

    missing_df = pd.DataFrame({
        "Column":        df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %":     (100 * df.isna().mean()).round(2).values,
        "DType":         df.dtypes.astype(str).values,
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

    if len(missing_df) == 0:
        st.success("✅ No missing values found!")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(missing_df, use_container_width=True)
    with col2:
        fig_miss = px.bar(
            missing_df, x="Column", y="Missing %",
            title="Missing Value % per Column",
            color="Missing %", color_continuous_scale="Reds",
            text="Missing %"
        )
        fig_miss.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside"
        )
        fig_miss.update_layout(**PLOT_THEME)
        st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("#### Missingness Pattern Heatmap")
    miss_matrix = df.isna().astype(int)
    fig_heat = px.imshow(
        miss_matrix.T, aspect="auto",
        color_continuous_scale=["#161b22", "#f85149"],
        title="Missingness Map (red = missing)",
        labels={"x": "Row Index", "y": "Feature"}
    )
    fig_heat.update_layout(**PLOT_THEME, height=400)
    st.plotly_chart(fig_heat, use_container_width=True)
