"""Results tab — per-target permutation importance + partial dependence on the joint MTL model."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from tabs._shared import PLOT_THEME


def render() -> None:
    st.markdown(
        '<p class="tab-header">📊 Results — Feature Importance & Partial Dependence</p>',
        unsafe_allow_html=True,
    )

    mtl = st.session_state.get("mtl_results")
    if mtl is None:
        st.info(
            "Run the **Multitask Learning** tab first to train a joint "
            "multi-output model. This tab then shows per-target feature "
            "importance and partial dependence plots (top-5 features per target)."
        )
        return

    from sklearn.inspection import (
        permutation_importance,
        PartialDependenceDisplay,
    )
    import matplotlib.pyplot as plt

    st.caption(
        f"Source: joint **{mtl['base_choice']}** model on targets "
        f"{', '.join(mtl['target_cols'])} (task: **{mtl['task']}**)."
    )

    X_full   = mtl["X"]
    Y_full   = mtl["Y"]
    feat_nm  = mtl["feature_names"]
    joint    = mtl["model"]            # MultiOutputRegressor/Classifier
    per_est  = joint.estimators_       # one fitted estimator per target

    top_n_pdp = st.slider(
        "PDP — top N features per target (by permutation importance)",
        min_value=1, max_value=min(10, len(feat_nm)),
        value=min(5, len(feat_nm)),
        key="results_top_n_pdp",
    )
    pi_repeats = st.slider(
        "Permutation importance — n_repeats",
        min_value=3, max_value=20, value=5,
        key="results_pi_repeats",
        help="More repeats = lower variance, slower. 5 is usually fine.",
    )

    tab_per_target = st.tabs(
        [f"Target: {t}" for t in mtl["target_cols"]]
    )
    for ti, t_col in enumerate(mtl["target_cols"]):
        with tab_per_target[ti]:
            sub_est = per_est[ti]
            y_t = Y_full[:, ti]

            # ── Feature importance ───────────────────────────────
            with st.spinner(
                f"Computing permutation importance for `{t_col}`…"
            ):
                try:
                    pi = permutation_importance(
                        sub_est, X_full, y_t,
                        n_repeats=pi_repeats,
                        random_state=42,
                        n_jobs=1,
                    )
                    importances_mean = pi.importances_mean
                    importances_std  = pi.importances_std
                except Exception as e:
                    st.warning(
                        f"Permutation importance failed ({e}); "
                        "falling back to model.feature_importances_."
                    )
                    if hasattr(sub_est, "feature_importances_"):
                        importances_mean = np.asarray(sub_est.feature_importances_)
                        importances_std  = np.zeros_like(importances_mean)
                    else:
                        importances_mean = np.zeros(X_full.shape[1])
                        importances_std  = np.zeros(X_full.shape[1])

            fi_df = pd.DataFrame({
                "Feature":    feat_nm,
                "Importance": importances_mean,
                "Std":        importances_std,
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            st.markdown(f"#### Feature Importance — `{t_col}`")
            fig_fi = px.bar(
                fi_df.head(20),
                x="Importance", y="Feature",
                orientation="h",
                error_x="Std",
                title=f"Permutation Importance (top 20) — `{t_col}`",
                color="Importance", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(**PLOT_THEME)
            fig_fi.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_fi, use_container_width=True)

            with st.expander("All feature importance values"):
                st.dataframe(fi_df, use_container_width=True)

            # ── Partial dependence (top-N) ───────────────────────
            st.markdown(
                f"#### Partial Dependence — top {top_n_pdp} features for `{t_col}`"
            )
            top_idx = (
                fi_df.head(top_n_pdp)["Feature"]
                .map(lambda f: feat_nm.index(f))
                .tolist()
            )
            if not top_idx:
                st.info("No features available for PDP.")
                continue

            try:
                n_cols_pdp = min(3, len(top_idx))
                n_rows_pdp = int(np.ceil(len(top_idx) / n_cols_pdp))
                fig_pdp, axes_pdp = plt.subplots(
                    n_rows_pdp, n_cols_pdp,
                    figsize=(4.5 * n_cols_pdp, 3.5 * n_rows_pdp),
                    squeeze=False,
                )
                # sklearn renders into the provided axes when ax= list
                ax_flat = axes_pdp.ravel().tolist()
                PartialDependenceDisplay.from_estimator(
                    sub_est, X_full, features=top_idx,
                    feature_names=feat_nm,
                    ax=ax_flat[:len(top_idx)],
                    grid_resolution=30,
                )
                # Blank any unused subplot
                for ax in ax_flat[len(top_idx):]:
                    ax.set_visible(False)
                fig_pdp.suptitle(
                    f"Partial Dependence — `{t_col}`",
                    fontsize=12,
                )
                fig_pdp.tight_layout()
                st.pyplot(fig_pdp, clear_figure=True)
                plt.close(fig_pdp)
            except Exception as e:
                st.warning(f"PDP failed for `{t_col}`: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
