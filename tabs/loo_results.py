"""LOO Results tab — Leave-One-Out cross-validation + per-target diagnostic plots."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble    import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics      import confusion_matrix

from pipeline.models.xgboost_model        import XGBoostModel
from pipeline.models.neural_network_model import NeuralNetworkModel
from pipeline.models.hpo import (
    optimize_xgboost,
    optimize_neural_network,
    optimize_tabpfn,
    TabPFNWrapper,
    _OPTUNA_AVAILABLE,
)

from tabs._shared    import PLOT_THEME, safe_regression_metrics, render_parity_plot
from tabs._loo_utils import run_loo_with_wrapper

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: F401
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False


def render(
    df: pd.DataFrame,
    *,
    target_cols: list[str],
    target_col_input: str,
    task_type: str,
    estimator_choice: str,
) -> None:
    st.markdown('<p class="tab-header">LOO Cross-Validation Results</p>',
                unsafe_allow_html=True)

    # LOO is a single-target operation. Let the user pick which of the
    # selected targets to evaluate.
    if len(target_cols) > 1:
        loo_target = st.selectbox(
            "Target for LOO",
            target_cols,
            index=0,
            help="LOO evaluates one target at a time. Pick which of the selected targets to use.",
        )
        if loo_target != target_col_input:
            st.warning(
                f"`{loo_target}` is not the primary target (`{target_col_input}`). "
                "The LLM-built preprocessing pipeline was fit with the primary as target. "
                "If you want LOO on `{loo_target}` with it as primary, list it first in the sidebar and re-run LLM Decisions."
            )
    else:
        loo_target = target_col_input

    st.info(
        f"Selected estimator: **{estimator_choice}** "
        f"(change in sidebar under *Pipeline Settings*)"
    )

    # ── HPO Controls ─────────────────────────────────────────────────
    with st.expander("⚙️ Hyperparameter Optimization (optional)", expanded=False):
        if not _OPTUNA_AVAILABLE:
            st.warning("Optuna not installed. Run: `pip install optuna`")
            enable_hpo  = False
            hpo_trials  = 20
            hpo_cv_folds = 5
        else:
            hpo_col1, hpo_col2, hpo_col3 = st.columns(3)
            with hpo_col1:
                enable_hpo = st.checkbox(
                    "Enable HPO before LOO",
                    value=False,
                    help=(
                        "Runs Optuna KFold search to find best hyperparameters, "
                        "then uses them for LOO evaluation. "
                        "Not available for Random Forest / Logistic Regression / Ridge."
                    ),
                )
            with hpo_col2:
                hpo_trials = st.slider(
                    "HPO Trials", min_value=5, max_value=100,
                    value=20, step=5,
                    help="Number of Optuna trials. More = better params, slower search.",
                )
            with hpo_col3:
                hpo_cv_folds = st.slider(
                    "HPO CV Folds", min_value=3, max_value=10,
                    value=5, step=1,
                    help="K-Fold splits used inside each Optuna trial.",
                )

        if enable_hpo and estimator_choice in ("Random Forest", "Logistic Regression / Ridge"):
            st.warning(
                f"HPO is not supported for **{estimator_choice}** — "
                "it will run with default parameters."
            )
            enable_hpo = False

        if enable_hpo:
            st.info(
                "**The LOO score below is not fully held out when HPO is on.** "
                "Hyperparameters are searched over *every* row, then LOO is run "
                "on those same rows — so each held-out point already influenced "
                "the parameters being evaluated. A clean estimate needs nested "
                "CV (a search inside every fold), which costs folds × trials.\n\n"
                "Measured on the bundled `Fish.csv` with XGBoost, nested vs "
                "not: **ΔR² = −0.0013** — i.e. nothing, and the wrong sign for "
                "a leak. Expect it to stay small where hyperparameters barely "
                "move the score, and to matter more on harder targets or "
                "smaller samples. Treat *Best CV score* as the search's own "
                "objective, not as expected performance: it is the maximum over "
                "trials, so it is optimistic by construction."
            )

        if st.session_state.hpo_results is not None:
            prev = st.session_state.hpo_results
            st.success(
                f"Last HPO run: **{prev['estimator']}** | "
                f"Best CV score: **{prev['best_value']:.4f}**"
            )
            st.json(prev["best_params"])

    if st.button("Run LOO Cross-Validation", key="btn_run_loo"):
        if st.session_state.pipeline is None:
            st.warning(
                "Run LLM Analysis first (Tab: LLM Decisions) "
                "to preprocess data."
            )
        elif estimator_choice == "TabPFN" and not _TABPFN_AVAILABLE:
            st.error(
                "TabPFN is not installed. "
                "Run: `pip install tabpfn` and restart."
            )
            st.code("pip install tabpfn", language="bash")
        else:
            if estimator_choice == "TabPFN":
                st.info(
                    "TabPFN LOO: each fold fits a fresh TabPFN instance. "
                    "This may take 1–3 minutes on CPU."
                )

            try:
                # ── Random Forest ─────────────────────────────
                if estimator_choice == "Random Forest":
                    with st.spinner("Running LOO with Random Forest…"):
                        est = (
                            RandomForestClassifier(
                                n_estimators=50, random_state=42
                            )
                            if task_type == "classification"
                            else RandomForestRegressor(
                                n_estimators=50, random_state=42
                            )
                        )
                        loo_out = st.session_state.pipeline.run_loo(df, est)

                # ── Logistic Regression / Ridge ───────────────
                elif estimator_choice == "Logistic Regression / Ridge":
                    with st.spinner("Running LOO with Logistic Regression / Ridge…"):
                        est = (
                            LogisticRegression(max_iter=1000)
                            if task_type == "classification"
                            else Ridge()
                        )
                        loo_out = st.session_state.pipeline.run_loo(df, est)

                # ── XGBoost ───────────────────────────────────
                elif estimator_choice == "XGBoost":
                    t_df  = st.session_state.pipeline.get_transformed_df(df)
                    X_loo = t_df.drop(
                        columns=target_cols, errors="ignore"
                    ).values.astype(np.float32)
                    y_loo = df[loo_target].values
                    if task_type == "classification":
                        y_loo = LabelEncoder().fit_transform(
                            y_loo
                        ).astype(np.float32)
                    else:
                        y_loo = y_loo.astype(np.float32)

                    xgb_hpo_params = {}
                    if enable_hpo:
                        with st.spinner(
                            f"HPO: searching XGBoost params "
                            f"({hpo_trials} trials, {hpo_cv_folds}-fold CV)…"
                        ):
                            hpo_out = optimize_xgboost(
                                X_loo, y_loo,
                                task         = task_type,
                                n_trials     = hpo_trials,
                                n_splits     = hpo_cv_folds,
                                random_state = 42,
                            )
                        xgb_hpo_params = hpo_out["best_params"]
                        st.session_state.hpo_results = {
                            "estimator":   "XGBoost",
                            "best_params": xgb_hpo_params,
                            "best_value":  hpo_out["best_value"],
                        }
                        st.success(
                            f"HPO done — best CV score: "
                            f"**{hpo_out['best_value']:.4f}**"
                        )
                        st.json(xgb_hpo_params)

                    with st.spinner(
                        f"Running LOO with XGBoost"
                        f"{' (HPO params)' if enable_hpo else ''}…"
                    ):
                        loo_out = run_loo_with_wrapper(
                            XGBoostModel(
                                task         = task_type,
                                n_splits     = 5,
                                random_state = 42,
                                hpo_params   = xgb_hpo_params,
                            ),
                            X_loo, y_loo, task=task_type,
                        )

                # ── Neural Network ────────────────────────────
                elif estimator_choice == "Neural Network":
                    t_df  = st.session_state.pipeline.get_transformed_df(df)
                    X_loo = t_df.drop(
                        columns=target_cols, errors="ignore"
                    ).values.astype(np.float32)
                    y_loo = df[loo_target].values
                    if task_type == "classification":
                        y_loo = LabelEncoder().fit_transform(
                            y_loo
                        ).astype(np.float32)
                    else:
                        y_loo = y_loo.astype(np.float32)

                    nn_kwargs = dict(
                        task         = task_type,
                        n_splits     = 5,
                        random_state = 42,
                        epochs       = 30,
                        patience     = 5,
                    )
                    if enable_hpo:
                        with st.spinner(
                            f"HPO: searching Neural Network params "
                            f"({hpo_trials} trials, {hpo_cv_folds}-fold CV)…"
                        ):
                            hpo_out = optimize_neural_network(
                                X_loo, y_loo,
                                task         = task_type,
                                n_trials     = hpo_trials,
                                n_splits     = hpo_cv_folds,
                                random_state = 42,
                            )
                        bp = hpo_out["best_params"]
                        nn_kwargs.update(bp)
                        # Match the training protocol used inside HPO
                        # (_nn_fold_score_v2 runs 150 epochs / patience 15).
                        # Without this, lr/dropout/arch are sub-optimal at 30 epochs.
                        nn_kwargs["epochs"]   = 150
                        nn_kwargs["patience"] = 15
                        arch = hpo_out.get("arch_detail", {})
                        st.session_state.hpo_results = {
                            "estimator":   "Neural Network",
                            "best_params": bp,
                            "best_value":  hpo_out["best_value"],
                        }
                        st.success(
                            f"HPO done — best CV score: **{hpo_out['best_value']:.4f}** | "
                            f"arch: {arch.get('arch_type','?')} {arch.get('base_size','?')}×{arch.get('n_layers','?')}"
                        )
                        display_bp = {k: (list(v) if isinstance(v, tuple) else v) for k, v in bp.items()}
                        if arch:
                            display_bp["_arch_detail"] = arch
                        st.json(display_bp)

                    with st.spinner(
                        f"Running LOO with Neural Network"
                        f"{' (HPO params)' if enable_hpo else ''}…"
                    ):
                        loo_out = run_loo_with_wrapper(
                            NeuralNetworkModel(**nn_kwargs),
                            X_loo, y_loo, task=task_type,
                        )

                # ── TabPFN ────────────────────────────────────
                elif estimator_choice == "TabPFN":
                    t_df  = st.session_state.pipeline.get_transformed_df(df)
                    X_loo = t_df.drop(
                        columns=target_cols, errors="ignore"
                    ).values.astype(np.float32)
                    y_loo = df[loo_target].values

                    n_loo_rows, n_loo_feats = X_loo.shape
                    if n_loo_rows > 10_000 or n_loo_feats > 100:
                        st.warning(
                            f"TabPFN: {n_loo_rows:,} rows × "
                            f"{n_loo_feats} features — exceeds "
                            "optimal limits. Proceeding, may be slow."
                        )

                    if task_type == "classification":
                        y_loo = LabelEncoder().fit_transform(
                            y_loo.astype(str)
                        ).astype(np.float32)
                    else:
                        y_loo = y_loo.astype(np.float32)

                    tabpfn_wrapper = TabPFNWrapper(
                        task              = task_type,
                        n_estimators      = 8,
                        preprocessor_type = "none",
                        random_state      = 42,
                    )

                    if enable_hpo:
                        with st.spinner(
                            f"HPO: searching TabPFN params "
                            f"({hpo_trials} trials, {hpo_cv_folds}-fold CV)…"
                        ):
                            hpo_out = optimize_tabpfn(
                                X_loo, y_loo,
                                task         = task_type,
                                n_trials     = hpo_trials,
                                n_splits     = hpo_cv_folds,
                                random_state = 42,
                            )
                        bp = hpo_out["best_params"]
                        _core_keys = {"n_estimators", "preprocessor_type"}
                        tabpfn_wrapper = TabPFNWrapper(
                            task              = task_type,
                            n_estimators      = bp.get("n_estimators", 8),
                            preprocessor_type = bp.get("preprocessor_type", "none"),
                            random_state      = 42,
                            extra_kwargs      = {k: v for k, v in bp.items()
                                                 if k not in _core_keys},
                        )
                        st.session_state.hpo_results = {
                            "estimator":   "TabPFN",
                            "best_params": bp,
                            "best_value":  hpo_out["best_value"],
                        }
                        st.success(
                            f"HPO done — best CV score: **{hpo_out['best_value']:.4f}** | "
                            f"n_estimators={bp.get('n_estimators',8)} | "
                            f"preprocessor={bp.get('preprocessor_type','none')}"
                        )
                        st.json(bp)

                    with st.spinner(
                        f"Running LOO with TabPFN"
                        f"{' (HPO params)' if enable_hpo else ''}…"
                    ):
                        loo_out = run_loo_with_wrapper(
                            tabpfn_wrapper, X_loo, y_loo, task=task_type,
                        )

                st.session_state.loo_results = loo_out
                st.success(f"LOO complete using **{estimator_choice}**!")

            except Exception as e:
                import traceback
                st.error(f"LOO Error: {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())

    # ── Display LOO results ───────────────────────────────────────────
    if st.session_state.loo_results is None:
        return

    loo_out = st.session_state.loo_results

    if task_type == "classification":
        acc_scores = loo_out.get("test_accuracy",    np.array([]))
        f1_scores  = loo_out.get("test_f1_weighted", np.array([]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Accuracy", f"{np.mean(acc_scores):.4f}")
        c2.metric("Std Accuracy",  f"{np.std(acc_scores):.4f}")
        c3.metric("Mean F1",       f"{np.mean(f1_scores):.4f}")
        c4.metric("Std F1",        f"{np.std(f1_scores):.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=acc_scores, mode="lines+markers",
                line=dict(color="#58a6ff", width=1.5),
                marker=dict(size=4), name="Accuracy"
            ))
            fig_acc.add_hline(
                y=np.mean(acc_scores), line_dash="dash",
                line_color="#f85149",
                annotation_text=f"Mean={np.mean(acc_scores):.3f}"
            )
            fig_acc.update_layout(
                title=f"LOO Accuracy — {estimator_choice}",
                xaxis_title="Fold", yaxis_title="Accuracy",
                **PLOT_THEME
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                y=f1_scores, mode="lines+markers",
                line=dict(color="#56d364", width=1.5),
                marker=dict(size=4), name="F1"
            ))
            fig_f1.add_hline(
                y=np.mean(f1_scores), line_dash="dash",
                line_color="#f85149",
                annotation_text=f"Mean={np.mean(f1_scores):.3f}"
            )
            fig_f1.update_layout(
                title=f"LOO F1 Score — {estimator_choice}",
                xaxis_title="Fold", yaxis_title="F1 Weighted",
                **PLOT_THEME
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        y_true_loo = loo_out.get("y_true")
        y_pred_loo = loo_out.get("y_pred")
        if y_true_loo is not None and y_pred_loo is not None:
            try:
                y_true_cm = np.array(y_true_loo, dtype=float).ravel()
                y_pred_cm = np.array(y_pred_loo, dtype=float).ravel()
                valid_cm  = ~(np.isnan(y_true_cm) | np.isnan(y_pred_cm))
                y_true_cm = y_true_cm[valid_cm].astype(int)
                y_pred_cm = np.round(y_pred_cm[valid_cm]).astype(int)
                known_labels = np.unique(y_true_cm)
                y_pred_cm = np.clip(
                    y_pred_cm, known_labels.min(), known_labels.max()
                )
                if len(y_true_cm) > 0:
                    cm     = confusion_matrix(y_true_cm, y_pred_cm)
                    labels = sorted(set(y_true_cm.tolist()))
                    fig_cm = px.imshow(
                        cm,
                        x=[str(l) for l in labels],
                        y=[str(l) for l in labels],
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix — {estimator_choice}",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                    )
                    fig_cm.update_layout(**PLOT_THEME, height=420)
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.markdown("#### Per-Class Accuracy")
                    per_class = []
                    for lbl in labels:
                        mask_lbl  = y_true_cm == lbl
                        correct   = int((y_pred_cm[mask_lbl] == lbl).sum())
                        total_lbl = int(mask_lbl.sum())
                        per_class.append({
                            "Class":    str(lbl),
                            "Correct":  correct,
                            "Total":    total_lbl,
                            "Accuracy": round(
                                correct / total_lbl if total_lbl > 0 else 0.0, 4
                            ),
                        })
                    st.dataframe(
                        pd.DataFrame(per_class), use_container_width=True
                    )
            except Exception as cm_err:
                st.warning(f"Could not render confusion matrix: {cm_err}")

        return

    # regression
    y_true_loo = loo_out.get("y_true")
    y_pred_loo = loo_out.get("y_pred")
    if y_true_loo is None or y_pred_loo is None:
        st.info(
            "No y_true / y_pred found in LOO results. "
            "Re-run LOO to generate predictions."
        )
        return

    y_true_arr = np.array(y_true_loo, dtype=float).ravel()
    y_pred_arr = np.array(y_pred_loo, dtype=float).ravel()
    valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[valid_mask]
    y_pred_arr = y_pred_arr[valid_mask]

    if len(y_true_arr) == 0:
        st.warning("No valid predictions available for parity plot.")
        return

    loo_metrics = safe_regression_metrics(y_true_arr, y_pred_arr)

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("R²",   f"{loo_metrics.get('r2',   float('nan')):.4f}")
    pm2.metric("RMSE", f"{loo_metrics.get('rmse', float('nan')):.4f}")
    pm3.metric("MAE",  f"{loo_metrics.get('mae',  float('nan')):.4f}")
    mape_val = loo_metrics.get("mape", float("nan"))
    pm4.metric("MAPE", f"{mape_val:.2f}%" if not np.isnan(mape_val) else "N/A")

    st.markdown("---")
    st.markdown("#### Parity Plot (Actual vs Predicted)")
    render_parity_plot(
        y_true=y_true_arr, y_pred=y_pred_arr,
        target_col=loo_target, metrics=loo_metrics,
    )

    st.markdown("#### Residual Distribution")
    residuals = y_pred_arr - y_true_arr
    fig_res   = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals vs Predicted", "Residual Histogram"]
    )
    fig_res.add_trace(
        go.Scatter(
            x=y_pred_arr, y=residuals, mode="markers",
            marker=dict(
                color=residuals, colorscale="RdYlGn_r",
                size=5, opacity=0.7,
            ),
            name="Residuals",
        ),
        row=1, col=1
    )
    fig_res.add_hline(
        y=0, line_dash="dash", line_color="#f85149", row=1, col=1
    )
    fig_res.add_trace(
        go.Histogram(
            x=residuals, nbinsx=30,
            marker_color="#58a6ff", name="Residual Dist",
        ),
        row=1, col=2
    )
    fig_res.update_layout(
        **PLOT_THEME, height=380,
        title_text=f"Residual Analysis — {estimator_choice}",
        showlegend=False,
    )
    fig_res.update_xaxes(title_text="Predicted", row=1, col=1)
    fig_res.update_yaxes(title_text="Residual",  row=1, col=1)
    fig_res.update_xaxes(title_text="Residual",  row=1, col=2)
    fig_res.update_yaxes(title_text="Count",     row=1, col=2)
    st.plotly_chart(fig_res, use_container_width=True)
