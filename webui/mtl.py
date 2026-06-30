"""Multitask Learning tab — trains a joint MultiOutputRegressor/Classifier on all selected targets."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from webui._shared import PLOT_THEME, warn_if_stale_pipeline


def render(df: pd.DataFrame, *, target_cols: list[str], task_type: str) -> None:
    st.markdown(
        '<p class="tab-header">🎯 Multitask Learning</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Trains a **joint multi-output model** that predicts all "
        f"**{len(target_cols)}** selected target(s) at once, using sklearn's "
        "`MultiOutputRegressor` / `MultiOutputClassifier` wrapper around the "
        "base estimator below. CV splits are shared across targets so each "
        "target sees the same train/test rows.\n\n"
        "If LLM pipeline has been run (Tab: *LLM Decisions*), its preprocessed "
        "features are used. Otherwise a simple StandardScaler + median fill is "
        "applied to the numeric columns."
    )

    n_splits_mtl = st.slider("K-Fold Splits", 3, 10, 5, key="mtl_splits")
    base_choice  = st.selectbox(
        "Base Estimator",
        ["Random Forest", "XGBoost", "Neural Network"],
        key="mtl_base",
        help="Single per-target estimator wrapped by MultiOutputRegressor/Classifier.",
    )

    if st.button("Run Multitask Learning", key="btn_run_mtl"):
        if warn_if_stale_pipeline(target_cols):
            st.stop()
        try:
            from sklearn.multioutput import (
                MultiOutputRegressor,
                MultiOutputClassifier,
            )
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler

            # ── Build X (drop ALL selected targets) ───────────────────
            # Track the inverse-transform recipe so BO/MOBO can (later)
            # map candidates back to user-original units. Pipeline path
            # leaves scaler_mtl=None — pipelines can be multi-step and
            # don't all expose inverse_transform; warn-only for those.
            if st.session_state.get("pipeline") is not None:
                # Strip ALL selected targets up front. AutoEDAPipeline
                # only knows about self.target_col (the primary), so a
                # secondary target (e.g. compressive_strength in a
                # 3-target MOBO run) would otherwise be transformed
                # into a feature like num__compressive_strength and
                # leak into X — the t_df_mtl.drop below cannot recover
                # it because the column name has changed.
                X_only = df.drop(columns=target_cols, errors="ignore")
                t_df_mtl = st.session_state.pipeline.get_transformed_df(X_only)
                X_full_df = t_df_mtl
                feature_names_mtl = list(X_full_df.columns)
                X_mtl = X_full_df.values.astype(np.float32)
                scaler_mtl = None
                x_space_mtl = "pipeline-transformed"
            else:
                X_raw = df.drop(columns=target_cols, errors="ignore")
                X_raw_num = X_raw.select_dtypes(include=np.number)
                if X_raw_num.shape[1] == 0:
                    st.error(
                        "No numeric feature columns available. Run LLM "
                        "Decisions first or supply numeric features."
                    )
                    st.stop()
                X_raw_num = X_raw_num.fillna(X_raw_num.median())
                feature_names_mtl = list(X_raw_num.columns)
                scaler_mtl = StandardScaler()
                X_mtl = scaler_mtl.fit_transform(
                    X_raw_num.values
                ).astype(np.float32)
                x_space_mtl = "standard-scaled"

            # ── Build Y (n_samples, n_targets) ────────────────────────
            missing_targets = [t for t in target_cols if t not in df.columns]
            if missing_targets:
                st.error(
                    f"Target column(s) not in dataframe: {missing_targets}"
                )
                st.stop()

            # Drop rows with NaN in ANY target. For classification,
            # astype(str) would silently turn NaN into a 'nan' class;
            # for regression, NaN in y breaks several base estimators.
            target_nan_mask = df[target_cols].isna().any(axis=1).values
            n_dropped_y = int(target_nan_mask.sum())
            if n_dropped_y:
                st.info(
                    f"Dropped {n_dropped_y} row(s) with NaN in target(s) "
                    f"(kept {int((~target_nan_mask).sum())})."
                )
                keep_y = ~target_nan_mask
                X_mtl = X_mtl[keep_y]
                df_y = df.loc[keep_y]
            else:
                df_y = df

            Y_cols, label_encoders_mtl = [], {}
            for t_col in target_cols:
                y_vals = df_y[t_col].values
                if task_type == "classification":
                    le_t = LabelEncoder()
                    y_vals = le_t.fit_transform(y_vals.astype(str))
                    label_encoders_mtl[t_col] = le_t
                Y_cols.append(y_vals)
            if task_type == "regression":
                Y_mtl = np.column_stack(Y_cols).astype(np.float32)
            else:
                Y_mtl = np.column_stack(Y_cols).astype(np.int64)

            # ── Base estimator selection ──────────────────────────────
            if base_choice == "Random Forest":
                base = (
                    RandomForestClassifier(n_estimators=100, random_state=42)
                    if task_type == "classification"
                    else RandomForestRegressor(n_estimators=100, random_state=42)
                )
            elif base_choice == "XGBoost":
                try:
                    from xgboost import XGBClassifier, XGBRegressor
                except ImportError:
                    st.error(
                        "XGBoost not installed. Run `pip install xgboost`."
                    )
                    st.stop()
                base = (
                    XGBClassifier(
                        n_estimators=100, random_state=42,
                        eval_metric="logloss", verbosity=0,
                    )
                    if task_type == "classification"
                    else XGBRegressor(
                        n_estimators=100, random_state=42, verbosity=0,
                    )
                )
            else:  # Neural Network
                from sklearn.neural_network import MLPClassifier, MLPRegressor
                base = (
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32), max_iter=500,
                        random_state=42,
                    )
                    if task_type == "classification"
                    else MLPRegressor(
                        hidden_layer_sizes=(64, 32), max_iter=500,
                        random_state=42,
                    )
                )

            wrapper_cls = (
                MultiOutputClassifier
                if task_type == "classification"
                else MultiOutputRegressor
            )
            kf = KFold(n_splits=n_splits_mtl, shuffle=True, random_state=42)

            # ── Per-fold per-target metrics ───────────────────────────
            fold_metrics = {t: [] for t in target_cols}
            with st.spinner(
                f"Running {n_splits_mtl}-fold CV on "
                f"{len(target_cols)} target(s) with {base_choice}…"
            ):
                for fold_idx, (tr_idx, te_idx) in enumerate(
                    kf.split(X_mtl)
                ):
                    model_f = wrapper_cls(base)
                    model_f.fit(X_mtl[tr_idx], Y_mtl[tr_idx])
                    Y_pred = model_f.predict(X_mtl[te_idx])
                    Y_true = Y_mtl[te_idx]
                    for ti, t_col in enumerate(target_cols):
                        yt = Y_true[:, ti]
                        yp = Y_pred[:, ti]
                        if task_type == "regression":
                            from sklearn.metrics import (
                                r2_score, mean_absolute_error,
                                mean_squared_error,
                            )
                            fold_metrics[t_col].append({
                                "R2":   float(r2_score(yt, yp)),
                                "MAE":  float(mean_absolute_error(yt, yp)),
                                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                            })
                        else:
                            from sklearn.metrics import (
                                accuracy_score, f1_score, precision_score,
                            )
                            fold_metrics[t_col].append({
                                "Accuracy":  float(accuracy_score(yt, yp)),
                                "F1":        float(f1_score(yt, yp, average="weighted", zero_division=0)),
                                "Precision": float(precision_score(yt, yp, average="weighted", zero_division=0)),
                            })

            # ── Final fit on full data (for Results tab) ──────────────
            with st.spinner("Fitting final joint model on full data…"):
                final_model = wrapper_cls(base)
                final_model.fit(X_mtl, Y_mtl)

            st.session_state.mtl_results = {
                "model":           final_model,
                "X":               X_mtl,
                "Y":               Y_mtl,
                "target_cols":     list(target_cols),
                "task":            task_type,
                "base_choice":     base_choice,
                "fold_metrics":    fold_metrics,
                "feature_names":   feature_names_mtl,
                "label_encoders":  label_encoders_mtl,
                "scaler":          scaler_mtl,        # None if pipeline path
                "x_space":         x_space_mtl,       # "standard-scaled" | "pipeline-transformed"
            }
            st.success(
                f"Joint {base_choice} multi-output model trained. "
                "See the **Results** tab for per-target feature importance "
                "and partial dependence plots."
            )

        except Exception as e:
            st.error(f"Multitask Learning failed: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    # ── Display: per-target metrics (mean ± std across folds) ─────────
    if st.session_state.get("mtl_results") is None:
        return

    mtl = st.session_state.mtl_results
    st.markdown("#### Per-Target CV Metrics (mean ± std)")
    rows = []
    for t_col in mtl["target_cols"]:
        m_list = mtl["fold_metrics"][t_col]
        if not m_list:
            continue
        row = {"Target": t_col}
        for k in m_list[0].keys():
            vals = [m[k] for m in m_list]
            row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        rows.append(row)
    if not rows:
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    primary_metric = (
        "R2" if mtl["task"] == "regression" else "F1"
    )
    metric_means = [
        float(np.mean([m[primary_metric] for m in mtl["fold_metrics"][t]]))
        for t in mtl["target_cols"]
    ]
    fig_mtl = px.bar(
        x=mtl["target_cols"], y=metric_means,
        labels={"x": "Target", "y": primary_metric},
        title=f"Per-Target {primary_metric} (mean across folds)",
        color=metric_means, color_continuous_scale="Blues",
    )
    fig_mtl.update_layout(**PLOT_THEME)
    st.plotly_chart(fig_mtl, use_container_width=True)

    st.caption(
        f"Base estimator: **{mtl['base_choice']}** · "
        f"Wrapper: "
        f"{'MultiOutputClassifier' if mtl['task'] == 'classification' else 'MultiOutputRegressor'} · "
        f"Targets: {', '.join(mtl['target_cols'])}"
    )
