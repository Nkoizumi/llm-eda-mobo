"""Test / Predict tab — apply a trained model to NEW data.

The model comes from the LOO Results tab (single target) or the Multitask
Learning tab (one or more targets), or from a bundle saved earlier. The test
file is preprocessed with the values learned on the TRAINING rows — nothing is
refit on the test data — then predicted. If the test file contains the target
column(s), it is scored: a parity plot and regression metrics, or a confusion
matrix and classification scores, next to the cross-validation scores.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.model_bundle import (
    ModelBundle,
    align_labels,
    classification_metrics,
    to_jsonable,
)
from tabs._shared import PLOT_THEME, render_parity_plot, safe_regression_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_UPLOAD = "Upload a saved model bundle (.zip)"


# ─────────────────────────────────────────────────────────────────────────────
# Used by other tabs
# ─────────────────────────────────────────────────────────────────────────────
def store_bundle(bundle: ModelBundle) -> None:
    """Keep a freshly trained model for this tab (one per source tab)."""
    bundles = st.session_state.get("model_bundles") or {}
    bundles[bundle.source] = bundle
    st.session_state["model_bundles"] = bundles
    st.session_state["test_predictions"] = None   # made with an older model


def best_params_download(hpo: dict, key: str) -> None:
    """Download / save-to-disk buttons for the last HPO result."""
    payload = json.dumps(to_jsonable(hpo), indent=2).encode("utf-8")
    name = (f"best_params_{hpo.get('estimator', 'model')}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.json").replace(" ", "_").replace("/", "_")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download best parameters (JSON)", payload,
                       file_name=name, mime="application/json", key=key)
    if c2.button("💾 Save best parameters to results/best_params/", key=key + "_disk"):
        out = RESULTS_DIR / "best_params" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        c2.success(f"Saved `{out}`")


# ─────────────────────────────────────────────────────────────────────────────
# Tab
# ─────────────────────────────────────────────────────────────────────────────
def _bundle_label(b: ModelBundle) -> str:
    return (f"{b.source}: {b.estimator_name} → {', '.join(b.target_cols)}  "
            f"({b.task}, trained {b.created_at[:16].replace('T', ' ')} UTC)")


def _select_bundle() -> ModelBundle | None:
    bundles = st.session_state.get("model_bundles") or {}
    options = {_bundle_label(b): b for b in bundles.values()}
    choice = st.radio("Model to test", list(options) + [_UPLOAD], key="tp_model_choice")
    if choice != _UPLOAD:
        return options[choice]

    st.caption("A bundle contains a pickle, which runs code when loaded — "
               "upload only bundles you saved yourself.")
    up = st.file_uploader("Model bundle", type=["zip"], key="tp_bundle_upload")
    if up is None:
        return None
    cached = st.session_state.get("tp_uploaded_bundle")
    if cached and cached[0] == up.file_id:
        return cached[1]
    try:
        bundle = ModelBundle.from_zip_bytes(up.getvalue())
    except Exception as e:
        st.error(f"Could not load the bundle: {e}")
        return None
    st.session_state["tp_uploaded_bundle"] = (up.file_id, bundle)
    st.session_state["test_predictions"] = None
    return bundle


def _render_model_card(b: ModelBundle) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimator", b.estimator_name)
    c2.metric("Targets", len(b.target_cols))
    c3.metric("Training rows", b.train_summary.get("n_rows", "?"))
    c4.metric("Input columns used", len(b.required_columns))
    if b.cv_metrics:
        st.markdown("**Cross-validation scores (training data)**")
        st.dataframe(pd.DataFrame(b.cv_metrics).T, use_container_width=True)
    if b.hpo:
        st.caption(f"HPO best CV score {b.hpo.get('best_value', float('nan')):.4f} "
                   f"over {b.hpo.get('n_trials', '?')} trials.")

    with st.expander("💾 Save this model, its preprocessing and best parameters"):
        st.markdown(
            "- **Model bundle (.zip)** — everything needed to reload this model "
            "here later: `model.joblib` + `preprocessing.json` + "
            "`best_params.json` + `manifest.json`.\n"
            "- **preprocessing.json** — every fitted preprocessing value "
            "(imputer fills, power-transform λ, scaler mean/scale, category "
            "levels, dropped correlated columns) and the training-data ranges.\n"
            "- **best_params.json** — estimator settings, HPO best parameters "
            "and CV scores.")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Model bundle (.zip)", b.to_zip_bytes(),
                           file_name=f"{b.default_name}.zip",
                           mime="application/zip", key="tp_dl_zip")
        d2.download_button("⬇️ preprocessing.json",
                           json.dumps(b.preprocessing_record(), indent=2),
                           file_name="preprocessing.json",
                           mime="application/json", key="tp_dl_prep")
        d3.download_button("⬇️ best_params.json",
                           json.dumps(b.best_params_record(), indent=2),
                           file_name="best_params.json",
                           mime="application/json", key="tp_dl_params")
        if st.button("💾 Save all to results/models/", key="tp_save_disk"):
            out = b.save_dir(RESULTS_DIR / "models")
            st.success(f"Saved to `{out}`")


def _render_checks(b: ModelBundle, test_df: pd.DataFrame) -> dict:
    chk = b.check_test_frame(test_df)
    n_out = int(chk["rows_outside_training_range"].sum())
    present = [t for t in b.target_cols if t in test_df.columns]

    m1, m2, m3 = st.columns(3)
    m1.metric("Test rows", len(test_df))
    m2.metric("Targets in test file", f"{len(present)} / {len(b.target_cols)}")
    m3.metric("Rows outside training range", n_out)

    if chk["missing"]:
        st.error(f"Missing column(s) the model needs: `{', '.join(chk['missing'])}`. "
                 "Column names must match the training file.")
    if chk["not_numeric"]:
        st.warning(f"Non-numeric text in numeric column(s) "
                   f"`{', '.join(chk['not_numeric'])}` — those cells are treated "
                   "as missing and filled the same way as in training.")
    if chk["unseen_levels"]:
        st.warning("Categories not seen in training (encoded as 'unknown'): "
                   + "; ".join(f"`{c}`: {v}" for c, v in chk["unseen_levels"].items()))
    if chk["out_of_range"]:
        st.warning(
            f"{n_out} row(s) have at least one value outside the training "
            "min–max, so their predictions are extrapolation. Per column: "
            + ", ".join(f"`{c}` {n}" for c, n in chk["out_of_range"].items()))
    if chk["extra"]:
        st.caption(f"Ignored columns (not used in training): {', '.join(chk['extra'])}")
    if not present:
        st.info("The test file has no target column, so predictions are made "
                "but cannot be scored.")
    return chk


def _render_regression(t: str, y_true, y_pred, cv: dict) -> dict:
    m = safe_regression_metrics(y_true, y_pred)
    if m.get("error") and not m.get("valid"):
        st.warning(f"`{t}`: {m['error']}")
    cols = st.columns(5)
    cols[0].metric("Test R²", f"{m['r2']:.4f}",
                   delta=f"{m['r2'] - cv['r2']:+.4f} vs CV" if "r2" in cv else None)
    cols[1].metric("Test RMSE", f"{m['rmse']:.4g}")
    cols[2].metric("Test MAE", f"{m['mae']:.4g}")
    cols[3].metric("Test MAPE", "N/A" if np.isnan(m["mape"]) else f"{m['mape']:.2f}%")
    cols[4].metric("n", len(y_true))
    render_parity_plot(y_true=y_true, y_pred=y_pred,
                       target_col=f"{t} (test set)", metrics=m)
    return {"n": int(len(y_true)), **{k: m[k] for k in ("r2", "rmse", "mae", "mape")}}


def _render_classification(t: str, y_true, y_pred, cv: dict) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    m = classification_metrics(y_true, y_pred)
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']:.4f}",
                   delta=f"{m['accuracy'] - cv['accuracy']:+.4f} vs CV"
                   if "accuracy" in cv else None)
    cols[1].metric("Balanced acc.", f"{m['balanced_accuracy']:.4f}")
    cols[2].metric("F1 (weighted)", f"{m['f1_weighted']:.4f}",
                   delta=f"{m['f1_weighted'] - cv['f1_weighted']:+.4f} vs CV"
                   if "f1_weighted" in cv else None)
    cols[3].metric("MCC", f"{m['mcc']:.4f}")
    cols[4].metric("n", m["n"])

    yt, yp = align_labels(y_true, y_pred)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()), key=str)
    cm = confusion_matrix(yt, yp, labels=labels)
    names = [str(l) for l in labels]
    fig = px.imshow(cm, x=names, y=names, text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title=f"Confusion matrix — {t} (test set)")
    fig.update_layout(**PLOT_THEME, height=420)
    st.plotly_chart(fig, use_container_width=True)

    rep = classification_report(yt, yp, labels=labels, target_names=names,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
    return m


def render(target_cols: list[str]) -> None:
    st.markdown('<p class="tab-header">🧪 Test / Predict on New Data</p>',
                unsafe_allow_html=True)
    st.info(
        "1. Train a model in **LOO Results** or **Multitask Learning** (a final "
        "model fitted on all training rows is kept here automatically), or "
        "upload a bundle you saved earlier.\n"
        "2. Upload a test CSV with the same column names as the training file. "
        "It is preprocessed with the values learned on the **training** data — "
        "nothing is refit on the test data.\n"
        "3. If the test file contains the target column(s), it is scored.")

    bundle = _select_bundle()
    if bundle is None:
        if not st.session_state.get("model_bundles"):
            st.caption("No model in this session yet.")
        return
    _render_model_card(bundle)

    st.markdown("---")
    st.markdown("#### Test data")
    up = st.file_uploader("Upload test CSV", type=["csv"], key="tp_test_upload")
    if up is None:
        return
    test_df = pd.read_csv(up)
    test_df.columns = test_df.columns.str.strip()   # same cleaning as training upload

    chk = _render_checks(bundle, test_df)
    if chk["missing"]:
        return

    run_id = (bundle.source, bundle.created_at, up.file_id)
    if st.button("Predict", key="tp_btn_predict", type="primary"):
        try:
            with st.spinner("Preprocessing with training-fitted values and predicting…"):
                preds = bundle.predict(test_df)
            st.session_state["test_predictions"] = {"run_id": run_id, "preds": preds}
        except Exception as e:
            import traceback
            st.error(f"Prediction failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return

    state = st.session_state.get("test_predictions")
    if not state or state["run_id"] != run_id:
        return
    preds: pd.DataFrame = state["preds"]

    # ── scores per target ────────────────────────────────────────────────
    summary = {}
    for t in bundle.target_cols:
        if t not in test_df.columns:
            continue
        known = test_df[t].notna().to_numpy()
        if not known.any():
            continue
        st.markdown(f"#### Test-set performance — `{t}`")
        if (~known).any():
            st.caption(f"{int((~known).sum())} row(s) without a `{t}` value are "
                       "predicted but not scored.")
        cv = bundle.cv_metrics.get(t, {})
        y_true = test_df.loc[known, t].to_numpy()
        y_pred = preds.loc[known, t].to_numpy()
        if bundle.task == "regression":
            summary[t] = _render_regression(t, pd.to_numeric(pd.Series(y_true),
                                                             errors="coerce").to_numpy(),
                                            y_pred.astype(float), cv)
        else:
            summary[t] = _render_classification(t, y_true, y_pred, cv)

    if summary:
        st.markdown("#### Summary: test vs cross-validation")
        rows = []
        for t, m in summary.items():
            for k, v in m.items():
                if k == "n":
                    continue
                rows.append({"target": t, "metric": k, "test": v,
                             "cv": bundle.cv_metrics.get(t, {}).get(k)})
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

    # ── prediction table + downloads ─────────────────────────────────────
    st.markdown("#### Predictions")
    out = test_df.copy()
    for t in bundle.target_cols:
        out[f"pred_{t}"] = preds[t].to_numpy()
        if bundle.task == "regression" and t in test_df.columns:
            out[f"residual_{t}"] = (out[f"pred_{t}"]
                                    - pd.to_numeric(test_df[t], errors="coerce"))
    out["outside_training_range"] = chk["rows_outside_training_range"]
    st.dataframe(out, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download predictions (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{Path(up.name).stem}.csv",
                       mime="text/csv", key="tp_dl_preds")
    if summary:
        report = {"model": bundle.best_params_record(),
                  "test_file": up.name, "n_test_rows": len(test_df),
                  "n_rows_outside_training_range": int(chk["rows_outside_training_range"].sum()),
                  "test_metrics": summary}
        c2.download_button("⬇️ Download test metrics (JSON)",
                           json.dumps(to_jsonable(report), indent=2),
                           file_name=f"test_metrics_{Path(up.name).stem}.json",
                           mime="application/json", key="tp_dl_metrics")
