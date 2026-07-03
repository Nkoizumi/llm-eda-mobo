"""Leave-One-Out helpers used by the LOO Results tab.

Kept out of ``tabs/_shared.py`` because only one tab uses them and
they pull in sklearn cross-validation machinery.
"""

import inspect

import numpy as np
import streamlit as st
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut


def loo_empty_result(task: str) -> dict:
    """Typed empty result dict so downstream code doesn't crash."""
    empty = np.array([], dtype=float)
    if task == "regression":
        return {
            "r2":        float("nan"),
            "rmse":      float("nan"),
            "mae":       float("nan"),
            "y_true":    empty,
            "y_pred":    empty,
            "residuals": empty,
        }
    return {
        "test_accuracy":    np.array([0.0]),
        "test_f1_weighted": np.array([0.0]),
        "y_true":           empty,
        "y_pred":           empty,
        "residuals":        empty,
    }


def clone_wrapper(wrapper_model):
    """Re-instantiate a wrapper model from its constructor signature.

    Unpacking __dict__ breaks most wrappers because they carry fitted state.
    Inspect __init__'s parameter list and read only those attributes from the
    source instance.
    """
    cls    = type(wrapper_model)
    sig    = inspect.signature(cls.__init__)
    params = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if hasattr(wrapper_model, param_name):
            params[param_name] = getattr(wrapper_model, param_name)
        elif param.default is not inspect.Parameter.empty:
            params[param_name] = param.default
        # else: required param not found, the call below will raise clearly

    return cls(**params)


def run_loo_with_wrapper(
    wrapper_model,
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
) -> dict:
    """True Leave-One-Out CV using any sklearn-compatible wrapper."""
    loo          = LeaveOneOut()
    y_true_list  = []
    y_pred_list  = []
    failed_folds = 0
    first_error  = None       # capture the REAL error from fold 0

    n_samples = len(y)
    progress  = st.progress(0, text="LOO progress...")

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    if nan_count > 0 or inf_count > 0:
        st.error(
            f"❌ Feature matrix X contains **{nan_count} NaN(s)** and "
            f"**{inf_count} Inf(s)** — impute before running LOO."
        )
        progress.empty()
        return loo_empty_result(task)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            fold_model = clone_wrapper(wrapper_model)
            fold_model.fit(X_train, y_train)

            raw_pred = fold_model.predict(X_test)
            pred_arr = np.array(raw_pred, dtype=float).ravel()

            if len(pred_arr) == 0:
                raise ValueError("predict() returned an empty array")

            pred_val = float(pred_arr[0])

            if np.isnan(pred_val) or np.isinf(pred_val):
                raise ValueError(
                    f"predict() returned non-finite value: {pred_val}"
                )

            y_pred_list.append(pred_val)

        except Exception as fold_err:
            failed_folds += 1
            y_pred_list.append(float("nan"))

            if first_error is None:
                import traceback
                first_error = (str(fold_err), traceback.format_exc())

        y_true_list.append(float(y_test[0]))

        progress.progress(
            (i + 1) / n_samples,
            text=f"LOO fold {i + 1}/{n_samples}"
        )

    progress.empty()

    if first_error is not None:
        st.error(
            f"❌ **{failed_folds}/{n_samples} folds failed.**\n\n"
            f"**First error message:** `{first_error[0]}`"
        )
        with st.expander("Full traceback of first failed fold"):
            st.code(first_error[1], language="python")

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)
    valid      = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]

    if len(y_true_arr) < 2:
        return loo_empty_result(task)

    residuals = y_true_arr - y_pred_arr

    if task == "regression":
        return {
            "r2":        float(r2_score(y_true_arr, y_pred_arr)),
            "rmse":      float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
            "mae":       float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "y_true":    y_true_arr,
            "y_pred":    y_pred_arr,
            "residuals": residuals,
        }
    y_t = y_true_arr.astype(int)
    y_p = np.clip(np.round(y_pred_arr).astype(int), y_t.min(), y_t.max())
    return {
        "test_accuracy":    np.array([float(accuracy_score(y_t, y_p))]),
        "test_f1_weighted": np.array([float(f1_score(y_t, y_p, average="weighted", zero_division=0))]),
        "y_true":           y_true_arr,
        "y_pred":           y_pred_arr,
        "residuals":        residuals,
    }
