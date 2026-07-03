"""Shared constants and helpers for the auto-EDA Streamlit UI.

Imported by ``app.py`` and the per-tab modules in ``tabs/``.
Grows as tabs are extracted; keep additions minimal and only promote
something here once it is genuinely reused across tabs.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    font_family="JetBrains Mono",
)


def render_latency(val: float, model_name: str) -> None:
    """Display a colour-coded latency badge for a local-LLM call."""
    if val <= 0:
        st.warning(
            f"⚠️ **{model_name}** latency = `{val}ms` — "
            f"Model may not have responded. Fallback was used."
        )
    elif val < 500:
        st.success(f"⚡ **{model_name}** responded in `{val:.0f}ms` — Fast!")
    elif val < 3000:
        st.info(f"🕐 **{model_name}** responded in `{val:.0f}ms` — Normal")
    else:
        st.warning(f"🐢 **{model_name}** responded in `{val:.0f}ms` — Slow!")


def pipeline_expected_features(pipeline_obj) -> set[str]:
    """Names of columns the cached preprocessor expects to receive as features."""
    expected: set[str] = set()
    try:
        pre = pipeline_obj.pipeline_.named_steps["preprocessor"]
        for _name, _trans, cols in pre.transformers:
            if cols and cols != "drop":
                expected.update(cols)
    except Exception:
        pass
    return expected


def stale_pipeline_targets(target_cols) -> set[str]:
    """Targets that are also in the cached pipeline's feature set.

    Non-empty when the pipeline was built before ``target_cols`` was expanded —
    e.g. user clicked Run LLM Ensemble with one target, then changed the sidebar
    to three. Returns ``set()`` when no pipeline is cached or the cache matches
    the current target list.
    """
    pipeline_obj = st.session_state.get("pipeline")
    if pipeline_obj is None:
        return set()
    return set(target_cols) & pipeline_expected_features(pipeline_obj)


def warn_if_stale_pipeline(target_cols) -> bool:
    """Render an st.error when the cached pipeline is stale. Returns True if stale."""
    stale = stale_pipeline_targets(target_cols)
    if not stale:
        return False
    st.error(
        f"⚠️ The cached preprocessing pipeline was built when "
        f"`{', '.join(sorted(stale))}` were still treated as features, not "
        "targets — running it now would leak them into X under transformed "
        "names like `num__"
        + sorted(stale)[0] +
        "`.\n\n"
        "**Fix:** go to **LLM Decisions** and click "
        "*Run Local LLM Ensemble Analysis* again so the pipeline is rebuilt "
        "with the current target list."
    )
    return True


def safe_regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics safely with explicit guards.

    Returns dict with keys: r2, rmse, mae, mape, valid, error.
    Never raises — failures surface via the ``error`` key.
    """
    result = {
        "r2":    float("nan"),
        "rmse":  float("nan"),
        "mae":   float("nan"),
        "mape":  float("nan"),
        "valid": False,
        "error": None,
    }

    try:
        y_true = np.array(y_true, dtype=float).ravel()
        y_pred = np.array(y_pred, dtype=float).ravel()
    except Exception as e:
        result["error"] = f"Cannot convert to float array: {e}"
        return result

    if y_true.shape != y_pred.shape:
        result["error"] = (
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
        return result

    nan_pred = int(np.isnan(y_pred).sum())
    nan_true = int(np.isnan(y_true).sum())
    inf_pred = int(np.isinf(y_pred).sum())

    if nan_pred > 0 or inf_pred > 0:
        result["error"] = (
            f"y_pred has {nan_pred} NaN(s) and {inf_pred} Inf(s). "
            f"Check model training or preprocessing pipeline."
        )
        return result

    if nan_true > 0:
        result["error"] = (
            f"y_true has {nan_true} NaN(s). "
            f"Drop or impute target column before evaluation."
        )
        return result

    if np.var(y_true) == 0:
        result["error"] = (
            "y_true has zero variance (all values identical). "
            "R² is undefined — check your target column selection."
        )
        return result

    if len(y_true) < 2:
        result["error"] = "Need at least 2 samples to compute metrics."
        return result

    try:
        result["r2"]    = float(r2_score(y_true, y_pred))
        result["rmse"]  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        result["mae"]   = float(mean_absolute_error(y_true, y_pred))
        result["valid"] = True

        if not np.any(y_true == 0):
            result["mape"] = float(
                mean_absolute_percentage_error(y_true, y_pred) * 100
            )
        else:
            result["mape"]  = float("nan")
            result["error"] = "MAPE skipped: y_true contains zeros."

    except Exception as e:
        result["error"] = f"Metric computation failed: {e}"

    return result


def render_parity_plot(
    y_true,
    y_pred,
    target_col: str = "Target",
    metrics: dict | None = None,
) -> None:
    """Render an interactive parity plot (Actual vs Predicted)."""
    try:
        y_true = np.array(y_true, dtype=float).ravel()
        y_pred = np.array(y_pred, dtype=float).ravel()
    except Exception as e:
        st.error(f"❌ Parity plot: cannot convert data to float — {e}")
        return

    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning("⚠️ Parity plot: empty predictions — model may not have run.")
        return

    if np.isnan(y_pred).any():
        n_nan = int(np.isnan(y_pred).sum())
        st.error(
            f"❌ Parity plot: y_pred contains **{n_nan} NaN value(s)**.\n\n"
            f"**Common causes:**\n"
            f"- Preprocessing left NaN in X features (check imputation)\n"
            f"- Target column contains NaN rows that were not dropped\n"
            f"- Model received wrong dtype (e.g. string column in X)"
        )
        return

    if np.isnan(y_true).any():
        st.warning(
            f"⚠️ y_true contains NaN — "
            f"dropping {int(np.isnan(y_true).sum())} rows."
        )
        mask   = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    min_val  = min(float(y_true.min()), float(y_pred.min()))
    max_val  = max(float(y_true.max()), float(y_pred.max()))
    pad      = (max_val - min_val) * 0.05
    line_rng = [min_val - pad, max_val + pad]

    residuals = y_pred - y_true

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=line_rng, y=line_rng,
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=residuals,
            colorscale="RdYlGn_r",
            size=6,
            opacity=0.75,
            colorbar=dict(title="Residual<br>(pred − actual)"),
            showscale=True,
        ),
        hovertemplate=(
            "<b>Actual</b>:    %{x:.4f}<br>"
            "<b>Predicted</b>: %{y:.4f}<br>"
            "<b>Residual</b>:  %{marker.color:.4f}"
            "<extra></extra>"
        ),
    ))

    if metrics and metrics.get("valid"):
        r2   = metrics.get("r2",   float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        mae  = metrics.get("mae",  float("nan"))
        fig.add_annotation(
            x=0.04, y=0.97,
            xref="paper", yref="paper",
            text=(
                f"R² = {r2:.4f}<br>"
                f"RMSE = {rmse:.4f}<br>"
                f"MAE = {mae:.4f}"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="#58a6ff",
            borderwidth=1,
            font=dict(size=13, color="#c9d1d9"),
        )

    fig.update_layout(
        title=f"Parity Plot — {target_col} (Actual vs Predicted)",
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **PLOT_THEME,
    )

    st.plotly_chart(fig, use_container_width=True)
