# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from io import StringIO

from pipeline.orchestrator       import AutoEDAPipeline
from pipeline.local_llm_engine   import LocalEnsembleLLMEngine, EnsembleDecision

from webui._shared import PLOT_THEME, warn_if_stale_pipeline
from webui import (
    overview, missing_values, distributions, outliers, correlations,
    llm_decisions, transformed_data, mtl, feedback_loop, bo_mobo,
)
# Aliased — the LOO tab assigns to a local named `results`, which would
# otherwise shadow the module import for the Results tab call below.
from webui import results as results_tab

from pipeline.models.xgboost_model        import XGBoostModel
from pipeline.models.neural_network_model import NeuralNetworkModel
from pipeline.models.hpo import (
    optimize_xgboost,
    optimize_neural_network,
    optimize_tabpfn,
    TabPFNWrapper,
    _OPTUNA_AVAILABLE,
)

from sklearn.ensemble  import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    cross_validate,
    KFold,
    LeaveOneOut,
    StratifiedKFold,          # ✅ FIX Bug 1: was missing
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,   # ✅ FIX Bug 2: was missing
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")



# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCY FLAGS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

try:
    import torch  # noqa: F401  (used inside the BO tab)
    import botorch  # noqa: F401
    import gpytorch  # noqa: F401
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto EDA Pipeline",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem; border-radius: 16px;
        border: 1px solid #30363d; margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem; color: #58a6ff; margin: 0;
    }
    .main-header p { color: #8b949e; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .label {
        font-size: 0.75rem; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem; color: #58a6ff; font-weight: 600;
    }
    .decision-card {
        background: #0d1117; border-left: 3px solid #58a6ff;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    }
    .decision-card .key { color: #79c0ff; }
    .decision-card .val { color: #56d364; }
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 600;
        width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3);
    }
    .tab-header {
        font-family: 'JetBrains Mono', monospace;
        color: #58a6ff; font-size: 1.1rem;
        font-weight: 600; margin-bottom: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
for key in [
    "pipeline", "df", "decisions", "loo_results",
    "profile", "transformed_df",
    "rf_kfold_results",
    "feedback_loop_results",
    "hpo_results",          # HPO best params per estimator
    "mtl_results",          # joint multi-output model from the Multitask Learning tab
    "target_cols",          # list of selected target columns (max 3)
    "bo_candidates",        # last batch suggested by the BO/MOBO tab
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# PRE-INITIALIZE ALL SIDEBAR VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file    = None
ollama_host      = "http://localhost:11434"
phi4_enabled     = True
mistral_enabled  = True
use_llm          = True
use_demo         = True
target_col_input = "target"
target_cols      = ["target"]
task_type        = "classification"
estimator_choice = "Random Forest"
override_imputer = "auto"
override_scaler  = "auto"
override_outlier = "auto"
corr_threshold   = 0.90

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Auto EDA Pipeline</h1>
  <p>Phi-4 + Mistral local ensemble | Privacy-first | Full visualizations</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔒 Local LLM Settings")

    ollama_host = st.text_input(
        "Ollama Host",
        value="http://localhost:11434",
        help="Local Ollama server endpoint"
    )

    col_phi, col_mis = st.columns(2)
    with col_phi:
        phi4_enabled = st.toggle("Phi-4", value=True)
    with col_mis:
        mistral_enabled = st.toggle("Mistral", value=True)

    use_llm = phi4_enabled or mistral_enabled

    if st.button("Test Connection", key="btn_test_connection"):
        import httpx
        try:
            resp   = httpx.get(f"{ollama_host}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            for name, tag in [
                ("Phi-4",   "phi4"),
                ("Mistral", "mistral"),
                ("Gemma2",  "gemma2"),
            ]:
                found = any(tag in m for m in models)
                st.markdown(
                    f"{'🟢' if found else '🔴'} **{name}** "
                    f"{'ready' if found else f'not found — run: ollama pull {tag}'}"
                )
        except Exception:
            st.error("❌ Cannot reach Ollama. Run: `ollama serve`")

    st.markdown("---")
    st.markdown("### 🛡️ Privacy Guarantee")
    st.success("All LLM inference is local.\n\n**No data is transmitted externally.**")

    st.markdown("---")
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_demo      = st.checkbox("Use Demo Dataset", value=True)

    st.markdown("---")
    st.markdown("### Pipeline Settings")
    target_cols_raw = st.text_input(
        "Target Column(s)",
        value="target",
        help="Single target, or comma-separated for multi-target (max 3). "
             "First target is treated as the primary for LOO and single-target steps.",
    )
    parsed = [t.strip() for t in target_cols_raw.split(",") if t.strip()]
    # Dedupe preserving order — "Y1, Y1" would otherwise fit twice downstream.
    parsed = list(dict.fromkeys(parsed))
    if len(parsed) > 3:
        st.warning(f"Capped at 3 targets; dropped: {parsed[3:]}")
        parsed = parsed[:3]
    if not parsed:
        parsed = ["target"]
    target_cols = parsed
    target_col_input = target_cols[0]
    st.session_state.target_cols = target_cols
    if len(target_cols) > 1:
        st.caption(f"Multi-target mode: {', '.join(target_cols)} (primary = `{target_col_input}`)")
    task_type        = st.selectbox("Task Type", ["classification", "regression"])
    estimator_choice = st.selectbox(
        "Estimator for LOO",
        [
            "Random Forest",
            "XGBoost",
            "Neural Network",
            "Logistic Regression / Ridge",
            "TabPFN",               # ✅ NEW
        ]
    )

    st.markdown("---")
    st.markdown("### Manual Overrides")
    with st.expander("Override LLM Decisions"):
        override_imputer = st.selectbox(
            "Imputation Strategy",
            ["auto", "mean", "median", "knn", "missforest"]
        )
        override_scaler  = st.selectbox(
            "Scaler", ["auto", "standard", "minmax", "robust"]
        )
        override_outlier = st.selectbox(
            "Outlier Method",
            ["auto", "iqr", "zscore", "isolation_forest", "none"]
        )
        corr_threshold   = st.slider(
            "Correlation Threshold", 0.7, 1.0, 0.90, step=0.01
        )

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    from sklearn.datasets import make_classification
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=12,
        n_informative=8, n_redundant=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    df["category_A"] = np.random.choice(["low", "medium", "high"], 200)
    df["category_B"] = np.random.choice(["red", "blue", "green"],  200)
    df.iloc[::10, 0] = np.nan
    df.iloc[::15, 3] = np.nan
    df.iloc[::20, 7] = np.nan
    df["feat_5"] = np.exp(df["feat_5"])
    df["feat_9"] = df["feat_0"] * 0.98 + np.random.normal(0, 0.01, 200)
    df["target"] = y
    return df


if uploaded_file is not None:
    # Strip leading/trailing whitespace from column names. CSVs exported from
    # Excel or hand-edited (e.g. data/slump_test.csv) sometimes carry a
    # trailing space on the last column ("compressive_strength "), which
    # silently breaks any sidebar input that matches the visible column
    # name without whitespace.
    raw = pd.read_csv(uploaded_file)
    raw.columns = raw.columns.str.strip()
    st.session_state.df = raw
elif use_demo:
    st.session_state.df = load_demo_data()

def _run_loo_with_wrapper(
    wrapper_model,
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
) -> dict:
    """
    Runs true Leave-One-Out CV using any sklearn-compatible wrapper.
    """
    loo          = LeaveOneOut()
    y_true_list  = []
    y_pred_list  = []
    failed_folds = 0
    first_error  = None       # ← capture the REAL error from fold 0

    n_samples = len(y)
    progress  = st.progress(0, text="LOO progress...")

    # ── Pre-flight: check X for NaN / Inf before looping ─────────────────
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    if nan_count > 0 or inf_count > 0:
        st.error(
            f"❌ Feature matrix X contains **{nan_count} NaN(s)** and "
            f"**{inf_count} Inf(s)** — impute before running LOO."
        )
        progress.empty()
        return _loo_empty_result(task)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            # ── Safe re-instantiation ─────────────────────────────────────
            # Instead of unpacking __dict__ (which breaks most wrappers),
            # create a fresh instance by reading only known safe init params.
            fold_model = _clone_wrapper(wrapper_model)

            # ── Fit ───────────────────────────────────────────────────────
            fold_model.fit(X_train, y_train)

            # ── Predict ───────────────────────────────────────────────────
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

            # ── Capture only the FIRST error for diagnosis ────────────────
            if first_error is None:
                import traceback
                first_error = (str(fold_err), traceback.format_exc())

        y_true_list.append(float(y_test[0]))

        progress.progress(
            (i + 1) / n_samples,
            text=f"LOO fold {i + 1}/{n_samples}"
        )

    progress.empty()

    # ── Always show first error so we can diagnose ────────────────────────
    if first_error is not None:
        st.error(
            f"❌ **{failed_folds}/{n_samples} folds failed.**\n\n"
            f"**First error message:** `{first_error[0]}`"
        )
        with st.expander("Full traceback of first failed fold"):
            st.code(first_error[1], language="python")

    # ── Convert & filter ──────────────────────────────────────────────────
    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)
    valid      = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]

    if len(y_true_arr) < 2:
        return _loo_empty_result(task)

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
    else:
        y_t = y_true_arr.astype(int)
        y_p = np.clip(np.round(y_pred_arr).astype(int), y_t.min(), y_t.max())
        return {
            "test_accuracy":    np.array([float(accuracy_score(y_t, y_p))]),
            "test_f1_weighted": np.array([float(f1_score(y_t, y_p, average="weighted", zero_division=0))]),
            "y_true":           y_true_arr,
            "y_pred":           y_pred_arr,
            "residuals":        residuals,
        }
        

def _clone_wrapper(wrapper_model):
    """
    Safely re-instantiate a wrapper model (XGBoostModel / NeuralNetworkModel)
    without unpacking internal fitted state.

    Reads only the constructor-safe attributes by inspecting __init__ signature.
    """
    import inspect

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
        # else: skip — required param not found, will raise clearly

    return cls(**params)


def _loo_empty_result(task: str) -> dict:
    """Returns a typed empty result dict so downstream code doesn't crash."""
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
    else:
        return {
            "test_accuracy":    np.array([0.0]),
            "test_f1_weighted": np.array([0.0]),
            "y_true":           empty,
            "y_pred":           empty,
            "residuals":        empty,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RF K-FOLD HELPER  (stays in app.py — no random_forest_model.py needed)
# ─────────────────────────────────────────────────────────────────────────────
def run_rf_kfold(
    X:            np.ndarray,
    y:            np.ndarray,
    task:         str = "classification",
    n_splits:     int = 5,
    random_state: int = 42,
) -> dict:
    if task == "classification":
        # ✅ Removed dead `est` variable
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        acc_scores, f1_scores = [], []

        for train_idx, val_idx in kf.split(X, y):
            est_clone = RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=random_state
            )
            est_clone.fit(X[train_idx], y[train_idx])
            preds = est_clone.predict(X[val_idx])
            acc_scores.append(accuracy_score(y[val_idx], preds))
            f1_scores.append(f1_score(y[val_idx], preds, average="weighted"))

        return {
            "accuracy": float(np.mean(acc_scores)),
            "f1":       float(np.mean(f1_scores)),
        }

    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        r2_scores, mae_scores, rmse_scores = [], [], []

        for train_idx, val_idx in kf.split(X):
            est_clone = RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=random_state
            )
            est_clone.fit(X[train_idx], y[train_idx])
            preds = est_clone.predict(X[val_idx])
            r2_scores.append(r2_score(y[val_idx], preds))
            mae_scores.append(mean_absolute_error(y[val_idx], preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], preds)))

        return {
            "r2":   float(np.mean(r2_scores)),
            "mae":  float(np.mean(mae_scores)),
            "rmse": float(np.mean(rmse_scores)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — still used by the heavy tabs that have not yet been split
# ─────────────────────────────────────────────────────────────────────────────
def safe_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute regression metrics safely.
    Returns dict with keys: r2, rmse, mae, mape, valid, error.
    """
    # ✅ No local imports needed — all imported at top of file

    result = {
        "r2":    float("nan"),
        "rmse":  float("nan"),
        "mae":   float("nan"),
        "mape":  float("nan"),
        "valid": False,
        "error": None,
    }

    # ── Guard 1: Convert to numpy ─────────────────────────────────────────
    try:
        y_true = np.array(y_true, dtype=float).ravel()
        y_pred = np.array(y_pred, dtype=float).ravel()
    except Exception as e:
        result["error"] = f"Cannot convert to float array: {e}"
        return result

    # ── Guard 2: Shape mismatch ───────────────────────────────────────────
    if y_true.shape != y_pred.shape:
        result["error"] = (
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
        return result

    # ── Guard 3: NaN / Inf ────────────────────────────────────────────────
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

    # ── Guard 4: Zero variance ────────────────────────────────────────────
    if np.var(y_true) == 0:
        result["error"] = (
            "y_true has zero variance (all values identical). "
            "R² is undefined — check your target column selection."
        )
        return result

    # ── Guard 5: Too few samples ──────────────────────────────────────────
    if len(y_true) < 2:
        result["error"] = "Need at least 2 samples to compute metrics."
        return result

    # ── Compute ───────────────────────────────────────────────────────────
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
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    target_col: str  = "Target",
    metrics:    dict = None,
) -> None:
    """
    Render an interactive parity plot (Actual vs Predicted).
    Shows a clear error message if data is invalid.
    Uses PLOT_THEME for consistent dark styling.
    """
    # ── Convert safely ────────────────────────────────────────────────────
    try:
        y_true = np.array(y_true, dtype=float).ravel()
        y_pred = np.array(y_pred, dtype=float).ravel()
    except Exception as e:
        st.error(f"❌ Parity plot: cannot convert data to float — {e}")
        return

    # ── Validate ──────────────────────────────────────────────────────────
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

    # ── Reference line ────────────────────────────────────────────────────
    min_val  = min(float(y_true.min()), float(y_pred.min()))
    max_val  = max(float(y_true.max()), float(y_pred.max()))
    pad      = (max_val - min_val) * 0.05
    line_rng = [min_val - pad, max_val + pad]

    residuals = y_pred - y_true

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()

    # Perfect prediction line (y = x)
    fig.add_trace(go.Scatter(
        x=line_rng, y=line_rng,
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash", width=2),
    ))

    # Scatter coloured by residual
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

    # Metrics annotation
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    tabs = st.tabs([
        "Overview",            # tabs[0]
        "Missing Values",      # tabs[1]
        "Distributions",       # tabs[2]
        "Outliers",            # tabs[3]
        "Correlations",        # tabs[4]
        "LLM Decisions",       # tabs[5]
        "Transformed Data",    # tabs[6]
        "LOO Results",         # tabs[7]
        "Multitask Learning",  # tabs[8]   joint multi-output model
        "Feedback Loop",       # tabs[9]
        "Results",             # tabs[10]  per-target FI + PDP
        "BO / MOBO",           # tabs[11]  Bayesian Optimization on the MTL surrogate
    ])


    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ──────────────────────────────────────────────────────────────────────
    with tabs[0]:
        overview.render(df)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 — MISSING VALUES
    # ──────────────────────────────────────────────────────────────────────
    with tabs[1]:
        missing_values.render(df)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 — DISTRIBUTIONS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[2]:
        distributions.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 — OUTLIERS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[3]:
        outliers.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5 — CORRELATIONS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[4]:
        correlations.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 6 — LLM DECISIONS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[5]:
        llm_decisions.render(
            df,
            target_col_input=target_col_input,
            task_type=task_type,
            ollama_host=ollama_host,
            use_llm=use_llm,
            target_cols=target_cols,
        )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 7 — TRANSFORMED DATA
    # ──────────────────────────────────────────────────────────────────────
    with tabs[6]:
        transformed_data.render(df, target_cols=target_cols)


    # ──────────────────────────────────────────────────────────────────────
    # TAB 8 — LOO RESULTS  (tabs[7])
    # ──────────────────────────────────────────────────────────────────────
    with tabs[7]:
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
            else:
                if estimator_choice == "TabPFN" and not _TABPFN_AVAILABLE:
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
                                results = st.session_state.pipeline.run_loo(
                                    df, est
                                )

                        # ── Logistic Regression / Ridge ───────────────
                        elif estimator_choice == "Logistic Regression / Ridge":
                            with st.spinner("Running LOO with Logistic Regression / Ridge…"):
                                est = (
                                    LogisticRegression(max_iter=1000)
                                    if task_type == "classification"
                                    else Ridge()
                                )
                                results = st.session_state.pipeline.run_loo(
                                    df, est
                                )

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
                                results = _run_loo_with_wrapper(
                                    XGBoostModel(
                                        task       = task_type,
                                        n_splits   = 5,
                                        random_state = 42,
                                        hpo_params = xgb_hpo_params,
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
                                nn_kwargs["epochs"]  = 150
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
                                results = _run_loo_with_wrapper(
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

                            # Default wrapper (no HPO)
                            tabpfn_wrapper = TabPFNWrapper(
                                task             = task_type,
                                n_estimators     = 8,
                                preprocessor_type = "none",
                                random_state     = 42,
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
                                results = _run_loo_with_wrapper(
                                    tabpfn_wrapper, X_loo, y_loo, task=task_type,
                                )

                        st.session_state.loo_results = results
                        st.success(f"LOO complete using **{estimator_choice}**!")

                    except Exception as e:
                        import traceback
                        st.error(f"LOO Error: {e}")
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc())

        # ── Display LOO results ───────────────────────────────────────────
        if st.session_state.loo_results is not None:
            results = st.session_state.loo_results

            if task_type == "classification":
                acc_scores = results.get("test_accuracy",    np.array([]))
                f1_scores  = results.get("test_f1_weighted", np.array([]))

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

                y_true_loo = results.get("y_true")
                y_pred_loo = results.get("y_pred")
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

            else:  # regression
                y_true_loo = results.get("y_true")
                y_pred_loo = results.get("y_pred")
                if y_true_loo is not None and y_pred_loo is not None:
                    y_true_arr = np.array(y_true_loo, dtype=float).ravel()
                    y_pred_arr = np.array(y_pred_loo, dtype=float).ravel()
                    valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
                    y_true_arr = y_true_arr[valid_mask]
                    y_pred_arr = y_pred_arr[valid_mask]

                    if len(y_true_arr) > 0:
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

                    else:
                        st.warning("No valid predictions available for parity plot.")
                else:
                    st.info(
                        "No y_true / y_pred found in LOO results. "
                        "Re-run LOO to generate predictions."
                    )
        
    
    # ──────────────────────────────────────────────────────────────────────
    # TAB 9 — MULTITASK LEARNING  (tabs[8])
    # ──────────────────────────────────────────────────────────────────────
    with tabs[8]:
        mtl.render(df, target_cols=target_cols, task_type=task_type)


    # ──────────────────────────────────────────────────────────────────────
    # TAB 10 — FEEDBACK LOOP  (tabs[9])
    # ──────────────────────────────────────────────────────────────────────
    with tabs[9]:
        feedback_loop.render()


    # ──────────────────────────────────────────────────────────────────────
    # TAB 11 — RESULTS  (tabs[10])
    # Feature importance + partial dependence plots, per target,
    # from the joint multi-output model trained in the Multitask Learning tab.
    # ──────────────────────────────────────────────────────────────────────
    with tabs[10]:
        results_tab.render()

    # ──────────────────────────────────────────────────────────────────────
    # TAB 12 — BO / MOBO  (tabs[11])
    # Bayesian Optimization on the joint multi-output GP surrogate.
    # Single target → Log Expected Improvement.
    # Multi-target → qLogExpectedHypervolumeImprovement (MOBO).
    # ──────────────────────────────────────────────────────────────────────
    with tabs[11]:
        bo_mobo.render(target_cols=target_cols)
