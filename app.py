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

from webui._shared import PLOT_THEME
from webui import overview, missing_values, distributions

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
    st.session_state.df = pd.read_csv(uploaded_file)
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
# ✅ HELPER FUNCTIONS — defined AFTER PLOT_THEME so render_parity_plot can use it
# ─────────────────────────────────────────────────────────────────────────────
def render_latency(val: float, model_name: str):
    """Display a colour-coded latency badge."""
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

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5 — CORRELATIONS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<p class="tab-header">Correlation Analysis</p>',
                    unsafe_allow_html=True)

        num_cols_corr = df.select_dtypes(include=np.number).columns.tolist()
        corr_method   = st.radio(
            "Correlation Method", ["pearson", "spearman", "kendall"],
            horizontal=True
        )
        corr_matrix = df[num_cols_corr].corr(method=corr_method)

        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            title=f"{corr_method.capitalize()} Correlation Heatmap",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(**PLOT_THEME, height=550)
        fig_corr.update_traces(textfont_size=9)
        st.plotly_chart(fig_corr, use_container_width=True)

        threshold = st.slider("Highlight Threshold", 0.5, 1.0, 0.80, 0.01)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_pairs = [
            {
                "Feature 1":   c1,
                "Feature 2":   c2,
                "Correlation": round(upper.loc[c2, c1], 4),
            }
            for c1 in upper.columns
            for c2 in upper.index
            if not pd.isna(upper.loc[c2, c1])
            and abs(upper.loc[c2, c1]) >= threshold
        ]
        if high_pairs:
            high_df = pd.DataFrame(high_pairs).sort_values(
                "Correlation", ascending=False, key=abs
            )
            st.markdown(f"#### Pairs with |corr| ≥ {threshold}")
            st.dataframe(
                high_df.style.background_gradient(
                    subset=["Correlation"], cmap="Reds"
                ),
                use_container_width=True
            )
        else:
            st.info(f"No pairs with |correlation| ≥ {threshold}")

        targets_in_df = [t for t in target_cols if t in df.columns]
        if targets_in_df and num_cols_corr:
            feat_choices = [c for c in num_cols_corr if c not in target_cols]
            if feat_choices:
                for t_idx, t_col in enumerate(targets_in_df):
                    st.markdown(f"#### Feature vs Target: `{t_col}`")
                    sel_corr_feat = st.selectbox(
                        "Select Feature", feat_choices, key=f"corr_feat_{t_idx}"
                    )
                    fig_scatter = px.scatter(
                        df, x=sel_corr_feat, y=t_col, trendline="ols",
                        color_discrete_sequence=["#58a6ff"],
                        title=f"{sel_corr_feat} vs {t_col}"
                    )
                    fig_scatter.update_layout(**PLOT_THEME)
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 6 — LLM DECISIONS
    # ──────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<p class="tab-header">Phi-4 + Mistral Ensemble Decisions</p>',
                    unsafe_allow_html=True)

        if st.button("Run Local LLM Ensemble Analysis", key="btn_run_llm"):
            with st.spinner("🤖 Querying Phi-4 + Mistral in parallel (local)..."):
                try:
                    eda = AutoEDAPipeline(
                        target_col=target_col_input,
                        task=task_type,
                        ollama_host=ollama_host,
                        use_local_llm=use_llm
                    )
                    # Drop every selected target from the feature matrix so the
                    # ensemble pipeline never sees one target as a feature
                    # of another.
                    X_only = df.drop(columns=target_cols, errors="ignore")
                    eda.build_pipeline(X_only)
                    st.session_state.pipeline  = eda
                    st.session_state.decisions = eda.ensemble_result_
                    st.success("✅ Ensemble analysis complete!")
                except Exception as e:
                    st.error(f"LLM Error: {e}")

        if st.session_state.decisions is not None:
            ens: EnsembleDecision = st.session_state.decisions

            # ── Agreement gauge ───────────────────────────────────────────
            score = ens.agreement_score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                title={"text": "Model Agreement Score",
                       "font": {"color": "#c9d1d9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#c9d1d9"},
                    "bar":  {"color": "#58a6ff"},
                    "steps": [
                        {"range": [0,  50], "color": "#3d1c1c"},
                        {"range": [50, 75], "color": "#2d2d0d"},
                        {"range": [75, 100], "color": "#0d2d1c"},
                    ],
                    "threshold": {
                        "line":      {"color": "#56d364", "width": 3},
                        "thickness": 0.8,
                        "value":     75,
                    },
                },
                number={"suffix": "%", "font": {"color": "#58a6ff"}},
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=40, b=10),
                **PLOT_THEME
            )
            col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
            with col_g1:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col_g2:
                st.metric("Phi-4 Confidence",
                          f"{ens.phi4_decision.confidence:.0%}")
                render_latency(ens.phi4_decision.latency_ms, "Phi-4")
            with col_g3:
                st.metric("Mistral Confidence",
                          f"{ens.mistral_decision.confidence:.0%}")
                render_latency(ens.mistral_decision.latency_ms, "Mistral")

            # ── ✅ FIX Bug 3: Correct tiebreaker message ──────────────────
            if ens.tiebreak_used:
                n_conflicts       = len(ens.conflicts)
                gemma2_succeeded  = any(
                    c.get("method") == "gemma2_tiebreak"
                    for c in ens.conflicts
                )
                if gemma2_succeeded:
                    st.warning(
                        f"⚖️ **{n_conflicts} conflict(s) detected** — "
                        f"Gemma2 acted as tiebreaker and resolved them."
                    )
                else:
                    st.warning(
                        f"⚖️ **{n_conflicts} conflict(s) detected** — "
                        f"Gemma2 tiebreaker failed. Phi-4 used as fallback."
                    )

            # ── Side-by-side comparison ───────────────────────────────────
            st.markdown("#### Model Decision Comparison")
            compare_data = {
                "Decision": [
                    "Imputation", "Power Transform",
                    "Outlier Method", "Outlier Threshold",
                    "Corr Threshold", "Scaler",
                ],
                "Phi-4": [
                    ens.phi4_decision.imputation_strategy,
                    ens.phi4_decision.power_transform,
                    ens.phi4_decision.outlier_method,
                    str(ens.phi4_decision.outlier_threshold),
                    str(ens.phi4_decision.correlation_threshold),
                    ens.phi4_decision.scaler,
                ],
                "Mistral": [
                    ens.mistral_decision.imputation_strategy,
                    ens.mistral_decision.power_transform,
                    ens.mistral_decision.outlier_method,
                    str(ens.mistral_decision.outlier_threshold),
                    str(ens.mistral_decision.correlation_threshold),
                    ens.mistral_decision.scaler,
                ],
                "Ensemble Final": [
                    ens.final.imputation_strategy,
                    ens.final.power_transform,
                    ens.final.outlier_method,
                    str(ens.final.outlier_threshold),
                    str(ens.final.correlation_threshold),
                    ens.final.scaler,
                ],
            }
            compare_df = pd.DataFrame(compare_data)

            def highlight_conflicts(row):
                if row["Phi-4"] != row["Mistral"]:
                    return ["background-color:#3d2200;color:#ffa657"] * len(row)
                return [""] * len(row)

            st.dataframe(
                compare_df.style.apply(highlight_conflicts, axis=1),
                use_container_width=True
            )

            if ens.conflicts:
                st.markdown("#### Conflict Resolution Details")
                st.dataframe(
                    pd.DataFrame(ens.conflicts), use_container_width=True
                )

            st.markdown("#### Model Reasoning")
            tab_phi4, tab_mistral, tab_json = st.tabs(
                ["Phi-4 Reasoning", "Mistral Reasoning", "Final JSON"]
            )
            with tab_phi4:
                st.info(
                    ens.phi4_decision.reasoning_summary or "No reasoning returned."
                )
                with st.expander("Raw Phi-4 Response"):
                    st.code(ens.phi4_decision.raw_response, language="json")
            with tab_mistral:
                st.info(
                    ens.mistral_decision.reasoning_summary or "No reasoning returned."
                )
                with st.expander("Raw Mistral Response"):
                    st.code(ens.mistral_decision.raw_response, language="json")
            with tab_json:
                from dataclasses import asdict
                st.json(asdict(ens.final))

    # ──────────────────────────────────────────────────────────────────────
    # TAB 7 — TRANSFORMED DATA
    # ──────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<p class="tab-header">Transformed Data Inspection</p>',
                    unsafe_allow_html=True)

        if st.button("Transform Dataset", key="btn_transform"):
            if st.session_state.pipeline is not None:
                with st.spinner("Transforming data..."):
                    try:
                        t_df = st.session_state.pipeline.get_transformed_df(df)
                        st.session_state.transformed_df = t_df
                        st.success(
                            f"Transformed: "
                            f"{t_df.shape[0]} rows × {t_df.shape[1]} columns"
                        )
                    except Exception as e:
                        st.error(f"Transform Error: {e}")
            else:
                st.warning("Run LLM Analysis first (Tab: LLM Decisions).")

        if st.session_state.transformed_df is not None:
            t_df = st.session_state.transformed_df

            td1, td2, td3 = st.columns(3)
            td1.metric("Features Before", len(df.columns))
            td2.metric("Features After",  len(t_df.columns))
            td3.metric("Delta", f"{len(t_df.columns) - len(df.columns):+d}")

            # ── Preview table ─────────────────────────────────────────────
            st.dataframe(t_df.head(20).round(4), use_container_width=True)

            # ── Download Section ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📥 Download Transformed Data")

            dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 1])

            with dl_col1:
                csv_features = t_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Transformed Features (CSV)",
                    data=csv_features,
                    file_name="transformed_features.csv",
                    mime="text/csv",
                    help="Downloads only the preprocessed feature columns (no target).",
                    use_container_width=True,
                )

            with dl_col2:
                try:
                    t_df_with_target = t_df.copy().reset_index(drop=True)
                    for t_col in target_cols:
                        if t_col in df.columns:
                            t_df_with_target[t_col] = (
                                df[t_col].reset_index(drop=True)
                            )

                    csv_with_target = t_df_with_target.to_csv(
                        index=False
                    ).encode("utf-8")
                    label = (
                        "⬇️ Download With Target (CSV)"
                        if len(target_cols) == 1
                        else "⬇️ Download With Targets (CSV)"
                    )
                    st.download_button(
                        label=label,
                        data=csv_with_target,
                        file_name="transformed_with_target.csv",
                        mime="text/csv",
                        help="Downloads transformed features with the original target column(s) appended.",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Could not append target column(s): {e}")

            with dl_col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="label">Download Size</div>
                        <div class="value">{t_df.shape[0]:,} × {t_df.shape[1]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("##### Excel Format")
            try:
                from io import BytesIO
                import openpyxl

                excel_buf = BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    t_df.to_excel(
                        writer, sheet_name="Transformed", index=False
                    )
                    if any(t in df.columns for t in target_cols):
                        t_df_with_target.to_excel(
                            writer, sheet_name="With Target", index=False
                        )
                excel_buf.seek(0)

                st.download_button(
                    label="⬇️ Download as Excel (.xlsx)",
                    data=excel_buf.read(),
                    file_name="transformed_data.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    help="Downloads both sheets: Transformed features + With Target.",
                    use_container_width=False,
                )
            except ImportError:
                st.info(
                    "💡 Install openpyxl for Excel export: "
                    "`pip install openpyxl`"
                )
            except Exception as e:
                st.warning(f"Excel export failed: {e}")

            # ── Before vs After Distribution Comparison ───────────────────
            st.markdown("---")
            st.markdown("#### Before vs After: Distribution Comparison")
            shared_num = [
                c for c in df.select_dtypes(include=np.number).columns
                if c in t_df.columns and c not in target_cols
            ]
            if shared_num:
                compare_feat = st.selectbox("Feature to Compare", shared_num)
                fig_compare  = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Before Transform", "After Transform"]
                )
                fig_compare.add_trace(
                    go.Histogram(
                        x=df[compare_feat].dropna(),
                        marker_color="#58a6ff", nbinsx=30, name="Before"
                    ),
                    row=1, col=1
                )
                fig_compare.add_trace(
                    go.Histogram(
                        x=t_df[compare_feat].dropna(),
                        marker_color="#56d364", nbinsx=30, name="After"
                    ),
                    row=1, col=2
                )
                fig_compare.update_layout(
                    **PLOT_THEME, height=350,
                    title_text=f"Distribution: {compare_feat}"
                )
                st.plotly_chart(fig_compare, use_container_width=True)
    
    
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
                    t_df_mtl = st.session_state.pipeline.get_transformed_df(df)
                    X_full_df = t_df_mtl.drop(
                        columns=target_cols, errors="ignore"
                    )
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
        if st.session_state.get("mtl_results") is not None:
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
            if rows:
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


    # ──────────────────────────────────────────────────────────────────────
    # TAB 10 — FEEDBACK LOOP  (tabs[9])
    # ──────────────────────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown(
            '<p class="tab-header">🔄 Feedback Loop</p>',
            unsafe_allow_html=True,
        )
        st.info(
            "The Feedback Loop tab will display iterative model improvement "
            "results driven by the AutoEDA controller.\n\n"
            "Run the pipeline from **LLM Decisions** first to populate results here."
        )

        if st.session_state.get("feedback_loop_results") is not None:
            st.json(st.session_state["feedback_loop_results"])
        else:
            st.warning("No feedback loop results yet. Run the LLM pipeline first.")
            

    # ──────────────────────────────────────────────────────────────────────
    # TAB 11 — RESULTS  (tabs[10])
    # Feature importance + partial dependence plots, per target,
    # from the joint multi-output model trained in the Multitask Learning tab.
    # ──────────────────────────────────────────────────────────────────────
    with tabs[10]:
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
        else:
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

    # ──────────────────────────────────────────────────────────────────────
    # TAB 12 — BO / MOBO  (tabs[11])
    # Bayesian Optimization on the joint multi-output GP surrogate.
    # Single target → Log Expected Improvement.
    # Multi-target → qLogExpectedHypervolumeImprovement (MOBO).
    # ──────────────────────────────────────────────────────────────────────
    with tabs[11]:
        st.markdown(
            '<p class="tab-header">🎯 Bayesian Optimization (BO / MOBO)</p>',
            unsafe_allow_html=True,
        )

        if not _BOTORCH_AVAILABLE:
            st.error(
                "BoTorch is not installed. Install with:\n\n"
                "```bash\npip install botorch gpytorch torch\n```\n"
                "(Already in requirements.txt — heavy dependency, install only "
                "if you intend to use BO/MOBO.)"
            )
        else:
            mtl_bo = st.session_state.get("mtl_results")
            if mtl_bo is None:
                st.info(
                    "Run the **Multitask Learning** tab first. BO/MOBO uses "
                    "the trained data (X, Y) and the joint surrogate. "
                    "(GPs are fit fresh here on the same X, Y.)"
                )
            else:
                import torch
                from botorch.models import SingleTaskGP, ModelListGP
                from botorch.fit import fit_gpytorch_mll
                from gpytorch.mlls import (
                    ExactMarginalLogLikelihood,
                    SumMarginalLogLikelihood,
                )
                from botorch.models.transforms import Normalize, Standardize
                from botorch.optim import optimize_acqf

                X_bo      = np.asarray(mtl_bo["X"], dtype=np.float64)
                Y_bo_raw  = np.asarray(mtl_bo["Y"], dtype=np.float64)
                feat_nm_b = list(mtl_bo["feature_names"])
                t_cols_b  = list(mtl_bo["target_cols"])
                n_targets = len(t_cols_b)

                # Sidebar target list changed since MTL was last run → the
                # stored mtl_results is stale. Warn rather than silently
                # using a different target set than the sidebar advertises.
                if t_cols_b != list(target_cols):
                    st.warning(
                        f"Sidebar targets `{', '.join(target_cols)}` do not "
                        f"match the last Multitask Learning run "
                        f"(`{', '.join(t_cols_b)}`). BO/MOBO below uses the "
                        "MTL run's targets — re-run the **Multitask Learning** "
                        "tab to refresh."
                    )

                st.caption(
                    f"Surrogate source: joint **{mtl_bo['base_choice']}** model "
                    f"with {len(feat_nm_b)} feature(s) and {n_targets} target(s) "
                    f"on {X_bo.shape[0]} training rows."
                )
                if mtl_bo["task"] == "classification":
                    st.warning(
                        "BO on classification targets uses the label-encoded "
                        "integer class indices as the objective. "
                        "*Maximize* pushes candidates toward higher class indices."
                    )

                # ── 1. Direction per target ──────────────────────────────
                st.markdown("#### 1. Optimization direction per target")
                if "bo_directions" not in st.session_state or \
                   st.session_state.get("_bo_dirs_for") != t_cols_b:
                    st.session_state.bo_directions = {
                        t: "maximize" for t in t_cols_b
                    }
                    st.session_state["_bo_dirs_for"] = list(t_cols_b)

                dir_cols = st.columns(min(3, max(1, n_targets)))
                for ti, t_col in enumerate(t_cols_b):
                    with dir_cols[ti % len(dir_cols)]:
                        st.session_state.bo_directions[t_col] = st.radio(
                            f"`{t_col}`",
                            ["maximize", "minimize"],
                            index=0 if st.session_state.bo_directions.get(t_col, "maximize") == "maximize" else 1,
                            key=f"bo_dir_{t_col}",
                            horizontal=True,
                        )
                directions = [st.session_state.bo_directions[t] for t in t_cols_b]
                direction_signs = np.array(
                    [1.0 if d == "maximize" else -1.0 for d in directions]
                )

                # Y oriented so higher = better in all targets
                Y_bo = Y_bo_raw * direction_signs[None, :]

                # ── 2. Pareto front / leaderboard of existing data ───────
                st.markdown("---")
                st.markdown("#### 2. Existing data view")

                def _pareto_mask(Yo):
                    """Return boolean mask of Pareto-optimal rows (higher = better)."""
                    n = Yo.shape[0]
                    keep = np.ones(n, dtype=bool)
                    for i in range(n):
                        if not keep[i]:
                            continue
                        # j dominates i if j ≥ i everywhere and > i somewhere
                        dom = (
                            np.all(Yo >= Yo[i], axis=1)
                            & np.any(Yo > Yo[i], axis=1)
                        )
                        dom[i] = False
                        if dom.any():
                            keep[i] = False
                    return keep

                # Pull predicted Y values of the last BO/MOBO batch, if any —
                # used to overlay candidates on the existing-data view.
                bo_cands_df = st.session_state.get("bo_candidates")
                cand_preds = None
                if bo_cands_df is not None:
                    pred_cols = [f"pred_{t}" for t in t_cols_b]
                    if all(c in bo_cands_df.columns for c in pred_cols):
                        cand_preds = bo_cands_df[pred_cols].to_numpy(dtype=float)

                if n_targets == 1:
                    # Sorted leaderboard
                    t = t_cols_b[0]
                    ascending = directions[0] == "minimize"
                    ldb = pd.DataFrame({"Source": ["data"] * X_bo.shape[0],
                                        t: Y_bo_raw[:, 0]})
                    for fi, fn in enumerate(feat_nm_b):
                        ldb[fn] = X_bo[:, fi]

                    # Append BO candidates with their predicted target
                    if cand_preds is not None and cand_preds.shape[0]:
                        cand_rows = pd.DataFrame({
                            "Source": ["BO suggestion"] * cand_preds.shape[0],
                            t: cand_preds[:, 0],
                        })
                        for fi, fn in enumerate(feat_nm_b):
                            cand_rows[fn] = bo_cands_df[fn].to_numpy()
                        ldb = pd.concat([ldb, cand_rows], ignore_index=True)

                    ldb_sorted = ldb.sort_values(t, ascending=ascending).reset_index(drop=True)
                    arrow = "↓" if ascending else "↑"
                    st.markdown(
                        f"**Sorted leaderboard for `{t}` ({directions[0]} {arrow})** — top rows are best."
                    )
                    if cand_preds is not None:
                        st.caption(
                            f"Includes {cand_preds.shape[0]} BO suggestion(s) "
                            "interleaved with existing data based on predicted target."
                        )

                    # Highlight BO suggestion rows
                    def _highlight_src(row):
                        if row["Source"] == "BO suggestion":
                            return ["background-color:#3d2a00;color:#ffa657"] * len(row)
                        return [""] * len(row)
                    st.dataframe(
                        ldb_sorted.head(20).style.apply(_highlight_src, axis=1),
                        use_container_width=True,
                    )
                else:
                    # Pareto front view (multi-target)
                    pareto_keep = _pareto_mask(Y_bo)
                    n_pareto = int(pareto_keep.sum())
                    st.markdown(
                        f"**{n_pareto} Pareto-optimal point(s)** out of "
                        f"{Y_bo.shape[0]} based on the chosen directions."
                    )

                    pareto_mask_arr = pareto_keep
                    dom_idx    = np.where(~pareto_mask_arr)[0]
                    pareto_idx = np.where(pareto_mask_arr)[0]

                    if n_targets == 2:
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(
                            x=Y_bo_raw[dom_idx, 0], y=Y_bo_raw[dom_idx, 1],
                            mode="markers", name="Dominated",
                            marker=dict(color="#6e7681", size=8, opacity=0.7),
                        ))
                        fig_p.add_trace(go.Scatter(
                            x=Y_bo_raw[pareto_idx, 0], y=Y_bo_raw[pareto_idx, 1],
                            mode="markers", name="Pareto",
                            marker=dict(color="#56d364", size=10,
                                        line=dict(width=1, color="#0d2d1c")),
                        ))
                        if cand_preds is not None and cand_preds.shape[0]:
                            fig_p.add_trace(go.Scatter(
                                x=cand_preds[:, 0], y=cand_preds[:, 1],
                                mode="markers+text",
                                name="BO suggestion (predicted)",
                                marker=dict(color="#f0883e", size=14,
                                            symbol="star",
                                            line=dict(width=1, color="#3d2a00")),
                                text=[f"#{i+1}" for i in range(cand_preds.shape[0])],
                                textposition="top center",
                                textfont=dict(color="#ffa657", size=10),
                            ))
                        fig_p.update_layout(
                            title="Existing data — Pareto front (BO suggestions overlaid)"
                                  if cand_preds is not None
                                  else "Existing data — Pareto front",
                            xaxis_title=f"{t_cols_b[0]} ({directions[0]})",
                            yaxis_title=f"{t_cols_b[1]} ({directions[1]})",
                            **PLOT_THEME,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                    elif n_targets == 3:
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter3d(
                            x=Y_bo_raw[dom_idx, 0], y=Y_bo_raw[dom_idx, 1],
                            z=Y_bo_raw[dom_idx, 2],
                            mode="markers", name="Dominated",
                            marker=dict(color="#6e7681", size=4, opacity=0.6),
                        ))
                        fig_p.add_trace(go.Scatter3d(
                            x=Y_bo_raw[pareto_idx, 0], y=Y_bo_raw[pareto_idx, 1],
                            z=Y_bo_raw[pareto_idx, 2],
                            mode="markers", name="Pareto",
                            marker=dict(color="#56d364", size=6,
                                        line=dict(width=0.5, color="#0d2d1c")),
                        ))
                        if cand_preds is not None and cand_preds.shape[0]:
                            fig_p.add_trace(go.Scatter3d(
                                x=cand_preds[:, 0], y=cand_preds[:, 1],
                                z=cand_preds[:, 2],
                                mode="markers+text",
                                name="BO suggestion (predicted)",
                                marker=dict(color="#f0883e", size=8,
                                            symbol="diamond",
                                            line=dict(width=1, color="#3d2a00")),
                                text=[f"#{i+1}" for i in range(cand_preds.shape[0])],
                                textfont=dict(color="#ffa657", size=10),
                            ))
                        fig_p.update_layout(
                            title="Existing data — 3D Pareto (BO suggestions overlaid)"
                                  if cand_preds is not None
                                  else "Existing data — 3D Pareto",
                            scene=dict(
                                xaxis_title=f"{t_cols_b[0]} ({directions[0]})",
                                yaxis_title=f"{t_cols_b[1]} ({directions[1]})",
                                zaxis_title=f"{t_cols_b[2]} ({directions[2]})",
                            ),
                            height=550,
                            **PLOT_THEME,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                    if cand_preds is not None:
                        st.caption(
                            f"Orange star/diamond markers = {cand_preds.shape[0]} BO suggestion(s) "
                            "plotted at the GP's predicted target values."
                        )

                    # Pareto rows table
                    pf_rows = pd.DataFrame(
                        np.hstack([Y_bo_raw[pareto_keep], X_bo[pareto_keep]]),
                        columns=t_cols_b + feat_nm_b,
                    )
                    st.markdown("**Pareto-optimal rows:**")
                    st.dataframe(pf_rows, use_container_width=True)

                # ── 3. Per-feature bounds ─────────────────────────────────
                st.markdown("---")
                st.markdown("#### 3. Per-feature search bounds")
                st.caption(
                    "Defaults to the observed min/max per feature. Edit cells to "
                    "widen or narrow the search space before BO."
                )
                st.warning(
                    f"**Feature space:** values below are in the "
                    f"`{mtl_bo.get('x_space', 'training')}` space used by the "
                    f"MTL surrogate, **not** your original units. "
                    f"BO candidates downloaded later are in the same space — "
                    f"inverse-transform before applying as new experiments.\n\n"
                    f"**Extrapolation:** bounds wider than observed data → "
                    f"the GP extrapolates; candidates often pile up at the "
                    f"bound edges where uncertainty is highest."
                )

                default_bounds = pd.DataFrame({
                    "Feature":  feat_nm_b,
                    "Min":      X_bo.min(axis=0).astype(float),
                    "Max":      X_bo.max(axis=0).astype(float),
                })
                edited_bounds = st.data_editor(
                    default_bounds,
                    key="bo_bounds_editor",
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "Feature": st.column_config.TextColumn(disabled=True),
                        "Min":     st.column_config.NumberColumn(format="%.6g"),
                        "Max":     st.column_config.NumberColumn(format="%.6g"),
                    },
                )

                # ── 4. Candidate count + Run ──────────────────────────────
                st.markdown("---")
                st.markdown("#### 4. Run BO / MOBO")
                n_candidates = st.slider(
                    "Number of candidate points (q)",
                    min_value=1, max_value=20, value=5, key="bo_q",
                )
                num_restarts = st.slider(
                    "Acquisition optimizer restarts",
                    min_value=4, max_value=32, value=10, key="bo_restarts",
                    help="More restarts = better acquisition optimum, slower.",
                )
                raw_samples = st.select_slider(
                    "Acquisition initial raw samples",
                    options=[128, 256, 512, 1024, 2048],
                    value=512, key="bo_raw_samples",
                )

                bo_label = "MOBO" if n_targets > 1 else "BO"
                if st.button(f"Run {bo_label}", key="btn_run_bo", type="primary"):
                    try:
                        # Validate bounds
                        lows  = edited_bounds["Min"].to_numpy(dtype=np.float64)
                        highs = edited_bounds["Max"].to_numpy(dtype=np.float64)
                        if np.any(highs <= lows):
                            bad = [
                                feat_nm_b[i]
                                for i in range(len(feat_nm_b))
                                if highs[i] <= lows[i]
                            ]
                            st.error(
                                f"Min must be < Max for every feature. "
                                f"Bad: {bad}"
                            )
                            st.stop()

                        bounds_t = torch.tensor(
                            np.stack([lows, highs]), dtype=torch.double
                        )
                        train_X_t = torch.tensor(X_bo, dtype=torch.double)
                        train_Y_t = torch.tensor(Y_bo, dtype=torch.double)

                        with st.spinner(
                            f"Fitting GP surrogate(s) and optimizing "
                            f"{bo_label} acquisition…"
                        ):
                            if n_targets == 1:
                                # qLogEI handles both q=1 and q>1 batch suggestions.
                                from botorch.acquisition.logei import (
                                    qLogExpectedImprovement,
                                )
                                from botorch.sampling.normal import SobolQMCNormalSampler

                                gp = SingleTaskGP(
                                    train_X_t, train_Y_t,
                                    input_transform=Normalize(
                                        d=train_X_t.shape[1], bounds=bounds_t,
                                    ),
                                    outcome_transform=Standardize(m=1),
                                )
                                mll = ExactMarginalLogLikelihood(
                                    gp.likelihood, gp
                                )
                                fit_gpytorch_mll(mll)

                                best_f = train_Y_t.max().item()
                                sampler_bo = SobolQMCNormalSampler(
                                    sample_shape=torch.Size([128]),
                                )
                                acqf = qLogExpectedImprovement(
                                    model    = gp,
                                    best_f   = best_f,
                                    sampler  = sampler_bo,
                                )

                                cand, _ = optimize_acqf(
                                    acq_function = acqf,
                                    bounds       = bounds_t,
                                    q            = int(n_candidates),
                                    num_restarts = int(num_restarts),
                                    raw_samples  = int(raw_samples),
                                )
                                cand_np = cand.detach().cpu().numpy()
                                with torch.no_grad():
                                    post_mean = gp.posterior(cand).mean.detach().cpu().numpy().reshape(-1, 1)
                            else:
                                # MOBO
                                from botorch.acquisition.multi_objective.logei import (
                                    qLogExpectedHypervolumeImprovement,
                                )
                                from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                                    FastNondominatedPartitioning,
                                )
                                from botorch.sampling.normal import SobolQMCNormalSampler

                                gp_models = []
                                for ti in range(n_targets):
                                    gp_i = SingleTaskGP(
                                        train_X_t,
                                        train_Y_t[:, ti:ti+1],
                                        input_transform=Normalize(
                                            d=train_X_t.shape[1],
                                            bounds=bounds_t,
                                        ),
                                        outcome_transform=Standardize(m=1),
                                    )
                                    gp_models.append(gp_i)
                                model_list = ModelListGP(*gp_models)
                                mll = SumMarginalLogLikelihood(
                                    model_list.likelihood, model_list,
                                )
                                fit_gpytorch_mll(mll)

                                Y_range = (
                                    train_Y_t.max(dim=0).values
                                    - train_Y_t.min(dim=0).values
                                ).clamp_min(1e-9)
                                ref_point_t = (
                                    train_Y_t.min(dim=0).values
                                    - 0.1 * Y_range
                                )

                                partitioning = FastNondominatedPartitioning(
                                    ref_point=ref_point_t, Y=train_Y_t,
                                )
                                sampler = SobolQMCNormalSampler(
                                    sample_shape=torch.Size([128]),
                                )
                                acqf = qLogExpectedHypervolumeImprovement(
                                    model        = model_list,
                                    ref_point    = ref_point_t.tolist(),
                                    partitioning = partitioning,
                                    sampler      = sampler,
                                )

                                cand, _ = optimize_acqf(
                                    acq_function = acqf,
                                    bounds       = bounds_t,
                                    q            = int(n_candidates),
                                    num_restarts = int(num_restarts),
                                    raw_samples  = int(raw_samples),
                                )
                                cand_np = cand.detach().cpu().numpy()
                                with torch.no_grad():
                                    pm_cols = []
                                    for gp_i in gp_models:
                                        pm_cols.append(
                                            gp_i.posterior(cand)
                                                .mean.detach().cpu().numpy()
                                                .reshape(-1)
                                        )
                                    post_mean = np.column_stack(pm_cols)

                        # Invert direction sign so predicted targets read
                        # in user-original scale (max stays max, min stays min)
                        post_mean_user = post_mean * direction_signs[None, :]

                        cand_df = pd.DataFrame(
                            cand_np, columns=feat_nm_b
                        )
                        for ti, t_col in enumerate(t_cols_b):
                            cand_df[f"pred_{t_col}"] = post_mean_user[:, ti]

                        st.session_state.bo_candidates = cand_df
                        st.success(
                            f"{bo_label} suggested **{len(cand_df)}** candidate "
                            f"point(s). See the table below."
                        )

                    except Exception as e:
                        st.error(f"{bo_label} failed: {e}")
                        import traceback
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())

                # ── Display latest candidates ─────────────────────────────
                if st.session_state.get("bo_candidates") is not None:
                    st.markdown("#### Suggested candidate points")
                    st.dataframe(
                        st.session_state.bo_candidates,
                        use_container_width=True,
                    )
                    st.download_button(
                        label="⬇️ Download candidates (CSV)",
                        data=st.session_state.bo_candidates
                            .to_csv(index=False).encode("utf-8"),
                        file_name="bo_candidates.csv",
                        mime="text/csv",
                    )

