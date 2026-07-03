# app.py

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from tabs import (
    overview, missing_values, distributions, outliers, correlations,
    llm_decisions, transformed_data, loo_results, mtl, feedback_loop,
    bo_mobo,
)
from tabs import results as results_tab

warnings.filterwarnings("ignore")


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
    "transformed_df",
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    panels = st.tabs([
        "Overview",            # panels[0]
        "Missing Values",      # panels[1]
        "Distributions",       # panels[2]
        "Outliers",            # panels[3]
        "Correlations",        # panels[4]
        "LLM Decisions",       # panels[5]
        "Transformed Data",    # panels[6]
        "LOO Results",         # panels[7]
        "Multitask Learning",  # panels[8]   joint multi-output model
        "Feedback Loop",       # panels[9]
        "Results",             # panels[10]  per-target FI + PDP
        "BO / MOBO",           # panels[11]  Bayesian Optimization on the MTL surrogate
    ])


    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ──────────────────────────────────────────────────────────────────────
    with panels[0]:
        overview.render(df)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 — MISSING VALUES
    # ──────────────────────────────────────────────────────────────────────
    with panels[1]:
        missing_values.render(df)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 — DISTRIBUTIONS
    # ──────────────────────────────────────────────────────────────────────
    with panels[2]:
        distributions.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 — OUTLIERS
    # ──────────────────────────────────────────────────────────────────────
    with panels[3]:
        outliers.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5 — CORRELATIONS
    # ──────────────────────────────────────────────────────────────────────
    with panels[4]:
        correlations.render(df, target_cols=target_cols)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 6 — LLM DECISIONS
    # ──────────────────────────────────────────────────────────────────────
    with panels[5]:
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
    with panels[6]:
        transformed_data.render(df, target_cols=target_cols)


    # ──────────────────────────────────────────────────────────────────────
    # TAB 8 — LOO RESULTS  (panels[7])
    # ──────────────────────────────────────────────────────────────────────
    with panels[7]:
        loo_results.render(df, target_cols=target_cols, target_col_input=target_col_input, task_type=task_type, estimator_choice=estimator_choice)
    # ──────────────────────────────────────────────────────────────────────
    # TAB 9 — MULTITASK LEARNING  (panels[8])
    # ──────────────────────────────────────────────────────────────────────
    with panels[8]:
        mtl.render(df, target_cols=target_cols, task_type=task_type)


    # ──────────────────────────────────────────────────────────────────────
    # TAB 10 — FEEDBACK LOOP  (panels[9])
    # ──────────────────────────────────────────────────────────────────────
    with panels[9]:
        feedback_loop.render()


    # ──────────────────────────────────────────────────────────────────────
    # TAB 11 — RESULTS  (panels[10])
    # Feature importance + partial dependence plots, per target,
    # from the joint multi-output model trained in the Multitask Learning tab.
    # ──────────────────────────────────────────────────────────────────────
    with panels[10]:
        results_tab.render()

    # ──────────────────────────────────────────────────────────────────────
    # TAB 12 — BO / MOBO  (panels[11])
    # Bayesian Optimization on the joint multi-output GP surrogate.
    # Single target → Log Expected Improvement.
    # Multi-target → qLogExpectedHypervolumeImprovement (MOBO).
    # ──────────────────────────────────────────────────────────────────────
    with panels[11]:
        bo_mobo.render(target_cols=target_cols)
