# pipeline/orchestrator.py

import logging
import pandas as pd
import numpy as np

from sklearn.base import clone as sk_clone
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    r2_score, mean_squared_error,
    mean_absolute_error, accuracy_score, f1_score,
)

from .local_llm_engine import (
    LocalEnsembleLLMEngine,
    EnsembleDecision,
    LLMDecision,
)
from .transformers import HighCorrelationRemover

from .models.xgboost_model        import XGBoostModel
from .models.neural_network_model import NeuralNetworkModel
from .models.evaluator import (
    compare_regression_results,
    compare_classification_results,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class Orchestrator:
    """Uses Transformer — import is lazy so it won't break AutoEDAPipeline."""

    def __init__(
        self,
        task             = "regression",
        n_splits         = 5,
        random_state     = 42,
        use_cuda         = True,
        baseline_results = None,
        ollama_host      = "http://localhost:11434",   # ✅ Fix 4: consistent with AutoEDAPipeline
    ):
        from .transformers import Transformer   # ✅ lazy — safe

        self.task         = task
        self.n_splits     = n_splits
        self.random_state = random_state
        self.use_cuda     = use_cuda
        self.transformer  = Transformer()
        self.llm          = LocalEnsembleLLMEngine(ollama_host=ollama_host)  # ✅ Fix 4
        self.results      = baseline_results or {}

    def _build_models(self):
        """XGBoost + Neural Network only — RF is in app.py."""
        shared = dict(
            task         = self.task,
            n_splits     = self.n_splits,
            random_state = self.random_state,
        )
        return {
            "XGBoost":        XGBoostModel(**shared, use_cuda=self.use_cuda),
            "Neural Network": NeuralNetworkModel(**shared, use_cuda=self.use_cuda),
        }

    def run(
        self,
        df              : pd.DataFrame,
        target_col      : str,
        pre_transformed : bool = False,
    ) -> dict:

        print(f"\n{'='*55}")
        print(f"  AUTO EDA PIPELINE | Task: {self.task.upper()}")
        print(f"{'='*55}")

        # ── 0. Guard ──────────────────────────────────────────────────────────
        if target_col not in df.columns:
            raise KeyError(
                f"[Orchestrator] target_col '{target_col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        # ── 1. Split target from features ─────────────────────────────────────
        y = df[target_col]
        X = df.drop(columns=[target_col])
        print(f"[Orchestrator] Raw data        : X={X.shape}, y={y.shape}")
        print(f"[Orchestrator] y dtype         : {y.dtype}")
        print(f"[Orchestrator] y unique values : {sorted(y.unique())[:10]} ...")

        # ── 2. Label-encode y for classification ──────────────────────────────
        if self.task == "classification":
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
            print(f"[Orchestrator] y after LabelEncode: {sorted(y.unique())} ✅")
            self._label_encoder = le   # ✅ store for inverse transform later if needed
        else:
            self._label_encoder = None

        # ── 3. Transform only if not already pre-processed ────────────────────
        if pre_transformed:
            X_transformed = X
            print(f"[Orchestrator] Pre-transformed : X={X_transformed.shape}, y={y.shape}")
        else:
            X_transformed = self.transformer.fit_transform(X, y)
            print(f"[Orchestrator] Transformed     : X={X_transformed.shape}, y={y.shape}")

        # ── 4. Train all models on transformed X ──────────────────────────────
        for name, model in self._build_models().items():
            print(f"\n[Orchestrator] Running → {name}")
            self.results[name] = model.run(X_transformed, y)

        # ── 5. Compare results ─────────────────────────────────────────────────
        if self.task == "regression":
            comparison_df = compare_regression_results(self.results)
        else:
            comparison_df = compare_classification_results(self.results)

        print(f"\n{'='*55}")
        print("  FINAL MODEL COMPARISON")
        print(f"{'='*55}")
        print(comparison_df.to_string())

        # ── 6. Build and return output ─────────────────────────────────────────
        pipeline_output = {
            "task":       self.task,
            "results":    self.results,
            "comparison": comparison_df,
        }
        pipeline_output["llm_summary"] = self.llm.interpret(pipeline_output)
        return pipeline_output


# ─────────────────────────────────────────────────────────────────────────────
class AutoEDAPipeline:

    def __init__(
        self,
        target_col    : str  = "target",
        task          : str  = "classification",
        ollama_host   : str  = "http://localhost:11434",
        use_local_llm : bool = True,
    ):
        self.target_col    = target_col
        self.task          = task
        self.ollama_host   = ollama_host
        self.use_local_llm = use_local_llm

        self.engine            = LocalEnsembleLLMEngine(ollama_host=ollama_host)
        self.ensemble_result_  : EnsembleDecision | None = None
        self.pipeline_         = None
        self.X_transformed     = None
        self.y                 = None

    # ─────────────────────────────────────────────────────────────────────
    def build_pipeline(self, df: pd.DataFrame):
        profile = self.engine.profile_dataset(df)

        if not isinstance(profile, dict):
            logger.error(
                f"[build_pipeline] profile_dataset() returned "
                f"{type(profile).__name__} instead of dict!"
            )
            profile = {}

        logger.info(f"[build_pipeline] Profile keys: {list(profile.keys())}")

        if self.use_local_llm:
            self.ensemble_result_ = self.engine.run_ensemble(profile)
        else:
            fallback = self.engine._fallback_decision("rule-based", profile)
            self.ensemble_result_ = EnsembleDecision(
                phi4_decision    = fallback,
                mistral_decision = fallback,
                final            = fallback,
                agreement_score  = 1.0,
                conflicts        = [],
                tiebreak_used    = False,
            )

        self.pipeline_ = self._build_sklearn_pipeline(
            self.ensemble_result_.final,
            profile,
        )
        return self

    # ─────────────────────────────────────────────────────────────────────
    def _build_sklearn_pipeline(self, decision: LLMDecision, profile: dict):
        from sklearn.pipeline      import Pipeline
        from sklearn.impute        import SimpleImputer, KNNImputer
        from sklearn.preprocessing import (
            StandardScaler, RobustScaler, MinMaxScaler,
            PowerTransformer, OrdinalEncoder, OneHotEncoder,
            FunctionTransformer,
        )
        from sklearn.compose import ColumnTransformer

        numeric_cols     = profile.get("numeric_cols",     [])
        categorical_cols = profile.get("categorical_cols", [])

        onehot_cols  = decision.encoding.get("onehot_features",  [])
        ordinal_cols = decision.encoding.get("ordinal_features", [])

        assigned     = set(onehot_cols) | set(ordinal_cols)
        default_cols = [c for c in categorical_cols if c not in assigned]
        ordinal_cols = ordinal_cols + default_cols

        strategy = decision.imputation_strategy
        if strategy == "knn":
            num_imputer = KNNImputer(n_neighbors=decision.knn_neighbors)
        elif strategy in ("mean", "median"):
            num_imputer = SimpleImputer(strategy=strategy)
        else:
            num_imputer = SimpleImputer(strategy="median")

        scaler = {
            "robust":   RobustScaler(),
            "minmax":   MinMaxScaler(),
            "standard": StandardScaler(),
        }.get(decision.scaler, StandardScaler())

        numeric_steps = [("imputer", num_imputer)]
        if decision.power_transform in ("yeo-johnson", "box-cox"):
            numeric_steps.append((
                "power",
                PowerTransformer(method=decision.power_transform),
            ))
        numeric_steps.append(("scaler", scaler))

        transformers = []

        if numeric_cols:
            transformers.append(("num", Pipeline(steps=numeric_steps), numeric_cols))

        if onehot_cols:
            transformers.append((
                "cat_onehot",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown = "ignore",
                        sparse_output  = False,
                        drop           = "if_binary",
                    )),
                ]),
                onehot_cols,
            ))

        if ordinal_cols:
            transformers.append((
                "cat_ordinal",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        handle_unknown = "use_encoded_value",
                        unknown_value  = -1,
                    )),
                ]),
                ordinal_cols,
            ))

        if not transformers:
            return Pipeline([("passthrough", FunctionTransformer(validate=False))])

        preprocessor = ColumnTransformer(
            transformers = transformers,
            remainder    = "drop",
        )

        # Apply the LLM-ensemble-chosen correlation threshold. The
        # HighCorrelationRemover wraps numpy → named DataFrame so it can drop
        # columns; get_transformed_df below runs it manually to preserve
        # column names from preprocessor.get_feature_names_out().
        corr_threshold = float(getattr(decision, "correlation_threshold", 0.90) or 0.90)
        return Pipeline([
            ("preprocessor", preprocessor),
            ("corr_remover", HighCorrelationRemover(threshold=corr_threshold)),
        ])

    # ─────────────────────────────────────────────────────────────────────
    def get_transformed_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Call build_pipeline() first.")

        X = df.drop(columns=[self.target_col], errors="ignore")

        # Standard sklearn flow: fit_transform on the full pipeline.
        # HighCorrelationRemover (in the pipeline) is now position-based and
        # implements get_feature_names_out, so Pipeline.get_feature_names_out
        # threads names correctly through preprocessor → corr_remover.
        X_transformed = self.pipeline_.fit_transform(X)
        try:
            feature_names = self.pipeline_.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]

        corr_step = self.pipeline_.named_steps.get("corr_remover")
        if corr_step is not None:
            n_dropped = len(corr_step.features_to_drop_positions_)
            n_before = len(corr_step.feature_names_in_)
            logger.info(
                "HighCorrelationRemover (threshold=%.2f) dropped %d / %d columns "
                "(kept %d).",
                corr_step.threshold, n_dropped, n_before, n_before - n_dropped,
            )

        arr = (X_transformed.values if isinstance(X_transformed, pd.DataFrame)
               else np.asarray(X_transformed))
        self.X_transformed = arr

        if self.target_col in df.columns:
            self.y = df[self.target_col].loc[X.index]
        else:
            self.y = None

        return pd.DataFrame(arr, columns=feature_names, index=X.index)

    # ─────────────────────────────────────────────────────────────────────
    def run_loo(self, df: pd.DataFrame, estimator) -> dict:
        """Leave-one-out CV with the preprocessing refit inside every fold.

        It used to LOO-split `self.X_transformed`, which `get_transformed_df`
        produces by calling `fit_transform` on the WHOLE frame. Every fold's
        held-out row had therefore already contributed to the imputer's medians,
        the power transform's lambda, the scaler's mean/std and the correlation
        filter's column choice — so the result was not the held-out estimate its
        name promises, and it disagreed in meaning with
        `tabs/_loo_utils.run_loo_with_wrapper`, which does refit per fold.

        Measured before changing it (2026-08-08), so the size of the effect is
        on record rather than assumed:

            Fish  (n=159) RF   R² 0.9711 -> 0.9711   (0.0000)
            Fish  (n=159) MLP  R² 0.8486 -> 0.8501  (+0.0015)
            slump (n=103) MLP  R² 0.1115 -> 0.1007  (-0.0109)

        i.e. negligible, and BENCHMARKS.md is unaffected. RF is invariant to
        monotone per-feature transforms, none of the pipeline steps look at the
        target, and dropping one row of ~150 barely moves a median. The reason
        to fix it anyway is that a method called `run_loo` should mean one
        thing, and the honest version costs a refit per fold.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call build_pipeline() first.")

        X_raw = df.drop(columns=[self.target_col], errors="ignore")
        if self.target_col not in df.columns:
            raise RuntimeError(
                f"run_loo needs the target column {self.target_col!r} in `df`."
            )
        y = df[self.target_col].loc[X_raw.index]

        loo         = LeaveOneOut()
        y_true_list = []
        y_pred_list = []

        for train_idx, test_idx in loo.split(X_raw):
            fold = SkPipeline([("prep", sk_clone(self.pipeline_)),
                               ("est",  sk_clone(estimator))])
            fold.fit(X_raw.iloc[train_idx], y.iloc[train_idx])
            pred = fold.predict(X_raw.iloc[test_idx])
            y_true_list.append(float(y.iloc[test_idx[0]]))
            y_pred_list.append(float(pred[0]))

        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        residuals  = y_true_arr - y_pred_arr

        if self.task == "regression":
            global_r2   = r2_score(y_true_arr, y_pred_arr)
            global_rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
            global_mae  = mean_absolute_error(y_true_arr, y_pred_arr)

            print(f"✅ LOO R²   : {global_r2:.4f}")
            print(f"✅ LOO RMSE : {global_rmse:.4f}")
            print(f"✅ LOO MAE  : {global_mae:.4f}")

            return {
                "r2":        global_r2,
                "rmse":      global_rmse,
                "mae":       global_mae,
                "y_true":    y_true_arr,
                "y_pred":    y_pred_arr,
                "residuals": residuals,
            }
        else:
            acc = accuracy_score(y_true_arr, y_pred_arr)
            f1  = f1_score(y_true_arr, y_pred_arr, average="weighted")

            print(f"✅ LOO Accuracy : {acc:.4f}")
            print(f"✅ LOO F1       : {f1:.4f}")

            return {
                "test_accuracy":    np.array([acc]),
                "test_f1_weighted": np.array([f1]),
                "y_true":           y_true_arr,
                "y_pred":           y_pred_arr,
                "residuals":        residuals,
            }
