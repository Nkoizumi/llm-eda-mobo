# eda_pipeline.py
#
# AutoEDA Pipeline — Core EDA, Feature Analysis & Model Training
# Produces an eda_report consumed by FeedbackController & FeedbackAnalyzer.

import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.inspection      import permutation_importance

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _safe_cols(df: pd.DataFrame, cols: list) -> list:
    """Return only columns that actually exist in df."""
    return [c for c in cols if c in df.columns]


def _numeric_cols(df: pd.DataFrame, exclude: list = None) -> list:
    """Return numeric columns, optionally excluding some."""
    exclude = exclude or []
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]


def _encode_target(series: pd.Series, task_type: str) -> pd.Series:
    """
    ✅ CENTRALISED target encoder.

    - classification : LabelEncode any non-numeric target
                       (catches object, category, and mixed types like 'RL')
    - regression     : coerce to float; raise clearly if impossible
    """
    if task_type == "classification":
        # Already integer-encoded → nothing to do
        if pd.api.types.is_integer_dtype(series):
            return series
        # Float that happens to hold integer values (0.0, 1.0 …) → cast
        if pd.api.types.is_float_dtype(series):
            return series.astype(int)
        # String / category / object  → LabelEncode
        le = LabelEncoder()
        return pd.Series(
            le.fit_transform(series.astype(str)),
            index  = series.index,
            name   = series.name,
            dtype  = int,
        )

    else:  # regression
        try:
            return pd.to_numeric(series, errors="raise")
        except Exception:
            raise ValueError(
                f"Target column '{series.name}' contains non-numeric values "
                f"({series.unique()[:5]}) but task_type='regression'. "
                f"Switch to classification or fix the target column."
            )


# ──────────────────────────────────────────────────────────────────
# AutoEDAPipeline
# ──────────────────────────────────────────────────────────────────

class FeedbackEDAPipeline:
    """
    Runs automated EDA on a DataFrame and trains a baseline model.

    NOT the same class as `pipeline.orchestrator.AutoEDAPipeline`, which is
    what the README, the Streamlit tabs and scripts/run_benchmarks.py all mean
    by that name. The two shared a name while having incompatible constructors
    (`config` here vs `target_col, task, ollama_host, use_local_llm` there) and
    no method in common — this one produces SHAP-feedback diagnostics, that one
    builds the LLM-chosen preprocessing pipeline.

    Renamed because a same-name/different-class pair is how a later fix lands
    in the wrong file. That nearly happened: the 2026-08-08 `run_loo` fix went
    into `pipeline/orchestrator.py`, and anyone assuming FeedbackController
    shared that code would have been wrong.

    Produces:
        eda_report  : dict — signals consumed by FeedbackController
        model       : fitted sklearn estimator
        model_name  : str — best model name
        score       : float — CV score
        feature_imp : pd.Series — feature importances
    """

    _MODELS = {
        "regression": {
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "ridge": Ridge(alpha=1.0),
        },
        "classification": {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "logistic": LogisticRegression(
                max_iter=500, random_state=42
            ),
        },
    }

    _SCORING = {
        "regression":     "r2",
        "classification": "f1_weighted",
    }

    # ── Init ──────────────────────────────────────────────────────

    def __init__(self, config):
        self.config     = config
        self.task_type  = config.task_type
        self.target_col = config.target_col
        self.verbose    = getattr(config, "verbose", True)

        self.model       = None
        self.model_name  = None
        self.score       = None
        self.feature_imp = None
        self.eda_report  = {}

    # ── Public API ────────────────────────────────────────────────

    def run(self, df: pd.DataFrame):
        """
        Full EDA + model training pipeline.

        Returns
        -------
        eda_report  : dict
        model       : fitted estimator
        model_name  : str
        score       : float
        feature_imp : pd.Series
        """
        if self.verbose:
            print("\n🔍 [AutoEDAPipeline] Starting EDA...")

        # ✅ Guard: target column must exist
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        df = self._prepare(df)

        self.eda_report = {
            **self._detect_ghost_features(df),
            **self._detect_outlier_features(df),
            **self._detect_damaged_features(df),
            **self._detect_skewed_features(df),
            **self._basic_stats(df),
        }

        self.model, self.model_name, self.score = self._train_best_model(df)
        self.feature_imp = self._compute_feature_importance(df)

        if self.verbose:
            print(
                f"\n✅ [AutoEDAPipeline] Done — "
                f"Model: {self.model_name} | Score: {self.score:.4f}"
            )

        return (
            self.eda_report,
            self.model,
            self.model_name,
            self.score,
            self.feature_imp,
        )

    # ── Data Preparation ──────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safe preparation:
            1. Drop fully-empty columns
            2. Encode non-target categorical columns
            3. Fill numeric NaNs with median
            4. ✅ Encode target using _encode_target() — handles ALL types
        """
        df = df.copy()

        # 1. Drop all-null columns
        df.dropna(axis=1, how="all", inplace=True)

        # 2. Encode non-target categoricals
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col == self.target_col:
                continue                        # ← target handled separately
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # 3. Fill numeric NaNs with median
        for col in _numeric_cols(df):
            df[col] = df[col].fillna(df[col].median())

        # 4. ✅ Encode target — centralised, handles 'RL', categories, etc.
        if self.target_col in df.columns:
            df[self.target_col] = _encode_target(
                df[self.target_col], self.task_type
            )

        if self.verbose:
            tgt_dtype = df[self.target_col].dtype if self.target_col in df.columns else "missing"
            print(
                f"   📐 Prepared: {df.shape[0]} rows × {df.shape[1]} cols | "
                f"target dtype after encode: {tgt_dtype}"
            )

        return df

    # ── EDA Signal Detectors ──────────────────────────────────────

    def _detect_ghost_features(
        self, df: pd.DataFrame, threshold: float = 0.90
    ) -> dict:
        num_cols = _numeric_cols(df, exclude=[self.target_col])
        ghost    = []

        if len(num_cols) < 2:
            return {"ghost_features": []}

        corr  = df[num_cols].corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        for col in upper.columns:
            if (upper[col] >= threshold).any():
                ghost.append(col)

        if self.verbose and ghost:
            print(f"   👻 Ghost features: {ghost}")

        return {"ghost_features": ghost}

    def _detect_outlier_features(
        self,
        df: pd.DataFrame,
        iqr_factor: float = 3.0,
        pct_thresh: float = 0.03,
    ) -> dict:
        num_cols = _numeric_cols(df, exclude=[self.target_col])
        outliers = []

        for col in num_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr    = q3 - q1
            lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            pct    = ((df[col] < lo) | (df[col] > hi)).sum() / len(df)
            if pct >= pct_thresh:
                outliers.append(col)

        if self.verbose and outliers:
            print(f"   📈 Outlier features: {outliers}")

        return {"outlier_features": outliers}

    def _detect_damaged_features(
        self, df: pd.DataFrame, zero_thresh: float = 0.50
    ) -> dict:
        num_cols = _numeric_cols(df, exclude=[self.target_col])
        damaged  = []

        for col in num_cols:
            if (df[col] == 0).sum() / len(df) >= zero_thresh:
                damaged.append(col)

        if self.verbose and damaged:
            print(f"   🩹 Damaged features: {damaged}")

        return {"damaged_features": damaged}

    def _detect_skewed_features(
        self, df: pd.DataFrame, skew_threshold: float = 2.0
    ) -> dict:
        num_cols = _numeric_cols(df, exclude=[self.target_col])
        skewed   = []

        for col in num_cols:
            if abs(df[col].skew()) >= skew_threshold:
                skewed.append(col)

        if self.verbose and skewed:
            print(f"   📉 Skewed features: {skewed}")

        return {"skewed_features": skewed}

    def _basic_stats(self, df: pd.DataFrame) -> dict:
        num_cols = _numeric_cols(df, exclude=[self.target_col])
        return {
            "n_rows":      len(df),
            "n_cols":      len(df.columns),
            "n_features":  len(num_cols),
            "missing_pct": df.isnull().mean().to_dict(),
            "dtypes":      df.dtypes.astype(str).to_dict(),
        }

    # ── Model Training ────────────────────────────────────────────

    def _get_xy(self, df: pd.DataFrame):
        """
        ✅ Split into X (numeric features only) and y (encoded target).
        Drops any remaining non-numeric feature columns safely.
        """
        feature_cols = [
            c for c in df.columns
            if c != self.target_col
            and pd.api.types.is_numeric_dtype(df[c])   # ← safety net
        ]
        X = df[feature_cols]
        y = df[self.target_col]

        # ✅ Final guard — re-encode y if something slipped through
        if not pd.api.types.is_numeric_dtype(y):
            if self.verbose:
                print(
                    f"   ⚠️  [_get_xy] target '{self.target_col}' is still "
                    f"{y.dtype} — re-encoding now."
                )
            y = _encode_target(y, self.task_type)

        return X, y

    def _train_best_model(self, df: pd.DataFrame):
        X, y    = self._get_xy(df)
        scoring = self._SCORING[self.task_type]
        models  = self._MODELS[self.task_type]

        best_model = None
        best_name  = None
        best_score = -np.inf

        if self.verbose:
            print(f"\n🏋️  Training models ({self.task_type})...")

        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=5, scoring=scoring, n_jobs=-1
                )
                mean_score = float(scores.mean())

                if self.verbose:
                    print(
                        f"   [{name}] {scoring}: "
                        f"{mean_score:.4f} (±{scores.std():.4f})"
                    )

                if mean_score > best_score:
                    best_score = mean_score
                    best_name  = name
                    best_model = model

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  [{name}] failed: {e}")

        if best_model is not None:
            best_model.fit(X, y)

        return best_model, best_name, best_score

    # ── Feature Importance ────────────────────────────────────────

    def _compute_feature_importance(
        self, df: pd.DataFrame, n_repeats: int = 5
    ) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)

        X, y = self._get_xy(df)

        try:
            pi = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            importance = pd.Series(
                pi.importances_mean, index=X.columns
            ).sort_values(ascending=False)

        except Exception:
            if hasattr(self.model, "feature_importances_"):
                importance = pd.Series(
                    self.model.feature_importances_, index=X.columns
                ).sort_values(ascending=False)
            else:
                importance = pd.Series(dtype=float)

        if self.verbose and not importance.empty:
            print("\n📊 Top Feature Importances:")
            for feat, imp in importance.head(5).items():
                print(f"   {feat:<25} {imp:.4f}")

        return importance


# Backwards-compatible alias. Prefer `FeedbackEDAPipeline` — the bare name
# `AutoEDAPipeline` means pipeline.orchestrator.AutoEDAPipeline everywhere else
# in this project.
AutoEDAPipeline = FeedbackEDAPipeline
