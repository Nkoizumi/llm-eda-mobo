# model_trainer.py
#
# ModelTrainer — Trains, evaluates, and selects the best model
# for a given task type. Used by FeedbackController each iteration.

import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import LabelEncoder

# ── Optional: XGBoost ─────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# ── Optional: TabPFN ──────────────────────────────────────────────
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# TabPFN works best under these limits
_TABPFN_MAX_ROWS     = 10_000
_TABPFN_MAX_FEATURES = 100


# ──────────────────────────────────────────────────────────────────
# ModelTrainer
# ──────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains multiple candidate models via k-fold cross-validation
    and returns the best fitted estimator, its name, and score.

    Parameters
    ----------
    config : FeedbackConfig
        Must have: task_type, target_col, verbose
    cv_folds : int
        Number of cross-validation folds (default: 5)
    """

    # ── Scoring metric per task ───────────────────────────────────
    _SCORING = {
        "regression"     : "r2",
        "classification" : "f1_weighted",
    }

    # ── Init ──────────────────────────────────────────────────────

    def __init__(self, config, cv_folds: int = 5):
        self.config     = config
        self.task_type  = config.task_type
        self.target_col = config.target_col
        self.verbose    = getattr(config, "verbose", True)
        self.cv_folds   = cv_folds

        self.best_model = None
        self.best_name  = None
        self.best_score = -np.inf

    # ── Public API ────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        """
        Train all candidate models and return the best one.

        Parameters
        ----------
        df : pd.DataFrame
            Fully preprocessed DataFrame (no NaNs, numeric only).

        Returns
        -------
        model      : fitted sklearn/xgboost/tabpfn estimator
        model_name : str
        score      : float — best CV score
        """
        df      = self._prepare(df)
        X, y    = self._split(df)
        models  = self._build_models(X)           # ← pass X for size check
        scoring = self._SCORING[self.task_type]

        if self.verbose:
            print(
                f"\n🏋️  [ModelTrainer] Training {len(models)} models "
                f"({self.task_type}, CV={self.cv_folds})..."
            )
            if _TABPFN_AVAILABLE:
                print(
                    f"   🧠 TabPFN available — "
                    f"eligible: {self._tabpfn_eligible(X)}"
                )

        self.best_model = None
        self.best_name  = None
        self.best_score = -np.inf

        for name, model in models.items():
            try:
                scores     = cross_val_score(
                    model, X, y,
                    cv      = self.cv_folds,
                    scoring = scoring,
                    n_jobs  = -1 if name != "tabpfn" else 1,  # TabPFN: no parallel
                )
                mean_score = float(scores.mean())
                std_score  = float(scores.std())

                if self.verbose:
                    print(
                        f"   [{name:<22}] "
                        f"{scoring}={mean_score:.4f} "
                        f"(±{std_score:.4f})"
                    )

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_name  = name
                    self.best_model = model

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  [{name}] skipped — {e}")

        # Refit best model on full training data
        if self.best_model is not None:
            self.best_model.fit(X, y)
            if self.verbose:
                print(
                    f"\n   🏆 Best: {self.best_name} | "
                    f"Score: {self.best_score:.4f}"
                )

        return self.best_model, self.best_name, self.best_score

    # ── Helpers ───────────────────────────────────────────────────

    def _tabpfn_eligible(self, X: pd.DataFrame) -> bool:
        """Check whether the dataset is within TabPFN's supported limits."""
        return (
            _TABPFN_AVAILABLE
            and X.shape[0] <= _TABPFN_MAX_ROWS
            and X.shape[1] <= _TABPFN_MAX_FEATURES
        )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safety preparation:
            - Encode any remaining categoricals (non-target)
            - Fill NaNs with column median
            - Encode classification target if still string
        """
        df = df.copy()

        # Encode leftover categoricals (non-target)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col == self.target_col:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # Fill NaNs in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

        # Encode classification target if still a string
        if self.task_type == "classification":
            if df[self.target_col].dtype == object:
                le = LabelEncoder()
                df[self.target_col] = le.fit_transform(
                    df[self.target_col].astype(str)
                )

        return df

    def _split(self, df: pd.DataFrame):
        """Split into X (features) and y (target)."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols]
        y = df[self.target_col]
        return X, y

    def _build_models(self, X: pd.DataFrame) -> dict:
        """
        Build candidate model dict based on task type.
        TabPFN and XGBoost are included if available/eligible.
        """
        use_tabpfn = self._tabpfn_eligible(X)

        if self.task_type == "regression":
            models = {
                "random_forest" : RandomForestRegressor(
                                    n_estimators = 100,
                                    random_state = 42,
                                    n_jobs       = -1,
                                  ),
                "gradient_boost": GradientBoostingRegressor(
                                    n_estimators = 100,
                                    random_state = 42,
                                  ),
                "ridge"         : Ridge(alpha=1.0),
            }
            if _XGBOOST_AVAILABLE:
                models["xgboost"] = XGBRegressor(
                    n_estimators = 100,
                    random_state = 42,
                    n_jobs       = -1,
                    verbosity    = 0,
                    eval_metric  = "rmse",
                )
            # ✅ TabPFN Regression
            if use_tabpfn:
                models["tabpfn"] = TabPFNRegressor(
                    n_estimators = 8,       # ensemble size
                    random_state = 42,
                )

        else:  # classification
            models = {
                "random_forest" : RandomForestClassifier(
                                    n_estimators = 100,
                                    random_state = 42,
                                    n_jobs       = -1,
                                  ),
                "gradient_boost": GradientBoostingClassifier(
                                    n_estimators = 100,
                                    random_state = 42,
                                  ),
                "logistic"      : LogisticRegression(
                                    max_iter     = 500,
                                    random_state = 42,
                                  ),
            }
            if _XGBOOST_AVAILABLE:
                models["xgboost"] = XGBClassifier(
                    n_estimators      = 100,
                    random_state      = 42,
                    n_jobs            = -1,
                    verbosity         = 0,
                    use_label_encoder = False,
                    eval_metric       = "logloss",
                )
            # ✅ TabPFN Classification
            if use_tabpfn:
                models["tabpfn"] = TabPFNClassifier(
                    n_estimators = 8,
                    random_state = 42,
                )

        return models
