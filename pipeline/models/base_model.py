# pipeline/models/base_model.py

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models inside pipeline/models/.
    
    Currently serves:
        - XGBoostModel
        - NeuralNetworkModel
    
    Random Forest lives in app.py and does NOT inherit from this.
    Its results dict is injected into Orchestrator as baseline_results.

    Every subclass MUST implement:
        - run(X, y)   → train + CV evaluate → returns metrics dict
        - predict(X)  → inference on new data → returns np.ndarray
    """

    def __init__(
        self,
        task: str = "regression",
        n_splits: int = 5,
        random_state: int = 42,
        use_cuda: bool = True,
    ):
        """
        Parameters
        ----------
        task         : "regression" or "classification"
        n_splits     : Number of K-Fold CV splits
        random_state : Reproducibility seed
        use_cuda     : Whether to attempt GPU acceleration
        """
        self._validate_task(task)

        self.task         = task
        self.n_splits     = n_splits
        self.random_state = random_state
        self.use_cuda     = use_cuda

        # ── Populated after run() is called ──────────────
        self.model        = None          # Trained model object
        self.results: dict = {}           # CV metrics
        self.is_fitted: bool = False      # Guard for predict()

    # ─────────────────────────────────────────────────────
    # Abstract Methods (MUST be implemented by subclasses)
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train model using K-Fold (or Stratified K-Fold) CV.

        Parameters
        ----------
        X : Feature matrix, shape (n_samples, n_features)
        y : Target vector, shape (n_samples,)

        Returns
        -------
        dict with keys depending on task:

        Regression:
            {
                "model" : trained model,
                "r2"    : float,
                "mae"   : float,
                "rmse"  : float,
            }

        Classification:
            {
                "model"    : trained model,
                "accuracy" : float,
                "f1"       : float,
                "auc"      : float,
            }
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on new/unseen data.

        Parameters
        ----------
        X : Feature matrix, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of predictions, shape (n_samples,)
        """
        pass

    # ─────────────────────────────────────────────────────
    # Concrete Shared Methods (available to all subclasses)
    # ─────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """
        Returns CV metrics after run() has been called.
        Raises RuntimeError if called before run().
        """
        self._check_fitted(for_method="get_metrics")

        # Strip the model object — return only numeric metrics
        return {
            k: v for k, v in self.results.items()
            if k != "model"
        }

    def get_task(self) -> str:
        """Returns the task type: 'regression' or 'classification'."""
        return self.task

    def get_model(self):
        """Returns the underlying trained model object."""
        self._check_fitted(for_method="get_model")
        return self.model

    def summary(self) -> str:
        """
        Returns a human-readable string summary of CV results.
        Useful for logging and LLM engine input.
        """
        self._check_fitted(for_method="summary")

        name = self.__class__.__name__
        lines = [f"\n[{name}] Task: {self.task.upper()} | CV Folds: {self.n_splits}"]

        if self.task == "regression":
            lines.append(f"  R2   : {self.results.get('r2',   'N/A'):.4f}")
            lines.append(f"  MAE  : {self.results.get('mae',  'N/A'):.4f}")
            lines.append(f"  RMSE : {self.results.get('rmse', 'N/A'):.4f}")
        else:
            lines.append(f"  Accuracy : {self.results.get('accuracy', 'N/A'):.4f}")
            lines.append(f"  F1       : {self.results.get('f1',       'N/A'):.4f}")
            lines.append(f"  AUC      : {self.results.get('auc',      'N/A'):.4f}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_task(task: str):
        """Ensures task is a supported value."""
        valid = ("regression", "classification")
        if task not in valid:
            raise ValueError(
                f"Invalid task '{task}'. Must be one of: {valid}"
            )

    def _check_fitted(self, for_method: str = "this method"):
        """Raises RuntimeError if model has not been trained yet."""
        if not self.is_fitted:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Cannot call '{for_method}' "
                f"before run(). Please call run(X, y) first."
            )

    def _store_results(self, results: dict):
        """
        Called at end of run() by subclasses to store results
        and flip the is_fitted flag.

        Usage inside subclass:
            self._store_results({
                "model": trained_model,
                "r2":    0.91,
                ...
            })
        """
        self.results   = results
        self.model     = results.get("model", None)
        self.is_fitted = True

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"task={self.task!r}, "
            f"n_splits={self.n_splits}, "
            f"use_cuda={self.use_cuda}, "
            f"status={status!r})"
        )

