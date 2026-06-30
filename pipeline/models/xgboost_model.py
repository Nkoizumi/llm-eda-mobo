import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_xgb_device():
    """
    Detect CUDA availability and return appropriate XGBoost device string.
    Falls back to CPU if CUDA is not available.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[XGBoost] CUDA detected. Using GPU (device='cuda')")
            return "cuda"
    except FileNotFoundError:
        pass
    print("[XGBoost] No CUDA detected. Using CPU (device='cpu')")
    return "cpu"


def run_xgboost_regression(X, y, n_splits=5, random_state=42):
    """
    XGBoost Regression with K-Fold Cross-Validation.
    Uses CUDA if available.
    """
    device = get_xgb_device()

    model_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist",       # Required for GPU
        device=device,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores, mae_scores, rmse_scores = [], [], []

    print(f"\n[XGBoost Regression] Running {n_splits}-Fold CV...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)

        r2  = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"  Fold {fold}: R2={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

    print("\n[XGBoost Regression] Cross-Validation Summary:")
    print(f"  Mean R2   : {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"  Mean MAE  : {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    print(f"  Mean RMSE : {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")

    return {
        "model": model,
        "r2": np.mean(r2_scores),
        "mae": np.mean(mae_scores),
        "rmse": np.mean(rmse_scores),
    }


def run_xgboost_classification(X, y, n_splits=5, random_state=42):
    """
    XGBoost Classification with Stratified K-Fold Cross-Validation.
    Automatically handles binary and multiclass problems.
    Uses CUDA if available.
    """
    device = get_xgb_device()
    n_classes = len(np.unique(y))
    is_binary = (n_classes == 2)

    model_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist",
        device=device,
        early_stopping_rounds=50,
        objective="binary:logistic" if is_binary else "multi:softprob",
        num_class=None if is_binary else n_classes,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores, f1_scores, auc_scores = [], [], []

    print(f"\n[XGBoost Classification] Running {n_splits}-Fold Stratified CV...")
    print(f"  Task Type: {'Binary' if is_binary else 'Multiclass'} ({n_classes} classes)\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, average="binary" if is_binary else "weighted")
        auc = roc_auc_score(
            y_val, probs[:, 1] if is_binary else probs,
            multi_class="ovr" if not is_binary else "raise",
            average="macro" if not is_binary else None
        )

        acc_scores.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)

        print(f"  Fold {fold}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    print("\n[XGBoost Classification] Cross-Validation Summary:")
    print(f"  Mean Accuracy : {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  Mean F1       : {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"  Mean AUC      : {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    return {
        "model": model,
        "accuracy": np.mean(acc_scores),
        "f1": np.mean(f1_scores),
        "auc": np.mean(auc_scores),
    }
    

# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS AT THE BOTTOM of xgboost_model.py — after all existing functions
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost wrapper exposing:
      - .fit(X, y)   — train on a given split   (used by LOO)
      - .predict(X)  — return predictions        (used by LOO)
      - .run(X, y)   — full K-Fold CV metrics    (used by Tab 9)

    hpo_params: optional dict of hyperparameters from optimize_xgboost()
                that override the defaults in _make_estimator().
    """

    def __init__(
        self,
        task="classification",
        n_splits=5,
        random_state=42,
        use_cuda=False,
        hpo_params=None,
    ):
        self.task         = task
        self.n_splits     = n_splits
        self.random_state = random_state
        self.use_cuda     = use_cuda
        self.hpo_params   = hpo_params or {}
        self._booster     = None

    # ── Build a fresh XGB estimator ───────────────────────────────────────
    def _make_estimator(self):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        device = "cuda" if self.use_cuda else "cpu"

        # Default params — overridden by hpo_params when present
        defaults = dict(
            n_estimators     = 200,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            random_state     = self.random_state,
            tree_method      = "hist",
            device           = device,
            verbosity        = 0,
        )
        defaults.update(self.hpo_params)
        # Ensure random_state and device from self always win
        defaults["random_state"] = self.random_state
        defaults["device"]       = device

        if self.task == "classification":
            defaults.setdefault("eval_metric", "logloss")
            return xgb.XGBClassifier(**defaults, use_label_encoder=False)
        else:
            defaults.setdefault("eval_metric", "rmse")
            return xgb.XGBRegressor(**defaults)

    # ── sklearn-compatible fit ────────────────────────────────────────────
    def fit(self, X, y):
        """Train on (X, y). Required before calling .predict()."""
        self._booster = self._make_estimator()
        self._booster.fit(X, y)
        return self

    # ── sklearn-compatible predict ────────────────────────────────────────
    def predict(self, X):
        """
        Return predictions for X.
        - Classification → integer class labels (numpy int array)
        - Regression     → float values (numpy float array)
        """
        if self._booster is None:
            raise RuntimeError(
                "XGBoostModel: .predict() called before .fit(). "
                "Call .fit(X_train, y_train) first."
            )
        preds = self._booster.predict(X)
        if self.task == "classification":
            return np.array(preds, dtype=int).ravel()
        return np.array(preds, dtype=float).ravel()

    # ── K-Fold CV runner for Tab 9 ────────────────────────────────────────
    def run(self, X, y):
        """
        Run stratified K-Fold CV.
        Returns a dict with metrics and full y_actual / y_pred arrays.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if self.task == "classification":
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kf.split(X, y.astype(int))
            acc_scores, f1_scores = [], []
            y_actual_all, y_pred_all = [], []

            for train_idx, val_idx in splits:
                est = self._make_estimator()
                est.fit(X[train_idx], y[train_idx])
                preds = est.predict(X[val_idx])

                acc_scores.append(
                    accuracy_score(y[val_idx].astype(int), preds)
                )
                f1_scores.append(
                    f1_score(
                        y[val_idx].astype(int),
                        preds,
                        average="weighted",
                        zero_division=0,
                    )
                )
                y_actual_all.extend(y[val_idx].tolist())
                y_pred_all.extend(preds.tolist())

            return {
                "accuracy": float(np.mean(acc_scores)),
                "f1":       float(np.mean(f1_scores)),
                "y_actual": np.array(y_actual_all, dtype=float),
                "y_pred":   np.array(y_pred_all,   dtype=float),
            }

        else:
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kf.split(X)
            r2_scores, mae_scores, rmse_scores = [], [], []
            y_actual_all, y_pred_all = [], []

            for train_idx, val_idx in splits:
                est = self._make_estimator()
                est.fit(X[train_idx], y[train_idx])
                preds = est.predict(X[val_idx])

                r2_scores.append(r2_score(y[val_idx], preds))
                mae_scores.append(mean_absolute_error(y[val_idx], preds))
                rmse_scores.append(
                    float(np.sqrt(mean_squared_error(y[val_idx], preds)))
                )
                y_actual_all.extend(y[val_idx].tolist())
                y_pred_all.extend(preds.tolist())

            return {
                "r2":       float(np.mean(r2_scores)),
                "mae":      float(np.mean(mae_scores)),
                "rmse":     float(np.mean(rmse_scores)),
                "y_actual": np.array(y_actual_all, dtype=float),
                "y_pred":   np.array(y_pred_all,   dtype=float),
            }
