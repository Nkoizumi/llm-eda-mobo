"""
Hyperparameter Optimization via Optuna for XGBoost, Neural Network, and TabPFN.
Each function runs KFold CV internally per trial, then returns best_params for use
in the LOO wrapper.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


def _check_optuna():
    if not _OPTUNA_AVAILABLE:
        raise ImportError("optuna not installed. Run: pip install optuna")


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost HPO  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_xgboost(
    X,
    y,
    task:         str = "regression",
    n_trials:     int = 30,
    n_splits:     int = 5,
    random_state: int = 42,
) -> dict:
    """
    Optuna HPO for XGBoost using KFold CV with per-fold pruning.
    Returns dict with keys: best_params, best_value, study.
    best_params can be passed to XGBoostModel(hpo_params=...).
    """
    _check_optuna()
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, r2_score

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 800),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            random_state     = random_state,
            verbosity        = 0,
            tree_method      = "hist",
        )

        if task == "classification":
            n_classes = len(np.unique(y.astype(int)))
            if n_classes > 2:
                params["objective"]  = "multi:softprob"
                params["num_class"]  = n_classes
            kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y.astype(int)))
        else:
            kf     = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            if task == "classification":
                model = xgb.XGBClassifier(**params)
                model.fit(X[train_idx], y[train_idx].astype(int))
                preds = model.predict(X[val_idx])
                fold_scores.append(float(accuracy_score(y[val_idx].astype(int), preds)))
            else:
                model = xgb.XGBRegressor(**params)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                fold_scores.append(float(r2_score(y[val_idx], preds)))

            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study  = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=random_state),
        pruner    = pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network HPO — improved
# ─────────────────────────────────────────────────────────────────────────────

def _build_hidden_dims(trial) -> tuple:
    """
    Sample a variable-depth, variable-width architecture from four shapes:
      flat        — [B, B, ..., B]
      decreasing  — [B, B/2, B/4, ...] (funnel)
      increasing  — [B/4, B/2, ..., B] (expanding)
      bottleneck  — [B, B/2, B/4, ..., B/2, B] (hourglass, symmetric)
    """
    n_layers  = trial.suggest_int("n_layers", 1, 4)
    base_size = trial.suggest_categorical("base_size", [64, 128, 256, 512])
    arch_type = trial.suggest_categorical(
        "arch_type", ["flat", "decreasing", "increasing", "bottleneck"]
    )

    if arch_type == "flat":
        return tuple([base_size] * n_layers)

    if arch_type == "decreasing":
        return tuple([max(32, base_size >> i) for i in range(n_layers)])

    if arch_type == "increasing":
        return tuple([max(32, base_size >> (n_layers - 1 - i)) for i in range(n_layers)])

    # bottleneck / hourglass
    mid = max(0, (n_layers - 1) // 2)
    return tuple([max(32, base_size >> abs(i - mid)) for i in range(n_layers)])


def _decode_hidden_dims(params: dict) -> tuple:
    """Re-create hidden_dims tuple from Optuna best_params dict."""
    n_layers  = params["n_layers"]
    base_size = params["base_size"]
    arch_type = params["arch_type"]

    if arch_type == "flat":
        return tuple([base_size] * n_layers)
    if arch_type == "decreasing":
        return tuple([max(32, base_size >> i) for i in range(n_layers)])
    if arch_type == "increasing":
        return tuple([max(32, base_size >> (n_layers - 1 - i)) for i in range(n_layers)])
    mid = max(0, (n_layers - 1) // 2)
    return tuple([max(32, base_size >> abs(i - mid)) for i in range(n_layers)])


def _nn_fold_score_v2(
    X_train_sc:  np.ndarray,
    X_val_sc:    np.ndarray,
    y_train:     np.ndarray,
    y_val:       np.ndarray,
    task:        str,
    hidden_dims: tuple,
    lr:          float,
    dropout:     float,
    weight_decay: float,
    batch_size:  int,
    n_classes:   int,
    epochs:      int = 150,
    patience:    int = 15,
) -> float:
    """
    Single-fold NN evaluation for HPO.
    Improvements over _nn_fold_score:
      - 150 epochs / patience=15 (was 60/10)
      - CosineAnnealingLR for smooth decay
      - Explicit weight_decay and batch_size from trial
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import r2_score, accuracy_score
    from .neural_network_model import RegressionNet, ClassificationNet

    device = torch.device("cpu")

    # ── Target preprocessing ─────────────────────────────────────────────
    if task == "regression":
        use_log = (
            float(np.min(y_train)) >= 0
            and float(np.std(y_train)) / (float(np.mean(y_train)) + 1e-8) > 0.5
        )
        y_tr_proc = np.log1p(y_train) if use_log else y_train.copy()
        y_vl_proc = np.log1p(y_val)   if use_log else y_val.copy()
        y_mean = float(y_tr_proc.mean())
        y_std  = float(y_tr_proc.std()) + 1e-8
        y_tr_t = torch.tensor(((y_tr_proc - y_mean) / y_std).astype(np.float32))
        y_vl_t = torch.tensor(((y_vl_proc - y_mean) / y_std).astype(np.float32))
        model     = RegressionNet(X_train_sc.shape[1], hidden_dims, dropout).to(device)
        criterion = nn.HuberLoss(delta=1.0)
    else:
        use_log = False
        y_mean, y_std = 0.0, 1.0
        y_tr_t = torch.tensor(y_train.astype(np.float32))
        y_vl_t = torch.tensor(y_val.astype(np.float32))
        model     = ClassificationNet(X_train_sc.shape[1], n_classes, hidden_dims, dropout).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_tr_t = torch.tensor(X_train_sc.astype(np.float32))
    X_vl_t = torch.tensor(X_val_sc.astype(np.float32))

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    actual_bs  = min(batch_size, max(8, len(X_train_sc)))
    loader     = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=actual_bs, shuffle=True)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb.long()) if task == "classification" else criterion(out.squeeze(), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(X_vl_t)
            val_loss = (
                criterion(val_out, y_vl_t.long()).item()
                if task == "classification"
                else criterion(val_out.squeeze(), y_vl_t).item()
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_out = model(X_vl_t)

    if task == "classification":
        preds = torch.argmax(pred_out, dim=1).numpy()
        return float(accuracy_score(y_val.astype(int), preds))
    else:
        preds_sc   = pred_out.squeeze().numpy()
        preds_orig = preds_sc * y_std + y_mean
        if use_log:
            preds_orig = np.expm1(preds_orig)
        return float(r2_score(y_val, preds_orig))


def optimize_neural_network(
    X,
    y,
    task:         str = "regression",
    n_trials:     int = 30,
    n_splits:     int = 5,
    random_state: int = 42,
) -> dict:
    """
    Robust Optuna HPO for NeuralNetworkModel.

    Search space (vs v1):
      Architecture  — flat / decreasing / increasing / bottleneck, base_size 64-512
      weight_decay  — 1e-6 … 1e-2  (log)          [NEW]
      batch_size    — 16 / 32 / 64 / 128            [NEW]
      dropout       — 0.0 … 0.5                     (widened)
      lr            — 5e-5 … 5e-3  (log)            (tightened)

    Training (vs v1):
      epochs   150  (was 60)
      patience  15  (was 10)
      CosineAnnealingLR with eta_min = lr*0.01      [NEW]

    Pruning:
      MedianPruner — cuts trials below median after ≥ 5 completed, ≥ 2 folds done  [NEW]
    """
    _check_optuna()
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    n_classes = int(len(np.unique(y.astype(int)))) if task == "classification" else 1

    def objective(trial):
        hidden_dims  = _build_hidden_dims(trial)
        dropout      = trial.suggest_float("dropout",      0.0,  0.5)
        lr           = trial.suggest_float("lr",           5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        if task == "classification":
            kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y.astype(int)))
        else:
            kf     = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(X[train_idx])
            X_vl_sc = scaler.transform(X[val_idx])

            score = _nn_fold_score_v2(
                X_tr_sc, X_vl_sc,
                y[train_idx], y[val_idx],
                task, hidden_dims, lr, dropout, weight_decay, batch_size, n_classes,
            )
            fold_scores.append(score)

            # Pruning: report running mean after each fold
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study  = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=random_state),
        pruner    = pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    return {
        "best_params": {
            "hidden_dims":  _decode_hidden_dims(bp),
            "lr":           bp["lr"],
            "dropout":      bp["dropout"],
            "weight_decay": bp["weight_decay"],
            "batch_size":   bp["batch_size"],
        },
        "best_value":  study.best_value,
        "study":       study,
        "arch_detail": {
            "arch_type": bp["arch_type"],
            "base_size": bp["base_size"],
            "n_layers":  bp["n_layers"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# TabPFN wrapper  (used by LOO when HPO is enabled)
# ─────────────────────────────────────────────────────────────────────────────

_TABPFN_PREPROCESSORS = ["none", "standard", "robust", "quantile_uniform", "quantile_normal"]


def _make_tabpfn_preprocessor(name: str):
    """Return a fitted-ready sklearn transformer, or None."""
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    return {
        "none":             None,
        "standard":         StandardScaler(),
        "robust":           RobustScaler(),
        "quantile_uniform": QuantileTransformer(output_distribution="uniform", random_state=42),
        "quantile_normal":  QuantileTransformer(output_distribution="normal",  random_state=42),
    }[name]


class TabPFNWrapper:
    """
    sklearn-compatible wrapper around TabPFNClassifier / TabPFNRegressor with
    an optional pre-processing step.  _clone_wrapper() in app.py picks up all
    constructor params automatically via inspect.signature.

    Parameters
    ----------
    task             : "classification" or "regression"
    n_estimators     : TabPFN ensemble size
    preprocessor_type: one of _TABPFN_PREPROCESSORS
    random_state     : reproducibility seed
    extra_kwargs     : additional TabPFN-specific kwargs found by optimize_tabpfn
                       (e.g. softmax_temperature, balance_probabilities).
                       Stored as a dict and forwarded to the TabPFN constructor.
    """

    def __init__(
        self,
        task:              str  = "classification",
        n_estimators:      int  = 8,
        preprocessor_type: str  = "none",
        random_state:      int  = 42,
        extra_kwargs:      dict = None,
    ):
        self.task              = task
        self.n_estimators      = n_estimators
        self.preprocessor_type = preprocessor_type
        self.random_state      = random_state
        self.extra_kwargs      = extra_kwargs or {}
        self._model            = None
        self._preprocessor     = None

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        self._preprocessor = _make_tabpfn_preprocessor(self.preprocessor_type)
        X_fit = self._preprocessor.fit_transform(X) if self._preprocessor else X
        if self.task == "classification":
            self._model = TabPFNClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.extra_kwargs,
            )
            self._model.fit(X_fit, y.astype(int))
        else:
            self._model = TabPFNRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.extra_kwargs,
            )
            self._model.fit(X_fit, y)
        return self

    def predict(self, X):
        X_pred = self._preprocessor.transform(X) if self._preprocessor else X
        return self._model.predict(X_pred)


# ─────────────────────────────────────────────────────────────────────────────
# TabPFN HPO — improved
# ─────────────────────────────────────────────────────────────────────────────

def _tabpfn_safe_kwargs(trial, task: str) -> dict:
    """
    Probe which TabPFN constructor kwargs are accepted in the installed version
    and add them to a dict.  Unknown kwargs are silently skipped so we stay
    compatible with both TabPFN v1 and v2.
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    import inspect

    Cls   = TabPFNClassifier if task == "classification" else TabPFNRegressor
    sig   = inspect.signature(Cls.__init__)
    known = set(sig.parameters.keys())
    extra = {}

    if task == "classification":
        if "softmax_temperature" in known:
            extra["softmax_temperature"] = trial.suggest_float(
                "softmax_temperature", 0.5, 2.0
            )
        if "balance_probabilities" in known:
            extra["balance_probabilities"] = trial.suggest_categorical(
                "balance_probabilities", [True, False]
            )
        if "average_before_softmax" in known:
            extra["average_before_softmax"] = trial.suggest_categorical(
                "average_before_softmax", [True, False]
            )

    return extra


def optimize_tabpfn(
    X,
    y,
    task:         str = "regression",
    n_trials:     int = 20,
    n_splits:     int = 5,
    random_state: int = 42,
) -> dict:
    """
    Robust Optuna HPO for TabPFN.

    Search space (vs v1):
      n_estimators   — 4 … 64 (step 4, up from 4–32)          [EXTENDED]
      preprocessor   — none / standard / robust /
                       quantile_uniform / quantile_normal       [NEW]
      classification — softmax_temperature (if available)       [NEW]
      classification — balance_probabilities (if available)     [NEW]
      classification — average_before_softmax (if available)    [NEW]

    Returns best_params with keys:
      n_estimators, preprocessor_type, and any TabPFN-specific kwargs found.
    """
    _check_optuna()
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError:
        raise ImportError("tabpfn not installed. Run: pip install tabpfn")

    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, r2_score

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    def objective(trial):
        n_ens          = trial.suggest_int("n_estimators", 4, 64, step=4)
        preprocessor   = trial.suggest_categorical("preprocessor", _TABPFN_PREPROCESSORS)
        tabpfn_kwargs  = _tabpfn_safe_kwargs(trial, task)

        if task == "classification":
            kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y.astype(int)))
        else:
            kf     = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            preproc = _make_tabpfn_preprocessor(preprocessor)
            X_tr = preproc.fit_transform(X[train_idx]) if preproc else X[train_idx]
            X_vl = preproc.transform(X[val_idx])       if preproc else X[val_idx]

            if task == "classification":
                m = TabPFNClassifier(
                    n_estimators=n_ens,
                    random_state=random_state,
                    **tabpfn_kwargs,
                )
                m.fit(X_tr, y[train_idx].astype(int))
                preds = m.predict(X_vl)
                fold_scores.append(float(accuracy_score(y[val_idx].astype(int), preds)))
            else:
                m = TabPFNRegressor(
                    n_estimators=n_ens,
                    random_state=random_state,
                    **tabpfn_kwargs,
                )
                m.fit(X_tr, y[train_idx])
                preds = m.predict(X_vl)
                fold_scores.append(float(r2_score(y[val_idx], preds)))

            # Prune after first fold if clearly bad
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study  = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=random_state),
        pruner    = pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    # Separate TabPFN-specific kwargs so the caller can pass them directly
    core_keys   = {"n_estimators", "preprocessor"}
    tabpfn_extra = {k: v for k, v in bp.items() if k not in core_keys}

    return {
        "best_params": {
            "n_estimators":     bp["n_estimators"],
            "preprocessor_type": bp["preprocessor"],
            **tabpfn_extra,
        },
        "best_value":  study.best_value,
        "study":       study,
    }
