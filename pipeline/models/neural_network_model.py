import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
)
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[NN] CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[NN] Apple MPS detected. Using MPS backend.")
    else:
        device = torch.device("cpu")
        print("[NN] No GPU detected. Using CPU.")
    return device


# ─────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm_out = nn.LayerNorm(dim)
        self.act_out  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_out(self.norm_out(x + self.block(x)))


# ─────────────────────────────────────────────────────────────
# Network Architectures
# ─────────────────────────────────────────────────────────────

class RegressionNet(nn.Module):
    def __init__(
        self,
        input_dim:   int,
        hidden_dims: tuple = (256, 256, 128),
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(hidden_dims)):
            if i > 0 and hidden_dims[i] != hidden_dims[i - 1]:
                blocks.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            blocks.append(ResidualBlock(hidden_dims[i], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


class ClassificationNet(nn.Module):
    def __init__(
        self,
        input_dim:   int,
        n_classes:   int,
        hidden_dims: tuple = (256, 256, 128),
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(hidden_dims)):
            if i > 0 and hidden_dims[i] != hidden_dims[i - 1]:
                blocks.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            blocks.append(ResidualBlock(hidden_dims[i], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# Training Helpers
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    return (
        total_loss / len(loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
    )


# ─────────────────────────────────────────────────────────────
# Bug 2 Fixed: Standalone run_nn_regression() restored
# ─────────────────────────────────────────────────────────────

def run_nn_regression(
    X, y,
    n_splits:     int   = 5,
    random_state: int   = 42,
    epochs:       int   = 300,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    patience:     int   = 30,
    hidden_dims:  tuple = (256, 256, 128),
    dropout:      float = 0.1,
):
    device = get_torch_device()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores, mae_scores, rmse_scores = [], [], []

    print(f"\n[NN Regression] Running {n_splits}-Fold CV | Device: {device}\n")

    # Detect skewed target and apply log1p
    use_log = (np.min(y) >= 0 and np.std(y) / (np.mean(y) + 1e-8) > 0.5)
    if use_log:
        print("  [Target] Right-skewed target detected — applying log1p transform.\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Apply log1p if skewed, then standardise target
        y_tr_raw = np.log1p(y_train) if use_log else y_train.copy()
        y_vl_raw = np.log1p(y_val)   if use_log else y_val.copy()

        y_mean, y_std  = y_tr_raw.mean(), y_tr_raw.std() + 1e-8
        y_tr_scaled    = (y_tr_raw - y_mean) / y_std
        y_vl_scaled    = (y_vl_raw - y_mean) / y_std

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_tr_scaled)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_vl_scaled)
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        model     = RegressionNet(X_train.shape[1], hidden_dims, dropout).to(device)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        steps_per_epoch = max(1, len(train_loader))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch * 50, T_mult=2
        )

        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.load_state_dict(best_weights)
        _, val_preds, val_targets = eval_epoch(model, val_loader, criterion, device)

        # Inverse-transform: undo StandardScaler then undo log1p
        preds_unscaled = val_preds.numpy() * y_std + y_mean
        targs_unscaled = val_targets.numpy() * y_std + y_mean
        if use_log:
            preds_orig   = np.expm1(preds_unscaled)
            targets_orig = np.expm1(targs_unscaled)
        else:
            preds_orig   = preds_unscaled
            targets_orig = targs_unscaled

        r2   = r2_score(targets_orig, preds_orig)
        mae  = mean_absolute_error(targets_orig, preds_orig)
        rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"  Fold {fold}: R2={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f} | Stopped @ Epoch {epoch+1}")

    print("\n[NN Regression] Cross-Validation Summary:")
    print(f"  Mean R2   : {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"  Mean MAE  : {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    print(f"  Mean RMSE : {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")

    return {
        "model":  model,
        "scaler": scaler,
        "r2":     np.mean(r2_scores),
        "mae":    np.mean(mae_scores),
        "rmse":   np.mean(rmse_scores),
    }


# ─────────────────────────────────────────────────────────────
# Standalone Classification (unchanged — was correct)
# ─────────────────────────────────────────────────────────────

def run_nn_classification(
    X, y,
    n_splits:     int   = 5,
    random_state: int   = 42,
    epochs:       int   = 300,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    patience:     int   = 30,
    hidden_dims:  tuple = (256, 256, 128),
    dropout:      float = 0.1,
):
    device = get_torch_device()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_classes = len(np.unique(y))
    is_binary = (n_classes == 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores, f1_scores_list, auc_scores = [], [], []

    print(f"\n[NN Classification] Running {n_splits}-Fold Stratified CV | Device: {device}")
    print(f"  Task: {'Binary' if is_binary else 'Multiclass'} ({n_classes} classes)\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        model     = ClassificationNet(X_train.shape[1], n_classes, hidden_dims, dropout).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        steps_per_epoch = max(1, len(train_loader))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch * 50, T_mult=2
        )

        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.load_state_dict(best_weights)
        _, val_preds_raw, val_targets = eval_epoch(model, val_loader, criterion, device)

        probs     = torch.softmax(val_preds_raw, dim=1).numpy()
        preds_cls = np.argmax(probs, axis=1)
        y_true    = val_targets.numpy()

        acc = accuracy_score(y_true, preds_cls)
        f1  = f1_score(y_true, preds_cls, average="binary" if is_binary else "weighted")

        if is_binary:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

        acc_scores.append(acc)
        f1_scores_list.append(f1)
        auc_scores.append(auc)

        print(f"  Fold {fold}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | Stopped @ Epoch {epoch+1}")

    print("\n[NN Classification] Cross-Validation Summary:")
    print(f"  Mean Accuracy : {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  Mean F1       : {np.mean(f1_scores_list):.4f} (+/- {np.std(f1_scores_list):.4f})")
    print(f"  Mean AUC      : {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    return {
        "model":    model,
        "scaler":   scaler,
        "accuracy": np.mean(acc_scores),
        "f1":       np.mean(f1_scores_list),
        "auc":      np.mean(auc_scores),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NeuralNetworkModel wrapper
# Bugs 1, 3, 4 fixed: run() is inside the class, target scaling added
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetworkModel:
    """
    PyTorch residual MLP wrapper with:
      - .run(X, y)   -> K-Fold CV metrics dict  (used by Tab 9)
      - .fit(X, y)   -> trains on given data     (used by LOO)
      - .predict(X)  -> returns predictions      (used by LOO)
    """

    def __init__(
        self,
        task:         str   = "classification",
        n_splits:     int   = 5,
        random_state: int   = 42,
        use_cuda:     bool  = False,
        epochs:       int   = 300,
        patience:     int   = 30,
        hidden_dims:  tuple = (256, 256, 128),
        lr:           float = 1e-3,
        dropout:      float = 0.1,
        weight_decay: float = 1e-5,
        batch_size:   int   = 0,
    ):
        self.task         = task
        self.n_splits     = n_splits
        self.random_state = random_state
        self.use_cuda     = use_cuda
        self.epochs       = epochs
        self.patience     = patience
        self.hidden_dims  = hidden_dims
        self.lr           = lr
        self.dropout      = dropout
        self.weight_decay = weight_decay
        # 0 = auto-compute as min(64, max(8, n_samples // 8))
        self.batch_size   = batch_size

        self._model        = None
        self._scaler       = None
        self._device       = None
        # Bug 4 fix: store target transform state for predict()
        self._y_mean       = None
        self._y_std        = None
        self._use_log      = False

    def _get_device(self):
        if self.use_cuda:
            if torch.cuda.is_available():
                return torch.device("cuda")
            try:
                if torch.backends.mps.is_available():
                    return torch.device("mps")
            except AttributeError:
                pass
        return torch.device("cpu")

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        if self.task == "classification":
            return ClassificationNet(input_dim, output_dim, self.hidden_dims, self.dropout)
        else:
            return RegressionNet(input_dim, self.hidden_dims, self.dropout)

    def _train(
        self,
        model:  nn.Module,
        X_tr:   torch.Tensor,
        y_tr:   torch.Tensor,
        X_vl:   torch.Tensor,
        y_vl:   torch.Tensor,
        device,
    ) -> nn.Module:
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = (
            nn.CrossEntropyLoss(label_smoothing=0.05)
            if self.task == "classification"
            else nn.HuberLoss(delta=1.0)
        )

        auto_bs    = min(64, max(8, len(X_tr) // 8))
        batch_size = min(self.batch_size, len(X_tr)) if self.batch_size > 0 else auto_bs
        dataset    = TensorDataset(X_tr, y_tr)
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # CosineAnnealingLR stepped once per epoch — matches the HPO training protocol
        # in _nn_fold_score_v2 so that lr/dropout/arch found by HPO transfer correctly.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.epochs), eta_min=self.lr * 0.01
        )

        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0

        for epoch in range(self.epochs):
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = model(X_batch)
                loss = (
                    criterion(out, y_batch.long())
                    if self.task == "classification"
                    else criterion(out.squeeze(), y_batch)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()  # once per epoch, not per batch

            model.eval()
            with torch.no_grad():
                val_out = model(X_vl)
                val_loss = (
                    criterion(val_out, y_vl.long()).item()
                    if self.task == "classification"
                    else criterion(val_out.squeeze(), y_vl).item()
                )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # ── Bug 4 Fixed: fit() now scales regression target ──────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        device       = self._get_device()
        self._device = device

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self._scaler = StandardScaler()
        X            = self._scaler.fit_transform(X)

        # Regression: detect skew and scale target
        if self.task == "regression":
            self._use_log = (np.min(y) >= 0 and np.std(y) / (np.mean(y) + 1e-8) > 0.5)
            y_proc = np.log1p(y) if self._use_log else y.copy()
            self._y_mean = float(y_proc.mean())
            self._y_std  = float(y_proc.std()) + 1e-8
            y_scaled = (y_proc - self._y_mean) / self._y_std
        else:
            y_scaled = y

        n_val = max(1, int(len(X) * 0.1))
        idx   = np.random.permutation(len(X))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        if len(train_idx) < 2:
            train_idx = idx
            val_idx   = idx

        X_tr = torch.tensor(X[train_idx],        dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_scaled[train_idx], dtype=torch.float32).to(device)
        X_vl = torch.tensor(X[val_idx],          dtype=torch.float32).to(device)
        y_vl = torch.tensor(y_scaled[val_idx],   dtype=torch.float32).to(device)

        output_dim = (
            len(np.unique(y.astype(int)))
            if self.task == "classification"
            else 1
        )

        self._model = self._build_model(X.shape[1], output_dim).to(device)
        self._model = self._train(self._model, X_tr, y_tr, X_vl, y_vl, device)
        return self

    # ── Bug 4 Fixed: predict() now inverse-transforms regression output ──
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() called before fit().")

        X = self._scaler.transform(np.array(X, dtype=np.float32))
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            out = self._model(X_t)

        if self.task == "classification":
            return torch.argmax(out, dim=1).cpu().numpy().astype(int).ravel()
        else:
            # Inverse-transform: undo StandardScaler, then undo log1p
            preds = out.squeeze().cpu().numpy().astype(float)
            preds = preds * self._y_std + self._y_mean
            if self._use_log:
                preds = np.expm1(preds)
            return preds.ravel()

    # ── Bug 1+3 Fixed: run() is INSIDE the class with full target scaling ──
    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        device = self._get_device()

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Detect skewed regression target once before CV loop
        if self.task == "regression":
            use_log = (np.min(y) >= 0 and np.std(y) / (np.mean(y) + 1e-8) > 0.5)
            if use_log:
                print("  [Target] Right-skewed target detected — applying log1p.\n")
        else:
            use_log = False

        if self.task == "classification":
            n_classes  = len(np.unique(y.astype(int)))
            output_dim = n_classes
            kf         = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            splits = kf.split(X, y.astype(int))
        else:
            output_dim = 1
            kf         = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            splits = kf.split(X)

        acc_list, f1_list            = [], []
        r2_list, mae_list, rmse_list = [], [], []
        y_actual_all, y_pred_all     = [], []

        for train_idx, val_idx in splits:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val   = scaler.transform(X[val_idx])

            if self.task == "regression":
                # Per-fold target transform + standardisation
                y_tr_raw = np.log1p(y[train_idx]) if use_log else y[train_idx].copy()
                y_vl_raw = np.log1p(y[val_idx])   if use_log else y[val_idx].copy()
                y_mean_f = y_tr_raw.mean()
                y_std_f  = y_tr_raw.std() + 1e-8
                y_tr_sc  = (y_tr_raw - y_mean_f) / y_std_f
                y_vl_sc  = (y_vl_raw - y_mean_f) / y_std_f

                X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_tr = torch.tensor(y_tr_sc, dtype=torch.float32).to(device)
                X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
                y_vl = torch.tensor(y_vl_sc, dtype=torch.float32).to(device)
            else:
                X_tr = torch.tensor(X_train,      dtype=torch.float32).to(device)
                y_tr = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
                X_vl = torch.tensor(X_val,        dtype=torch.float32).to(device)
                y_vl = torch.tensor(y[val_idx],   dtype=torch.float32).to(device)

            model = self._build_model(X_train.shape[1], output_dim).to(device)
            model = self._train(model, X_tr, y_tr, X_vl, y_vl, device)

            model.eval()
            with torch.no_grad():
                out = model(X_vl)

            if self.task == "classification":
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc_list.append(accuracy_score(y[val_idx].astype(int), preds))
                f1_list.append(
                    f1_score(
                        y[val_idx].astype(int), preds,
                        average="weighted", zero_division=0,
                    )
                )
                y_actual_all.extend(y[val_idx].tolist())
                y_pred_all.extend(preds.tolist())

            else:
                # Inverse-transform back to original scale
                preds_sc     = out.squeeze().cpu().numpy()
                preds_log    = preds_sc * y_std_f + y_mean_f
                preds_orig   = np.expm1(preds_log) if use_log else preds_log
                targets_orig = y[val_idx]

                r2_list.append(r2_score(targets_orig, preds_orig))
                mae_list.append(mean_absolute_error(targets_orig, preds_orig))
                rmse_list.append(np.sqrt(mean_squared_error(targets_orig, preds_orig)))

                y_actual_all.extend(targets_orig.tolist())
                y_pred_all.extend(preds_orig.tolist())

        if self.task == "classification":
            return {
                "accuracy": float(np.mean(acc_list)),
                "f1":       float(np.mean(f1_list)),
                "y_actual": np.array(y_actual_all, dtype=float),
                "y_pred":   np.array(y_pred_all,   dtype=float),
            }
        else:
            return {
                "r2":       float(np.mean(r2_list)),
                "mae":      float(np.mean(mae_list)),
                "rmse":     float(np.mean(rmse_list)),
                "y_actual": np.array(y_actual_all, dtype=float),
                "y_pred":   np.array(y_pred_all,   dtype=float),
            }
