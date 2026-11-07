"""Save a trained model with its preprocessing, and apply both to new test data.

A ``ModelBundle`` holds everything needed to predict on data the model has
never seen:

* the preprocessing pipeline, **fitted on the training rows** (imputer medians,
  power-transform lambdas, scaler mean/std, category levels, dropped
  correlated columns). New data is only ``transform``-ed with it, never refit —
  refitting on test data would leak the test set into its own preprocessing;
* the final estimator, fitted on all training rows;
* label encoders for classification targets, so predictions come back in the
  original class labels;
* a record of the training data (columns, dtypes, numeric ranges, category
  levels) used to check a test file before predicting.

On disk a bundle is a zip with four files:

    model.joblib        the fitted objects (what actually runs)
    preprocessing.json  readable record of every fitted preprocessing value
    best_params.json    estimator settings, HPO best parameters, CV scores
    manifest.json       versions + SHA-256 of model.joblib

The JSON files are the human-readable record; ``model.joblib`` is what is
executed. They are written together and the manifest hash ties them to each
other, so a JSON edited by hand cannot silently disagree with the model.

``model.joblib`` is a pickle: loading one runs code from it. Only load bundles
you made yourself.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone as sk_clone
from sklearn.preprocessing import LabelEncoder

BUNDLE_FORMAT_VERSION = 1

# Fitted attributes worth recording, per sklearn step. Anything not listed is
# still in model.joblib; this only controls what the readable JSON shows.
_FITTED_ATTRS = (
    "statistics_",            # SimpleImputer
    "lambdas_",               # PowerTransformer
    "mean_", "scale_", "var_",  # StandardScaler
    "center_",                # RobustScaler
    "data_min_", "data_max_",  # MinMaxScaler
    "categories_",            # One-hot / ordinal encoders
)


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_jsonable(obj):
    """Convert numpy / pandas / torch objects into plain JSON values.

    NaN and inf become None (JSON has no NaN). Unknown objects become their
    ``repr`` so writing a record never fails on an exotic parameter value.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.Index)):
        return to_jsonable(obj.tolist())
    return repr(obj)


def estimator_params(est) -> dict:
    """Constructor parameters of any estimator, as JSON-safe values.

    sklearn estimators expose ``get_params``; the project's own wrappers
    (XGBoostModel, NeuralNetworkModel, TabPFNWrapper) store their constructor
    arguments as attributes of the same name, so read those instead.
    """
    if hasattr(est, "get_params"):
        try:
            params = est.get_params(deep=False)
            # MultiOutput wrappers: show the base estimator's params too.
            inner = params.get("estimator")
            if inner is not None and hasattr(inner, "get_params"):
                params = {k: v for k, v in params.items() if k != "estimator"}
                params["estimator"] = type(inner).__name__
                params["estimator_params"] = inner.get_params(deep=False)
            return to_jsonable(params)
        except Exception:
            pass
    params = {}
    for name in inspect.signature(type(est).__init__).parameters:
        if name != "self" and hasattr(est, name):
            params[name] = getattr(est, name)
    return to_jsonable(params)


def fresh_estimator(est):
    """An unfitted copy of ``est`` with the same constructor arguments."""
    if isinstance(est, BaseEstimator):
        return sk_clone(est)
    cls = type(est)
    kwargs = {}
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self":
            continue
        if hasattr(est, name):
            kwargs[name] = copy.deepcopy(getattr(est, name))
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
    return cls(**kwargs)


def _make_portable(est):
    """Move fitted models to CPU so a bundle loads on any machine.

    Also keeps test-time prediction off the GPU, which may be busy (or full)
    with other work.
    """
    try:
        import torch
    except ImportError:
        torch = None

    targets = [est]
    targets += list(getattr(est, "estimators_", []) or [])   # MultiOutput*
    for e in targets:
        model = getattr(e, "_model", None)
        if torch is not None and isinstance(model, torch.nn.Module):
            e._model = model.to("cpu")
            e._device = torch.device("cpu")
        for attr in ("_booster", "_model"):
            inner = getattr(e, attr, None)
            if inner is not None and type(inner).__module__.startswith("xgboost"):
                try:
                    inner.set_params(device="cpu")
                except Exception:
                    pass
        if type(e).__module__.startswith("xgboost"):
            try:
                e.set_params(device="cpu")
            except Exception:
                pass
    return est


# ─────────────────────────────────────────────────────────────────────────────
# Training-data record
# ─────────────────────────────────────────────────────────────────────────────
def summarize_training_frame(X: pd.DataFrame) -> dict:
    """Columns, dtypes, numeric ranges and category levels of the training X."""
    numeric, categorical = {}, {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric[col] = {
                "min":  to_jsonable(s.min()),
                "max":  to_jsonable(s.max()),
                "mean": to_jsonable(s.mean()),
                "std":  to_jsonable(s.std()),
                "n_missing": int(s.isna().sum()),
            }
        else:
            levels = sorted(map(str, s.dropna().unique()))
            categorical[col] = {
                "levels": levels[:500],
                "n_levels": len(levels),
                "n_missing": int(s.isna().sum()),
            }
    return {
        "n_rows": int(len(X)),
        "columns": list(map(str, X.columns)),
        "dtypes": {str(c): str(t) for c, t in X.dtypes.items()},
        "numeric": numeric,
        "categorical": categorical,
    }


def describe_preprocessor(prep) -> dict:
    """Readable account of a fitted preprocessing pipeline.

    Walks ``Pipeline -> ColumnTransformer -> per-group Pipeline`` (the shape
    ``AutoEDAPipeline._build_sklearn_pipeline`` builds) and records the fitted
    values of every step, keyed by the input column they apply to.
    """
    if prep is None:
        return {"type": "none"}

    def step_record(step, cols):
        rec = {"class": type(step).__name__,
               "params": estimator_params(step)}
        for attr in _FITTED_ATTRS:
            if not hasattr(step, attr):
                continue
            val = getattr(step, attr)
            if attr == "categories_":
                rec[attr] = {c: to_jsonable(v) for c, v in zip(cols, val)}
            elif np.ndim(val) == 1 and len(val) == len(cols):
                rec[attr] = {c: to_jsonable(v) for c, v in zip(cols, val)}
            else:
                rec[attr] = to_jsonable(val)
        return rec

    out = {"type": type(prep).__name__, "steps": []}
    steps = getattr(prep, "steps", [("preprocessor", prep)])
    for name, step in steps:
        if hasattr(step, "transformers_"):          # ColumnTransformer
            groups = []
            for g_name, g_pipe, g_cols in step.transformers_:
                if g_name == "remainder" or g_pipe == "drop":
                    continue
                g_cols = list(map(str, g_cols))
                inner = getattr(g_pipe, "steps", [(g_name, g_pipe)])
                groups.append({
                    "name": g_name,
                    "columns": g_cols,
                    "steps": [{"name": s_name, **step_record(s, g_cols)}
                              for s_name, s in inner],
                })
            out["steps"].append({"name": name, "class": type(step).__name__,
                                 "remainder": str(step.remainder),
                                 "groups": groups})
        elif hasattr(step, "features_to_drop_positions_"):   # HighCorrelationRemover
            names_in = list(map(str, step.feature_names_in_))
            out["steps"].append({
                "name": name,
                "class": type(step).__name__,
                "threshold": to_jsonable(step.threshold),
                "dropped_features": [names_in[i] for i in step.features_to_drop_positions_],
                "n_in": len(names_in),
            })
        else:
            out["steps"].append({"name": name, **step_record(step, [])})
    try:
        out["output_features"] = list(map(str, prep.get_feature_names_out()))
    except Exception:
        pass
    return out


def preprocessor_input_columns(prep) -> list[str]:
    """Raw columns the fitted preprocessor actually reads (not the ones it drops)."""
    cols: list[str] = []
    for _, step in getattr(prep, "steps", []):
        for g_name, g_pipe, g_cols in getattr(step, "transformers_", []):
            if g_name == "remainder" or g_pipe == "drop":
                continue
            cols.extend(map(str, g_cols))
    return list(dict.fromkeys(cols))


# ─────────────────────────────────────────────────────────────────────────────
# Bundle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    estimator: object
    task: str
    target_cols: list[str]
    feature_names: list[str]
    train_summary: dict
    preprocessor: object = None          # fitted sklearn Pipeline, or None
    # Fallback preprocessing (Multitask tab without an LLM pipeline):
    # numeric columns only, median fill, StandardScaler.
    fallback_columns: list[str] | None = None
    fallback_medians: dict | None = None
    fallback_scaler: object = None
    label_encoders: dict = field(default_factory=dict)
    source: str = ""                     # "LOO" | "Multitask"
    estimator_name: str = ""
    hpo: dict | None = None              # {"best_params", "best_value", "n_trials", "cv_folds"}
    cv_metrics: dict = field(default_factory=dict)
    llm_decision: dict | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ── what the bundle needs from a test file ───────────────────────────
    @property
    def required_columns(self) -> list[str]:
        if self.preprocessor is not None:
            return preprocessor_input_columns(self.preprocessor)
        return list(self.fallback_columns or [])

    @property
    def fit_columns(self) -> list[str]:
        """Every column the preprocessor saw at fit time, in order."""
        if self.preprocessor is not None:
            names = getattr(self.preprocessor, "feature_names_in_", None)
            if names is None:
                names = getattr(self.preprocessor.steps[0][1], "feature_names_in_", None)
            if names is not None:
                return list(map(str, names))
        return list(self.fallback_columns or [])

    # ── checks ───────────────────────────────────────────────────────────
    def check_test_frame(self, df: pd.DataFrame) -> dict:
        """Compare a test frame against the training record before predicting.

        Returns ``missing`` (blocking), ``extra`` (ignored), ``not_numeric``
        (numeric in training, text in the test file), ``unseen_levels``
        (categories the encoders never saw — encoded as unknown) and
        ``out_of_range`` (count of rows outside the training min/max, per
        column: predictions there are extrapolation).
        """
        required = self.required_columns
        missing = [c for c in required if c not in df.columns]
        known = set(self.fit_columns) | set(self.target_cols)
        extra = [c for c in df.columns if c not in known]

        not_numeric, out_of_range, unseen = [], {}, {}
        num_rec = self.train_summary.get("numeric", {})
        cat_rec = self.train_summary.get("categorical", {})
        row_outside = np.zeros(len(df), dtype=bool)
        for col in required:
            if col not in df.columns:
                continue
            if col in num_rec:
                vals = pd.to_numeric(df[col], errors="coerce")
                bad = vals.isna() & df[col].notna()
                if bad.any():
                    not_numeric.append(col)
                lo, hi = num_rec[col]["min"], num_rec[col]["max"]
                if lo is not None and hi is not None:
                    outside = ((vals < lo) | (vals > hi)).to_numpy()
                    if outside.any():
                        out_of_range[col] = int(outside.sum())
                        row_outside |= outside
            elif col in cat_rec:
                levels = set(cat_rec[col]["levels"])
                new = sorted(set(map(str, df[col].dropna().unique())) - levels)
                if new:
                    unseen[col] = new[:20]
        return {
            "missing": missing,
            "extra": extra,
            "not_numeric": not_numeric,
            "unseen_levels": unseen,
            "out_of_range": out_of_range,
            "rows_outside_training_range": row_outside,
        }

    # ── prediction ───────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the TRAINING-fitted preprocessing to new rows (no refit)."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Test data is missing required column(s): {missing}")

        num_rec = self.train_summary.get("numeric", {})
        if self.preprocessor is not None:
            # Same columns, same order as at fit time. Columns the pipeline
            # never reads (correlation- or remainder-dropped) are filled with
            # NaN so the test file does not need them.
            X = df.reindex(columns=self.fit_columns).copy()
            for col in X.columns:
                if col in num_rec:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
            Xt = self.preprocessor.transform(X)
        else:
            X = df[self.fallback_columns].apply(pd.to_numeric, errors="coerce")
            X = X.fillna(pd.Series(self.fallback_medians))
            Xt = self.fallback_scaler.transform(X.values)
        Xt = np.asarray(Xt, dtype=np.float32)
        if not np.isfinite(Xt).all():
            raise ValueError(
                "Preprocessed test features contain NaN/inf. A required column "
                "is probably empty or non-numeric in the test file.")
        return Xt

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predictions in original units / class labels, one column per target."""
        Xt = self.transform(df)
        raw = np.asarray(self.estimator.predict(Xt))
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        out = {}
        for i, t in enumerate(self.target_cols):
            col = raw[:, i]
            le = self.label_encoders.get(t)
            if self.task == "classification" and le is not None:
                idx = np.clip(np.round(col.astype(float)).astype(int), 0, len(le.classes_) - 1)
                col = le.inverse_transform(idx)
            out[t] = col
        return pd.DataFrame(out, index=df.index)

    # ── records ──────────────────────────────────────────────────────────
    def preprocessing_record(self) -> dict:
        rec = {
            "task": self.task,
            "target_cols": self.target_cols,
            "required_input_columns": self.required_columns,
            "transformed_feature_names": self.feature_names,
            "training_data": self.train_summary,
            "llm_decision": self.llm_decision,
        }
        if self.preprocessor is not None:
            rec["pipeline"] = describe_preprocessor(self.preprocessor)
        else:
            rec["pipeline"] = {
                "type": "fallback: numeric columns, median fill, StandardScaler",
                "columns": self.fallback_columns,
                "medians": to_jsonable(self.fallback_medians),
                "scaler_mean": dict(zip(self.fallback_columns,
                                        to_jsonable(self.fallback_scaler.mean_))),
                "scaler_scale": dict(zip(self.fallback_columns,
                                         to_jsonable(self.fallback_scaler.scale_))),
            }
        rec["label_encoders"] = {t: to_jsonable(le.classes_)
                                 for t, le in self.label_encoders.items()}
        return to_jsonable(rec)

    def best_params_record(self) -> dict:
        hpo = self.hpo or {}
        return to_jsonable({
            "source": self.source,
            "estimator": self.estimator_name,
            "estimator_class": type(self.estimator).__name__,
            "task": self.task,
            "target_cols": self.target_cols,
            "hpo_used": bool(hpo),
            "hpo_best_params": hpo.get("best_params"),
            "hpo_best_cv_score": hpo.get("best_value"),
            "hpo_n_trials": hpo.get("n_trials"),
            "hpo_cv_folds": hpo.get("cv_folds"),
            "final_estimator_params": estimator_params(self.estimator),
            "cv_metrics": self.cv_metrics,
            "n_training_rows": self.train_summary.get("n_rows"),
            "created_at": self.created_at,
        })

    @property
    def default_name(self) -> str:
        stamp = self.created_at.replace(":", "").replace("-", "")[:15]
        est = "".join(ch if ch.isalnum() else "_" for ch in self.estimator_name).strip("_")
        return f"{stamp}_{self.source}_{est}".lower()

    # ── save / load ──────────────────────────────────────────────────────
    def _files(self) -> dict[str, bytes]:
        buf = io.BytesIO()
        joblib.dump(self, buf, compress=3)
        model_bytes = buf.getvalue()
        versions = {}
        for mod in ("sklearn", "numpy", "pandas", "xgboost", "torch", "tabpfn"):
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:
                pass
        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "created_at": self.created_at,
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "library_versions": versions,
            "note": "model.joblib is a pickle: load only bundles you created.",
        }
        dump = lambda d: json.dumps(d, indent=2, ensure_ascii=False).encode("utf-8")
        return {
            "model.joblib": model_bytes,
            "preprocessing.json": dump(self.preprocessing_record()),
            "best_params.json": dump(self.best_params_record()),
            "manifest.json": dump(manifest),
        }

    def to_zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in self._files().items():
                zf.writestr(name, data)
        return buf.getvalue()

    def save_dir(self, root: str | Path) -> Path:
        """Write the four files into ``root/<default_name>/`` and return that path."""
        out = Path(root) / self.default_name
        out.mkdir(parents=True, exist_ok=True)
        for name, data in self._files().items():
            (out / name).write_bytes(data)
        return out

    @classmethod
    def from_zip_bytes(cls, data: bytes) -> "ModelBundle":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "model.joblib" not in names:
                raise ValueError("Not a model bundle: model.joblib is missing.")
            model_bytes = zf.read("model.joblib")
            if "manifest.json" in names:
                manifest = json.loads(zf.read("manifest.json"))
                if manifest.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest():
                    raise ValueError(
                        "model.joblib does not match the hash in manifest.json — "
                        "the bundle was modified or corrupted.")
        bundle = joblib.load(io.BytesIO(model_bytes))
        if not isinstance(bundle, cls):
            raise ValueError(f"model.joblib holds a {type(bundle).__name__}, not a ModelBundle.")
        return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Building bundles
# ─────────────────────────────────────────────────────────────────────────────
def encode_target(y: pd.Series, task: str):
    """Return (encoded y as numpy, LabelEncoder or None)."""
    if task != "classification":
        return pd.to_numeric(y, errors="raise").to_numpy(dtype=np.float32), None
    le = LabelEncoder()
    return le.fit_transform(y.to_numpy()).astype(np.int64), le


def fit_single_target_bundle(
    *,
    df: pd.DataFrame,
    preprocessor,
    estimator,
    target: str,
    target_cols: list[str],
    task: str,
    estimator_name: str,
    cv_metrics: dict | None = None,
    hpo: dict | None = None,
    llm_decision: dict | None = None,
) -> ModelBundle:
    """Refit the preprocessing and a fresh copy of ``estimator`` on all
    training rows with a known target, and bundle them.

    ``preprocessor`` is the LLM-built (unfitted or fitted) sklearn pipeline;
    it is cloned, so the session's copy is not touched. Rows with a missing
    target are left out of both fits.
    """
    keep = df[target].notna()
    X = df.loc[keep].drop(columns=target_cols, errors="ignore")
    y, le = encode_target(df.loc[keep, target], task)

    prep = sk_clone(preprocessor)
    Xt = np.asarray(prep.fit_transform(X), dtype=np.float32)
    try:
        feature_names = list(map(str, prep.get_feature_names_out()))
    except Exception:
        feature_names = [f"feat_{i}" for i in range(Xt.shape[1])]

    est = fresh_estimator(estimator)
    est.fit(Xt, y)
    _make_portable(est)

    return ModelBundle(
        estimator=est,
        task=task,
        target_cols=[target],
        feature_names=feature_names,
        train_summary=summarize_training_frame(X),
        preprocessor=prep,
        label_encoders={target: le} if le is not None else {},
        source="LOO",
        estimator_name=estimator_name,
        hpo=hpo,
        cv_metrics=cv_metrics or {},
        llm_decision=llm_decision,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test-set metrics
# ─────────────────────────────────────────────────────────────────────────────
def align_labels(y_true, y_pred):
    """Make true and predicted class labels comparable.

    A CSV column of 0/1 with one blank reads as float (0.0/1.0) while the
    training labels were int, so compare as numbers when both sides are
    numeric and as strings otherwise.
    """
    t = pd.Series(np.asarray(y_true, dtype=object))
    p = pd.Series(np.asarray(y_pred, dtype=object))
    tn, pn = pd.to_numeric(t, errors="coerce"), pd.to_numeric(p, errors="coerce")
    if tn.notna().all() and pn.notna().all():
        return tn.to_numpy(float), pn.to_numpy(float)
    return t.astype(str).to_numpy(), p.astype(str).to_numpy()


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, matthews_corrcoef,
    )
    t, p = align_labels(y_true, y_pred)
    kw = dict(average="weighted", zero_division=0)
    return {
        "n": int(len(t)),
        "accuracy": float(accuracy_score(t, p)),
        "balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "f1_weighted": float(f1_score(t, p, **kw)),
        "precision_weighted": float(precision_score(t, p, **kw)),
        "recall_weighted": float(recall_score(t, p, **kw)),
        "mcc": float(matthews_corrcoef(t, p)),
    }
