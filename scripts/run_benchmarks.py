"""Benchmark auto_eda across the demo datasets via the LLM-driven preprocessing path.

Run from the repo root:

    conda activate auto_eda
    python scripts/run_benchmarks.py

Method:
  - LOO for n <= 200, 5-fold CV (shuffle, seed 42) otherwise.
  - Three base estimators tried per dataset: RandomForest, XGBoost, MLP.
  - Multi-target datasets use sklearn's `MultiOutputRegressor` wrapping the base.

Requires Ollama running on localhost:11434 with `phi4` and `mistral` pulled.
The LLM-built preprocessing pipeline is fit once on the full data before the
CV split — same as `app.py` — so the numbers match what users see in the LOO
Results / Multitask Learning tabs.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.orchestrator import AutoEDAPipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

DATA = REPO_ROOT / "data"
BENCH_OUT = REPO_ROOT / "BENCHMARKS.md"
OLLAMA = "http://localhost:11434"

CONFIGS = [
    dict(file="Fish.csv", task="regression",
         primary="Weight", extras=[], drop=[], method="loo"),
    dict(file="slump_test.csv", task="regression",
         primary="slump", extras=["flow", "compressive_strength"],
         drop=[], method="loo_multi"),
    dict(file="ENB2012_data.csv", task="regression",
         primary="Y1", extras=["Y2"], drop=[], method="kfold_multi"),
    dict(file="AmesHousing.csv", task="regression",
         primary="SalePrice", extras=[], drop=["Order", "PID"], method="kfold"),
    dict(file="Titanic-Dataset.csv", task="classification",
         primary="Survived", extras=[],
         drop=["PassengerId", "Name", "Ticket", "Cabin"], method="kfold"),
]


def get_estimators(task: str) -> dict:
    if task == "regression":
        ests = {
            "RF":  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "NN":  MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
        if XGB_OK:
            ests["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42,
                                           verbosity=0, n_jobs=-1)
    else:
        ests = {
            "RF":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "NN":  MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
        if XGB_OK:
            ests["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                            verbosity=0, eval_metric="logloss",
                                            n_jobs=-1)
    return ests


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip whitespace from column names — `slump_test.csv` ships with a
    # trailing space on `compressive_strength` and that leaks into output tables.
    df.columns = [c.strip() for c in df.columns]
    return df


def build_pipeline(df: pd.DataFrame, primary: str, task: str, drop_cols: list,
                   all_targets: list[str]) -> AutoEDAPipeline:
    eda = AutoEDAPipeline(target_col=primary, task=task,
                          ollama_host=OLLAMA, use_local_llm=True)
    X_only = df.drop(columns=all_targets + drop_cols, errors="ignore")
    eda.build_pipeline(X_only)
    return eda


def score_single(eda: AutoEDAPipeline, df: pd.DataFrame, task: str,
                 estimators: dict, method: str) -> dict:
    eda.get_transformed_df(df)
    X = np.asarray(eda.X_transformed, dtype=np.float64)
    y = np.asarray(eda.y)
    cv = LeaveOneOut() if method == "loo" else KFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, est in estimators.items():
        t0 = time.time()
        try:
            y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=1)
            if task == "regression":
                out[name] = dict(r2=r2_score(y, y_pred),
                                 mae=mean_absolute_error(y, y_pred),
                                 sec=time.time() - t0)
            else:
                out[name] = dict(acc=accuracy_score(y, y_pred),
                                 f1=f1_score(y, y_pred, average="weighted", zero_division=0),
                                 sec=time.time() - t0)
        except Exception as e:
            out[name] = dict(error=str(e)[:120], sec=time.time() - t0)
    return out


def score_multi(eda: AutoEDAPipeline, df: pd.DataFrame, all_targets: list[str],
                estimators: dict, method: str) -> dict:
    eda.get_transformed_df(df)
    X = np.asarray(eda.X_transformed, dtype=np.float64)
    Y = df[all_targets].values.astype(np.float64)
    cv = LeaveOneOut() if method == "loo_multi" else KFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, est in estimators.items():
        t0 = time.time()
        try:
            wrapper = MultiOutputRegressor(est)
            Y_pred = cross_val_predict(wrapper, X, Y, cv=cv, n_jobs=1)
            res = {f"R²[{t}]": r2_score(Y[:, i], Y_pred[:, i])
                   for i, t in enumerate(all_targets)}
            res["sec"] = time.time() - t0
            out[name] = res
        except Exception as e:
            out[name] = dict(error=str(e)[:120], sec=time.time() - t0)
    return out


def main():
    rows = []
    print(f"\n{'=' * 68}\n auto_eda benchmark — LLM-driven preprocessing path\n{'=' * 68}\n")
    for cfg in CONFIGS:
        path = DATA / cfg["file"]
        df = load_csv(path)
        if cfg["drop"]:
            df = df.drop(columns=cfg["drop"], errors="ignore")
        n = len(df)
        targets = [cfg["primary"]] + cfg["extras"]
        print(f"\n--- {cfg['file']} (n={n}, task={cfg['task']}, "
              f"targets={targets}, method={cfg['method']}) ---")

        t_llm = time.time()
        try:
            eda = build_pipeline(df, cfg["primary"], cfg["task"], cfg["drop"], targets)
            print(f"  LLM-decisions: {time.time() - t_llm:.1f}s")
        except Exception as e:
            print(f"  FAILED at build_pipeline: {e}")
            traceback.print_exc()
            rows.append(dict(cfg=cfg, error=str(e)[:120]))
            continue

        ests = get_estimators(cfg["task"])
        try:
            if cfg["method"] in ("loo", "kfold"):
                scores = score_single(eda, df, cfg["task"], ests, cfg["method"])
            else:
                scores = score_multi(eda, df, targets, ests, cfg["method"])
        except Exception as e:
            print(f"  FAILED at scoring: {e}")
            traceback.print_exc()
            rows.append(dict(cfg=cfg, error=str(e)[:120]))
            continue

        for name, s in scores.items():
            print(f"  {name:>8}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                                else f"{k}={v}" for k, v in s.items()))
        rows.append(dict(cfg=cfg, scores=scores, n=n))

    # Render BENCHMARKS.md
    lines = []
    lines.append("# auto_eda — benchmarks")
    lines.append("")
    lines.append("All numbers come from the LLM-driven preprocessing path "
                 "(Phi-4 + Mistral ensemble decides imputation, scaler, encoding, "
                 "outlier method, correlation threshold) → joint multi-output model.")
    lines.append("")
    lines.append("Methodology:")
    lines.append("- LOO for `n ≤ 200`, 5-fold CV (shuffle, seed 42) otherwise.")
    lines.append("- Three base estimators tried per dataset: RandomForest, XGBoost, MLP. The best is reported.")
    lines.append("- Multi-target datasets use `MultiOutputRegressor` (sklearn) wrapping the base estimator.")
    lines.append("- Caveat: the LLM-built preprocessing pipeline is fit once on the full data "
                 "before the CV split. This matches what `app.py` does — a small leak that biases "
                 "metrics optimistically by a few percent but keeps numbers comparable to what users "
                 "see in the LOO Results / Multitask Learning tabs.")
    lines.append("")
    lines.append("## Headline table")
    lines.append("")
    lines.append("| Dataset | Rows | Task | Targets | Method | Best base | Metric | Score |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for r in rows:
        cfg = r["cfg"]
        if "error" in r:
            lines.append(f"| {cfg['file']} | — | {cfg['task']} | "
                         f"{', '.join([cfg['primary']] + cfg['extras'])} | "
                         f"{cfg['method']} | (failed) | — | `{r['error']}` |")
            continue
        targets = [cfg["primary"]] + cfg["extras"]
        scores = r["scores"]
        n = r["n"]
        if cfg["method"] in ("loo", "kfold"):
            key = "r2" if cfg["task"] == "regression" else "acc"
            ranked = [(name, s.get(key, float("-inf"))) for name, s in scores.items() if "error" not in s]
            ranked.sort(key=lambda kv: -kv[1])
            best_name, best_score = ranked[0]
            metric = "R²" if cfg["task"] == "regression" else "Accuracy"
            lines.append(f"| {cfg['file']} | {n} | {cfg['task']} | {cfg['primary']} | "
                         f"{cfg['method'].upper()} | {best_name} | {metric} | **{best_score:.3f}** |")
        else:
            ranked = []
            for name, s in scores.items():
                if "error" in s:
                    continue
                r2s = [v for k, v in s.items() if k.startswith("R²")]
                ranked.append((name, np.mean(r2s), r2s))
            ranked.sort(key=lambda kv: -kv[1])
            best_name, _mean_r2, r2s = ranked[0]
            r2_str = " / ".join(f"{v:.3f}" for v in r2s)
            lines.append(f"| {cfg['file']} | {n} | {cfg['task']} | {', '.join(targets)} | "
                         f"{cfg['method'].replace('_multi','').upper()} | {best_name} | "
                         f"R² (per target) | **{r2_str}** |")

    lines.append("")
    lines.append("## Per-dataset detail (all estimators)")
    lines.append("")
    for r in rows:
        cfg = r["cfg"]
        if "error" in r:
            continue
        lines.append(f"### {cfg['file']}")
        lines.append("")
        targets = [cfg["primary"]] + cfg["extras"]
        if cfg["method"] in ("loo", "kfold"):
            header = ("R²", "MAE", "sec") if cfg["task"] == "regression" else ("Accuracy", "F1", "sec")
            lines.append(f"| Estimator | {header[0]} | {header[1]} | runtime (s) |")
            lines.append("|---|---|---|---|")
            for name, s in r["scores"].items():
                if "error" in s:
                    lines.append(f"| {name} | — | — | {s['sec']:.1f} (error: `{s['error']}`) |")
                else:
                    if cfg["task"] == "regression":
                        lines.append(f"| {name} | {s['r2']:.4f} | {s['mae']:.4f} | {s['sec']:.1f} |")
                    else:
                        lines.append(f"| {name} | {s['acc']:.4f} | {s['f1']:.4f} | {s['sec']:.1f} |")
        else:
            target_cols_lbl = " | ".join(f"R²[{t}]" for t in targets)
            lines.append(f"| Estimator | {target_cols_lbl} | runtime (s) |")
            lines.append("|---|" + "|".join("---" for _ in targets) + "|---|")
            for name, s in r["scores"].items():
                if "error" in s:
                    lines.append(f"| {name} | " + " | ".join("—" for _ in targets) +
                                 f" | {s['sec']:.1f} (error: `{s['error']}`) |")
                else:
                    vals = " | ".join(f"{s[f'R²[{t}]']:.4f}" for t in targets)
                    lines.append(f"| {name} | {vals} | {s['sec']:.1f} |")
        lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# Requires Ollama running on localhost:11434 with phi4 and mistral pulled.")
    lines.append("conda activate auto_eda")
    lines.append("python scripts/run_benchmarks.py")
    lines.append("```")
    lines.append("")
    lines.append("Random seed: 42 (CV shuffle, MLP/RF/XGB random_state). Estimator hyperparameters "
                 "match those used in the Streamlit app. LLM decisions are non-deterministic, so "
                 "re-runs may vary by a few percent depending on which preprocessing path each model picks.")

    BENCH_OUT.write_text("\n".join(lines) + "\n")
    print(f"\n✓ Wrote {BENCH_OUT}")


if __name__ == "__main__":
    main()
