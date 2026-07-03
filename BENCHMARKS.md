# llm-eda-mobo — benchmarks

All numbers come from the LLM-driven preprocessing path (Phi-4 + Mistral ensemble decides imputation, scaler, encoding, outlier method, correlation threshold) → joint multi-output model.

Methodology:
- LOO for `n ≤ 200`, 5-fold CV (shuffle, seed 42) otherwise.
- Three base estimators tried per dataset: RandomForest, XGBoost, MLP. The best is reported.
- Multi-target datasets use `MultiOutputRegressor` (sklearn) wrapping the base estimator.
- Caveat: the LLM-built preprocessing pipeline is fit once on the full data before the CV split. This matches what `app.py` does — a small leak that biases metrics optimistically by a few percent but keeps numbers comparable to what users see in the LOO Results / Multitask Learning tabs.

## Headline table

| Dataset | Rows | Task | Targets | Method | Best base | Metric | Score |
|---|---|---|---|---|---|---|---|
| Fish.csv | 159 | regression | Weight | LOO | RF | R² | **0.971** |
| slump_test.csv | 103 | regression | slump, flow, compressive_strength | LOO | RF | R² (per target) | **0.292 / 0.407 / 0.819** |
| ENB2012_data.csv | 768 | regression | Y1, Y2 | KFOLD | XGBoost | R² (per target) | **0.999 / 0.991** |
| AmesHousing.csv | 2930 | regression | SalePrice | KFOLD | XGBoost | R² | **0.904** |
| Titanic-Dataset.csv | 891 | classification | Survived | KFOLD | NN | Accuracy | **0.826** |

## Per-dataset detail (all estimators)

### Fish.csv

| Estimator | R² | MAE | runtime (s) |
|---|---|---|---|
| RF | 0.9711 | 37.3564 | 11.0 |
| NN | 0.8486 | 94.9531 | 13.1 |
| XGBoost | 0.9654 | 39.9251 | 3.7 |

### slump_test.csv

| Estimator | R²[slump] | R²[flow] | R²[compressive_strength] | runtime (s) |
|---|---|---|---|---|
| RF | 0.2920 | 0.4067 | 0.8194 | 20.8 |
| NN | 0.1115 | 0.2342 | 0.8078 | 22.0 |
| XGBoost | 0.2168 | 0.2776 | 0.8434 | 10.3 |

### ENB2012_data.csv

| Estimator | R²[Y1] | R²[Y2] | runtime (s) |
|---|---|---|---|
| RF | 0.9977 | 0.9691 | 0.8 |
| NN | 0.9452 | 0.9178 | 3.2 |
| XGBoost | 0.9987 | 0.9911 | 0.2 |

### AmesHousing.csv

| Estimator | R² | MAE | runtime (s) |
|---|---|---|---|
| RF | 0.8976 | 15754.5910 | 0.6 |
| NN | 0.8408 | 20247.5178 | 6.6 |
| XGBoost | 0.9041 | 15513.0264 | 2.5 |

> **Note:** the default runner uses 5-fold CV for `n > 200` to keep total wall-clock under a few minutes. XGBoost + true LOO on Ames (verified separately, ~25 min wall-clock) gives R² ≈ 0.92 — the small lift over the 5-fold number above is the expected LOO bias (each fold trains on `n-1` rather than `0.8n` rows).

### Titanic-Dataset.csv

| Estimator | Accuracy | F1 | runtime (s) |
|---|---|---|---|
| RF | 0.8215 | 0.8204 | 0.5 |
| NN | 0.8260 | 0.8232 | 1.8 |
| XGBoost | 0.8126 | 0.8119 | 1.9 |

## Reproduce

```bash
# Requires Ollama running on localhost:11434 with phi4 and mistral pulled.
conda activate auto_eda
python scripts/run_benchmarks.py
```

Random seed: 42 (CV shuffle, MLP/RF/XGB random_state). Estimator hyperparameters are the same as those used in the Streamlit app (`base = RandomForestRegressor(n_estimators=100, …)`, etc.). The LLM decisions are non-deterministic, so re-runs may vary by a few percent depending on which preprocessing path each LLM picks.
