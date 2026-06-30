import pandas as pd
import numpy as np


def compare_regression_results(results: dict) -> pd.DataFrame:
    """
    Takes a dict of {model_name: result_dict} and returns a comparison DataFrame.
    Example input:
        {
            "Random Forest": {"r2": 0.899, "mae": 15705, "rmse": 24000},
            "XGBoost":       {"r2": 0.920, "mae": 13200, "rmse": 21000},
            "Neural Network": {"r2": 0.905, "mae": 14800, "rmse": 22500},
        }
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model":  model_name,
            "R2":     round(metrics.get("r2",   0), 4),
            "MAE":    round(metrics.get("mae",  0), 2),
            "RMSE":   round(metrics.get("rmse", 0), 2),
        })

    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    df.index += 1  # Rank starts at 1
    df.index.name = "Rank"
    return df


def compare_classification_results(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model":    model_name,
            "Accuracy": round(metrics.get("accuracy", 0), 4),
            "F1":       round(metrics.get("f1",       0), 4),
            "AUC":      round(metrics.get("auc",      0), 4),
        })

    df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    return df

