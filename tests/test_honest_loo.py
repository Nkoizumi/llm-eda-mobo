"""run_loo refits its preprocessing inside every fold.

It used to LOO-split `self.X_transformed`, which `get_transformed_df` produces by
calling fit_transform on the WHOLE frame — so every held-out row had already
contributed to the imputer's medians, the power transform's lambda, the scaler's
moments and the correlation filter's column choice. It also disagreed in meaning
with `tabs/_loo_utils.run_loo_with_wrapper`, which does refit per fold.

Measured before the fix, so the size is on record rather than assumed:

    Fish  (n=159) RF   R2 0.9711 -> 0.9711   (0.0000)
    Fish  (n=159) MLP  R2 0.8486 -> 0.8501  (+0.0015)
    slump (n=103) MLP  R2 0.1115 -> 0.1007  (-0.0109)

Negligible — BENCHMARKS.md is unaffected. Fixed because a method called run_loo
should mean one thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from pipeline.orchestrator import AutoEDAPipeline


@pytest.fixture
def eda_and_df():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=12), "b": rng.normal(size=12)})
    df["y"] = 2 * df["a"] - df["b"]
    eda = AutoEDAPipeline(target_col="y", task="regression", use_local_llm=False)
    eda.build_pipeline(df.drop(columns=["y"]))
    return eda, df


def test_run_loo_does_not_require_get_transformed_df_first(eda_and_df):
    """It is now self-sufficient; it used to raise unless the cached matrix
    existed, which is exactly what made it leak."""
    eda, df = eda_and_df

    out = eda.run_loo(df, Ridge())

    assert set(out) >= {"r2", "rmse", "mae", "y_true", "y_pred"}
    assert len(out["y_true"]) == len(df)


def test_the_preprocessing_is_refit_once_per_fold(eda_and_df):
    """The property the fix is about: n folds means n fits, not one."""
    eda, df = eda_and_df
    fits = {"n": 0}
    original_fit = type(eda.pipeline_).fit

    def counting_fit(self, X, y=None, **kw):
        fits["n"] += 1
        return original_fit(self, X, y, **kw)

    type(eda.pipeline_).fit = counting_fit
    try:
        eda.run_loo(df, Ridge())
    finally:
        type(eda.pipeline_).fit = original_fit

    assert fits["n"] == len(df), f"expected one fit per fold, got {fits['n']}"


def test_a_missing_target_is_an_explicit_error(eda_and_df):
    eda, df = eda_and_df

    with pytest.raises(RuntimeError, match="target column"):
        eda.run_loo(df.drop(columns=["y"]), Ridge())


def test_predictions_are_not_the_training_fit(eda_and_df):
    """Held-out predictions must differ from in-sample ones, or nothing is
    actually being held out."""
    eda, df = eda_and_df
    out = eda.run_loo(df, Ridge())

    assert not np.allclose(out["y_true"], out["y_pred"])
