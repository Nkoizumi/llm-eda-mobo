"""FeedbackEDAPipeline does not do work nobody reads.

`run()` trained a baseline model with 5-fold CV and computed permutation
importance on every call. `feedback_controller.py:91` unpacks those four values
and never references them again — it calls ModelTrainer.train_and_evaluate and
uses that score instead.

Measured on the bundled AmesHousing.csv (2930 rows x 28 numeric cols):

    detectors            (used)      0.01 s
    _train_best_model    (discarded) 1.23 s
    _compute_feature_imp (discarded) 1.26 s

99% of run() was discarded, once per feedback iteration.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from conftest import DATA
from eda_pipeline import FeedbackEDAPipeline


class _Cfg:
    target_col = "SalePrice"
    task_type = "regression"
    verbose = False


@pytest.fixture
def frame():
    df = pd.read_csv(DATA / "AmesHousing.csv")
    df.columns = df.columns.str.strip()
    return df.select_dtypes(include="number").dropna(axis=1, how="any")


def test_the_baseline_model_is_skipped_by_default(frame):
    report, model, name, score, imp = FeedbackEDAPipeline(_Cfg()).run(frame)

    assert model is None and name is None and score is None
    assert report["ghost_features"] is not None, "signals must still be produced"


def test_the_return_shape_is_unchanged(frame):
    """feedback_controller unpacks exactly five values."""
    out = FeedbackEDAPipeline(_Cfg()).run(frame)

    assert len(out) == 5


def test_the_baseline_is_still_available_on_request(frame):
    _, model, name, score, _ = FeedbackEDAPipeline(_Cfg()).run(
        frame, train_baseline=True)

    assert model is not None
    assert name in {"random_forest", "ridge"}
    assert score is not None


def test_fitting_does_not_mutate_the_shared_class_level_estimator(frame):
    """`_MODELS` holds INSTANTIATED estimators at class level, so fitting one
    without cloning mutates state every instance shares — and the controller
    builds a fresh pipeline each iteration."""
    FeedbackEDAPipeline(_Cfg()).run(frame, train_baseline=True)

    for estimator in FeedbackEDAPipeline._MODELS["regression"].values():
        with pytest.raises(NotFittedError):
            check_is_fitted(estimator)
