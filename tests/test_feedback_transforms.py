"""The feedback loop applies each correction to each column at most once.

`pipeline/feedback_loop.py` feeds its corrected frame back in as the next
iteration's input, and nothing recorded what had been applied — so each round
re-diagnosed already-corrected data and could repeat the same correction.

log1p is the one that hurts. On the bundled AmesHousing.csv, at the detector's
default skew threshold of 2.0, four of six flagged columns are STILL flagged
after one log1p (their skew is a mass at zero, which a log cannot fix), so the
loop would apply log1p(log1p(log1p(x))). The existing negative-value guard cannot
catch the repeat: log1p output is non-negative by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feedback_analyzer import FeedbackAnalyzer


class _Cfg:
    target_col = "y"
    verbose = False


@pytest.fixture
def fa():
    return FeedbackAnalyzer(_Cfg())


@pytest.fixture
def df():
    return pd.DataFrame({"a": [0., 0., 0., 1., 2., 3., 400.],
                         "y": range(7)})


def _report(**kw):
    base = {"ghost_features": [], "outlier_features": [],
            "damaged_features": [], "skewed_features": []}
    base.update(kw)
    return base


def test_log1p_is_not_applied_twice(fa, df):
    r = _report(skewed_features=["a"])
    once = fa.suggest_corrections(df, r, 0.0, 1)
    twice = fa.suggest_corrections(once, r, 0.0, 2)

    assert np.allclose(once["a"], twice["a"]), "column was log-transformed again"
    assert np.allclose(once["a"], np.log1p(df["a"])), "first pass should apply it"


def test_rescale_is_not_applied_twice(fa, df):
    r = _report(outlier_features=["a"])
    once = fa.suggest_corrections(df, r, 0.0, 1)
    twice = fa.suggest_corrections(once, r, 0.0, 2)

    assert np.allclose(once["a"], twice["a"])


def test_a_column_skipped_for_negatives_stays_retryable(fa):
    """`_unapplied` filters and `_mark` records, deliberately separate: a column
    log1p declined has not been transformed, so barring a later retry would be
    wrong once an earlier correction makes it non-negative.
    """
    neg = pd.DataFrame({"a": [-5., -1., 0., 1., 100.], "y": range(5)})
    fa.suggest_corrections(neg, _report(skewed_features=["a"]), 0.0, 1)

    assert ("log1p", "a") not in fa._applied

    shifted = neg.assign(a=neg["a"] + 10)
    out = fa.suggest_corrections(shifted, _report(skewed_features=["a"]), 0.0, 2)
    assert np.allclose(out["a"], np.log1p(shifted["a"])), "retry should now apply"


def test_different_transforms_on_one_column_are_independent(fa, df):
    """Claiming log1p must not block a later rescale of the same column."""
    fa.suggest_corrections(df, _report(skewed_features=["a"]), 0.0, 1)
    out = fa.suggest_corrections(df, _report(outlier_features=["a"]), 0.0, 2)

    assert ("robust", "a") in fa._applied


def test_the_llm_correction_mode_shares_the_same_guard(fa, df):
    d = {"log_transform_columns": ["a"], "drop_columns": [],
         "rescale_columns": {}, "clip_columns": {}}
    once = fa.apply_llm_corrections(df, d)
    twice = fa.apply_llm_corrections(once, d)

    assert np.allclose(once["a"], twice["a"])


def test_the_motivating_data_really_does_re_flag():
    """Pins the measurement the fix exists for, so it is not folklore."""
    ames = pd.read_csv("data/AmesHousing.csv")
    ames.columns = ames.columns.str.strip()
    col = ames["Low Qual Fin SF"]

    assert abs(col.skew()) > 2.0
    assert abs(np.log1p(col).skew()) > 2.0, "would be logged again next iteration"
