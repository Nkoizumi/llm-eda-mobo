"""OutlierHandler learns its replacement values at fit time.

The isolation_forest branch used to compute replacement medians inside
`transform`, which made the transform a function of its own input rather than of
the fitted state. Two consequences, the second silent:

  * test data was repaired towards TEST medians;
  * a single-row transform took the median of one row — that row itself — so the
    repair became a no-op. Leave-one-out CV transforms exactly one row per fold,
    so outlier handling applied during training and quietly did nothing on every
    held-out point.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transformers import OutlierHandler


@pytest.fixture
def train():
    return pd.DataFrame({"a": [1., 2., 3., 4., 5., 100.],
                         "b": [1., 1., 1., 1., 1., 50.]})


def test_medians_are_learned_at_fit(train):
    oh = OutlierHandler(method="isolation_forest").fit(train)

    assert oh.iso_medians_["a"] == 3.5
    assert oh.iso_medians_["b"] == 1.0


def test_a_single_row_is_repaired_not_left_alone(train):
    """The regression: median-of-one-row is that row, so this was a no-op."""
    oh = OutlierHandler(method="isolation_forest").fit(train)
    extreme = train.iloc[[5]]
    assert (oh.iso_forest_.predict(extreme) == -1).any(), "fixture must be flagged"

    out = oh.transform(extreme)

    assert out.iloc[0]["a"] == 3.5
    assert out.iloc[0]["b"] == 1.0


def test_repair_uses_train_medians_not_the_transformed_frame(train):
    """Transforming a different frame must not re-derive the medians from it."""
    oh = OutlierHandler(method="isolation_forest").fit(train)
    other = pd.DataFrame({"a": [1000., 2000.], "b": [500., 600.]})

    out = oh.transform(other)

    assert set(out["a"]).issubset({3.5, 1000., 2000.})
    assert 1500.0 not in set(out["a"]), "used the transformed frame's own median"


def test_iqr_and_zscore_branches_still_clip_to_fitted_bounds(train):
    for method in ("iqr", "zscore"):
        oh = OutlierHandler(method=method).fit(train)
        out = oh.transform(train)
        lo, hi = oh.bounds_["a"]
        assert out["a"].max() <= hi + 1e-9
        assert out["a"].min() >= lo - 1e-9
