"""The conservative merge is conservative for the destructive corrections.

This path runs only after the arbitrator LLM has already failed — precisely when
there is least reason to trust one agent's unreviewed proposal. It took the UNION
of both agents' log-transform columns, so a column only one agent proposed
logging got logged: the least conservative option available, for the correction
that is hardest to undo, while drops already required agreement.
"""
from __future__ import annotations

from arbitrator import ArbitratorLLM

A = {"drop_columns": ["x", "y"], "log_transform_columns": ["p", "q"],
     "clip_columns": {"c": [5, 95]}, "rescale_columns": {"r": "MinMaxScaler"}}
B = {"drop_columns": ["y"],       "log_transform_columns": ["q", "z"],
     "clip_columns": {"c": [10, 90]}, "rescale_columns": {"r": "RobustScaler"}}


def _merge(a=A, b=B):
    return ArbitratorLLM._conservative_merge(ArbitratorLLM.__new__(ArbitratorLLM), a, b)


def test_only_jointly_agreed_columns_are_dropped():
    assert sorted(_merge()["drop_columns"]) == ["y"]


def test_only_jointly_agreed_columns_are_log_transformed():
    """The regression: `p` and `z` each had one proposer and must not apply."""
    assert sorted(_merge()["log_transform_columns"]) == ["q"]


def test_clip_bounds_widen_to_the_safer_of_the_two():
    assert _merge()["clip_columns"]["c"] == [5, 95]


def test_rescale_conflicts_prefer_robust():
    """Rescale stays a union deliberately — choosing a scaler is reversible."""
    assert _merge()["rescale_columns"]["r"] == "RobustScaler"


def test_the_merge_is_flagged_as_non_consensus():
    assert _merge()["consensus"] is False


def test_disjoint_proposals_produce_no_destructive_action():
    a = {"drop_columns": ["x"], "log_transform_columns": ["p"],
         "clip_columns": {}, "rescale_columns": {}}
    b = {"drop_columns": ["y"], "log_transform_columns": ["q"],
         "clip_columns": {}, "rescale_columns": {}}
    m = _merge(a, b)

    assert m["drop_columns"] == []
    assert m["log_transform_columns"] == []


# ── response parsing ─────────────────────────────────────────────────────────
import pytest


def _parser():
    a = ArbitratorLLM.__new__(ArbitratorLLM)
    a.arbitrator_model = "phi4"
    return a


def test_a_proper_object_parses():
    parsed, failed = _parser()._parse_json_response('{"drop_columns": ["x"]}')

    assert failed is False
    assert parsed["drop_columns"] == ["x"]


def test_missing_keys_are_filled_so_callers_cannot_keyerror():
    parsed, _ = _parser()._parse_json_response('{"drop_columns": ["x"]}')

    for key in ("rescale_columns", "clip_columns", "log_transform_columns"):
        assert key in parsed


@pytest.mark.parametrize("raw", ['["scaler"]', '"none"', '42', 'true'])
def test_valid_json_that_is_not_an_object_falls_back(raw):
    """These are all valid JSON and none is a decision. `_fill_defaults` used to
    subscript them and raise an uncaught TypeError, killing the run instead of
    reaching the conservative-merge fallback built for this."""
    parsed, failed = _parser()._parse_json_response(raw)

    assert failed is True
    assert isinstance(parsed, dict)


def test_malformed_json_still_falls_back():
    parsed, failed = _parser()._parse_json_response("not json at all")

    assert failed is True
    assert isinstance(parsed, dict)


def test_a_fenced_json_block_is_unwrapped():
    parsed, failed = _parser()._parse_json_response(
        'here you go:\n```json\n{"drop_columns": ["a"]}\n```'
    )

    assert failed is False
    assert parsed["drop_columns"] == ["a"]
