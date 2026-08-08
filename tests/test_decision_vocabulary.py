"""The preprocessing decision that is reported is the one that runs.

`_build_sklearn_pipeline` resolves values it does not implement by falling back:
unknown scaler -> StandardScaler, unknown imputation -> median, unknown power
transform -> no power step at all. Nothing validated values on the way in, so the
LLM Decisions tab displayed the model's answer while something else executed.

This was not merely a hallucination risk. The prompt ADVERTISED "missforest" and
instructed the model to use it, and the builder never implemented it — a model
following the prompt correctly produced the divergence.
"""
from __future__ import annotations

import pytest

from pipeline.local_llm_engine import LLMDecision, LocalEnsembleLLMEngine as Eng
from pipeline.orchestrator import AutoEDAPipeline

PROFILE = {"numeric_cols": ["a", "b"], "categorical_cols": []}


def _decision(**kw):
    eng = Eng.__new__(Eng)                    # no Ollama connection needed
    return Eng._dict_to_decision(eng, kw, model="phi4", raw="", latency=0.0)


def _applied(decision):
    eda = AutoEDAPipeline(target_col="y", task="regression", use_local_llm=False)
    pipe = eda._build_sklearn_pipeline(decision, PROFILE)
    ct = pipe.named_steps.get("preprocessor", pipe)
    num = dict((n, t) for n, t, _ in ct.transformers)["num"]
    steps = dict(num.steps)
    return (type(steps["scaler"]).__name__,
            type(steps["imputer"]).__name__,
            "power" in steps)


@pytest.mark.parametrize("key,bad,expected", [
    ("scaler",              "quantile",   "standard"),
    ("imputation_strategy", "missforest", "median"),
    ("imputation_strategy", "iterative",  "median"),
    ("power_transform",     "log",        "none"),
    ("outlier_method",      "dbscan",     "iqr"),
])
def test_unimplemented_values_are_coerced(key, bad, expected):
    assert getattr(_decision(**{key: bad}), key) == expected


@pytest.mark.parametrize("key,good", [
    ("scaler", "robust"), ("scaler", "minmax"),
    ("imputation_strategy", "knn"), ("imputation_strategy", "mean"),
    ("power_transform", "box-cox"), ("outlier_method", "zscore"),
])
def test_implemented_values_pass_through(key, good):
    assert getattr(_decision(**{key: good}), key) == good


def test_reported_matches_applied_for_an_unimplemented_decision():
    """The property that matters: whatever the tab shows is what ran."""
    d = _decision(scaler="quantile", imputation_strategy="missforest",
                  power_transform="log")
    scaler, imputer, has_power = _applied(d)

    assert (d.scaler, d.imputation_strategy, d.power_transform) == \
           ("standard", "median", "none")
    assert (scaler, imputer, has_power) == ("StandardScaler", "SimpleImputer", False)


def test_reported_matches_applied_for_an_implemented_decision():
    d = _decision(scaler="robust", imputation_strategy="knn",
                  power_transform="yeo-johnson")

    assert _applied(d) == ("RobustScaler", "KNNImputer", True)


def test_the_prompt_only_advertises_what_the_builder_implements():
    """The root cause: the prompt offered `missforest`, which never existed."""
    from pipeline.local_llm_engine import DECISION_PROMPT_TEMPLATE as P

    assert "missforest" not in P
    for allowed, _default in Eng._VOCAB.values():
        assert any(v in P for v in allowed), allowed


def test_the_tiebreaker_cannot_invent_a_third_option():
    """Gemma2's prompt says "choose the BEST value" without restricting it to
    the two on offer, and its reply used to be passed through unchecked."""
    eng = Eng.__new__(Eng)
    conflicts = [{"key": "scaler", "phi4": "robust", "mistral": "minmax"}]

    class _FakeClient:
        @staticmethod
        def generate(**kw):
            return '{"scaler": "quantile", "unrelated_key": "x"}', 1.0
        @staticmethod
        def extract_json(raw):
            return {"scaler": "quantile", "unrelated_key": "x"}

    eng.client = _FakeClient()
    votes, _ = eng._tiebreak_with_gemma2(conflicts, {}, None, None)

    assert votes == {"scaler": "standard"}, "out-of-vocab value must be coerced"
    assert "unrelated_key" not in votes, "keys not in conflict must be dropped"
