"""Two unrelated classes must not share the name AutoEDAPipeline.

`eda_pipeline` (SHAP-feedback diagnostics, ctor `config`) and
`pipeline.orchestrator` (the LLM preprocessing pipeline, ctor `target_col, task,
ollama_host, use_local_llm`) had the same name, incompatible constructors and no
method in common. Not a live bug — a trap. It nearly sprang: the run_loo fix went
into the orchestrator, and anyone assuming FeedbackController shared that code
would have been wrong about which implementation they had just corrected.
"""
from __future__ import annotations

import inspect

from eda_pipeline import AutoEDAPipeline as Alias, FeedbackEDAPipeline
from pipeline.orchestrator import AutoEDAPipeline as Orchestrated


def test_the_two_classes_are_distinct():
    assert Orchestrated is not FeedbackEDAPipeline


def test_the_old_name_still_imports():
    assert Alias is FeedbackEDAPipeline


def test_their_constructors_differ_as_documented():
    feedback = list(inspect.signature(FeedbackEDAPipeline.__init__).parameters)[1:]
    orch = list(inspect.signature(Orchestrated.__init__).parameters)[1:]

    assert feedback == ["config"]
    assert "target_col" in orch and "ollama_host" in orch


def test_feedback_controller_uses_the_feedback_one():
    """Changing only the import would have left a NameError at the call site."""
    src = inspect.getsource(__import__("feedback_controller"))

    assert "FeedbackEDAPipeline(self.config)" in src
    assert "AutoEDAPipeline(self.config)" not in src


def test_the_loop_reports_the_model_behind_its_best_frame():
    """run_feedback prints `Best Model : {model_name or 'N/A'}` — the
    max-iterations exit returned a hardcoded None, so every run that did not exit
    early reported N/A."""
    from pipeline.feedback_loop import FeedbackLoop

    src = inspect.getsource(FeedbackLoop)
    assert "self.best_model_name = model_name" in src
    assert "return self.best_df, None, self.history" not in src


# ── offline operation ────────────────────────────────────────────────────────
def test_the_pipeline_constructs_without_ollama():
    """`use_local_llm=False` selects the rule-based fallback and never calls a
    model, but AutoEDAPipeline built LocalEnsembleLLMEngine unconditionally and
    OllamaClient verified the daemon in its constructor — so the one path meant
    to work offline could not even be instantiated.

    Found by CI, not locally: the whole suite passed on a machine with Ollama
    running and failed the moment it ran anywhere else.
    """
    from pipeline.orchestrator import AutoEDAPipeline

    eda = AutoEDAPipeline(target_col="y", task="regression",
                          ollama_host="http://127.0.0.1:9",   # nothing listening
                          use_local_llm=False)
    assert eda.engine is not None


def test_the_rule_based_path_runs_without_ollama():
    import pandas as pd
    from pipeline.orchestrator import AutoEDAPipeline

    df = pd.DataFrame({"a": [1., 2., 3., 4.], "b": [4., 3., 2., 1.]})
    eda = AutoEDAPipeline(target_col="y", task="regression",
                          ollama_host="http://127.0.0.1:9", use_local_llm=False)

    eda.build_pipeline(df)

    assert eda.pipeline_ is not None
    assert eda.ensemble_result_.final.scaler in {"standard", "robust", "minmax"}


def test_the_client_is_only_built_when_actually_used():
    from pipeline.local_llm_engine import LocalEnsembleLLMEngine

    eng = LocalEnsembleLLMEngine(ollama_host="http://127.0.0.1:9")
    assert eng._client is None, "client must not be constructed eagerly"

    import pytest as _pytest
    with _pytest.raises(ConnectionError):
        eng.client
