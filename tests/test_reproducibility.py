"""BO suggestions and preprocessing decisions are reproducible.

`tabs/bo_mobo.py` built its SobolQMCNormalSampler without a seed, and nothing
pinned torch's global RNG, which optimize_acqf's random restarts also draw from.
Two runs on identical data proposed entirely different experiments:

    run 1 candidate 0: [0.603553 0.156326 0.119633]
    run 2 candidate 0: [0.191986 0.464732 0.454498]

For a tool whose output is "which experiment should I run next", the user could
not separate the effect of changing a setting from the effect of pressing the
button twice.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from bo_seeding import DEFAULT_SEED, make_sampler, seed_everything


def _candidates(seed):
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    seed_everything(seed)
    g = torch.Generator().manual_seed(0)          # fix the DATA
    X = torch.rand(12, 3, generator=g, dtype=torch.double)
    Y = X.sum(-1, keepdim=True)
    gp = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acqf = qLogExpectedImprovement(model=gp, best_f=Y.max().item(),
                                   sampler=make_sampler(64, seed))
    bounds = torch.stack([torch.zeros(3, dtype=torch.double),
                          torch.ones(3, dtype=torch.double)])
    cand, _ = optimize_acqf(acq_function=acqf, bounds=bounds, q=2,
                            num_restarts=3, raw_samples=32)
    return cand.detach().numpy()


def test_the_same_seed_gives_the_same_experiments():
    assert np.allclose(_candidates(DEFAULT_SEED), _candidates(DEFAULT_SEED))


def test_a_different_seed_gives_different_experiments():
    """Otherwise the seed control would be decorative."""
    assert not np.allclose(_candidates(DEFAULT_SEED), _candidates(DEFAULT_SEED + 5))


def test_the_sampler_carries_its_seed():
    assert make_sampler(16, 123).seed == 123


def test_seed_everything_pins_torch():
    seed_everything(7); a = torch.rand(5)
    seed_everything(7); b = torch.rand(5)
    assert torch.allclose(a, b)


def test_every_ollama_model_is_seeded():
    """Sampling happens at temperature > 0; without a seed two identical dataset
    profiles could yield different preprocessing, and feedback_loop rebuilds the
    pipeline every iteration."""
    from pipeline.local_llm_engine import RTX4090_CONFIG

    for role, cfg in RTX4090_CONFIG.items():
        assert "seed" in cfg, f"{role} has no seed"
