"""Make the BO / MOBO suggestions reproducible.

WHY THIS EXISTS
---------------
`tabs/bo_mobo.py` built its `SobolQMCNormalSampler` without a `seed=`, and
nothing pinned torch's global RNG, which `optimize_acqf`'s random restarts also
draw from. Two runs on identical data therefore proposed different experiments:

    run 1 candidate 0: [0.603553 0.156326 0.119633]
    run 2 candidate 0: [0.191986 0.464732 0.454498]
    max abs diff: 0.41   (on a [0, 1] domain — entirely different points)

For a tool whose output is "which experiment should I run next", that is worse
than an ordinary nondeterminism bug. A user who re-runs the same analysis gets a
different answer with nothing on screen to say why, and cannot tell an effect of
changing a setting from an effect of having pressed the button twice.

Kept as a standalone top-level module rather than a helper inside
`tabs/bo_mobo.py` for the same reason as `ollama_keepalive`: `pipeline` is a
package name this project shares with inverse-material-design, and top-level
names cannot be shadowed by it.

USAGE
-----
    from bo_seeding import seed_everything, make_sampler

    seed_everything(seed)                       # before building the model
    sampler = make_sampler(128, seed)           # instead of a bare sampler
"""
from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Pin python / numpy / torch RNGs. Returns the seed for logging."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def make_sampler(num_samples: int = 128, seed: int = DEFAULT_SEED):
    """A `SobolQMCNormalSampler` that draws the same quasi-random sequence
    every run.

    Seeding torch globally is not sufficient on its own: BoTorch's sampler
    draws its Sobol engine at construction, so two samplers built at different
    points in the same process still diverge unless the seed is passed here.
    """
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler

    return SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]),
                                 seed=seed)
