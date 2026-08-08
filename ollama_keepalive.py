"""How long Ollama keeps a model resident in VRAM after it answers.

WHY THIS EXISTS
---------------
Ollama's default is to hold a model in VRAM for 5 minutes after the last
request. That is the right default for an interactive chat session and the
wrong one here, because this project is usually driven as one step of a larger
GPU workload that wants the card back immediately.

Measured 2026-08-08 on a 24 GB RTX 4090 driving the inverse-material-design
pipeline (which calls this project for its EDA step, then runs BoTorch on the
same GPU): the optimizer peaks at ~3.2 GB *allocated*, so a report-sized model
left resident is harmless, but a 20 GB model (qwen3:32b) leaves only 2.7 GB
free and the next run dies with a CUDA OOM. The ensemble here loads phi4
(9.1 GB) and mistral (4.4 GB) concurrently, plus gemma2 (5.4 GB) as tiebreaker.

The cost of unloading is a model reload on the next call — roughly 10 s for
phi4, against inference calls that already carry a 120 s timeout. That trade is
worth it by default; set the environment variable to opt out.

USAGE
-----
    AUTO_EDA_OLLAMA_KEEP_ALIVE="5m"   # Ollama's own default: stay resident
    AUTO_EDA_OLLAMA_KEEP_ALIVE="-1"   # never unload
    AUTO_EDA_OLLAMA_KEEP_ALIVE="0"    # unload immediately (this project's default)

Kept as a standalone top-level module rather than a constant inside
`pipeline/local_llm_engine.py` on purpose: `pipeline` is a package name this
project shares with inverse-material-design, which loads this code with its own
`pipeline/` on sys.path. A `from pipeline...` import here could resolve to the
wrong package. A unique top-level name cannot.
"""
from __future__ import annotations

import os

_ENV_VAR = "AUTO_EDA_OLLAMA_KEEP_ALIVE"


def _parse(raw: str) -> int | float | str:
    """Ollama accepts seconds as a number, a duration string ("5m"), or -1."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw          # duration string — hand it to Ollama as-is


KEEP_ALIVE = _parse(os.environ.get(_ENV_VAR, "0"))
