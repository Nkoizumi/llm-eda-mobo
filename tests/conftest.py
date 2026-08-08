"""Make the project importable from tests without installing it.

The modules under test live at the repo root (`bo_seeding`, `arbitrator`,
`feedback_analyzer`) and in `pipeline/`, and the app is run as `streamlit run
app.py` from that root, so tests put the root on sys.path the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
