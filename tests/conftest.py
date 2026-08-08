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


APP_PY = ROOT / "app.py"
"""Absolute path to the Streamlit entrypoint.

`AppTest.from_file` resolves a RELATIVE path against the file that calls it, not
the working directory — so "app.py" from tests/ resolves to tests/app.py. Older
Streamlit resolved against the CWD, which is why passing a relative path worked
locally and failed in CI.
"""

DATA = ROOT / "data"
"""Bundled CSVs. Absolute, so tests do not depend on the working directory."""
