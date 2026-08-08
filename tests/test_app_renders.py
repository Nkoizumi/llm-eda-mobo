"""The Streamlit app renders without raising.

This is the check that caught `Styler.applymap`, removed in pandas 3.0 while
`pandas` sits unpinned in requirements.txt. A fresh install picked up 3.x and
app.py — which renders the Distributions tab unconditionally — died at startup.
It stayed invisible in development because the author's environment still had
pandas 2.3, where the deprecated alias survives.

Kept in the suite rather than only as a CI step so that `pytest` alone catches
an API removal in any dependency, on whatever versions the developer happens to
have installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")


def test_the_app_renders_with_no_uncaught_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=180).run()

    assert not at.exception, [e.value for e in at.exception]


def test_the_distributions_styler_call_survives_current_pandas():
    """Pins the specific API. `Styler.map` replaced `applymap` in pandas 2.1;
    the alias was removed in 3.0."""
    import pandas as pd

    styler = pd.DataFrame({"Needs Transform": ["Yes", "No"]}).style
    assert hasattr(styler, "map"), "pandas < 2.1 — Styler.map unavailable"

    out = styler.map(lambda v: "color:red" if v == "Yes" else "",
                     subset=["Needs Transform"])
    assert out is not None
