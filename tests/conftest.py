"""Shared test guards.

The home cast (data/home-canon.json) holds the family's REAL registered characters on
Choksi's machine. Tests that exercise the fixers with fake names (Jammus, Marley-like
fixtures) must never see it — a real registration would silently change fixture
behavior (the same machine-dependence trap the roster hit). This autouse fixture points
the default cast path at a per-test temp file; tests that WANT a cast write one there
or pass home_path explicitly."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_home_canon(tmp_path, monkeypatch):
    import home_canon
    monkeypatch.setattr(home_canon, "DEFAULT_PATH", tmp_path / "home-canon.json")
