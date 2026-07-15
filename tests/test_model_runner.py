"""Tests for run_model — the spawn primitive behind every model call.

These spawn real subprocesses (the worker fns are module-level so they pickle), so they
also guard the picklability contract a future refactor could silently break.
"""
import time

import pytest

from model_runner import run_model


def _double(x):
    return x * 2


def _raise_value_error(msg):
    raise ValueError(msg)


def _sleep_then_return(seconds):
    time.sleep(seconds)
    return "done"


def test_round_trip_picklable_arg_and_return():
    assert run_model(_double, 21) == 42


def test_worker_exception_propagates_to_caller():
    with pytest.raises(ValueError, match="boom"):
        run_model(_raise_value_error, "boom")


def test_timeout_bounds_wallclock_and_kills_worker():
    """A worker that overruns is killed near `timeout`, not after it finishes — this is
    the safety valve every model path relies on (regression guard for the inert-timeout bug)."""
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        run_model(_sleep_then_return, 30, timeout=1)
    elapsed = time.monotonic() - t0
    assert elapsed < 15, f"timeout did not bound wall-clock: returned after {elapsed:.1f}s"
