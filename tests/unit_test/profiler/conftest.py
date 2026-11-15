# SPDX-License-Identifier: Apache-2.0
"""Shared profiler test lifecycle fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sglang_omni.profiler.event_recorder import get_recorder, reset_active_stage


@pytest.fixture(autouse=True)
def reset_recorder() -> Iterator[None]:
    """Close the process-global recorder and clear stage context per test."""
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop(wait=True)
    else:
        pass
    recorder.wait_for_finalizations(timeout_seconds=5)
    reset_active_stage(None)
    yield
    if recorder.is_active():
        recorder.stop(wait=True)
    else:
        pass
    recorder.wait_for_finalizations(timeout_seconds=5)
