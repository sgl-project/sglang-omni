# SPDX-License-Identifier: Apache-2.0
"""Regression for review finding P2: stop without run_id was a no-op.

The original bug:

1. ``POST /start_request_profile`` with no ``run_id`` generates one of the
   form ``run_<timestamp>``.
2. ``POST /stop_request_profile`` with no ``run_id`` used ``"default"`` as
   the value to broadcast.
3. Each stage's stop handler only stopped when its active run id matched
   the message — so a default-value stop was silently dropped, and the
   recorder kept writing.

The fix makes ``ProfilerStopMessage.run_id`` optional. ``None`` is a
wildcard: stop whatever's active.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import get_recorder, reset_active_stage
from sglang_omni.proto.messages import ProfilerStopMessage


@pytest.fixture(autouse=True)
def _reset_recorder():
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    reset_active_stage(None)


def test_profiler_stop_message_supports_optional_run_id() -> None:
    """The proto must accept run_id=None and round-trip cleanly."""
    msg = ProfilerStopMessage()
    assert msg.run_id is None
    d = msg.to_dict()
    assert d == {"type": "profiler_stop", "run_id": None}
    round_trip = ProfilerStopMessage.from_dict(d)
    assert round_trip.run_id is None


def test_profiler_stop_message_legacy_run_id_round_trip() -> None:
    """Existing callers that DO pass a run id must continue to work."""
    msg = ProfilerStopMessage(run_id="abc")
    d = msg.to_dict()
    round_trip = ProfilerStopMessage.from_dict(d)
    assert round_trip.run_id == "abc"


def test_recorder_stop_with_none_run_id_unconditionally_stops(
    tmp_path: Path,
) -> None:
    """A wildcard (None) stop must close the active recorder even when
    the recorder was started with a specific run_id."""
    rec = get_recorder()
    rec.start(run_id="generated-run-12345", event_dir=str(tmp_path), stage="s")
    assert rec.is_active()
    rec.stop(run_id=None)
    assert not rec.is_active()


def test_recorder_stop_with_mismatched_run_id_is_still_noop(
    tmp_path: Path,
) -> None:
    """A non-None mismatched run_id must NOT close the recorder
    (preserved from earlier behavior)."""
    rec = get_recorder()
    rec.start(run_id="real-run", event_dir=str(tmp_path), stage="s")
    rec.stop(run_id="other-run")
    assert rec.is_active(), "mismatched non-None run_id should not stop"
    rec.stop()  # cleanup
