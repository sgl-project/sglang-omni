# SPDX-License-Identifier: Apache-2.0
"""Gather window of batched session hooks: an idle stage waits briefly for more appends."""

import asyncio
import queue
import time

import pytest

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.session import SessionAppend, SessionScheduler
from tests.unit_test.scheduling.test_session_batch import (
    RecordingHooks,
    SequentialHooks,
    message,
    opens,
)

WINDOW_SECONDS = 0.03
HOOK_SECONDS = 0.01


class WindowHooks(RecordingHooks):
    gather_window_ms = WINDOW_SECONDS * 1000


class ScriptedInbox:
    """Inbox whose messages arrive at scripted times on a clock the test advances."""

    def __init__(self) -> None:
        self.now_seconds = 0.0
        self.arrivals: list[tuple[float, IncomingMessage]] = []
        self.timeouts: list[float] = []

    def get(self, timeout: float) -> IncomingMessage:
        self.timeouts.append(timeout)
        if self.arrivals and self.arrivals[0][0] <= self.now_seconds + timeout:
            self.now_seconds = max(self.now_seconds, self.arrivals[0][0])
            return self.arrivals.pop(0)[1]
        else:
            self.now_seconds += timeout
            raise queue.Empty

    def get_nowait(self) -> IncomingMessage:
        if self.empty():
            raise queue.Empty
        else:
            return self.arrivals.pop(0)[1]

    def empty(self) -> bool:
        return not self.arrivals or self.arrivals[0][0] > self.now_seconds


class TimedWindowHooks(WindowHooks):
    """Each batched call takes HOOK_SECONDS on the scripted clock."""

    def __init__(self, inbox: ScriptedInbox) -> None:
        super().__init__()
        self.inbox = inbox

    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        self.inbox.now_seconds += HOOK_SECONDS
        return super().append_batch(appends)


def script_arrivals(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: SessionScheduler,
    inbox: ScriptedInbox,
    session_ids: str,
    arrivals: list[tuple[str, float]],
) -> None:
    """Open the sessions, then make the scripted arrivals the scheduler's inbox and clock."""
    for queued in opens(*session_ids):
        scheduler.register_operation(queued)
        scheduler.compute(queued.data)
    for request_id, seconds in arrivals:
        operation, _, session_id = request_id.rpartition("-")
        if operation:
            queued = message(request_id, operation, session_id)
        else:
            queued = message(request_id, "append", request_id[0])
        scheduler.register_operation(queued)
        inbox.arrivals.append((seconds, queued))
    scheduler.inbox = inbox
    monkeypatch.setattr(time, "monotonic", lambda: inbox.now_seconds)


def run_next_batch(scheduler: SessionScheduler, inbox: ScriptedInbox) -> list[str]:
    """Move the clock to the next arrival, then collect a batch and run it through run_batch."""
    inbox.now_seconds = max(inbox.now_seconds, inbox.arrivals[0][0])
    batch = scheduler.collect_batch(inbox.get_nowait())
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_batch(batch, loop)
    finally:
        loop.close()
    return [queued.request_id for queued in batch]


def run_batches(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: SessionScheduler,
    session_ids: str,
    arrivals: list[tuple[str, float]],
) -> list[list[str]]:
    inbox = ScriptedInbox()
    script_arrivals(monkeypatch, scheduler, inbox, session_ids, arrivals)
    batches: list[list[str]] = []
    while inbox.arrivals:
        batches.append(run_next_batch(scheduler, inbox))
    return batches


@pytest.mark.parametrize(
    "hooks_class, session_ids, arrivals, batches, waits",
    [
        (WindowHooks, "abc", [("a0", 0), ("b0", 0.005)], [["a0", "b0"]], 2),
        (WindowHooks, "abc", [("a0", 0), ("b0", 0.05)], [["a0"], ["b0"]], 2),
        (WindowHooks, "ab", [("open-c", 0), ("a0", 0)], [["open-c", "a0"]], 0),
        (WindowHooks, "ab", [("close-b", 0), ("a0", 0)], [["close-b", "a0"]], 0),
        (WindowHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0", "b0"]], 1),
        (SequentialHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0"], ["b0"]], 0),
        (RecordingHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0"], ["b0"]], 0),
    ],
    ids=[
        "appends-inside-the-window-share-one-call",
        "append-after-the-window-runs-in-a-second-call",
        "open-is-not-held",
        "close-is-not-held",
        "wait-ends-once-every-open-session-has-an-append",
        "unbatched-hooks-never-wait",
        "zero-window-never-waits",
    ],
)
def test_idle_stage_gathers_appends_within_the_window(
    monkeypatch, hooks_class, session_ids, arrivals, batches, waits
):
    scheduler = SessionScheduler(hooks_class(), max_concurrency=1)
    assert run_batches(monkeypatch, scheduler, session_ids, arrivals) == batches
    assert len(scheduler.inbox.timeouts) == waits


def test_work_arriving_during_a_hook_skips_the_next_wait_until_the_queue_drains(
    monkeypatch,
):
    inbox = ScriptedInbox()
    scheduler = SessionScheduler(TimedWindowHooks(inbox))
    arrivals = [("a0", 0), ("b0", WINDOW_SECONDS + HOOK_SECONDS / 2), ("c0", 0.1)]
    script_arrivals(monkeypatch, scheduler, inbox, "abc", arrivals)

    assert run_next_batch(scheduler, inbox) == ["a0"]
    assert scheduler.is_backlogged
    assert len(inbox.timeouts) == 1

    assert run_next_batch(scheduler, inbox) == ["b0"]
    assert not scheduler.is_backlogged
    assert len(inbox.timeouts) == 1

    assert run_next_batch(scheduler, inbox) == ["c0"]
    assert len(inbox.timeouts) == 2
