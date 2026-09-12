# SPDX-License-Identifier: Apache-2.0
"""Shutdown handoff at hook completion and owner unlock."""
from __future__ import annotations

import inspect
import sys
import threading
from dataclasses import asdict

import pytest

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY, SessionRef, TimedChunk
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler
from tests.unit_test.fixtures.session_pipeline import compute_registered


def command(op):
    data = {
        "op": op,
        "ref": asdict(SessionRef("session", epoch=int(op == "abort"))),
        "chunk": asdict(TimedChunk("audio", 0, 20, 0, b"pcm")),
        "output_limits": {"chunks": 4, "bytes": 1024},
    }
    return StagePayload(
        op, OmniRequest(None, metadata={SESSION_METADATA_KEY: data}), {}
    )


class Hooks(SessionHooks):
    def __init__(self, block=None):
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = []

    def open(self, ref, request):
        return object()

    def append(self, state, chunk, payload, context):
        self.pause("append")
        return payload

    def abort(self, state, ref):
        self.pause("abort")

    def pause(self, op):
        if self.block == op:
            self.entered.set()
            assert self.release.wait(5)

    def close(self, state):
        self.closed.append(state)


def run_command(scheduler, op, trace=None):
    errors = []

    def run():
        if trace is not None:
            sys.settrace(trace)
        try:
            compute_registered(scheduler, command(op))
        except BaseException as exc:
            errors.append(exc)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


@pytest.mark.parametrize("op", ["abort", "append"])
def test_stop_hands_cleanup_to_active_hook_completion(op):
    hooks = Hooks(block=op)
    scheduler = SessionScheduler(hooks)
    compute_registered(scheduler, command("open"))
    thread, errors = run_command(scheduler, op)
    try:
        assert hooks.entered.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        hooks.release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(hooks.closed) == 1
    assert not scheduler._sessions
    scheduler.stop()
    assert len(hooks.closed) == 1


@pytest.mark.parametrize("op", ["open", "append"])
def test_stop_after_last_hook_check_before_owner_unlock(op):
    hooks = Hooks()
    scheduler = SessionScheduler(hooks)
    if op != "open":
        compute_registered(scheduler, command("open"))
    lines, first_line = inspect.getsourcelines(scheduler._compute_session)
    if op == "open":
        pause_line = max(
            first_line + i
            for i, line in enumerate(lines)
            if line.strip() == "session.lock.release()"
        )
    else:
        pause_line = next(
            first_line + i
            for i, line in enumerate(lines)
            if line.strip() == "with session.lock:"
        )
    paused, release = threading.Event(), threading.Event()

    def trace(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is scheduler._compute_session.__func__.__code__
            and frame.f_lineno == pause_line
            and (op == "open" or "result" in frame.f_locals)
        ):
            paused.set()
            assert release.wait(5)
        return trace

    thread, errors = run_command(scheduler, op, trace)
    try:
        assert paused.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(hooks.closed) == 1
    assert not scheduler._sessions
    scheduler.stop()
    assert len(hooks.closed) == 1
