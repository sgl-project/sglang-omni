# SPDX-License-Identifier: Apache-2.0
"""Shutdown handoff at hook completion and owner unlock."""
from __future__ import annotations

import sys
import threading

import pytest

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionRef, TimedChunk
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler
from tests.unit_test.fixtures.session_pipeline import (
    command_metadata,
    compute_registered,
)


def command(op):
    metadata = command_metadata(
        op,
        SessionRef("session", epoch=int(op == "abort")),
        TimedChunk("audio", 0, 20, 0, b"pcm"),
    )
    return StagePayload(op, OmniRequest(None, metadata=metadata), {})


class Hooks(SessionHooks):
    def __init__(self, block=None):
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = []

    def open(self, ref, request):
        self.pause("open")
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


def run_command(scheduler, op, profile=None):
    errors = []

    def run():
        sys.setprofile(profile)
        try:
            compute_registered(scheduler, command(op))
        except BaseException as exc:
            errors.append(exc)
        finally:
            sys.setprofile(None)

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
    assert not scheduler.sessions
    scheduler.stop()
    assert len(hooks.closed) == 1


def test_stop_during_open_rejects_the_session():
    hooks = Hooks(block="open")
    scheduler = SessionScheduler(hooks)
    thread, errors = run_command(scheduler, "open")
    try:
        assert hooks.entered.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        hooks.release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert len(hooks.closed) == 1
    assert not scheduler.sessions


def test_stop_after_open_checks_before_owner_unlock():
    hooks = Hooks()
    scheduler = SessionScheduler(hooks)
    paused, release = threading.Event(), threading.Event()

    def profile(frame, event, arg):
        # Note (Junnan Li): No hook runs between the last closing check and the unlock.
        if (
            event == "c_call"
            and arg.__name__ == "release"
            and frame.f_code is SessionScheduler.open_session.__code__
        ):
            paused.set()
            assert release.wait(5)

    thread, errors = run_command(scheduler, "open", profile)
    try:
        assert paused.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(hooks.closed) == 1
    assert not scheduler.sessions
