# SPDX-License-Identifier: Apache-2.0
"""Shutdown handoff at hook completion and owner unlock."""
from __future__ import annotations

import sys
import threading

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from tests.unit_test.fixtures.session_pipeline import (
    compute_registered,
    operation_metadata,
)


def operation_payload(operation):
    metadata = operation_metadata(
        operation,
        SessionIdentity("session"),
        TimedChunk("audio", 0, 20, 0, b"pcm"),
    )
    return StagePayload(operation, OmniRequest(None, metadata=metadata), {})


class Hooks(SessionHooks):
    def __init__(self, block: str | None = None) -> None:
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.opened: set[SessionIdentity] = set()
        self.closed: list[SessionIdentity] = []

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.pause("open")
        self.opened.add(session_identity)

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        assert context.session_identity in self.opened
        self.pause("append")
        return payload

    def pause(self, operation: str) -> None:
        if self.block == operation:
            self.entered.set()
            assert self.release.wait(5)

    def close(self, session_identity: SessionIdentity) -> None:
        self.opened.remove(session_identity)
        self.closed.append(session_identity)


def run_operation(scheduler, operation, profile=None):
    errors = []

    def run():
        sys.setprofile(profile)
        try:
            compute_registered(scheduler, operation_payload(operation))
        except BaseException as exc:
            errors.append(exc)
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def test_stop_hands_cleanup_to_active_hook_completion():
    hooks = Hooks(block="append")
    scheduler = SessionScheduler(hooks)
    compute_registered(scheduler, operation_payload("open"))
    thread, errors = run_operation(scheduler, "append")
    try:
        assert hooks.entered.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        hooks.release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(hooks.closed) == 1
    assert not scheduler.open_sessions
    scheduler.stop()
    assert len(hooks.closed) == 1


def test_stop_during_open_rejects_the_session():
    hooks = Hooks(block="open")
    scheduler = SessionScheduler(hooks)
    thread, errors = run_operation(scheduler, "open")
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
    assert not scheduler.open_sessions


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

    thread, errors = run_operation(scheduler, "open", profile)
    try:
        assert paused.wait(5)
        scheduler.stop()
        assert not hooks.closed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(hooks.closed) == 1
    assert not scheduler.open_sessions
