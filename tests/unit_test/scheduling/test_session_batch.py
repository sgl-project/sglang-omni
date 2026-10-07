# SPDX-License-Identifier: Apache-2.0
"""Batched appends across sessions on one stage scheduler."""

import threading
from typing import Literal

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.session import (
    BatchedSessionHooks,
    SessionAppend,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)

OUTBOX_TIMEOUT_S = 5
STATE_BYTES_PER_UNIT = 100


class SequentialHooks(SessionHooks):
    def __init__(self) -> None:
        self.opened: set[SessionIdentity] = set()
        self.calls: list[list[str]] = []
        self.units_by_session: dict[SessionIdentity, int] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.opened.add(session_identity)
        self.units_by_session[session_identity] = 0

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        self.calls.append([payload.request_id])
        payload.data = {"seq": chunk.seq}
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.opened.discard(session_identity)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage(
            bytes=STATE_BYTES_PER_UNIT * self.units_by_session.get(session_identity, 0)
        )


class RecordingHooks(SequentialHooks, BatchedSessionHooks):
    def __init__(
        self, fail_batch: bool = False, growing_session: str | None = None
    ) -> None:
        super().__init__()
        self.fail_batch = fail_batch
        self.growing_session = growing_session

    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        self.calls.append([append.payload.request_id for append in appends])
        if self.fail_batch and len(appends) > 1:
            raise RuntimeError("batch failed")
        else:
            pass
        for append in appends:
            session_identity = append.context.session_identity
            if session_identity.id == self.growing_session:
                self.units_by_session[session_identity] += 1
            else:
                pass
            append.context.emit(append.chunk)
            append.payload.data = {"seq": append.chunk.seq}
        return [append.payload for append in appends]


def message(
    request_id: str,
    operation: Literal["open", "append", "close"],
    session_id: str,
    seq: int = 0,
) -> IncomingMessage:
    session_operation = SessionOperation(
        operation=operation,
        session_identity=SessionIdentity(session_id),
        stages=("speech",),
        chunk=(
            TimedChunk("audio", seq * 1000, 1000, seq, b"x")
            if operation == "append"
            else None
        ),
    )
    request = OmniRequest(
        None, metadata={SESSION_METADATA_KEY: session_operation.to_dict()}
    )
    return IncomingMessage(
        request_id, "new_request", StagePayload(request_id, request, {})
    )


def run_backlog(
    scheduler: SessionScheduler,
    messages: list[IncomingMessage],
    aborted: tuple[str, ...] = (),
) -> dict[str, OutgoingMessage]:
    """Queue every message before the worker starts, so they arrive as one backlog."""
    for queued in messages:
        scheduler.inbox.put(queued)
    for request_id in aborted:
        scheduler.abort(request_id)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    finals: dict[str, OutgoingMessage] = {}
    try:
        while len(finals) < len(messages) - len(aborted):
            output = scheduler.outbox.get(timeout=OUTBOX_TIMEOUT_S)
            if output.type != "stream":
                finals[output.request_id] = output
            else:
                pass
    finally:
        scheduler.stop()
        worker.join(timeout=OUTBOX_TIMEOUT_S)
    return finals


def opens(*session_ids: str) -> list[IncomingMessage]:
    return [message(f"open-{sid}", "open", sid) for sid in session_ids]


def test_backlog_appends_of_distinct_sessions_share_one_hook_call():
    hooks = RecordingHooks()
    scheduler = SessionScheduler(hooks)
    outputs = run_backlog(
        scheduler,
        opens("a", "b", "c")
        + [
            message("a0", "append", "a", 0),
            message("b0", "append", "b", 0),
            message("a1", "append", "a", 1),
            message("c0", "append", "c", 0),
        ],
    )
    assert hooks.calls == [["a0", "b0", "c0"], ["a1"]]
    appended = {rid: out for rid, out in outputs.items() if not rid.startswith("open")}
    assert {rid: out.data.data["seq"] for rid, out in appended.items()} == {
        "a0": 0,
        "b0": 0,
        "a1": 1,
        "c0": 0,
    }
    assert all(out.type == "result" for out in outputs.values())


def test_open_and_close_run_alone_in_arrival_order():
    hooks = RecordingHooks()
    scheduler = SessionScheduler(hooks)
    outputs = run_backlog(
        scheduler,
        opens("a")
        + [
            message("a0", "append", "a", 0),
            message("open-b", "open", "b"),
            message("b0", "append", "b", 0),
            message("close-a", "close", "a"),
            message("a1", "append", "a", 1),
        ],
    )
    assert hooks.calls == [["a0", "b0"]]
    assert outputs["open-b"].type == "result"
    assert outputs["close-a"].data.data == {"closed": True}
    assert outputs["a1"].type == "error"


def test_hooks_without_a_batch_hook_run_one_unit_per_call():
    hooks = SequentialHooks()
    scheduler = SessionScheduler(hooks, max_concurrency=1)
    run_backlog(
        scheduler,
        opens("a", "b")
        + [message("a0", "append", "a", 0), message("b0", "append", "b", 0)],
    )
    assert hooks.calls == [["a0"], ["b0"]]


def test_session_over_its_state_budget_fails_only_its_own_unit():
    hooks = RecordingHooks(growing_session="b")
    scheduler = SessionScheduler(
        hooks, max_state_bytes_per_session=STATE_BYTES_PER_UNIT // 2
    )
    outputs = run_backlog(
        scheduler,
        opens("a", "b")
        + [
            message("a0", "append", "a", 0),
            message("b0", "append", "b", 0),
            message("a1", "append", "a", 1),
            message("b1", "append", "b", 1),
        ],
    )
    assert hooks.calls == [["a0", "b0"], ["a1"]]
    assert isinstance(outputs["b0"].data, QueueFullError)
    assert outputs["b1"].type == "error"
    assert outputs["a0"].type == outputs["a1"].type == "result"


def test_failed_batch_ends_only_its_sessions():
    hooks = RecordingHooks(fail_batch=True)
    scheduler = SessionScheduler(hooks)
    outputs = run_backlog(
        scheduler,
        opens("a", "b")
        + [
            message("a0", "append", "a", 0),
            message("b0", "append", "b", 0),
            message("open-c", "open", "c"),
            message("a1", "append", "a", 1),
        ],
    )
    assert outputs["a0"].type == outputs["b0"].type == "error"
    assert isinstance(outputs["a0"].data, RuntimeError)
    assert outputs["open-c"].type == "result"
    # note (Junnan Li): a1 was queued behind the failed a0, so it must not run on a's state.
    assert outputs["a1"].type == "error"
    assert hooks.calls == [["a0", "b0"]]


def test_aborted_unit_leaves_the_batch_without_blocking_its_session():
    hooks = RecordingHooks()
    scheduler = SessionScheduler(hooks)
    outputs = run_backlog(
        scheduler,
        opens("a", "b")
        + [
            message("a0", "append", "a", 0),
            message("b0", "append", "b", 0),
            message("b1", "append", "b", 1),
        ],
        aborted=("b0",),
    )
    assert set(outputs) == {"open-a", "open-b", "a0", "b1"}
    assert hooks.calls == [["a0", "b1"]]
