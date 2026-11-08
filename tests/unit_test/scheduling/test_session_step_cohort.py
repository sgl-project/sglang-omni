# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    STEP_COHORT_KEY,
    Cohort,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.session import SessionScheduler
from tests.unit_test.scheduling.test_session_batch import RecordingHooks, opens
from tests.unit_test.scheduling.test_session_gather_window import (
    ScriptedInbox,
    run_next_batch,
)

NS_PER_SECOND = 1_000_000_000
STEP_WAIT_S = 0.01


def step(index: int, size: int, end_s: float) -> Cohort:
    return Cohort(
        index=index, size=size, due_ns=round((end_s + STEP_WAIT_S) * NS_PER_SECOND)
    )


def append_payload(session_id: str, cohort: Cohort | None) -> StagePayload:
    request_id = f"{session_id}0"
    session_operation = SessionOperation(
        operation="append",
        session_identity=SessionIdentity(session_id),
        stages=("speech",),
        chunk=TimedChunk("audio", 0, 80, 0, b"x"),
    )
    metadata = {SESSION_METADATA_KEY: session_operation.to_dict()}
    if cohort is not None:
        metadata[STEP_COHORT_KEY] = cohort.to_dict()
    else:
        pass
    return StagePayload(request_id, OmniRequest(None, metadata=metadata), {})


def scripted_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    arrivals: list[tuple[str, float, Cohort | None]],
) -> tuple[SessionScheduler, ScriptedInbox]:
    scheduler = SessionScheduler(RecordingHooks())
    inbox = ScriptedInbox()
    for queued in opens(*[session_id for session_id, _, _ in arrivals]):
        scheduler.register_operation(queued)
        scheduler.compute(queued.data)
    for session_id, seconds, cohort in arrivals:
        payload = append_payload(session_id, cohort)
        queued = IncomingMessage(payload.request_id, "new_request", payload)
        scheduler.register_operation(queued)
        inbox.arrivals.append((seconds, queued))
    scheduler.inbox = inbox
    monkeypatch.setattr(time, "monotonic", lambda: inbox.now_seconds)
    monkeypatch.setattr(
        time, "monotonic_ns", lambda: round(inbox.now_seconds * NS_PER_SECOND)
    )
    return scheduler, inbox


def test_a_steps_results_run_in_one_batch_once_the_last_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = step(7, 3, end_s=1.0)
    scheduler, inbox = scripted_scheduler(
        monkeypatch, [("a", 1.0, cohort), ("b", 1.001, cohort), ("c", 1.004, cohort)]
    )
    assert run_next_batch(scheduler, inbox) == ["a0", "b0", "c0"]
    assert inbox.now_seconds == pytest.approx(1.004)


def test_a_steps_batch_runs_when_its_wait_is_over_without_the_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = step(7, 3, end_s=1.0)
    scheduler, inbox = scripted_scheduler(
        monkeypatch, [("a", 1.0, cohort), ("b", 1.001, cohort), ("c", 1.05, cohort)]
    )
    assert run_next_batch(scheduler, inbox) == ["a0", "b0"]
    assert inbox.now_seconds == pytest.approx(1.0 + STEP_WAIT_S)
    assert run_next_batch(scheduler, inbox) == ["c0"]


def test_untagged_results_are_not_held(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, inbox = scripted_scheduler(
        monkeypatch, [("a", 1.0, None), ("b", 1.001, None)]
    )
    assert run_next_batch(scheduler, inbox) == ["a0"]
    assert inbox.timeouts == []


def stub_lm(lm_step_ms: float | None) -> SimpleNamespace:
    pytest.importorskip("sglang")
    from sglang_omni.scheduling.omni_scheduler import StepOutbox

    return SimpleNamespace(outbox=StepOutbox(), lm_step_ms=lm_step_ms)


def test_step_outbox_holds_only_the_holding_threads_puts() -> None:
    lm = stub_lm(None)
    held = lm.outbox.hold()
    lm.outbox.put(OutgoingMessage(request_id="mine", type="result"))
    other = threading.Thread(
        target=lambda: lm.outbox.put(OutgoingMessage(request_id="other", type="stream"))
    )
    other.start()
    other.join()
    lm.outbox.release()
    assert [message.request_id for message in held] == ["mine"]
    assert lm.outbox.get_nowait().request_id == "other"
    with pytest.raises(queue.Empty):
        lm.outbox.get_nowait()


def test_a_steps_unit_results_are_tagged_with_the_step_and_published_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    lm = stub_lm(lm_step_ms=10.0)
    monkeypatch.setattr(time, "monotonic_ns", lambda: 5 * NS_PER_SECOND)
    messages = [
        OutgoingMessage(request_id="a0", type="result", data=append_payload("a", None)),
        OutgoingMessage(request_id="x", type="error", data=ValueError("x")),
        OutgoingMessage(request_id="b0", type="result", data=append_payload("b", None)),
    ]
    OmniScheduler.publish_step_results(lm, SimpleNamespace(forward_iter=42), messages)
    published = [lm.outbox.get_nowait() for _ in range(3)]
    assert [message.request_id for message in published] == ["a0", "x", "b0"]
    expected = Cohort(index=42, size=2, due_ns=5 * NS_PER_SECOND + 10_000_000).to_dict()
    for message in (published[0], published[2]):
        assert message.data.request.metadata[STEP_COHORT_KEY] == expected
