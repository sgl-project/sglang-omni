# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the coordinator-backed realtime interaction adapter."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.pipeline.types import SessionIdleTimeoutError, SessionOutputBudgetError
from sglang_omni.profiler.duplex_events import SessionUnitReadyObservation
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionLimits, TimedChunk
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import OutputEvent, TextDelta, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import OutputBudgetError, RuntimeLimits, Unit
from tests.unit_test.fixtures.session_client import FakeSessionClient

SAMPLE_RATE = 16000
UNIT_SAMPLES = 320
PROCESS_TIMEOUT_S = 5
ADAPTER_LOGGER_NAME = "sglang_omni.serve.realtime.adapters"
CONTEXT_EXHAUSTED_MESSAGE = (
    "context_exhausted: the session reached the thinker context length of 8192 tokens"
)
SESSION_CONFIG: SessionConfiguration = {
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}}}
}


class SessionCoordinator(FakeSessionClient):
    """Answers each appended chunk with scripted text outputs, then input_done."""

    def __init__(self, replies: list[bytes], stale_reply: bytes | None = None) -> None:
        super().__init__()
        self.replies = replies
        self.stale_reply = stale_reply
        self.ready_timestamps_ns: list[int | None] = []
        self.ready_run_ids: list[str | None] = []

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        self.ready_timestamps_ns.append(ready_timestamp_ns)
        self.ready_run_ids.append(ready_run_id)
        return await super().append_session(session_identity, chunk)

    def answer(self, chunk: TimedChunk) -> None:
        if self.stale_reply is not None:
            self.emit("text", None, self.stale_reply, input_seq=chunk.seq + 1)
        else:
            pass
        for reply in self.replies:
            self.emit("text", None, reply, input_seq=chunk.seq)


class RecordingSink:
    def __init__(self) -> None:
        self.published: list[tuple[OutputEvent, Unit | None]] = []
        self.has_published = asyncio.Event()

    async def __call__(self, event: OutputEvent, unit: Unit | None = None) -> None:
        self.published.append((event, unit))
        self.has_published.set()


def build_request(config: SessionConfiguration) -> OmniRequest:
    return OmniRequest(inputs=None, params={"instructions": config.get("instructions")})


def convert_output(output: OutputChunk) -> Iterable[OutputEvent]:
    assert isinstance(output.payload, bytes)
    return [TextDelta("resp", "item", output.payload.decode())]


def build_adapter(
    coordinator: SessionCoordinator,
    *,
    limits: SessionLimits | None = None,
) -> CoordinatorAdapter:
    return CoordinatorAdapter(
        Client(coordinator),
        request_builder=build_request,
        output_converter=convert_output,
        atomic_consumption=True,
        limits=limits,
    )


def build_unit(
    seq: int,
    *,
    is_eos: bool = False,
    ready_timestamp_ns: int | None = None,
    ready_run_id: str | None = None,
) -> Unit:
    return Unit(
        seq,
        seq * UNIT_SAMPLES,
        b"\1\0" * UNIT_SAMPLES,
        UNIT_SAMPLES,
        is_eos,
        ("text",),
        ready_observation=(
            SessionUnitReadyObservation(
                timestamp_ns=ready_timestamp_ns, run_id=ready_run_id
            )
            if ready_timestamp_ns is not None and ready_run_id is not None
            else None
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pipeline_idle_seconds", "runtime_idle_seconds", "expected_backup_seconds"),
    [(1, 60, 300), (900, 60, 900), (1, 300, 640)],
)
async def test_runtime_limits_set_pipeline_idle_backup(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_idle_seconds: float,
    runtime_idle_seconds: float,
    expected_backup_seconds: float,
) -> None:
    coordinator = SessionCoordinator([])
    open_session = AsyncMock(wraps=coordinator.open_session)
    monkeypatch.setattr(coordinator, "open_session", open_session)
    adapter = build_adapter(
        coordinator,
        limits=SessionLimits(idle_timeout_s=pipeline_idle_seconds, max_output_chunks=7),
    )
    adapter.set_limits(
        RuntimeLimits(idle_timeout_s=runtime_idle_seconds, cleanup_timeout_s=10)
    )
    try:
        await adapter.open("sess_1", SESSION_CONFIG, RecordingSink())
        session_limits = open_session.call_args.kwargs["limits"]
        assert isinstance(session_limits, SessionLimits)
        assert session_limits.idle_timeout_s == expected_backup_seconds
        assert session_limits.operation_timeout_s == 30
        assert session_limits.max_output_chunks == 7
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_open_opens_the_pipeline_session() -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)

    await adapter.open("sess_1", SESSION_CONFIG, RecordingSink())
    await adapter.close()

    request, _, session_id = coordinator.opened[0]
    assert request.params == {"instructions": None}
    assert session_id == "sess_1"
    assert coordinator.closed.is_set()


@pytest.mark.asyncio
async def test_process_sends_timed_chunk_and_consumes_whole_unit() -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)
    await adapter.open("sess_1", SESSION_CONFIG, RecordingSink())

    consumed = await adapter.process(
        build_unit(2, is_eos=True, ready_timestamp_ns=123, ready_run_id="run-1")
    )
    await adapter.close()

    assert consumed == UNIT_SAMPLES
    assert coordinator.appended == [
        TimedChunk(
            "audio",
            40.0,
            20.0,
            2,
            b"\1\0" * UNIT_SAMPLES,
            format="pcm16",
            eos=True,
        )
    ]
    assert coordinator.ready_timestamps_ns == [123]
    assert coordinator.ready_run_ids == ["run-1"]


@pytest.mark.asyncio
async def test_unit_outputs_are_published_with_unit_after_input_done() -> None:
    coordinator = SessionCoordinator([b"hel", b"lo"], stale_reply=b"stale")
    sink = RecordingSink()
    adapter = build_adapter(coordinator)
    await adapter.open("sess_1", SESSION_CONFIG, sink)
    unit = build_unit(0)

    await adapter.process(unit)
    await adapter.close()

    assert sink.published == [
        (TextDelta("resp", "item", "hel"), unit),
        (TextDelta("resp", "item", "lo"), unit),
    ]


@pytest.mark.asyncio
async def test_output_over_unit_budget_fails_the_unit() -> None:
    coordinator = SessionCoordinator([b"a", b"b"])
    sink = RecordingSink()
    adapter = build_adapter(coordinator, limits=SessionLimits(max_output_chunks=1))
    await adapter.open("sess_1", SESSION_CONFIG, sink)

    with pytest.raises(OutputBudgetError, match="output budget"):
        await adapter.process(build_unit(0))
    await adapter.close()

    assert sink.published == []


@pytest.mark.asyncio
async def test_unexpected_output_stream_end_fails_the_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = SessionCoordinator([])
    sink = RecordingSink()
    adapter = build_adapter(coordinator)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER_NAME):
        await adapter.open("sess_1", SESSION_CONFIG, sink)
        coordinator.outputs.put_nowait(None)
        await sink.has_published.wait()

        with pytest.raises(RuntimeError, match="output stream closed"):
            await asyncio.wait_for(adapter.process(build_unit(0)), PROCESS_TIMEOUT_S)
        await adapter.close()

    event, _ = sink.published[0]
    assert isinstance(event, TurnFailure)
    adapter_records = [
        record for record in caplog.records if record.name == ADAPTER_LOGGER_NAME
    ]
    assert len(adapter_records) == 1
    assert adapter_records[0].exc_info is not None


@pytest.mark.asyncio
async def test_context_exhaustion_without_pending_unit_fails_the_session_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = SessionCoordinator([])
    sink = RecordingSink()
    adapter = build_adapter(coordinator)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER_NAME):
        await adapter.open("sess_1", SESSION_CONFIG, sink)
        adapter.active_unit = build_unit(0)
        adapter.output_converter = Mock(
            side_effect=RuntimeError(CONTEXT_EXHAUSTED_MESSAGE)
        )
        coordinator.emit("text", None, b"reply")
        await asyncio.wait_for(sink.has_published.wait(), PROCESS_TIMEOUT_S)
        await adapter.close()

    assert sink.published == [
        (
            TurnFailure("server_error", "context_exhausted", CONTEXT_EXHAUSTED_MESSAGE),
            None,
        )
    ]
    assert not any(record.name == ADAPTER_LOGGER_NAME for record in caplog.records)


@pytest.mark.asyncio
async def test_context_exhaustion_with_pending_unit_fails_the_unit_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = SessionCoordinator([b"reply"])
    sink = RecordingSink()
    adapter = build_adapter(coordinator)
    adapter.output_converter = Mock(side_effect=RuntimeError(CONTEXT_EXHAUSTED_MESSAGE))
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER_NAME):
        await adapter.open("sess_1", SESSION_CONFIG, sink)

        with pytest.raises(RuntimeError, match="context_exhausted"):
            await asyncio.wait_for(adapter.process(build_unit(0)), PROCESS_TIMEOUT_S)
        await adapter.close()

    assert sink.published == []
    assert not any(record.name == ADAPTER_LOGGER_NAME for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (SessionIdleTimeoutError(300), "session_idle_timeout"),
        (SessionOutputBudgetError(), "output_budget_exhausted"),
    ],
)
async def test_coordinator_failure_without_pending_unit_fails_the_session_without_logging(
    error: Exception, code: str, caplog: pytest.LogCaptureFixture
) -> None:
    coordinator = SessionCoordinator([])
    sink = RecordingSink()
    adapter = build_adapter(coordinator)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER_NAME):
        await adapter.open("sess_1", SESSION_CONFIG, sink)
        coordinator.outputs.put_nowait(error)
        await asyncio.wait_for(sink.has_published.wait(), PROCESS_TIMEOUT_S)
        await adapter.close()

    assert sink.published == [(TurnFailure("server_error", code, str(error)), None)]
    assert not any(record.name == ADAPTER_LOGGER_NAME for record in caplog.records)
