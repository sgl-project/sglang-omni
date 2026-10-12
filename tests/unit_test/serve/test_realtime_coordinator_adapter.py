# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the coordinator-backed realtime interaction adapter."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Literal
from unittest.mock import Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import OutputEvent, TextDelta, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import Unit

SAMPLE_RATE = 16000
UNIT_SAMPLES = 320
STAGES = ["thinker"]
PROCESS_TIMEOUT_S = 5
ADAPTER_LOGGER_NAME = "sglang_omni.serve.realtime.adapters"
CONTEXT_EXHAUSTED_MESSAGE = (
    "context_exhausted: the session reached the thinker context length of 8192 tokens"
)
SESSION_CONFIG: SessionConfiguration = {
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}}}
}


class SessionCoordinator:
    """Answers each appended chunk with scripted text outputs, then input_done."""

    def __init__(self, replies: list[bytes], stale_reply: bytes | None = None) -> None:
        self.replies = replies
        self.stale_reply = stale_reply
        self.appended: list[TimedChunk] = []
        self.opened: list[tuple[OmniRequest, list[str], str | None]] = []
        self.is_closed = False
        self.outputs: asyncio.Queue[OutputChunk | None] = asyncio.Queue()

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None,
        session_id: str | None,
    ) -> SessionIdentity:
        self.opened.append((request, stages, session_id))
        return SessionIdentity(session_id or "session")

    async def append_session(
        self, session_identity: SessionIdentity, chunk: TimedChunk
    ) -> int:
        self.appended.append(chunk)
        if self.stale_reply is not None:
            self.put(session_identity, chunk.seq + 1, "data", self.stale_reply)
        else:
            pass
        for reply in self.replies:
            self.put(session_identity, chunk.seq, "data", reply)
        self.put(session_identity, chunk.seq, "input_done", None)
        return chunk.seq

    def put(
        self,
        session_identity: SessionIdentity,
        input_seq: int,
        kind: Literal["data", "input_done"],
        payload: bytes | None,
    ) -> None:
        self.outputs.put_nowait(
            OutputChunk(
                session_identity, 0, input_seq, "text", 0.0, 0.0, payload, kind=kind
            )
        )

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (output := await self.outputs.get()) is not None:
            yield output

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.is_closed = True
        self.outputs.put_nowait(None)


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
        stages=STAGES,
        request_builder=build_request,
        output_converter=convert_output,
        atomic_consumption=True,
        limits=limits,
    )


def build_unit(seq: int, *, is_eos: bool = False) -> Unit:
    return Unit(
        seq, seq * UNIT_SAMPLES, b"\1\0" * UNIT_SAMPLES, UNIT_SAMPLES, is_eos, ("text",)
    )


@pytest.mark.asyncio
async def test_open_routes_session_through_configured_stages() -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)

    await adapter.open("sess_1", SESSION_CONFIG, RecordingSink())
    await adapter.close()

    request, stages, session_id = coordinator.opened[0]
    assert request.params == {"instructions": None}
    assert (stages, session_id) == (STAGES, "sess_1")
    assert coordinator.is_closed is True


@pytest.mark.asyncio
async def test_process_sends_timed_chunk_and_consumes_whole_unit() -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)
    await adapter.open("sess_1", SESSION_CONFIG, RecordingSink())

    consumed = await adapter.process(build_unit(2, is_eos=True))
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

    with pytest.raises(RuntimeError, match="output budget"):
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
        coordinator.put(adapter.session_identity, 0, "data", b"reply")
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
