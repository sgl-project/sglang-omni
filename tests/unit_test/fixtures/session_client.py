# SPDX-License-Identifier: Apache-2.0
"""A Session-API double for serving tests: it records what the server opens and appends, and the test pushes its outputs."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    ChunkPayload,
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    AUDIO,
    CONTROL,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    AudioPort,
    SessionPorts,
)

PORTS = SessionPorts(
    input=(AudioPort(PCM16, 16000),),
    output=AudioPort(PCM_F32, 24000),
    unit_ms=200,
    hot_settings=frozenset({"instructions", "sampling", "turn_detection"}),
    sampling_parameters=("temperature", "max_new_tokens_per_unit"),
    turn_detection_types=("server_vad",),
)


class FakeSessionClient:
    """Stands in for Client or the Coordinator; answer() is the outputs one append produces before its input_done."""

    def __init__(self, *, auto_done: bool = True) -> None:
        self.identity = SessionIdentity("session", 3)
        self.auto_done = auto_done
        self.opened: list[tuple[OmniRequest, SessionLimits | None, str | None]] = []
        self.appended: list[TimedChunk] = []
        self.controlled: list[tuple[TimedChunk, list[str] | None, bool]] = []
        self.outputs: asyncio.Queue[OutputChunk | Exception | None] = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.closed = threading.Event()

    async def open_session(
        self,
        request: OmniRequest,
        *,
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> SessionIdentity:
        self.loop = asyncio.get_running_loop()
        self.opened.append((request, limits, session_id))
        return self.identity

    async def append_session(
        self, session_identity: SessionIdentity, chunk: TimedChunk
    ) -> int:
        assert session_identity == self.identity
        self.appended.append(chunk)
        self.answer(chunk)
        if self.auto_done:
            self.done(chunk.seq)
        else:
            pass
        return chunk.seq

    def answer(self, chunk: TimedChunk) -> None:
        pass

    async def control_session(
        self,
        session_identity: SessionIdentity,
        event: TimedChunk,
        *,
        stages: list[str] | None = None,
        should_preempt: bool = False,
        is_ordered: bool = True,
    ) -> None:
        assert session_identity == self.identity
        self.controlled.append((event, stages, should_preempt))

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (output := await self.outputs.get()) is not None:
            if isinstance(output, Exception):
                raise output
            else:
                yield output

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.closed.set()
        self.outputs.put_nowait(None)

    def emit(
        self,
        modality: str,
        output_format: str | None,
        payload: ChunkPayload,
        *,
        input_seq: int = 0,
        turn: int | None = None,
    ) -> None:
        self.outputs.put_nowait(
            OutputChunk(
                self.identity,
                0,
                input_seq,
                modality,
                0,
                0,
                payload,
                output_format,
                turn=turn,
            )
        )

    def control(
        self, control_type: str, *, turn: int | None = None, **fields: object
    ) -> None:
        self.emit(CONTROL, SESSION_CONTROL, {"type": control_type, **fields}, turn=turn)

    def done(self, input_seq: int) -> None:
        self.outputs.put_nowait(
            OutputChunk(
                self.identity, 0, input_seq, AUDIO, 0, 0, None, kind="input_done"
            )
        )

    def fail(self, error: Exception) -> None:
        """Fail the output stream from a thread other than the server's loop."""
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.outputs.put_nowait, error)
