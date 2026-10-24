# SPDX-License-Identifier: Apache-2.0
"""Run one streaming component as one CPU Session stage."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace

from sglang_omni.models.streaming_cascade.base import control_chunk
from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    CONFIGURE,
    CONFIGURED,
    PCM16,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    control_type,
)
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_CONCURRENCY,
    NODE_STAGE_MAX_OPEN_SESSIONS,
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)

INPUT_BYTES_PER_MS = PCM16_BYTES_PER_SAMPLE * SESSION_INPUT_SAMPLE_RATE_HZ / 1000


class ComponentSession:
    """A component's input, output and lifecycle; all edges belong to the coordinator."""

    def __init__(self, name: str, settings: ComponentGraphConfig) -> None:
        self.name = name
        self.settings = settings
        self.component = COMPONENTS.build(settings.node(name).component)
        self.queue: asyncio.Queue[tuple[TimedChunk, asyncio.Future[None]] | None] = (
            asyncio.Queue(maxsize=1)
        )
        self.task: asyncio.Task[None] | None = None
        self.emit: ChunkEmitter | None = None
        self.fail: Callable[[BaseException], None] | None = None
        self.failure: BaseException | None = None
        self.append_emit: ChunkEmitter | None = None
        self.is_speaking = False
        self.pending_settings: ComponentGraphConfig | None = None
        self.is_closing = False
        self.is_input_ended = False

    def start(self) -> None:
        self.task = asyncio.create_task(
            self.component.process(self.inputs(), self.output)
        )
        self.task.add_done_callback(self.finished)

    def finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            self.failure = task.exception()
            if self.fail is not None and not self.is_closing:
                self.fail(self.failure)
            else:
                pass
        elif not task.cancelled() and not self.is_closing and not self.is_input_ended:
            self.failure = RuntimeError(f"component {self.name!r} stopped before EOS")
            if self.fail is not None:
                self.fail(self.failure)
            else:
                pass
        else:
            pass

    def output(self, chunk: TimedChunk) -> None:
        emit = self.append_emit or self.emit
        if emit is None:
            raise RuntimeError("component has no output reader")
        else:
            emit(chunk)

    async def inputs(self) -> AsyncIterator[TimedChunk]:
        while (item := await self.queue.get()) is not None:
            chunk, consumed = item
            yield chunk
            consumed.set_result(None)

    async def append(self, chunk: TimedChunk, emit: ChunkEmitter) -> None:
        assert self.task is not None
        if self.task.done():
            self.task.result()
            return
        else:
            pass
        if chunk.format == PCM16 and (
            not isinstance(chunk.payload, bytes)
            or len(chunk.payload) % 2
            or abs(len(chunk.payload) / INPUT_BYTES_PER_MS - chunk.duration_ms) > 0.001
        ):
            raise ValueError("PCM16 byte length must match duration_ms")
        else:
            pass
        self.append_emit = emit
        try:
            await self.apply_pending()
            event = control_type(chunk)
            if event == CONFIGURE:
                await self.configure(chunk)
                return
            elif event == SPEECH_STARTED or (chunk.format == PCM16 and chunk.payload):
                self.is_speaking = True
            elif event in (SPEECH_STOPPED, COMMIT):
                self.is_speaking = False
            else:
                pass
            consumed = asyncio.get_running_loop().create_future()
            await self.queue.put((replace(chunk, eos=False), consumed))
            await asyncio.wait(
                {consumed, self.task}, return_when=asyncio.FIRST_COMPLETED
            )
            if self.task.done():
                self.task.result()
                raise RuntimeError("component stopped while handling input")
            elif chunk.eos:
                self.is_input_ended = True
                await self.queue.put(None)
                await self.task
            else:
                pass
            await self.apply_pending()
        finally:
            self.append_emit = None

    async def configure(self, chunk: TimedChunk) -> None:
        settings = (self.pending_settings or self.settings).apply(
            {k: v for k, v in chunk.payload.items() if k != "type"}
        )
        current = self.settings.node(self.name).component
        updated = settings.node(self.name).component
        if current != updated and self.is_speaking:
            raise ValueError(
                f"cannot replace {self.name!r} while speech is in progress"
            )
        else:
            self.pending_settings = settings
        await self.apply_pending()
        self.output(control_chunk(CONFIGURED, nodes={self.name: updated.kind}))

    async def apply_pending(self) -> None:
        if self.pending_settings is None or not self.component.can_replace():
            return
        else:
            settings, self.pending_settings = self.pending_settings, None
        if (
            settings.node(self.name).component
            != self.settings.node(self.name).component
        ):
            replacement = COMPONENTS.build(settings.node(self.name).component)
            replacement.adopt(self.component)
            previous, self.component = self.component, replacement
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            await previous.close()
            self.start()
        else:
            pass
        self.settings = settings

    async def control(self, event: TimedChunk) -> None:
        await self.component.control(event)
        await self.apply_pending()

    async def close(self) -> None:
        self.is_closing = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        else:
            pass
        await self.component.close()


class ComponentHooks(SessionHooks):
    """Keep streaming work on a persistent I/O loop while stage controls remain callable."""

    def __init__(
        self, name: str, settings: ComponentGraphConfig, timeout_s: float
    ) -> None:
        self.name, self.settings, self.timeout_s = name, settings, timeout_s
        self.sessions: dict[SessionIdentity, ComponentSession] = {}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name=f"{name}-io", daemon=True
        )
        self.thread.start()

    def call(self, coroutine: Coroutine) -> object:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(self.timeout_s)
        except FutureTimeout:
            future.cancel()
            raise

    def warm_up_serving_thread(self) -> None:
        config = self.settings.node(self.name).component
        probe = COMPONENTS.kind(config.kind).probe
        if probe is not None:
            self.call(probe(config))
        else:
            pass

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        async def create() -> None:
            self.sessions[session_identity] = ComponentSession(
                self.name, self.settings.apply(dict(request.params))
            )

        self.call(create())

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        session = self.sessions[session_identity]
        session.emit = emit
        self.loop.call_soon_threadsafe(session.start)

    def attach_output_error_handler(
        self, session_identity: SessionIdentity, fail: Callable[[BaseException], None]
    ) -> None:
        self.sessions[session_identity].fail = fail

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.sessions[context.session_identity]
        future = asyncio.run_coroutine_threadsafe(
            session.append(chunk, context.emit), self.loop
        )
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                future.result(0.02)
                break
            except FutureTimeout:
                if future.done():
                    raise
                elif context.cancelled.is_set():
                    future.cancel()
                    raise RuntimeError("component append cancelled")
                elif time.monotonic() >= deadline:
                    future.cancel()
                    raise TimeoutError(f"component {self.name!r} append timed out")
                else:
                    pass
        payload.data = {"accepted": True}
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        self.call(self.sessions[session_identity].control(event))

    def close(self, session_identity: SessionIdentity) -> None:
        session = self.sessions.pop(session_identity, None)
        if session is not None:
            self.call(session.close())
        else:
            pass

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage(
            bytes=self.sessions[session_identity].component.state_bytes()
        )


def create_scheduler(
    name: str, settings: dict[str, object], timeout_s: float
) -> SessionScheduler:
    return SessionScheduler(
        ComponentHooks(name, ComponentGraphConfig.model_validate(settings), timeout_s),
        max_open_sessions=NODE_STAGE_MAX_OPEN_SESSIONS,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )
