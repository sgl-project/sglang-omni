"""Interaction bridges. No transport parsing or model-name dispatch."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

from sglang_omni.client.client import Client
from sglang_omni.proto.session import SessionLimits, TimedChunk
from sglang_omni.serve.realtime.events import SessionConfig
from sglang_omni.serve.realtime.output import ContextLimitError, TurnFailure
from sglang_omni.serve.realtime.runtime import (
    InteractionAdapter,
    ProtocolError,
    RuntimeLimits,
)
from sglang_omni.serve.realtime.semantic_vad import SemanticEOUModel
from sglang_omni.serve.realtime.session import TurnBasedSession, TurnConfigurationError
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks


class CoordinatorAdapter(InteractionAdapter):
    """Publish output after unit success; cancel preserves its consumption receipt."""

    def __init__(
        self,
        client,
        *,
        stages: list[str],
        request_builder: Callable,
        output_converter: Callable,
        input_rate: int = 16000,
        atomic_consumption: bool = False,
        limits: SessionLimits | None = None,
    ):
        if not atomic_consumption:
            raise ValueError("a producer atomic-consumption contract is required")
        self.client = client
        self.stages = stages
        self.request_builder = request_builder
        self.convert = output_converter
        self.rate = input_rate
        self.limits = limits or SessionLimits()
        self.local_cleanup_timeout = self.limits.command_timeout_s
        self.ref = None
        self.reader = None
        self.active = None
        self.future = None
        self.buffer = []
        self.buffer_bytes = 0
        self.emit = None
        self.epoch = 0
        self.reader_error = None
        self.closing = False

    def set_limits(self, limits):
        self.local_cleanup_timeout = limits.cleanup_timeout_s

    async def open(self, session_id, config, emit):
        self.emit = emit
        self.ref = await self.client.open_session(
            self.request_builder(config),
            stages=self.stages,
            limits=self.limits,
            session_id=session_id,
        )
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            # Note (Junnan Li): Keep the output iterator across abort; closing it closes the coordinator session.
            async for output in self.client.session_outputs(self.ref):
                if self.active is None or output.input_seq != self.active.seq:
                    continue
                if output.kind == "input_done":
                    if output.ref.epoch == self.epoch:
                        for event in self.buffer:
                            await self.emit(event, self.epoch, self.active)
                    self.buffer.clear()
                    self.buffer_bytes = 0
                    if self.future is not None and not self.future.done():
                        self.future.set_result(self.active.real_samples)
                elif output.ref.epoch == self.epoch:
                    for event in self.convert(output):
                        size = len(repr(event).encode())
                        if (
                            len(self.buffer) >= self.limits.max_output_chunks
                            or self.buffer_bytes + size > self.limits.max_output_bytes
                        ):
                            raise RuntimeError("native unit output budget exhausted")
                        self.buffer.append(event)
                        self.buffer_bytes += size
            if not self.closing:
                raise RuntimeError("session output stream closed")
        except Exception as exc:
            self.reader_error = exc
            if self.future is not None and not self.future.done():
                self.future.set_exception(exc)
            else:
                await self.emit(
                    TurnFailure("server_error", "internal", str(exc)), self.epoch
                )

    async def process(self, unit, epoch):
        if self.reader_error is not None:
            raise self.reader_error
        self.active = unit
        self.epoch = epoch
        self.future = asyncio.get_running_loop().create_future()
        chunk = TimedChunk(
            "audio",
            unit.start_sample * 1000 / self.rate,
            unit.real_samples * 1000 / self.rate,
            unit.seq,
            unit.pcm,
            format="pcm16",
            eos=unit.eos,
        )
        try:
            await self.client.append_session(self.ref, chunk)
            return await self.future
        finally:
            self.active = None
            self.future = None
            self.buffer.clear()
            self.buffer_bytes = 0

    async def cancel(self):
        # Note (Junnan Li): The coordinator still completes the unit and preserves its receipt.
        future = self.future
        self.ref = await self.client.abort_session(self.ref)
        if future is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(future), self.local_cleanup_timeout
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "completed unit consumption receipt is missing"
                ) from exc
        self.epoch = self.ref.epoch
        self.buffer.clear()
        self.buffer_bytes = 0

    async def close(self):
        self.closing = True
        try:
            if self.ref is not None:
                await self.client.close_session(self.ref)
        finally:
            await cancel_local_tasks([self.reader], self.local_cleanup_timeout)


@dataclass(frozen=True)
class TurnBasedAdapterFactory:
    client: Client
    model_name: str
    supports_audio_output: bool = False
    smart_turn_model: SemanticEOUModel | None = None

    def __call__(self) -> TurnBasedAdapter:
        return TurnBasedAdapter(self)


class TurnBasedAdapter(InteractionAdapter):
    """Use the existing request/VAD/history computation with a typed sink."""

    def __init__(self, factory: TurnBasedAdapterFactory) -> None:
        self.factory = factory
        self.engine = None
        self.emit = None
        self.output_epoch = 0
        self.epoch = ContextVar("turn_epoch", default=0)
        self.processing_input = False
        self.input_idle = asyncio.Event()
        self.input_idle.set()
        self.committed = 0
        self.discarded = 0
        self.cancel_handler = None
        self.limits = RuntimeLimits()
        self.pending_update = None

    supports_hot_update = True

    def set_limits(self, limits):
        self.limits = limits

    def set_cancel_handler(self, handler):
        self.cancel_handler = handler

    async def _server_cancel(self):
        epoch = await self.cancel_handler(None)
        self.epoch.set(epoch)

    async def open(self, session_id, config, emit):
        self.emit = emit
        self.engine = TurnBasedSession(
            client=self.factory.client,
            model_name=self.factory.model_name,
            session_id=session_id,
            emit=self._emit,
            enable_vad=False,
            on_input_committed=self._committed,
            max_queued_turns=self.limits.max_input_chunks,
            server_cancel=self._server_cancel,
            max_queued_audio_bytes=self.limits.max_input_bytes,
            capture_turn_context=True,
            prepare_turn_context=self._prepare_turn_context,
            strict_cleanup=True,
            max_text_chars=self.limits.max_history_chars,
            supports_audio_output=self.factory.supports_audio_output,
            smart_turn_model=self.factory.smart_turn_model,
        )
        td = config.get("audio", {}).get("input", {}).get("turn_detection")
        update = dict(
            modalities=(
                ["text", "audio"]
                if "audio" in config.get("output_modalities", [])
                else ["text"]
            )
        )
        if "instructions" in config:
            update["instructions"] = config["instructions"]
        if td is not None:
            update["turn_detection"] = td
        candidate, detector = await self.engine.prepare_update(
            SessionConfig.model_validate(update)
        )
        self.engine.apply_update(candidate, detector)

    async def update(self, config):
        td = config.get("audio", {}).get("input", {}).get("turn_detection")
        update = dict(
            modalities=(
                ["text", "audio"]
                if "audio" in config["output_modalities"]
                else ["text"]
            )
        )
        if td is not None:
            update["turn_detection"] = td
        try:
            candidate, detector = await self.engine.prepare_update(
                SessionConfig.model_validate(update)
            )
        except (TurnConfigurationError, ValueError) as exc:
            raise ProtocolError(
                "invalid_request", str(exc), "session.audio.input.turn_detection"
            ) from exc
        if td is None:
            candidate.turn_detection = None
        self.pending_update = candidate, detector

    def _apply_pending_update(self):
        if self.pending_update is None:
            return
        candidate, detector = self.pending_update
        if detector is not None or candidate.turn_detection is None:
            self.engine.vad = detector
            self.engine.vad_origin_samples = (
                self.engine.buffer_origin_samples + self.engine.audio_buffer.num_samples
            )
        self.engine.session_object = candidate
        self.pending_update = None

    def _prepare_turn_context(self):
        self.epoch.set(self.output_epoch)

    def _committed(self, samples, discarded):
        self.committed += samples
        self.discarded += discarded

    async def _emit(self, event):
        await self.emit(event, self.epoch.get())

    async def process(self, unit, epoch):
        self._apply_pending_update()
        self.epoch.set(epoch)
        before = self.committed
        discarded_before = self.discarded
        if (
            sum(len(item.text) for item in self.engine.conversation)
            > self.limits.max_history_chars
        ):
            raise ContextLimitError("turn history context limit")
        self.processing_input = True
        self.input_idle.clear()
        try:
            await self.engine.append_audio(unit.pcm[: unit.real_samples * 2])
        finally:
            self.processing_input = False
            self.input_idle.set()
        if unit.eos and not self.engine.audio_buffer.is_empty():
            end_offset = (
                self.engine.buffer_origin_samples
                + self.engine.audio_buffer.num_samples
                - self.engine.vad_origin_samples
            )
            await self.engine.auto_commit_utterance(end_offset)
            self.engine.speech_idle.set()
        if unit.eos and self.engine.queue_drainer is not None:
            await self.engine.queue_idle.wait()
        return self.committed - before, self.discarded - discarded_before

    async def clear(self):
        if self.processing_input:
            return 0  # Note (Junnan Li): The detector owns this buffer until its in-flight call ends.
        samples = self.engine.audio_buffer.num_samples
        self.engine.drop_buffer_and_reset_vad()
        self.engine.speech_idle.set()
        return samples

    async def cancel(self):
        self.output_epoch += 1
        # Note (Junnan Li): Preserve active-turn history while the runtime fences its output.
        task = self.engine.active_task
        if task is not None:
            await asyncio.shield(task)

    async def close(self):
        if self.engine is not None:
            try:
                await self.input_idle.wait()
                await self.engine.teardown()
            finally:
                self.engine.closed = True
                await cancel_local_tasks(
                    [
                        self.engine.queue_drainer,
                        self.engine.active_task,
                        self.engine.active_response_task,
                        self.engine.active_response_abort_task,
                    ],
                    self.limits.cleanup_timeout_s,
                )
            self.engine.audio_buffer.clear()
            self.engine.conversation.clear()
            while not self.engine.response_queue.empty():
                self.engine.response_queue.get_nowait()
