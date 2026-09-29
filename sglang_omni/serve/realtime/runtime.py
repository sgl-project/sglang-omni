"""Bounded, transport independent realtime session ownership and media clocks."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cleared,
    Closed,
    ControlEvent,
    Created,
    Drained,
    Ended,
    Failure,
    UnitCompleted,
    Updated,
)
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.output import OutputEvent, ResponseStatus, TurnFailure
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    SessionConfiguration,
    SessionState,
)
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    PCM16_BYTES_PER_SAMPLE,
    AdapterFactory,
    Capabilities,
    Envelope,
    InteractionAdapter,
    ProtocolError,
    RuntimeLimits,
    Unit,
)

logger = logging.getLogger(__name__)

MAX_FAILURE_MESSAGE_CHARS = 512
MEDIA_TIME_TOLERANCE_MS = 1e-7


class SessionRuntime:
    def __init__(
        self,
        model: str,
        capabilities: Capabilities,
        adapter_factory: AdapterFactory,
        limits: RuntimeLimits,
    ) -> None:
        self.session_id = "sess_" + uuid.uuid4().hex
        self.model = model
        self.capabilities = capabilities
        self.adapter_factory = adapter_factory
        self.limits = limits
        self.negotiation = SessionNegotiation(
            model=model, capabilities=capabilities, limits=limits
        )
        self.adapter: InteractionAdapter | None = None
        self.state: SessionState = "CREATED"
        self.config: SessionConfiguration = {}
        self.granted: GrantedCapabilities = {}
        self.next_append_sequence = 0
        self.accepted_samples = 0
        self.consumed_samples = 0
        self.discarded_samples = 0
        self.padding_samples = 0
        self.pending_pcm = bytearray()
        self.next_unit_index = 0
        self.is_input_ended = False
        self.end_event_id: str | None = None
        self.command_lock = asyncio.Lock()
        self.input_ready = asyncio.Event()
        self.output_buffer = OutputBuffer(limits)
        self.pump_task: asyncio.Task[None] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.processing_unit: ContextVar[Unit | None] = ContextVar(
            "realtime_unit", default=None
        )

    @property
    def pending_samples(self) -> int:
        return len(self.pending_pcm) // PCM16_BYTES_PER_SAMPLE

    def notify(self, event: ControlEvent) -> None:
        self.output_buffer.enqueue(Envelope(event=event, is_control=True))

    def notify_created(self) -> None:
        self.notify(Created(self.session_id, self.model, "realtime"))

    async def outputs(self) -> AsyncIterator[Envelope]:
        while True:
            while (envelope := self.output_buffer.dequeue()) is not None:
                yield envelope
            if self.state == "CLOSED":
                return
            else:
                self.output_buffer.output_ready.clear()
                await self.output_buffer.output_ready.wait()

    async def emit(self, event: OutputEvent, unit: Unit | None = None) -> None:
        if self.state != "OPEN":
            return
        elif isinstance(event, TurnFailure):
            self.fail(event.message, event.code)
        else:
            producing_unit = unit or self.processing_unit.get()
            modalities = (
                producing_unit.output_modalities if producing_unit is not None else None
            ) or tuple(
                self.granted.get(
                    "output_modalities", self.capabilities.output_modalities
                )
            )
            self.output_buffer.emit(event, producing_unit, modalities)

    def require_open(self) -> None:
        if self.state != "OPEN":
            raise ProtocolError("invalid_state", "session is not OPEN")
        else:
            pass

    async def update(self, patch: object, event_id: str) -> None:
        async with self.command_lock:
            if self.state not in ("CREATED", "OPEN"):
                raise ProtocolError("invalid_state", "session is closing")
            else:
                pass
            candidate, granted = self.negotiation.negotiate(
                self.config, self.state, patch
            )
            if self.state == "CREATED":
                adapter = self.adapter_factory()
                adapter.set_limits(self.limits)
                try:
                    await asyncio.wait_for(
                        adapter.open(self.session_id, candidate, self.emit),
                        self.limits.cleanup_timeout_s,
                    )
                except Exception as exc:
                    logger.exception(
                        f"Realtime session {self.session_id} admission failed"
                    )
                    try:
                        await asyncio.wait_for(
                            adapter.close(), self.limits.cleanup_timeout_s
                        )
                    except Exception:
                        self.adapter = adapter
                        raise RuntimeError("admission cleanup failed") from exc
                    raise ProtocolError("admission_rejected", str(exc)) from exc
                self.adapter = adapter
                if self.close_task is not None:
                    # Note (Junnan Li): Admission cleanup belongs to the closing owner; do not publish OPEN here.
                    return
                else:
                    pass
                self.state = "OPEN"
                self.pump_task = asyncio.create_task(self.pump())
            else:
                pass
            self.config, self.granted = candidate, granted
            self.notify(
                Updated(
                    self.session_id,
                    self.model,
                    candidate["type"],
                    granted,
                    event_id,
                    copy.deepcopy(candidate),
                )
            )

    async def append(
        self, pcm: bytes, sequence: int, t_start_ms: float | None, event_id: str
    ) -> None:
        async with self.command_lock:
            self.require_open()
            buffered_samples = (
                self.accepted_samples - self.consumed_samples - self.discarded_samples
            )
            if self.is_input_ended:
                raise ProtocolError("invalid_state", "audio input has ended")
            elif sequence != self.next_append_sequence:
                raise ProtocolError("invalid_state", "audio seq must be contiguous")
            elif not pcm or len(pcm) % PCM16_BYTES_PER_SAMPLE:
                raise ProtocolError(
                    "invalid_request", "audio must contain whole PCM16 samples"
                )
            elif t_start_ms is not None and not math.isclose(
                t_start_ms,
                self.capabilities.input_duration_ms(self.accepted_samples),
                rel_tol=0,
                abs_tol=MEDIA_TIME_TOLERANCE_MS,
            ):
                raise ProtocolError(
                    "invalid_state", "input media time must be sample-contiguous"
                )
            elif (
                buffered_samples * PCM16_BYTES_PER_SAMPLE + len(pcm)
                > self.limits.max_input_bytes
            ):
                raise ProtocolError(
                    "buffer_overflow", "input budget exhausted; retry this seq"
                )
            else:
                pass
            self.pending_pcm.extend(pcm)
            self.accepted_samples += len(pcm) // PCM16_BYTES_PER_SAMPLE
            self.next_append_sequence += 1
            self.notify(
                Accepted(
                    sequence,
                    self.capabilities.input_duration_ms(self.accepted_samples),
                    event_id,
                )
            )
            self.input_ready.set()

    async def clear(self, event_id: str) -> None:
        async with self.command_lock:
            self.require_open()
            assert self.adapter is not None
            cleared_samples = self.pending_samples + await self.adapter.clear()
            self.pending_pcm.clear()
            self.discarded_samples += cleared_samples
            self.notify(
                Cleared(self.capabilities.input_duration_ms(cleared_samples), event_id)
            )

    async def end(self, event_id: str) -> None:
        async with self.command_lock:
            self.require_open()
            if self.is_input_ended:
                raise ProtocolError("invalid_state", "audio input already ended")
            elif self.capabilities.tail_policy == "reject" and (
                len(self.pending_pcm) % self.capabilities.native_unit_bytes
            ):
                raise ProtocolError(
                    "invalid_state", "partial native unit; append more audio before EOS"
                )
            else:
                pass
            self.is_input_ended = True
            self.end_event_id = event_id
            self.notify(
                Ended(
                    self.capabilities.input_duration_ms(self.accepted_samples),
                    self.capabilities.tail_policy,
                    event_id,
                )
            )
            self.input_ready.set()

    def cut_next_unit(self) -> Unit:
        unit_bytes = self.capabilities.native_unit_bytes
        unit_pcm_bytes = min(unit_bytes, len(self.pending_pcm))
        start_sample = self.accepted_samples - self.pending_samples
        pcm = bytes(self.pending_pcm[:unit_pcm_bytes])
        del self.pending_pcm[:unit_pcm_bytes]
        real_samples = len(pcm) // PCM16_BYTES_PER_SAMPLE
        if (
            real_samples
            and unit_pcm_bytes < unit_bytes
            and self.capabilities.tail_policy == "pad"
        ):
            padding_bytes = unit_bytes - unit_pcm_bytes
            self.padding_samples += padding_bytes // PCM16_BYTES_PER_SAMPLE
            pcm += b"\0" * padding_bytes
        else:
            pass
        unit = Unit(
            self.next_unit_index,
            start_sample,
            pcm,
            real_samples,
            self.is_input_ended and not self.pending_pcm,
            tuple(self.granted["output_modalities"]),
        )
        self.next_unit_index += 1
        return unit

    async def pump(self) -> None:
        assert self.adapter is not None
        try:
            while self.state == "OPEN":
                await self.input_ready.wait()
                async with self.command_lock:
                    self.input_ready.clear()
                    if self.state != "OPEN":
                        return
                    elif (
                        len(self.pending_pcm) < self.capabilities.native_unit_bytes
                        and not self.is_input_ended
                    ):
                        continue
                    else:
                        unit = self.cut_next_unit()
                self.processing_unit.set(unit)
                consumption = await self.adapter.process(unit)
                if isinstance(consumption, tuple):
                    consumed_samples, discarded_samples = consumption
                else:
                    consumed_samples = consumption
                    discarded_samples = unit.real_samples - consumption
                if (
                    any(
                        type(sample_count) is not int or sample_count < 0
                        for sample_count in (consumed_samples, discarded_samples)
                    )
                    or self.consumed_samples
                    + self.discarded_samples
                    + consumed_samples
                    + discarded_samples
                    > self.accepted_samples
                ):
                    raise RuntimeError(
                        "adapter did not provide valid media consumption"
                    )
                else:
                    pass
                self.consumed_samples += consumed_samples
                self.discarded_samples += discarded_samples
                if self.close_task is None:
                    self.output_buffer.enqueue(
                        Envelope(event=UnitCompleted(unit.unit_id), unit=unit)
                    )
                else:
                    pass
                if not unit.eos:
                    self.input_ready.set()
                elif self.close_task is not None:
                    return
                else:
                    assert self.end_event_id is not None
                    self.notify(
                        Drained(
                            self.capabilities.input_duration_ms(self.accepted_samples),
                            self.capabilities.input_duration_ms(self.consumed_samples),
                            self.capabilities.input_duration_ms(self.discarded_samples),
                            self.capabilities.input_duration_ms(self.padding_samples),
                            self.end_event_id,
                        )
                    )
                    return
        except ProtocolError as exc:
            self.fail(str(exc), exc.code)
        except Exception as exc:
            logger.exception(f"Realtime session {self.session_id} input pump failed")
            self.fail(str(exc))

    def fail(
        self, message: str, code: str = "internal", event_id: str | None = None
    ) -> None:
        if self.close_task is None:
            self.output_buffer.enqueue_terminal(
                Failure(code, message[:MAX_FAILURE_MESSAGE_CHARS], True, event_id)
            )
            self.close_task = asyncio.create_task(self.run_close(code))
        else:
            pass

    async def close(self, reason: str, event_id: str | None = None) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self.run_close(reason, event_id))
        else:
            pass
        await asyncio.shield(self.close_task)

    async def run_close(self, reason: str, event_id: str | None = None) -> None:
        # Note (Junnan Li): Set CLOSING under the command lock, then release it: adapter
        # teardown can run a VAD callback that must observe CLOSING.
        async with self.command_lock:
            self.state = "CLOSING"
        self.discarded_samples += self.pending_samples
        self.pending_pcm.clear()
        self.input_ready.set()
        cleanup_error: Exception | None = None
        try:
            if self.adapter is not None:
                await asyncio.wait_for(
                    self.adapter.close(), self.limits.cleanup_timeout_s
                )
            else:
                pass
        except Exception as exc:
            cleanup_error = exc
        finally:
            try:
                await cancel_local_tasks(
                    [self.pump_task], self.limits.cleanup_timeout_s
                )
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            if cleanup_error is not None:
                self.output_buffer.clear()
                self.output_buffer.enqueue_terminal(
                    Failure(
                        "cleanup_timeout",
                        str(cleanup_error)[:MAX_FAILURE_MESSAGE_CHARS],
                        True,
                        event_id,
                    )
                )
            else:
                status: ResponseStatus = (
                    "cancelled"
                    if reason in ("client_closed", "disconnect")
                    else "failed"
                )
                self.output_buffer.finish_responses(status, reason)
                self.output_buffer.enqueue_terminal(Closed(reason, event_id))
        finally:
            self.state = "CLOSED"
            self.output_buffer.output_ready.set()
