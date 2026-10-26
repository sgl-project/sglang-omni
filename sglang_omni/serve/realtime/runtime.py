"""Bounded, transport independent realtime session ownership and media clocks."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.profiler.duplex_events import (
    SessionUnitReadyObservation,
    capture_session_unit_ready,
)
from sglang_omni.serve.realtime.constants import (
    CLOSE_REASON_CLEANUP_TIMEOUT,
    CLOSE_REASON_CLIENT_CLOSED,
    CLOSE_REASON_DISCONNECT,
    CLOSE_REASON_IDLE_TIMEOUT,
    CLOSE_REASON_INTERNAL,
    CLOSE_REASON_SESSION_UPDATE_TIMEOUT,
)
from sglang_omni.serve.realtime.control import (
    Accepted,
    Cleared,
    Closed,
    ControlEvent,
    Created,
    Drained,
    Ended,
    Failure,
    ImageAccepted,
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

MAX_FRAME_LOOKAHEAD_UNITS = 2
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
        self.pending_commits = 0
        self.pending_frames: dict[int, list[tuple[float, bytes]]] = {}
        self.ready_observations: dict[int, SessionUnitReadyObservation] | None = None
        self.next_unit_index = 0
        self.is_input_ended = False
        self.end_event_id: str | None = None
        self.command_lock = asyncio.Lock()
        self.input_ready = asyncio.Event()
        self.last_activity_seconds: float = time.monotonic()
        self.admission_deadline_seconds: float | None = None
        self.is_admitting: bool = False
        self.is_unit_in_flight: bool = False
        self.output_buffer = OutputBuffer(limits)
        self.pump_task: asyncio.Task[None] | None = None
        self.watchdog_task: asyncio.Task[None] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.closed_event: asyncio.Event = asyncio.Event()
        self.close_reason: str | None = None
        self.processing_unit: ContextVar[Unit | None] = ContextVar(
            "realtime_unit", default=None
        )

    @property
    def pending_samples(self) -> int:
        return len(self.pending_pcm) // PCM16_BYTES_PER_SAMPLE

    def mark_ready_units(
        self, first_ready_unit_index: int, ready_unit_count: int
    ) -> None:
        """Retain the first profiler observation for each runnable Unit."""
        ready_observation = capture_session_unit_ready()
        if ready_observation is None:
            return
        else:
            pass
        if self.ready_observations is None:
            self.ready_observations = {}
        else:
            pass
        for unit_index in range(
            first_ready_unit_index, first_ready_unit_index + ready_unit_count
        ):
            if unit_index not in self.ready_observations:
                self.ready_observations[unit_index] = ready_observation
            else:
                pass

    def notify(self, event: ControlEvent) -> None:
        self.output_buffer.enqueue(Envelope(event=event, is_control=True))

    def mark_activity(self) -> None:
        """Restart the idle timeout."""
        self.last_activity_seconds = time.monotonic()

    def start(self) -> None:
        self.admission_deadline_seconds = (
            time.monotonic() + self.limits.session_update_timeout_s
        )
        self.notify(Created(self.session_id, self.model, "realtime"))
        self.watchdog_task = asyncio.create_task(self.watch_activity())

    def expire_admission_if_due(self) -> bool:
        """Reject a new admission after the admission deadline."""
        if (
            self.state == "CREATED"
            and self.admission_deadline_seconds is not None
            and time.monotonic() >= self.admission_deadline_seconds
        ):
            self.state = "CLOSING"
            self.fail(
                f"no session.update within {self.limits.session_update_timeout_s}s",
                CLOSE_REASON_SESSION_UPDATE_TIMEOUT,
            )
            return True
        else:
            return False

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
            self.mark_activity()

    def require_open(self) -> None:
        if self.state != "OPEN":
            raise ProtocolError("invalid_state", "session is not OPEN")
        else:
            pass

    async def update(self, patch: object, event_id: str) -> None:
        async with self.command_lock:
            if self.expire_admission_if_due():
                return
            else:
                pass
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
                self.is_admitting = True
                try:
                    is_open = await self.open_adapter(adapter, candidate, event_id)
                except asyncio.CancelledError:
                    self.abandon_admission(
                        adapter, "session admission cancelled", event_id
                    )
                    raise
                finally:
                    self.is_admitting = False
                if not is_open:
                    return
                else:
                    pass
                self.adapter = adapter
                if self.close_task is not None:
                    # Note (Junnan Li): Admission cleanup belongs to the closing owner; do not publish OPEN here.
                    return
                else:
                    pass
                self.state = "OPEN"
                self.mark_activity()
                self.pump_task = asyncio.create_task(self.pump())
            else:
                assert self.adapter is not None
                await self.adapter.update(candidate)
                self.mark_activity()
            self.config, self.granted = candidate, granted
            public_config = {
                **candidate,
                "sglang": {
                    key: value
                    for key, value in candidate.get("sglang", {}).items()
                    if key not in ("reference_audio", "tts_reference_audio")
                },
            }
            self.notify(
                Updated(
                    self.session_id,
                    self.model,
                    candidate["type"],
                    granted,
                    event_id,
                    copy.deepcopy(public_config),
                )
            )

    async def open_adapter(
        self,
        adapter: InteractionAdapter,
        candidate: SessionConfiguration,
        event_id: str,
    ) -> bool:
        """Open the adapter; return False if failed admission expires after cleanup."""
        try:
            await asyncio.wait_for(
                adapter.open(self.session_id, candidate, self.emit),
                self.limits.cleanup_timeout_s,
            )
            return True
        except Exception as exc:
            logger.exception(f"Realtime session {self.session_id} admission failed")
            try:
                await asyncio.wait_for(adapter.close(), self.limits.cleanup_timeout_s)
            except Exception:
                self.abandon_admission(adapter, "admission cleanup failed", event_id)
                raise RuntimeError("admission cleanup failed") from exc
            if self.expire_admission_if_due():
                return False
            else:
                raise ProtocolError("admission_rejected", str(exc)) from exc

    def abandon_admission(
        self, adapter: InteractionAdapter, message: str, event_id: str
    ) -> None:
        """Hand a half-opened adapter to the close task, which releases it."""
        self.adapter = adapter
        self.state = "CLOSING"
        self.fail(message, event_id=event_id)

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
            pending_unit_count_before = (
                len(self.pending_pcm) // self.capabilities.native_unit_bytes
            )
            self.pending_pcm.extend(pcm)
            self.accepted_samples += len(pcm) // PCM16_BYTES_PER_SAMPLE
            self.next_append_sequence += 1
            pending_unit_count_after = (
                len(self.pending_pcm) // self.capabilities.native_unit_bytes
            )
            newly_ready_unit_count = (
                pending_unit_count_after - pending_unit_count_before
            )
            if newly_ready_unit_count:
                self.mark_ready_units(
                    self.next_unit_index + pending_unit_count_before,
                    newly_ready_unit_count,
                )
            else:
                pass
            self.notify(
                Accepted(
                    sequence,
                    self.capabilities.input_duration_ms(self.accepted_samples),
                    event_id,
                )
            )
            self.mark_activity()
            self.input_ready.set()

    async def append_image(self, image: bytes, t_ms: float, event_id: str) -> None:
        async with self.command_lock:
            self.require_open()
            pending_start_ms = self.capabilities.input_duration_ms(
                self.accepted_samples - self.pending_samples
            )
            pending_unit_offset = math.floor(
                (t_ms - pending_start_ms) / self.capabilities.native_unit_ms
            )
            unit_index = self.next_unit_index + pending_unit_offset
            pending_units = math.ceil(
                len(self.pending_pcm) / self.capabilities.native_unit_bytes
            )
            if "image" not in self.capabilities.input_modalities:
                raise ProtocolError("not_supported", "image input is not granted")
            elif self.is_input_ended:
                raise ProtocolError("invalid_state", "input has ended")
            elif len(image) > self.capabilities.max_image_bytes:
                raise ProtocolError(
                    "buffer_overflow", "image exceeds input budget", "image"
                )
            elif not image.startswith((b"\xff\xd8", b"\x89PNG")):
                raise ProtocolError(
                    "invalid_request", "image must be JPEG or PNG", "image"
                )
            elif unit_index < self.next_unit_index:
                raise ProtocolError("invalid_state", "frame unit already cut")
            elif (
                len(self.pending_frames.get(unit_index, ()))
                >= self.granted["input_image_format"]["max_frames_per_unit"]
            ):
                raise ProtocolError("buffer_overflow", "unit frame count exceeds limit")
            elif pending_unit_offset > pending_units + MAX_FRAME_LOOKAHEAD_UNITS:
                raise ProtocolError("buffer_overflow", "frame exceeds lookahead budget")
            else:
                self.pending_frames.setdefault(unit_index, []).append((t_ms, image))
                self.notify(ImageAccepted(f"unit_{unit_index}", event_id))
                self.mark_activity()

    async def clear(self, event_id: str) -> None:
        async with self.command_lock:
            self.require_open()
            assert self.adapter is not None
            cleared_samples = self.pending_samples + await self.adapter.clear()
            self.pending_pcm.clear()
            self.pending_frames.clear()
            self.ready_observations = None
            self.discarded_samples += cleared_samples
            self.notify(
                Cleared(self.capabilities.input_duration_ms(cleared_samples), event_id)
            )
            self.mark_activity()

    async def cancel(self) -> None:
        """Stop the reply in progress; the session answers with its own cancelled response.done."""
        async with self.command_lock:
            self.require_open()
            assert self.adapter is not None
            await self.adapter.cancel()
            self.mark_activity()

    async def truncate(self, item_id: str, audio_end_ms: float) -> None:
        """Tell the session how much of the reply item the listener heard."""
        async with self.command_lock:
            self.require_open()
            assert self.adapter is not None
            await self.adapter.truncate(item_id, audio_end_ms)
            self.mark_activity()

    async def commit(self) -> None:
        """End the user's turn where the client says it ends, after the audio appended before it."""
        async with self.command_lock:
            self.require_open()
            if not self.granted.get("client_commit"):
                raise ProtocolError(
                    "not_applicable", "manual turn commands are not granted"
                )
            elif self.is_input_ended:
                raise ProtocolError("invalid_state", "audio input already ended")
            else:
                self.pending_commits += 1
                self.mark_activity()
            self.input_ready.set()

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
            pending_unit_count = (
                len(self.pending_pcm) // self.capabilities.native_unit_bytes
            )
            has_partial_unit = bool(
                len(self.pending_pcm) % self.capabilities.native_unit_bytes
            )
            if has_partial_unit:
                self.mark_ready_units(self.next_unit_index + pending_unit_count, 1)
            elif not pending_unit_count:
                self.mark_ready_units(self.next_unit_index, 1)
            else:
                pass
            self.end_event_id = event_id
            self.notify(
                Ended(
                    self.capabilities.input_duration_ms(self.accepted_samples),
                    self.capabilities.tail_policy,
                    event_id,
                )
            )
            self.mark_activity()
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
        if self.ready_observations is None:
            ready_observation = None
        else:
            ready_observation = self.ready_observations.pop(self.next_unit_index, None)
            if not self.ready_observations:
                self.ready_observations = None
            else:
                pass
        unit = Unit(
            self.next_unit_index,
            start_sample,
            pcm,
            real_samples,
            self.is_input_ended and not self.pending_pcm,
            tuple(self.granted["output_modalities"]),
            images=tuple(
                image
                for _, image in sorted(
                    self.pending_frames.pop(self.next_unit_index, []),
                    key=lambda frame: frame[0],
                )
            ),
            ready_observation=ready_observation,
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
                        and not self.pending_commits
                    ):
                        continue
                    elif not self.pending_pcm and self.pending_commits:
                        self.pending_commits -= 1
                        unit = None
                    else:
                        unit = self.cut_next_unit()
                    self.is_unit_in_flight = True
                if unit is None:
                    try:
                        await self.adapter.commit()
                    finally:
                        self.is_unit_in_flight = False
                    self.mark_activity()
                    self.input_ready.set()
                    continue
                else:
                    pass
                self.processing_unit.set(unit)
                try:
                    consumption = await self.adapter.process(unit)
                finally:
                    self.is_unit_in_flight = False
                self.mark_activity()
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
        except asyncio.CancelledError:
            if self.close_task is None:
                logger.error(f"Realtime session {self.session_id} input pump cancelled")
                self.state = "CLOSING"
                self.fail("input processing cancelled")
            else:
                pass
            raise
        except ProtocolError as exc:
            self.fail(str(exc), exc.code)
        except Exception as exc:
            if ContextExhaustedError.matches(exc):
                logger.error(f"Realtime session {self.session_id} closed: {exc}")
                failure_code = ContextExhaustedError.CODE
            else:
                logger.exception(
                    f"Realtime session {self.session_id} input pump failed"
                )
                failure_code = CLOSE_REASON_INTERNAL
            self.fail(str(exc), failure_code)

    async def watch_activity(self) -> None:
        # note (Haiyang Luo): OPEN has no wake-up event; cap waits at the idle timeout.
        maximum_wait_seconds = self.limits.idle_timeout_s
        while self.state in ("CREATED", "OPEN"):
            # note (Haiyang Luo): No await between the idle check and CLOSING avoids command races.
            if self.state == "CREATED":
                timeout_seconds = self.limits.session_update_timeout_s
                is_busy = self.is_admitting
                assert self.admission_deadline_seconds is not None
                remaining_seconds = self.admission_deadline_seconds - time.monotonic()
            else:
                timeout_seconds = self.limits.idle_timeout_s
                is_busy = self.is_unit_in_flight or self.command_lock.locked()
                remaining_seconds = (
                    self.last_activity_seconds + timeout_seconds - time.monotonic()
                )
            if is_busy:
                await asyncio.sleep(min(timeout_seconds, maximum_wait_seconds))
            elif remaining_seconds > 0:
                await asyncio.sleep(min(remaining_seconds, maximum_wait_seconds))
            elif self.state == "CREATED":
                self.expire_admission_if_due()
                return
            else:
                self.state = "CLOSING"
                self.fail(
                    f"no input or output within {timeout_seconds}s",
                    CLOSE_REASON_IDLE_TIMEOUT,
                )
                return

    def fail(
        self,
        message: str,
        code: str = CLOSE_REASON_INTERNAL,
        event_id: str | None = None,
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
        self.close_reason = reason
        # Note (Junnan Li): Set CLOSING under the command lock, then release it: adapter
        # teardown can run a VAD callback that must observe CLOSING.
        async with self.command_lock:
            self.state = "CLOSING"
            self.pending_frames.clear()
        self.discarded_samples += self.pending_samples
        self.pending_pcm.clear()
        self.ready_observations = None
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
                    [self.pump_task, self.watchdog_task], self.limits.cleanup_timeout_s
                )
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            if cleanup_error is not None:
                self.close_reason = CLOSE_REASON_CLEANUP_TIMEOUT
                self.output_buffer.clear()
                self.output_buffer.enqueue_terminal(
                    Failure(
                        CLOSE_REASON_CLEANUP_TIMEOUT,
                        str(cleanup_error)[:MAX_FAILURE_MESSAGE_CHARS],
                        True,
                        event_id,
                    )
                )
            else:
                status: ResponseStatus = (
                    "cancelled"
                    if reason in (CLOSE_REASON_CLIENT_CLOSED, CLOSE_REASON_DISCONNECT)
                    else "failed"
                )
                self.output_buffer.finish_responses(status, reason)
                self.output_buffer.enqueue_terminal(Closed(reason, event_id))
        finally:
            self.state = "CLOSED"
            self.output_buffer.output_ready.set()
            self.closed_event.set()
