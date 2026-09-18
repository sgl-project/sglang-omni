"""Bounded, transport independent realtime session ownership and media clocks."""

from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cancelled,
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
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ContextLimitError,
    OutputEvent,
    ResponseEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
    TranscriptionDelta,
    TranscriptionFinished,
    TranscriptionRevision,
    TurnFailure,
)
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, param: str | None = None):
        self.code = code
        self.param = param
        super().__init__(message)


@dataclass(frozen=True)
class RuntimeLimits:
    max_input_bytes: int = 1920000
    max_input_chunks: int = 128
    max_output_bytes: int = 4194304
    max_output_events: int = 256
    max_responses: int = 128
    max_segments: int = 256
    max_history_chars: int = 65536
    session_timeout_s: float = 300
    cleanup_timeout_s: float = 30

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("runtime limits must be finite and positive")


@dataclass(frozen=True)
class Capabilities:
    interaction: Literal["native", "turn_based", "transcription"] = "native"
    input_rate: int = 16000
    output_rate: int = 16000
    output_modalities: tuple[str, ...] = ("text",)
    native_unit_ms: int = 20
    tail_policy: Literal["flush", "pad", "reject"] = "flush"
    cancel_is_noop: bool = False
    partial_style: str = "append_only"

    def __post_init__(self):
        if self.interaction not in ("native", "turn_based", "transcription"):
            raise ValueError("unsupported interaction")
        if self.input_rate <= 0 or self.output_rate <= 0 or self.native_unit_ms <= 0:
            raise ValueError("positive rates and cadence required")
        if self.input_rate * self.native_unit_ms % 1000:
            raise ValueError("native cadence must contain whole samples")
        if self.interaction == "turn_based" and (
            self.input_rate != 16000 or self.tail_policy != "flush"
        ):
            raise ValueError("turn-based input requires 16 kHz PCM16 and flush tail")
        if self.tail_policy not in ("flush", "pad", "reject"):
            raise ValueError("unsupported tail policy")
        if not self.output_modalities or set(self.output_modalities) - {
            "text",
            "audio",
        }:
            raise ValueError("unsupported output modalities")

    def describe(self):
        return dict(
            interaction=self.interaction,
            native_full_duplex=self.interaction == "native",
            proactive_output=False,
            turn_control=(
                ["server_vad", None] if self.interaction == "turn_based" else [None]
            ),
            client_commit=False,
            input_modalities=["audio"],
            output_modalities=list(self.output_modalities),
            input_audio_format=dict(type="audio/pcm", rate=self.input_rate),
            output_audio_format=dict(type="audio/pcm", rate=self.output_rate),
            native_unit_ms=self.native_unit_ms,
            first_unit_ms=self.native_unit_ms,
            microturn_ms="variable",
            tail_policy=self.tail_policy,
            supports_server_interrupt=self.interaction == "turn_based",
            cancel_is_noop=self.cancel_is_noop,
            supports_truncate=False,
            supports_resume=False,
            partial_style=self.partial_style,
            pressure_policy="reject",
            strict_order=True,
        )


@dataclass(frozen=True)
class Unit:
    seq: int
    start_sample: int
    pcm: bytes
    real_samples: int
    eos: bool = False
    output_modalities: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Envelope:
    epoch: int
    event: OutputEvent | ControlEvent
    # Note (Junnan Li): Control acknowledgements and response terminals survive an epoch advance.
    control: bool = False
    unit: Unit | None = None
    chunk_seq: int = 0
    output_modalities: tuple[str, ...] | None = None


class InteractionAdapter:
    """Adapters override what they support; the runtime never probes for methods."""

    supports_hot_update = False

    def set_limits(self, limits: RuntimeLimits) -> None:
        pass

    def set_cancel_handler(self, handler: Callable) -> None:
        pass

    async def open(self, session_id: str, config: dict, emit: Callable) -> None:
        raise NotImplementedError

    async def process(self, unit: Unit, epoch: int) -> int | tuple[int, int]:
        raise NotImplementedError

    async def update(self, config: dict) -> None:
        raise NotImplementedError

    async def clear(self) -> int:
        return 0

    async def cancel(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


def merge_config(current: dict, patch: dict) -> dict:
    result = copy.deepcopy(current)
    for key, value in patch.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


class SessionRuntime:
    def __init__(
        self,
        model: str,
        capabilities: Capabilities,
        adapter_factory: Callable[[], InteractionAdapter],
        limits: RuntimeLimits,
    ):
        self.session_id = "sess_" + uuid.uuid4().hex
        self.model, self.capabilities, self.limits = model, capabilities, limits
        self.factory = adapter_factory
        self.adapter = None
        self.state: Literal["CREATED", "OPEN", "CLOSING", "CLOSED"] = "CREATED"
        self.epoch = 0
        self.config: dict = {}
        self.granted: dict = {}
        self.next_seq = 0
        self.accepted_samples = 0
        self.consumed_samples = 0
        self.discarded_samples = 0
        self.padding_samples = 0
        self.pending: deque[tuple[int, bytes]] = deque()
        self.pending_bytes = 0
        self.unit_seq = 0
        self.output_seq = 0
        self.eos = False
        self.eos_event_id = None
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.output_wake = asyncio.Event()
        self.output: deque[tuple[Envelope, int]] = deque()
        self.output_bytes = 0
        self.responses: dict[str, dict] = {}
        self.segments: dict[tuple, dict] = {}
        self.audio_sent: dict[tuple, tuple[float, float]] = {}
        self.worker = None
        self.close_task = None
        self.producer_unit = ContextVar("realtime_unit", default=None)
        self.created_at = time.monotonic()

    def notify(self, event: ControlEvent):
        self._enqueue(Envelope(self.epoch, event, True))

    def created(self):
        self.notify(
            Created(
                self.session_id,
                self.model,
                (
                    "transcription"
                    if self.capabilities.interaction == "transcription"
                    else "realtime"
                ),
            )
        )

    def _enqueue(self, envelope: Envelope):
        size = len(repr(envelope).encode())
        if (
            len(self.output) >= self.limits.max_output_events
            or self.output_bytes + size > self.limits.max_output_bytes
        ):
            raise RuntimeError("outbound event budget exhausted")
        self.output.append((envelope, size))
        self.output_bytes += size
        self.output_wake.set()

    async def outputs(self):
        while True:
            while self.output:
                value, size = self.output.popleft()
                self.output_bytes -= size
                yield value
            if self.state == "CLOSED":
                return
            self.output_wake.clear()
            await self.output_wake.wait()

    async def emit(self, event: OutputEvent, epoch: int, unit: Unit | None = None):
        if self.state != "OPEN":
            return
        if isinstance(event, TurnFailure):
            self.fail(event.message, event.code)
            return
        if epoch != self.epoch:
            return
        rid = event.response_id if isinstance(event, ResponseEvent) else None
        unit = unit or self.producer_unit.get()
        # Note (Junnan Li): Keep each response's negotiated modalities across hot updates.
        modalities = self.responses.get(rid, {}).get("output_modalities") or (
            (unit.output_modalities if unit is not None else None)
            or tuple(
                self.granted.get(
                    "output_modalities", self.capabilities.output_modalities
                )
            )
        )
        if isinstance(event, (AudioDelta, AudioFinished)) and "audio" not in modalities:
            return
        if isinstance(event, ResponseFinished) and "audio" not in modalities:
            event = replace(event, include_audio=False)
        if isinstance(event, ResponseStarted):
            if rid in self.responses:
                raise RuntimeError("duplicate response creation")
            if len(self.responses) >= self.limits.max_responses:
                raise ContextLimitError("response context limit")
            self.responses[rid] = dict(
                epoch=epoch,
                output_modalities=modalities,
                terminal=False,
                visible=False,
                terminal_sent=False,
                text_done_sent=False,
                audio_done_sent=False,
                audio_visible=False,
                item_id="",
                text="",
                audio=False,
            )
        elif rid is not None:
            state = self.responses.get(rid)
            if state is None:
                raise RuntimeError("response output precedes creation")
            if state["terminal"]:
                return
            if state["item_id"] and event.item_id != state["item_id"]:
                raise RuntimeError("only one message item per response is supported")
            state["item_id"] = event.item_id
            if isinstance(event, (TextDelta, TextFinished)):
                state["text"] = (
                    state["text"] + event.text
                    if isinstance(event, TextDelta)
                    else event.text
                )
                if len(state["text"]) > self.limits.max_history_chars:
                    raise ContextLimitError("response text context limit")
            if isinstance(event, AudioDelta):
                if len(event.pcm) % 2:
                    raise RuntimeError("producer emitted invalid PCM16")
                state["audio"] = True
            if isinstance(event, ResponseFinished):
                state["terminal"] = True
        if isinstance(event, TranscriptionDelta) and event.segment_id is not None:
            if not event.item_id or not (
                0 <= event.start_ms <= event.end_ms <= self.ms(self.accepted_samples)
            ):
                raise RuntimeError("invalid transcription segment bounds")
            key = event.item_id, event.segment_id
            if key not in self.segments:
                if len(self.segments) >= self.limits.max_segments:
                    raise ContextLimitError("transcription context limit")
                self.segments[key] = dict(version=0, final=False, text="")
            segment = self.segments[key]
            if segment["final"]:
                return
            if isinstance(event, TranscriptionRevision):
                if (
                    event.base_revision_id != segment["version"]
                    or event.revision_id != segment["version"] + 1
                ):
                    raise RuntimeError("invalid transcription revision")
                segment["version"] = event.revision_id
                segment["text"] = event.text
            elif isinstance(event, TranscriptionFinished):
                if segment["text"] != event.text:
                    segment["version"] += 1
                segment["text"] = event.text
                segment["final"] = True
                event = replace(event, revision_id=segment["version"])
            else:
                segment["version"] += 1
                segment["text"] += event.text
            if len(segment["text"]) > self.limits.max_history_chars:
                raise ContextLimitError("transcription text limit")
        self._enqueue(
            Envelope(
                epoch,
                event,
                unit=unit or self.producer_unit.get(),
                chunk_seq=self.output_seq,
                output_modalities=tuple(modalities),
            )
        )
        self.output_seq += 1

    def ms(self, samples):
        return samples * 1000 / self.capabilities.input_rate

    def _open(self):
        if self.state != "OPEN":
            raise ProtocolError("invalid_state", "session is not OPEN")

    def _negotiate(self, patch):
        allowed = {
            "type",
            "model",
            "instructions",
            "output_modalities",
            "audio",
            "sglang",
        }
        if not isinstance(patch, dict) or set(patch) - allowed:
            raise ProtocolError(
                "invalid_request", "unsupported session field", "session"
            )
        candidate = merge_config(self.config, patch)
        if self.state == "OPEN":

            def frozen(config):
                result = copy.deepcopy(config)
                result.pop("output_modalities", None)
                audio = result.get("audio", {})
                if not isinstance(audio, dict) or not isinstance(
                    audio.get("input", {}), dict
                ):
                    raise ProtocolError(
                        "invalid_request", "invalid audio config", "session.audio"
                    )
                audio.get("input", {}).pop("turn_detection", None)
                return result

            if frozen(candidate) != frozen(self.config):
                raise ProtocolError("invalid_state", "session field is frozen")
        if "instructions" in candidate and (
            not isinstance(candidate["instructions"], str)
            or len(candidate["instructions"]) > self.limits.max_history_chars
        ):
            raise ProtocolError(
                "invalid_request", "instructions exceed context or have invalid type"
            )
        if candidate.get("model", self.model) != self.model:
            raise ProtocolError("invalid_request", "model differs from deployment")
        typ = candidate.get(
            "type",
            (
                "transcription"
                if self.capabilities.interaction == "transcription"
                else "realtime"
            ),
        )
        if typ != (
            "transcription"
            if self.capabilities.interaction == "transcription"
            else "realtime"
        ):
            raise ProtocolError("invalid_request", "session type is unavailable")
        self._validate_audio(candidate)
        requested = candidate.get(
            "output_modalities", list(self.capabilities.output_modalities)
        )
        if not isinstance(requested, list) or any(
            not isinstance(x, str) for x in requested
        ):
            raise ProtocolError("invalid_request", "output_modalities must be an array")
        outputs = [x for x in requested if x in self.capabilities.output_modalities][:1]
        if not outputs:
            raise ProtocolError("invalid_request", "no supported output combination")
        micro = self._validate_extension(candidate)
        return self._grant(candidate, typ, requested, outputs, micro)

    def _validate_audio(self, candidate):
        audio = candidate.get("audio", {})
        if not isinstance(audio, dict) or set(audio) - {"input", "output"}:
            raise ProtocolError("invalid_request", "invalid audio config")
        for direction, rate in [
            ("input", self.capabilities.input_rate),
            ("output", self.capabilities.output_rate),
        ]:
            config = audio.get(direction, {})
            if not isinstance(config, dict) or set(config) - (
                {"format", "turn_detection"} if direction == "input" else {"format"}
            ):
                raise ProtocolError("invalid_request", "invalid audio fields")
            fmt = config.get("format", dict(type="audio/pcm", rate=rate))
            if (
                fmt != dict(type="audio/pcm", rate=rate)
                or type(fmt.get("rate")) is not int
            ):
                raise ProtocolError(
                    "invalid_request",
                    "unsupported PCM format or sample rate",
                    f"session.audio.{direction}.format",
                )
            if direction == "input" and config.get("turn_detection") is not None:
                if self.capabilities.interaction != "turn_based":
                    raise ProtocolError(
                        "not_applicable",
                        "VAD is only available for turn-based sessions",
                    )
                td = config["turn_detection"]
                if not isinstance(td, dict) or set(td) - {
                    "type",
                    "threshold",
                    "prefix_padding_ms",
                    "silence_duration_ms",
                    "eagerness",
                    "interrupt_response",
                }:
                    raise ProtocolError(
                        "invalid_request", "invalid turn detection fields"
                    )
                if (
                    "interrupt_response" in td
                    and type(td["interrupt_response"]) is not bool
                ):
                    raise ProtocolError(
                        "invalid_request", "interrupt_response must be boolean"
                    )

    def _validate_extension(self, candidate):
        extension = candidate.get("sglang", {})
        if not isinstance(extension, dict) or set(extension) - {
            "interaction",
            "timebase",
            "tail_policy",
        }:
            raise ProtocolError("invalid_request", "unsupported sglang configuration")
        if (
            extension.get("interaction", self.capabilities.interaction)
            != self.capabilities.interaction
        ):
            raise ProtocolError("invalid_request", "interaction is unavailable")
        if (
            extension.get("tail_policy", self.capabilities.tail_policy)
            != self.capabilities.tail_policy
        ):
            raise ProtocolError("invalid_request", "tail policy is unavailable")
        timebase = extension.get("timebase", {})
        if not isinstance(timebase, dict) or set(timebase) - {
            "microturn_ms",
            "native_unit_ms",
        }:
            raise ProtocolError("invalid_request", "invalid timebase")
        if (
            timebase.get("native_unit_ms", self.capabilities.native_unit_ms)
            != self.capabilities.native_unit_ms
        ):
            raise ProtocolError("invalid_request", "native cadence is fixed")
        micro = timebase.get("microturn_ms")
        if micro is not None and (
            type(micro) not in (float, int) or not math.isfinite(micro) or micro <= 0
        ):
            raise ProtocolError("invalid_request", "invalid microturn duration")
        return micro

    def _grant(self, candidate, typ, requested, outputs, micro):
        grant = self.capabilities.describe()
        grant.update(
            output_modalities=outputs, limits=asdict(self.limits), rejections=[]
        )
        if micro is not None:
            grant["rejections"].append(
                dict(
                    field="sglang.timebase.microturn_ms",
                    requested=micro,
                    reason="external chunks are variable length; no fixed external cadence",
                    granted=None,
                )
            )
        grant["microturn_ms"] = None
        if outputs != requested:
            grant["rejections"].append(
                dict(
                    field="output_modalities",
                    requested=requested,
                    reason="deployment modalities",
                    granted=outputs,
                )
            )
        if self.capabilities.interaction == "turn_based":
            turn = (
                candidate.setdefault("audio", {})
                .setdefault("input", {})
                .setdefault("turn_detection", {"type": "server_vad"})
            )
            grant["supports_server_interrupt"] = turn is not None and turn.get(
                "interrupt_response", True
            )
        for direction, rate in (
            ("input", self.capabilities.input_rate),
            ("output", self.capabilities.output_rate),
        ):
            candidate.setdefault("audio", {}).setdefault(direction, {}).setdefault(
                "format", dict(type="audio/pcm", rate=rate)
            )
        candidate.update(model=self.model, type=typ, output_modalities=outputs)
        return candidate, grant

    async def update(self, patch, event_id):
        async with self.lock:
            if self.state not in ("CREATED", "OPEN"):
                raise ProtocolError("invalid_state", "session is closing")
            candidate, grant = self._negotiate(patch)
            if self.state == "CREATED":
                adapter = self.factory()
                adapter.set_limits(self.limits)
                adapter.set_cancel_handler(self.cancel)
                try:
                    await asyncio.wait_for(
                        adapter.open(self.session_id, candidate, self.emit),
                        self.limits.cleanup_timeout_s,
                    )
                except Exception as exc:
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
                self.state = "OPEN"
                self.worker = asyncio.create_task(self._pump())
            elif candidate != self.config:
                if self.adapter.supports_hot_update:
                    await self.adapter.update(candidate)
                else:
                    before = (
                        self.config.get("audio", {})
                        .get("input", {})
                        .get("turn_detection")
                    )
                    after = (
                        candidate.get("audio", {})
                        .get("input", {})
                        .get("turn_detection")
                    )
                    if before != after:
                        raise ProtocolError(
                            "not_applicable", "adapter cannot update turn detection"
                        )
            self.config, self.granted = candidate, grant
            self.notify(
                Updated(
                    self.session_id,
                    self.model,
                    candidate["type"],
                    grant,
                    event_id,
                    copy.deepcopy(candidate),
                )
            )

    async def append(self, pcm: bytes, seq: int, start_ms, event_id):
        async with self.lock:
            self._open()
            if self.eos:
                raise ProtocolError("invalid_state", "audio input has ended")
            if type(seq) is not int or seq != self.next_seq:
                raise ProtocolError("invalid_state", "audio seq must be contiguous")
            if not pcm or len(pcm) % 2:
                raise ProtocolError(
                    "invalid_request", "audio must contain whole PCM16 samples"
                )
            if start_ms is not None and (
                type(start_ms) not in (int, float)
                or not math.isfinite(start_ms)
                or not math.isclose(
                    start_ms, self.ms(self.accepted_samples), rel_tol=0, abs_tol=1e-7
                )
            ):
                raise ProtocolError(
                    "invalid_state", "input media time must be sample-contiguous"
                )
            if (
                self.accepted_samples - self.consumed_samples - self.discarded_samples
            ) * 2 + len(pcm) > self.limits.max_input_bytes or len(
                self.pending
            ) >= self.limits.max_input_chunks:
                raise ProtocolError(
                    "buffer_overflow", "input budget exhausted; retry this seq"
                )
            self.pending.append((self.accepted_samples, pcm))
            self.pending_bytes += len(pcm)
            self.accepted_samples += len(pcm) // 2
            self.next_seq += 1
            self.notify(Accepted(seq, self.ms(self.accepted_samples), event_id))
            self.wake.set()

    async def clear(self, event_id):
        async with self.lock:
            self._open()
            discarded = self.pending_bytes // 2
            discarded += await self.adapter.clear()
            self.pending.clear()
            self.pending_bytes = 0
            self.discarded_samples += discarded
            self.notify(Cleared(self.ms(discarded), event_id))

    async def end(self, event_id):
        async with self.lock:
            self._open()
            if self.eos:
                raise ProtocolError("invalid_state", "audio input already ended")
            unit_bytes = (
                self.capabilities.input_rate
                * self.capabilities.native_unit_ms
                // 1000
                * 2
            )
            if (
                self.capabilities.tail_policy == "reject"
                and self.pending_bytes % unit_bytes
            ):
                raise ProtocolError(
                    "invalid_state", "partial native unit; append more audio before EOS"
                )
            self.eos = True
            self.eos_event_id = event_id
            self.notify(
                Ended(
                    self.ms(self.accepted_samples),
                    self.capabilities.tail_policy,
                    event_id,
                )
            )
            self.wake.set()

    def _take(self, count):
        start = self.pending[0][0]
        chunks = []
        while count:
            position, pcm = self.pending.popleft()
            size = min(count, len(pcm))
            chunks.append(pcm[:size])
            if size < len(pcm):
                self.pending.appendleft((position + size // 2, pcm[size:]))
            count -= size
            self.pending_bytes -= size
        return start, b"".join(chunks)

    async def _pump(self):
        unit_bytes = (
            self.capabilities.input_rate * self.capabilities.native_unit_ms // 1000 * 2
        )
        try:
            while self.state == "OPEN":
                await self.wake.wait()
                async with self.lock:
                    self.wake.clear()
                    if self.state != "OPEN":
                        return
                    if self.pending_bytes < unit_bytes and not self.eos:
                        continue
                    size = min(unit_bytes, self.pending_bytes)
                    start, pcm = (
                        self._take(size) if size else (self.accepted_samples, b"")
                    )
                    real = len(pcm) // 2
                    eos = self.eos and not self.pending
                    if (
                        real
                        and size < unit_bytes
                        and self.capabilities.tail_policy == "pad"
                    ):
                        self.padding_samples += (unit_bytes - size) // 2
                        pcm += b"\0" * (unit_bytes - size)
                    unit = Unit(
                        self.unit_seq,
                        start,
                        pcm,
                        real,
                        eos,
                        tuple(self.granted["output_modalities"]),
                    )
                    self.unit_seq += 1
                    epoch = self.epoch
                self.producer_unit.set(unit)
                consumed = await self.adapter.process(unit, epoch)
                if isinstance(consumed, tuple):
                    consumed, discarded = consumed
                else:
                    discarded = real - consumed
                if (
                    any(type(x) is not int or x < 0 for x in (consumed, discarded))
                    or self.consumed_samples
                    + self.discarded_samples
                    + consumed
                    + discarded
                    > self.accepted_samples
                ):
                    raise RuntimeError(
                        "adapter did not provide valid media consumption"
                    )
                self.consumed_samples += consumed
                self.discarded_samples += discarded
                if self.close_task is None and epoch == self.epoch:
                    self._enqueue(
                        Envelope(epoch, UnitCompleted(f"unit_{unit.seq}"), unit=unit)
                    )
                if eos:
                    if self.close_task is not None:
                        return
                    self.notify(
                        Drained(
                            self.ms(self.accepted_samples),
                            self.ms(self.consumed_samples),
                            self.ms(self.discarded_samples),
                            self.ms(self.padding_samples),
                            self.eos_event_id,
                        )
                    )
                    return
                self.wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(
                str(exc), exc.code if isinstance(exc, ProtocolError) else "internal"
            )

    async def cancel(self, event_id):
        async with self.lock:
            self._open()
            old = self.epoch
            self.epoch += 1
            await asyncio.wait_for(self.adapter.cancel(), self.limits.cleanup_timeout_s)
            affected = self._finish_responses(old, "cancelled", "client_cancelled")
            self.audio_sent.clear()
            self.notify(Cancelled(old, self.epoch, tuple(affected), event_id))
            return self.epoch

    def _finish_responses(self, old, status, reason):
        affected = []
        for rid, state in self.responses.items():
            if state["epoch"] != old or not state["visible"] or state["terminal_sent"]:
                continue
            affected.append(rid)
            state["terminal"] = True
            item_id = state["item_id"] or "item_" + rid
            events = []
            if not state["text_done_sent"] and state["output_modalities"]:
                events.append(TextFinished(rid, item_id, state["text"]))
            if state["audio_visible"] and not state["audio_done_sent"]:
                events.append(AudioFinished(rid, item_id))
            events.append(
                ResponseFinished(
                    rid, item_id, state["text"], state["audio_visible"], status, reason
                )
            )
            for event in events:
                envelope = Envelope(
                    old,
                    event,
                    True,
                    output_modalities=tuple(state["output_modalities"]),
                )
                size = len(repr(envelope).encode())
                self.output.append((envelope, size))
                self.output_bytes += size
            self.output_wake.set()
        return affected

    def before_send(self, envelope: Envelope):
        """Cancellation must observe lifecycle visibility before the socket send yields."""
        event = envelope.event
        if isinstance(event, ResponseStarted):
            self.responses[event.response_id]["visible"] = True
        elif isinstance(event, ResponseFinished):
            self.responses[event.response_id]["terminal_sent"] = True
        elif isinstance(event, TextFinished):
            self.responses[event.response_id]["text_done_sent"] = True
        elif isinstance(event, AudioFinished):
            self.responses[event.response_id]["audio_done_sent"] = True
        if isinstance(event, AudioDelta):
            self.responses[event.response_id]["audio_visible"] = True

    def sent(self, envelope: Envelope):
        # Note (Junnan Li): Playback is acknowledged only after a successful transport send.
        event = envelope.event
        if isinstance(event, AudioDelta):
            key = envelope.epoch, event.response_id, event.item_id, 0
            sent, ack = self.audio_sent.get(key, (0, 0))
            self.audio_sent[key] = (
                sent + len(event.pcm) * 500 / self.capabilities.output_rate,
                ack,
            )

    async def playback_ack(
        self, epoch, response_id, item_id, content_index, audio_end_ms
    ):
        async with self.lock:
            self._open()
            if type(epoch) is not int or epoch != self.epoch:
                raise ProtocolError("stale_epoch", "ACK refers to another output epoch")
            key = epoch, response_id, item_id, content_index
            if (
                type(content_index) is not int
                or type(audio_end_ms) not in (int, float)
                or not math.isfinite(audio_end_ms)
            ):
                raise ProtocolError("invalid_request", "invalid playback position")
            previous = self.audio_sent.get(key)
            if previous is None or not previous[1] <= audio_end_ms <= previous[0]:
                raise ProtocolError(
                    "invalid_request",
                    "ACK must be monotonic and no later than sent audio",
                )
            self.audio_sent[key] = previous[0], audio_end_ms

    def _terminal(self, event):
        # Note (Junnan Li): Terminal notifications must remain deliverable after media overflow.
        terminals = [
            (env, size)
            for env, size in self.output
            if isinstance(env.event, Failure)
            or (
                env.control
                and isinstance(
                    env.event, (TextFinished, AudioFinished, ResponseFinished)
                )
            )
        ]
        self.output = deque(terminals)
        self.output_bytes = sum(size for _, size in self.output)
        envelope = Envelope(self.epoch, event, True)
        size = len(repr(envelope).encode())
        self.output.append((envelope, size))
        self.output_bytes += size
        self.output_wake.set()

    def fail(self, message, code="internal", event_id=None):
        if self.close_task is not None:
            return
        self._terminal(Failure(code, message[:512], True, event_id))
        self.close_task = asyncio.create_task(self._close(code))

    async def timeout(self):
        await asyncio.sleep(
            max(0, self.limits.session_timeout_s - (time.monotonic() - self.created_at))
        )
        self.fail("session duration exceeded", "session_timeout")

    async def close(self, reason, event_id=None):
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close(reason, event_id))
        await asyncio.shield(self.close_task)

    async def _close(self, reason, event_id=None):
        # Note (Junnan Li): Set CLOSING under the command lock, then release it: adapter
        # teardown can run a VAD callback that must observe CLOSING.
        async with self.lock:
            self.state = "CLOSING"
        await self._close_state(reason, event_id)

    async def _close_state(self, reason, event_id=None):
        old_epoch = self.epoch
        self.epoch += 1
        self.discarded_samples += self.pending_bytes // 2
        self.pending.clear()
        self.pending_bytes = 0
        self.wake.set()
        cleanup_error = None
        try:
            if self.adapter is not None:
                await asyncio.wait_for(
                    self.adapter.close(), self.limits.cleanup_timeout_s
                )
        except Exception as exc:
            cleanup_error = exc
        finally:
            try:
                await cancel_local_tasks([self.worker], self.limits.cleanup_timeout_s)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            if cleanup_error is not None:
                self.output.clear()
                self.output_bytes = 0
                self._terminal(
                    Failure("cleanup_timeout", str(cleanup_error)[:512], True, event_id)
                )
            else:
                status = (
                    "cancelled"
                    if reason in ("client_closed", "disconnect")
                    else "failed"
                )
                self._finish_responses(old_epoch, status, reason)
                self._terminal(Closed(reason, event_id))
        finally:
            self.state = "CLOSED"
            self.output_wake.set()
