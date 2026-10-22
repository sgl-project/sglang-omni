# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler base.

``StreamingVocoderBase`` owns the request-local streaming-vocoder lifecycle
shared by model schedulers:

- per-request state registry: created on first chunk/payload via
  ``create_stream_state``, popped by ``clear_stream_state``
- ``is_streaming_payload`` = ``bool(params["stream"])`` with params validation
- the chunk validation scaffold (metadata dict / modality / stream flag /
  tensor type) with per-request contract latching via ``latch_stream_contract``
- threshold-accumulate -> decode-delta -> emit ordering, wrapping waveforms in
  ``OutgoingMessage(type="stream", metadata={"modality": "audio"})``
- request-level ``initial_codec_chunk_frames`` resolution
- ``on_stream_done`` sequencing: flush remainder -> final stream chunk ->
  terminal result -> state cleared
- whole-utterance ``fallback_full_decode`` when nothing was emitted
- abort and scheduler stop -> ``release_stream_resources`` -> state pop; late
  chunks for aborted/cleared request ids are dropped and never recreate state
- opt-in cross-request coalescing (``can_batch_stream_chunks``) through
  ``select_step_participants`` / ``build_step_plan`` / ``run_step`` /
  ``on_step_failure``; a failed step errors and aborts all its participants,
  and chunk delivery is batch-only (direct ``on_stream_chunk`` is rejected)

Models own cursor math, buffers, codec invocation/sessions, CUDA-graph
handling, and result shapes through the hooks. The base adds no locking of its
own beyond the inherited scheduler locks: subclasses that share codec state
across threads keep their own ``state_lock`` discipline.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Mapping, Sequence
from types import ModuleType

import torch
from typing_extensions import Generic, TypeVar

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.proto.session import (
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.proto.streams import (
    AUDIO,
    CONTROL,
    INTERRUPT,
    PCM_F32,
    REPLY_PROGRESS,
    SESSION_CONTROL,
    TEXT,
    TURN_DONE,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

INITIAL_CODEC_CHUNK_FRAMES_PARAM = "initial_codec_chunk_frames"

_COMPLETED_STREAM_REQUEST_ID_LIMIT = 10000
_COMPLETED_STREAM_REQUEST_ID_RETAINED = 5000

StreamStateT = TypeVar("StreamStateT")
StepPlanT = TypeVar("StepPlanT")


def resolve_initial_codec_chunk_frames(
    params: Mapping[str, object] | None,
    *,
    steady_chunk_frames: int,
    default_frames: int = 0,
) -> int:
    """Resolve the model default or request override, clamped to steady size."""
    if steady_chunk_frames <= 0:
        raise ValueError(
            f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
        )
    else:
        pass
    value = params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) if params is not None else None
    if value is None:
        value = default_frames
    else:
        pass

    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be an integer"
        ) from exc
    if frames < 0:
        raise ValueError(f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be >= 0")
    else:
        pass

    return min(frames, int(steady_chunk_frames))


def vocoder_decode_stream_priority(device_module: ModuleType) -> int:
    """The second-highest stream priority of the device: ahead of the talker's
    default-priority stream, with the top level left free. A device with two levels
    uses the top one."""
    least_priority, greatest_priority = device_module.Stream.priority_range()
    if greatest_priority + 1 < least_priority:
        return greatest_priority + 1
    else:
        pass
    return greatest_priority


class StreamingVocoderBase(
    StreamingSimpleScheduler, ABC, Generic[StreamStateT, StepPlanT]
):
    """Template-method base for streaming vocoder schedulers.

    The base owns the lifecycle skeleton; subclasses implement the abstract
    hooks (``create_stream_state``, ``validate_chunk``, ``ingest``,
    ``decode_delta``, ``final_result_data``) and optionally the contract
    latch, fallback, resource-release, serving-lifecycle, and coalescing
    hooks. Non-streaming requests use ``compute_fn`` / ``batch_compute_fn``
    exactly like :class:`StreamingSimpleScheduler`.

    ``stream_source_hint`` names the model in client-visible payloads and in
    the base-owned scaffold error/log text; it defaults to the subclass name.
    ``stream_input_modality`` defaults to discrete ``audio_codes`` and can be
    set to ``audio_latents`` by continuous-latent vocoders.
    ``reply_unit_format`` names the Session unit format whose one unit is a
    whole reply; the vocoder then publishes ``turn.started`` before that
    unit's first chunk and ``turn.done`` after its last. Other units carry
    text deltas of a reply and publish neither.
    """

    pump_on_chunk_batch = True

    def __init__(
        self,
        compute_fn: (
            Callable[
                [StagePayload],
                StagePayload | Coroutine[None, None, StagePayload],
            ]
            | None
        ),
        *,
        sample_rate: int,
        stream_source_hint: str | None = None,
        stream_input_modality: str = "audio_codes",
        reply_unit_format: str | None = None,
        batch_compute_fn: (
            Callable[
                [list[StagePayload]],
                Sequence[StagePayload] | Coroutine[None, None, Sequence[StagePayload]],
            ]
            | None
        ) = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Callable[[StagePayload], int] | None = None,
        max_batch_cost: int | None = None,
        abort_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.stream_states: dict[str, StreamStateT] = {}
        self.emitted_stream_ids: set[str] = set()
        self.completed_stream_request_ids: dict[str, None] = {}
        # Note (Junnan Li): A Session's appends reach the client as timed chunks in arrival order; a control unit waits for the units before it.
        self.session_appends: dict[str, SessionOperation] = {}
        # Note (Junnan Li): A unit that is a whole reply publishes turn.started before its first chunk and turn.done after its last.
        self.started_reply_units: set[str] = set()
        self.session_units: dict[str, list[str]] = {}
        self.session_unit_audio_ms: dict[str, float] = {}
        self.pending_session_controls: dict[str, list[OutgoingMessage]] = {}
        self.session_reads: dict[str, IncomingMessage] = {}
        self.reply_progress: dict[str, tuple[int | None, int, float]] = {}
        self.sample_rate = int(sample_rate)
        self.reply_unit_format = reply_unit_format
        self.stream_source_hint = stream_source_hint or type(self).__name__
        self.stream_input_modality = str(stream_input_modality)
        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
            abort_callback=abort_callback,
        )

    def start(self) -> None:
        try:
            self.on_serving_start()
            super().start()
        finally:
            self.shutdown_stream_states()

    def stop(self) -> None:
        was_running = self.running
        super().stop()
        if not was_running:
            self.shutdown_stream_states()
        else:
            pass

    def shutdown_stream_states(self) -> None:
        """Release every live state through ``clear_stream_state`` (the same
        path as completion/abort, so ``release_stream_resources`` runs), then
        let the subclass tear down session-level resources. A failing release
        is logged so the remaining states and ``on_serving_stop`` still run."""
        with self.state_lock:
            for request_id in list(self.stream_states):
                try:
                    self.clear_stream_state(request_id)
                except Exception:
                    logger.exception(
                        "%s failed to release stream state for %s during shutdown",
                        self.stream_source_hint,
                        request_id,
                    )
            self.emitted_stream_ids.clear()
            self.completed_stream_request_ids.clear()
            self.on_serving_stop()

    def handle_new_request_batch_with_loop(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        """A Session's open, read, close and control are answered here; an interrupt drops the units of its turn."""
        requests: list[IncomingMessage] = []
        for message in batch:
            operation = find_session_operation(message.data.request.metadata)
            if operation is None or operation.operation == "append":
                requests.append(message)
                continue
            else:
                pass
            session_id = operation.session_identity.id
            if operation.operation == "open":
                self.session_units.setdefault(session_id, [])
                self.answer_session_operation(message, {"opened": True})
            elif operation.operation == "read":
                # Note (Junnan Li): The read stays open until the close answers it; the units' chunks stream on their own request ids.
                self.session_reads[session_id] = message
            elif operation.operation == "close":
                for request_id in list(self.session_units.get(session_id, [])):
                    self.pending_session_controls.pop(request_id, None)
                    self.abort(request_id)
                self.session_units.pop(session_id, None)
                self.reply_progress.pop(session_id, None)
                read = self.session_reads.pop(session_id, None)
                if read is not None:
                    self.answer_session_operation(read, {"closed": True})
                else:
                    pass
                self.answer_session_operation(message, {"closed": True})
            elif operation.operation == "control":
                event = operation.chunk
                if control_type(event) == INTERRUPT and event.turn is not None:
                    for request_id, unit in list(self.session_appends.items()):
                        if (
                            unit.session_identity == operation.session_identity
                            and unit.chunk.turn == event.turn
                        ):
                            self.abort(request_id)
                        else:
                            pass
                else:
                    pass
                self.answer_session_operation(message, {"controlled": True})
            else:
                raise ValueError("unknown session operation")
        super().handle_new_request_batch_with_loop(requests, loop)

    def answer_session_operation(
        self, message: IncomingMessage, data: dict[str, bool]
    ) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=message.request_id,
                type="result",
                data=StagePayload(message.request_id, message.data.request, data),
            )
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        """A Session unit always streams; a control unit relays at its stream done."""
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"{self.stream_source_hint} request params must be a dict, got "
                f"{type(params).__name__}"
            )
        elif find_session_operation(payload.request.metadata) is not None:
            return True
        else:
            return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.completed_stream_request_ids.pop(request_id, None)
        operation = find_session_operation(payload.request.metadata)
        if operation is not None and operation.operation == "append":
            self.session_appends[request_id] = operation
            self.session_units.setdefault(operation.session_identity.id, []).append(
                request_id
            )
            self.session_unit_audio_ms[request_id] = 0.0
        else:
            pass
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return
        else:
            pass
        self.latch_stream_contract(request_id, state, payload, origin="payload")

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if self.can_batch_stream_chunks:
            # note (Gaokai): the coalesced pump must run under ``state_lock``
            # with abort cleanup deferred; only ``on_stream_chunk_batch``
            # provides that, so reject the unlocked entry point outright.
            raise RuntimeError(
                f"{type(self).__name__} coalesces stream chunks; deliver them "
                "through on_stream_chunk_batch"
            )
        else:
            pass
        state = self.ingest_stream_item(request_id, item)
        if state is None:
            return []
        else:
            pass
        return self.decode_and_emit(request_id, state)

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Ingest every chunk then run one coalesced pump under ``state_lock``;
        external abort callbacks for failed requests run after the lock is
        released, matching the serving loop."""
        failed: list[str] = []
        with self.state_lock:
            for request_id, item in items:
                if self.is_aborted(request_id):
                    continue
                else:
                    pass
                try:
                    self.ingest_stream_item(request_id, item)
                except Exception as exc:
                    self.emit_error(request_id, exc)
                    self.abort_state(request_id)
                    failed.append(request_id)
            if self.pump_on_chunk_batch:
                failed.extend(self.pump_streams())
            else:
                pass
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def handle_stream_chunk(self, request_id: str, item: object) -> None:
        """Coalescing schedulers route single-chunk deliveries through the
        batch backbone, so the deferred abort cleanup stays off ``state_lock``
        (the inherited path would run ``on_stream_chunk`` with the lock held)."""
        if not self.can_batch_stream_chunks:
            super().handle_stream_chunk(request_id, item)
            return
        else:
            pass
        item = self.validate_stream_chunk_item(request_id, item)
        self.on_stream_chunk_batch([(request_id, item)])

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        return self.finish_stream(request_id)

    def finish_stream(self, request_id: str) -> list[OutgoingMessage]:
        """Flush the remainder (or the nothing-emitted fallback), then build
        the terminal result; records the id as completed."""
        payload = self.stream_payloads[request_id]
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return []
        else:
            pass
        operation = self.session_appends.get(request_id)
        if operation is not None and operation.chunk.format == SESSION_CONTROL:
            return self.finish_session_control(request_id, payload, state)
        else:
            pass
        waveform = self.decode_delta(request_id, state, is_final=True)
        if (
            waveform is None
            and not self.stream_has_emitted(request_id)
            and operation is None
        ):
            waveform = self.fallback_full_decode(request_id, payload, state)
        else:
            pass
        messages: list[OutgoingMessage] = []
        if waveform is not None:
            self.mark_stream_emitted(request_id)
            messages.append(self.stream_chunk_message(request_id, waveform))
        else:
            pass
        messages.extend(self.terminal_messages(request_id, payload, state))
        return messages

    def terminal_messages(
        self, request_id: str, payload: StagePayload, state: StreamStateT
    ) -> list[OutgoingMessage]:
        """The request's result, preceded for a Session unit by its reply progress and followed by the controls that waited for it."""
        messages: list[OutgoingMessage] = []
        operation = self.session_appends.get(request_id)
        if operation is not None:
            messages.extend(self.reply_unit_start(request_id, operation))
            messages.append(self.reply_progress_message(request_id, operation))
            if self.is_reply_unit(operation):
                messages.append(
                    self.session_control_message(request_id, operation, TURN_DONE)
                )
            else:
                pass
        else:
            pass
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=(
                        {}
                        if operation is not None
                        else self.final_result_data(request_id, payload, state)
                    ),
                ),
            )
        )
        self.record_completed_stream_request_id(request_id)
        messages.extend(self.release_session_unit(request_id))
        return messages

    def reply_progress_message(
        self, request_id: str, operation: SessionOperation
    ) -> OutgoingMessage:
        chunk = operation.chunk
        session_id = operation.session_identity.id
        turn, text_end, audio_ms = self.reply_progress.get(session_id, (None, 0, 0.0))
        if turn != chunk.turn:
            text_end, audio_ms = 0, 0.0
        else:
            pass
        if not self.is_reply_unit(operation):
            text_end += len(str(chunk.payload["text"]))
        else:
            pass
        audio_ms += self.session_unit_audio_ms.get(request_id, 0.0)
        self.reply_progress[session_id] = (chunk.turn, text_end, audio_ms)
        progress = TimedChunk(
            "control",
            0,
            0,
            chunk.seq,
            {"type": REPLY_PROGRESS, "text_end": text_end, "audio_ms": audio_ms},
            format=SESSION_CONTROL,
            turn=chunk.turn,
        )
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=progress.to_dict(),
            metadata={"modality": progress.modality},
        )

    def finish_session_control(
        self, request_id: str, payload: StagePayload, state: StreamStateT
    ) -> list[OutgoingMessage]:
        """Relay an appended control to the client once every unit appended before it has finished."""
        operation = self.session_appends[request_id]
        chunk = operation.chunk
        session_id = operation.session_identity.id
        if self.reply_progress.get(session_id, (None,))[0] == chunk.turn:
            del self.reply_progress[session_id]
        else:
            pass
        messages = [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk.to_dict(),
                metadata={"modality": chunk.modality},
            ),
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id, request=payload.request, data={}
                ),
            ),
        ]
        if self.session_units[session_id][0] == request_id:
            self.record_completed_stream_request_id(request_id)
            return [*messages, *self.release_session_unit(request_id)]
        else:
            self.pending_session_controls[request_id] = messages
            return []

    def release_session_unit(self, request_id: str) -> list[OutgoingMessage]:
        """Forget a finished unit; the controls queued behind it that are now first in line go out."""
        operation = self.session_appends.pop(request_id, None)
        self.session_unit_audio_ms.pop(request_id, None)
        self.started_reply_units.discard(request_id)
        if operation is None:
            return []
        else:
            pass
        units = self.session_units[operation.session_identity.id]
        units.remove(request_id)
        released: list[OutgoingMessage] = []
        while units and units[0] in self.pending_session_controls:
            control_id = units[0]
            released.extend(self.pending_session_controls.pop(control_id))
            self.record_completed_stream_request_id(control_id)
            released.extend(self.release_session_unit(control_id))
        if not units:
            self.session_units.pop(operation.session_identity.id, None)
        else:
            pass
        return released

    def clear_stream_state(self, request_id: str) -> None:
        self.emitted_stream_ids.discard(request_id)
        if request_id in self.pending_session_controls and not self.is_aborted(
            request_id
        ):
            # Note (Junnan Li): A relayed control keeps waiting for the units before it after its own request state is gone.
            pass
        elif request_id in self.session_appends:
            # Note (Junnan Li): An aborted unit still lets the controls queued behind it go out.
            self.pending_session_controls.pop(request_id, None)
            for message in self.release_session_unit(request_id):
                self.outbox.put(message)
        else:
            pass
        state = self.stream_states.pop(request_id, None)
        if state is not None:
            self.release_stream_resources(request_id, state)
        else:
            pass

    def get_or_create_stream_state(self, request_id: str) -> StreamStateT | None:
        """Registry accessor. Returns None for aborted or completed request ids
        whose state was already cleared: late chunks are dropped and never
        recreate state (or re-acquire per-request resources)."""
        state = self.stream_states.get(request_id)
        if state is not None:
            return state
        else:
            pass
        if (
            self.is_aborted(request_id)
            or request_id in self.completed_stream_request_ids
        ):
            return None
        else:
            pass
        state = self.create_stream_state(request_id)
        if state is None:
            raise RuntimeError(
                f"{type(self).__name__}.create_stream_state returned None for "
                f"{request_id!r}"
            )
        else:
            pass
        self.stream_states[request_id] = state
        return state

    def stream_state_items(self) -> list[tuple[str, StreamStateT]]:
        return list(self.stream_states.items())

    def stream_has_emitted(self, request_id: str) -> bool:
        return request_id in self.emitted_stream_ids

    def mark_stream_emitted(self, request_id: str) -> None:
        self.emitted_stream_ids.add(request_id)

    def record_completed_stream_request_id(self, request_id: str) -> None:
        self.completed_stream_request_ids[request_id] = None
        if len(self.completed_stream_request_ids) <= _COMPLETED_STREAM_REQUEST_ID_LIMIT:
            return
        else:
            pass
        excess = (
            len(self.completed_stream_request_ids)
            - _COMPLETED_STREAM_REQUEST_ID_RETAINED
        )
        for stale_request_id in list(self.completed_stream_request_ids)[:excess]:
            del self.completed_stream_request_ids[stale_request_id]

    def ingest_stream_item(
        self, request_id: str, item: StreamItem
    ) -> StreamStateT | None:
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return None
        else:
            pass
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"{self.stream_source_hint} stream chunk for {request_id!r} is "
                "missing metadata"
            )
        else:
            pass
        operation = self.session_appends.get(request_id)
        if operation is not None and metadata.get("modality") == TEXT:
            # Note (Junnan Li): A reply's text reaches the client through its audio's owner, in order with the audio.
            for message in self.reply_unit_start(request_id, operation):
                self.outbox.put(message)
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=metadata["chunk"],
                    metadata={"modality": TEXT},
                )
            )
            return None
        else:
            pass
        if metadata.get("modality") not in (None, self.stream_input_modality):
            raise ValueError(
                f"{self.stream_source_hint} stream chunk modality must be "
                f"{self.stream_input_modality}, got {metadata.get('modality')!r}"
            )
        else:
            pass
        if not isinstance(metadata.get("stream"), bool):
            raise RuntimeError(
                f"{self.stream_source_hint} stream chunk for {request_id!r} must "
                "include a bool metadata['stream']"
            )
        else:
            pass
        self.latch_stream_contract(
            request_id, state, metadata, origin="stream metadata"
        )
        codes = item.data
        if not isinstance(codes, torch.Tensor):
            raise TypeError(
                f"{self.stream_source_hint} stream chunk for {request_id!r} must "
                f"carry a torch.Tensor, got {type(codes).__name__}"
            )
        else:
            pass
        codes = self.validate_chunk(request_id, state, codes)
        self.ingest(request_id, state, codes)
        return state

    def decode_and_emit(
        self, request_id: str, state: StreamStateT
    ) -> list[OutgoingMessage]:
        if not self.should_decode(state, is_final=False):
            return []
        else:
            pass
        waveform = self.decode_delta(request_id, state, is_final=False)
        if waveform is None:
            return []
        else:
            pass
        self.mark_stream_emitted(request_id)
        return [self.stream_chunk_message(request_id, waveform)]

    def stream_chunk_message(
        self, request_id: str, waveform: torch.Tensor
    ) -> OutgoingMessage:
        operation = self.session_appends.get(request_id)
        if operation is None:
            return OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=self.stream_payload(request_id, waveform),
                metadata={"modality": "audio"},
            )
        else:
            for message in self.reply_unit_start(request_id, operation):
                self.outbox.put(message)
            samples = waveform.detach().reshape(-1).to(torch.float32).cpu().numpy()
            duration_ms = len(samples) * 1000 / self.sample_rate
            chunk = TimedChunk(
                AUDIO,
                self.session_unit_audio_ms[request_id],
                duration_ms,
                operation.chunk.seq,
                samples.astype("<f4").tobytes(),
                format=PCM_F32,
                turn=operation.chunk.turn,
            )
            self.session_unit_audio_ms[request_id] += duration_ms
            return OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk.to_dict(),
                metadata={"modality": AUDIO},
            )

    def is_reply_unit(self, operation: SessionOperation) -> bool:
        return (
            self.reply_unit_format is not None
            and operation.chunk.format == self.reply_unit_format
        )

    def reply_unit_start(
        self, request_id: str, operation: SessionOperation
    ) -> list[OutgoingMessage]:
        if not self.is_reply_unit(operation) or request_id in self.started_reply_units:
            return []
        else:
            self.started_reply_units.add(request_id)
            return [self.session_control_message(request_id, operation, TURN_STARTED)]

    @staticmethod
    def session_control_message(
        request_id: str, operation: SessionOperation, event: str
    ) -> OutgoingMessage:
        chunk = TimedChunk(
            CONTROL,
            0,
            0,
            operation.chunk.seq,
            {"type": event},
            format=SESSION_CONTROL,
            turn=operation.chunk.turn,
        )
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=chunk.to_dict(),
            metadata={"modality": chunk.modality},
        )

    def pump_one_step(self) -> list[str] | None:
        """Returns None when no stream is ready, otherwise the request ids
        on_step_failure aborted (empty after a successful step)."""
        participants = self.select_step_participants()
        if not participants:
            return None
        else:
            pass
        plan = self.build_step_plan(participants)
        try:
            decoded = self.run_step(participants, plan)
        except Exception as exc:
            return list(self.on_step_failure(participants, exc))
        for request_id, _ in participants:
            waveform = decoded.get(request_id)
            if waveform is not None and not self.is_aborted(request_id):
                self.mark_stream_emitted(request_id)
                self.outbox.put(self.stream_chunk_message(request_id, waveform))
            else:
                pass
        return []

    def pump_streams(self) -> list[str]:
        """Coalesced decode loop: steps run until no participants remain, so
        capped-step backlogs drain within one pump; a failed step ends the
        pump and returns the aborted request ids."""
        while True:
            failed = self.pump_one_step()
            if failed is None:
                return []
            else:
                pass
            if failed:
                return failed
            else:
                pass

    def run_ready_step(self) -> None:
        with self.state_lock:
            failed = self.pump_one_step() or []
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    @abstractmethod
    def create_stream_state(self, request_id: str) -> StreamStateT:
        """Fresh per-request state; created on the first chunk or payload."""

    def latch_stream_contract(
        self,
        request_id: str,
        state: StreamStateT,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        """Latch the per-request codec contract. ``source`` is the
        :class:`StagePayload` when ``origin == "payload"`` and the chunk
        metadata mapping when ``origin == "stream metadata"``. Implementations
        must enforce immutability of already-latched values."""
        del request_id, state, source, origin

    @abstractmethod
    def validate_chunk(
        self, request_id: str, state: StreamStateT, codes: torch.Tensor
    ) -> torch.Tensor:
        """Model shape/dtype checks; returns the code tensor to buffer."""

    @abstractmethod
    def ingest(self, request_id: str, state: StreamStateT, codes: torch.Tensor) -> None:
        """Append validated codes to the model buffer."""

    def should_decode(self, state: StreamStateT, *, is_final: bool) -> bool:
        """Threshold gate for the per-chunk decode. Consulted only with
        ``is_final=False``: the stream-done flush always calls ``decode_delta``
        directly. Defaults to letting ``decode_delta`` decide (it may return
        None)."""
        del state, is_final
        return True

    @abstractmethod
    def decode_delta(
        self, request_id: str, state: StreamStateT, *, is_final: bool
    ) -> torch.Tensor | None:
        """All cursor/overlap/crossfade/holdback math plus the codec call;
        ``is_final`` flushes the remainder at stream-done. None emits nothing."""

    def stream_payload(
        self, request_id: str, waveform: torch.Tensor
    ) -> dict[str, bytes | list[int] | str | int]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self.sample_rate,
            modality="audio",
            source_hint=self.stream_source_hint,
        )

    @abstractmethod
    def final_result_data(
        self, request_id: str, payload: StagePayload, state: StreamStateT
    ) -> Mapping[str, object]:
        """Terminal ``result`` payload data (metadata-only or full audio)."""

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: StreamStateT
    ) -> torch.Tensor | None:
        """Whole-utterance decode used when the stream finished without ever
        emitting audio; None keeps the terminal result audio-free."""
        del request_id, payload, state
        return None

    def release_stream_resources(self, request_id: str, state: StreamStateT) -> None:
        """Release per-request resources (e.g. codec session slots); invoked
        from ``clear_stream_state`` on completion, abort, and scheduler stop."""
        del request_id, state

    def on_serving_start(self) -> None:
        """Runs once before the serving loop."""

    def on_serving_stop(self) -> None:
        """Session teardown; runs after the serving loop exits (or on ``stop``
        of a never-started scheduler) with the state registry already cleared."""

    def warmup_now(self) -> None:
        """Synchronous warmup at factory-build time, before the stage is
        marked ready (e.g. CUDA-graph capture)."""

    def select_step_participants(self) -> list[tuple[str, StreamStateT]]:
        """Streams joining the next coalesced step; [] ends the pump."""
        raise RuntimeError(
            f"{type(self).__name__} enables stream-chunk coalescing but does "
            "not implement select_step_participants"
        )

    def build_step_plan(
        self, participants: list[tuple[str, StreamStateT]]
    ) -> StepPlanT:
        raise RuntimeError(
            f"{type(self).__name__} enables stream-chunk coalescing but does "
            "not implement build_step_plan"
        )

    def run_step(
        self, participants: list[tuple[str, StreamStateT]], plan: StepPlanT
    ) -> dict[str, torch.Tensor]:
        """Advance all participants by one shared step; consumes their buffers
        and returns waveforms keyed by request id. A participant it omits
        emits nothing this step and stays eligible for the nothing-emitted
        fallback at stream-done."""
        raise RuntimeError(
            f"{type(self).__name__} enables stream-chunk coalescing but does "
            "not implement run_step"
        )

    def on_step_failure(
        self, participants: list[tuple[str, StreamStateT]], exc: BaseException
    ) -> list[str]:
        """Runs under ``state_lock``: emits the error and clears state for
        every participant, then returns the request ids whose external abort
        callback the caller must run once the lock is released (never invoke
        ``abort``/``cleanup_aborted_request`` here — the callback must stay
        off the GPU-serializing lock)."""
        logger.exception(
            "%s streaming decode step failed; aborting %d participating request(s)",
            self.stream_source_hint,
            len(participants),
        )
        failed: list[str] = []
        for request_id, _ in participants:
            self.emit_error(request_id, exc)
            self.abort_state(request_id)
            failed.append(request_id)
        return failed


__all__ = [
    "INITIAL_CODEC_CHUNK_FRAMES_PARAM",
    "StreamingVocoderBase",
    "resolve_initial_codec_chunk_frames",
    "vocoder_decode_stream_priority",
]
