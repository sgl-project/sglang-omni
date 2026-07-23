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
- opt-in cross-request coalescing (``_can_batch_stream_chunks``) through
  ``select_step_participants`` / ``build_step_plan`` / ``run_step`` /
  ``on_step_failure``; a failed step errors and aborts all its participants,
  and chunk delivery is batch-only (direct ``on_stream_chunk`` is rejected)

Models own cursor math, buffers, codec invocation/sessions, CUDA-graph
handling, and result shapes through the hooks. The base adds no locking of its
own beyond the inherited scheduler locks: subclasses that share codec state
across threads keep their own ``_state_lock`` discipline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Mapping, TypeVar

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

INITIAL_CODEC_CHUNK_FRAMES_PARAM = "initial_codec_chunk_frames"

_COMPLETED_STREAM_REQUEST_ID_LIMIT = 10000
_COMPLETED_STREAM_REQUEST_ID_RETAINED = 5000

StreamStateT = TypeVar("StreamStateT")
StepPlanT = TypeVar("StepPlanT")


def resolve_initial_codec_chunk_frames(
    params: Mapping[str, Any] | None,
    *,
    steady_chunk_frames: int,
) -> int:
    """Return the request-level first codec chunk size, clamped to steady size."""
    if steady_chunk_frames <= 0:
        raise ValueError(
            f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
        )
    if params is None:
        return 0

    value = params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM)
    if value is None:
        return 0

    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be an integer"
        ) from exc
    if frames < 0:
        raise ValueError(f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be >= 0")

    return min(frames, int(steady_chunk_frames))


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
    """

    def __init__(
        self,
        compute_fn: Callable[[Any], Any] | None,
        *,
        sample_rate: int,
        stream_source_hint: str | None = None,
        batch_compute_fn: Callable[[list[Any]], list[Any]] | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
        abort_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._stream_states: dict[str, StreamStateT] = {}
        self._emitted_stream_ids: set[str] = set()
        self._completed_stream_request_ids: dict[str, None] = {}
        self._sample_rate = int(sample_rate)
        self._stream_source_hint = stream_source_hint or type(self).__name__
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
            self._shutdown_stream_states()

    def stop(self) -> None:
        was_running = self._running
        super().stop()
        if not was_running:
            self._shutdown_stream_states()

    def _shutdown_stream_states(self) -> None:
        """Release every live state through ``clear_stream_state`` (the same
        path as completion/abort, so ``release_stream_resources`` runs), then
        let the subclass tear down session-level resources. A failing release
        is logged so the remaining states and ``on_serving_stop`` still run."""
        with self._state_lock:
            for request_id in list(self._stream_states):
                try:
                    self.clear_stream_state(request_id)
                except Exception:
                    logger.exception(
                        "%s failed to release stream state for %s during shutdown",
                        self._stream_source_hint,
                        request_id,
                    )
            self._emitted_stream_ids.clear()
            self._completed_stream_request_ids.clear()
            self.on_serving_stop()

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"{self._stream_source_hint} request params must be a dict, got "
                f"{type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self._completed_stream_request_ids.pop(request_id, None)
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return
        self.latch_stream_contract(request_id, state, payload, origin="payload")

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if self._can_batch_stream_chunks:
            # note (Gaokai): the coalesced pump must run under ``_state_lock``
            # with abort cleanup deferred; only ``on_stream_chunk_batch``
            # provides that, so reject the unlocked entry point outright.
            raise RuntimeError(
                f"{type(self).__name__} coalesces stream chunks; deliver them "
                "through on_stream_chunk_batch"
            )
        state = self._ingest_stream_item(request_id, item)
        if state is None:
            return []
        return self._decode_and_emit(request_id, state)

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Ingest every chunk then run one coalesced pump under ``_state_lock``;
        external abort callbacks for failed requests run after the lock is
        released, matching the serving loop."""
        failed: list[str] = []
        with self._state_lock:
            for request_id, item in items:
                if self._is_aborted(request_id):
                    continue
                try:
                    self._ingest_stream_item(request_id, item)
                except Exception as exc:
                    self._emit_error(request_id, exc)
                    self._abort_state(request_id)
                    failed.append(request_id)
            failed.extend(self._pump_streams())
        for request_id in failed:
            self._cleanup_aborted_request(request_id)

    def _handle_stream_chunk(self, request_id: str, item: Any) -> None:
        """Coalescing schedulers route single-chunk deliveries through the
        batch backbone, so the deferred abort cleanup stays off ``_state_lock``
        (the inherited path would run ``on_stream_chunk`` with the lock held)."""
        if not self._can_batch_stream_chunks:
            super()._handle_stream_chunk(request_id, item)
            return
        item = self._validate_stream_chunk_item(request_id, item)
        self.on_stream_chunk_batch([(request_id, item)])

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return []
        waveform = self.decode_delta(request_id, state, is_final=True)
        if waveform is None and not self._stream_has_emitted(request_id):
            waveform = self.fallback_full_decode(request_id, payload, state)
        messages: list[OutgoingMessage] = []
        if waveform is not None:
            self._mark_stream_emitted(request_id)
            messages.append(self._stream_chunk_message(request_id, waveform))
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=self.final_result_data(request_id, payload, state),
                ),
            )
        )
        self._record_completed_stream_request_id(request_id)
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._emitted_stream_ids.discard(request_id)
        state = self._stream_states.pop(request_id, None)
        if state is not None:
            self.release_stream_resources(request_id, state)

    def _get_or_create_stream_state(self, request_id: str) -> StreamStateT | None:
        """Registry accessor. Returns None for aborted or completed request ids
        whose state was already cleared: late chunks are dropped and never
        recreate state (or re-acquire per-request resources)."""
        state = self._stream_states.get(request_id)
        if state is not None:
            return state
        if (
            self._is_aborted(request_id)
            or request_id in self._completed_stream_request_ids
        ):
            return None
        state = self.create_stream_state(request_id)
        if state is None:
            raise RuntimeError(
                f"{type(self).__name__}.create_stream_state returned None for "
                f"{request_id!r}"
            )
        self._stream_states[request_id] = state
        return state

    def _stream_state_items(self) -> list[tuple[str, StreamStateT]]:
        return list(self._stream_states.items())

    def _stream_has_emitted(self, request_id: str) -> bool:
        return request_id in self._emitted_stream_ids

    def _mark_stream_emitted(self, request_id: str) -> None:
        self._emitted_stream_ids.add(request_id)

    def _record_completed_stream_request_id(self, request_id: str) -> None:
        self._completed_stream_request_ids[request_id] = None
        if (
            len(self._completed_stream_request_ids)
            <= _COMPLETED_STREAM_REQUEST_ID_LIMIT
        ):
            return
        excess = (
            len(self._completed_stream_request_ids)
            - _COMPLETED_STREAM_REQUEST_ID_RETAINED
        )
        for stale_request_id in list(self._completed_stream_request_ids)[:excess]:
            del self._completed_stream_request_ids[stale_request_id]

    def _ingest_stream_item(
        self, request_id: str, item: StreamItem
    ) -> StreamStateT | None:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return None
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"{self._stream_source_hint} stream chunk for {request_id!r} is "
                "missing metadata"
            )
        if metadata.get("modality") not in (None, "audio_codes"):
            raise ValueError(
                f"{self._stream_source_hint} stream chunk modality must be "
                f"audio_codes, got {metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"{self._stream_source_hint} stream chunk for {request_id!r} must "
                "include metadata['stream'] == True"
            )
        self.latch_stream_contract(
            request_id, state, metadata, origin="stream metadata"
        )
        codes = item.data
        if not isinstance(codes, torch.Tensor):
            raise TypeError(
                f"{self._stream_source_hint} stream chunk for {request_id!r} must "
                f"carry a torch.Tensor, got {type(codes).__name__}"
            )
        codes = self.validate_chunk(request_id, state, codes)
        self.ingest(request_id, state, codes)
        return state

    def _decode_and_emit(
        self, request_id: str, state: StreamStateT
    ) -> list[OutgoingMessage]:
        if not self.should_decode(state, is_final=False):
            return []
        waveform = self.decode_delta(request_id, state, is_final=False)
        if waveform is None:
            return []
        self._mark_stream_emitted(request_id)
        return [self._stream_chunk_message(request_id, waveform)]

    def _stream_chunk_message(
        self, request_id: str, waveform: torch.Tensor
    ) -> OutgoingMessage:
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=self.stream_payload(request_id, waveform),
            metadata={"modality": "audio"},
        )

    def _pump_streams(self) -> list[str]:
        """Coalesced decode loop: the hooks pick the participants of one shared
        step and run it; a failed step errors and aborts every participant; the
        loop re-pumps until no participants remain, so capped-step backlogs
        drain within one pump. Returns the request ids ``on_step_failure``
        aborted, whose external abort cleanup the caller must run off
        ``_state_lock``."""
        while True:
            participants = self.select_step_participants()
            if not participants:
                return []
            plan = self.build_step_plan(participants)
            try:
                decoded = self.run_step(participants, plan)
            except Exception as exc:
                return list(self.on_step_failure(participants, exc))
            for request_id, _ in participants:
                waveform = decoded.get(request_id)
                if waveform is not None and not self._is_aborted(request_id):
                    self._mark_stream_emitted(request_id)
                    self.outbox.put(self._stream_chunk_message(request_id, waveform))

    @abstractmethod
    def create_stream_state(self, request_id: str) -> StreamStateT:
        """Fresh per-request state; created on the first chunk or payload."""

    def latch_stream_contract(
        self,
        request_id: str,
        state: StreamStateT,
        source: StagePayload | Mapping[str, Any],
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

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=self._stream_source_hint,
        )

    @abstractmethod
    def final_result_data(
        self, request_id: str, payload: StagePayload, state: StreamStateT
    ) -> dict[str, Any]:
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
        """Runs under ``_state_lock``: emits the error and clears state for
        every participant, then returns the request ids whose external abort
        callback the caller must run once the lock is released (never invoke
        ``abort``/``_cleanup_aborted_request`` here — the callback must stay
        off the GPU-serializing lock)."""
        logger.exception(
            "%s streaming decode step failed; aborting %d participating request(s)",
            self._stream_source_hint,
            len(participants),
        )
        failed: list[str] = []
        for request_id, _ in participants:
            self._emit_error(request_id, exc)
            self._abort_state(request_id)
            failed.append(request_id)
        return failed


__all__ = [
    "INITIAL_CODEC_CHUNK_FRAMES_PARAM",
    "StreamingVocoderBase",
    "resolve_initial_codec_chunk_frames",
]
