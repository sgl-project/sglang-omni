# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""

from __future__ import annotations

import json
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.platforms import current_platform
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.profiler.event_recorder import get_recorder as _get_recorder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from sglang_omni.utils.device import resolve_device_spec

logger = logging.getLogger(__name__)


# Note (ruoyu): the decomposition, the captured batched keys and the batch
# ceiling all derive from this tuple so a captured key can never go missing.
_DECOMPOSE_SIZES = (8, 4, 2, 1)


def _serial_threshold_graph_keys(
    stream_chunk_size: int,
    left_context_size: int,
) -> tuple[GraphKey, ...]:
    # Each serial threshold decode advances by one chunk while its context grows
    # to the configured cap.
    frames = (
        *range(
            stream_chunk_size,
            stream_chunk_size + left_context_size,
            stream_chunk_size,
        ),
        stream_chunk_size + left_context_size,
    )
    return tuple(
        GraphKey(batch_size=1, frames=window_frames) for window_frames in frames
    )


def _batched_graph_keys(
    stream_chunk_size: int,
    left_context_size: int,
    batch_ceiling: int,
) -> tuple[GraphKey, ...]:
    # Same window lengths as the serial keys: the batch-former only coalesces
    # same-bucket windows, so batching adds batch sizes, not new frame counts.
    serial = _serial_threshold_graph_keys(stream_chunk_size, left_context_size)
    return serial + tuple(
        GraphKey(batch_size=batch_size, frames=key.frames)
        for batch_size in sorted(size for size in _DECOMPOSE_SIZES if size > 1)
        if batch_size <= batch_ceiling
        for key in serial
    )


def load_code2wav_model(
    model_path: str, *, device: str = "cuda", dtype: str | None = None
):
    """Load Code2Wav model from HF checkpoint."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = config.code2wav_config

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)
    model = load_module(
        model,
        model_path,
        prefix="code2wav.",
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    return model.eval()


@dataclass
class _PendingWindow:
    """Depth-2 pipeline slot reference held by a stream state.

    The sample count is recorded at launch so the flush never has to read a
    shape back from the device.
    """

    slot: PinnedTransferSlot
    samples: int


@dataclass
class Code2WavStreamState:
    chunks: list[torch.Tensor] = field(default_factory=list)
    emitted: int = 0
    audio_parts: list[np.ndarray] = field(default_factory=list)
    stream_enabled: bool | None = None
    due_since: float | None = None
    checked: int = 0
    pending: _PendingWindow | None = None


class Code2WavScheduler(StreamingVocoderBase[Code2WavStreamState, "list[int]"]):
    """Streaming vocoder scheduler. Same inbox/outbox interface as OmniScheduler."""

    # Note (edwardzh): one slot per stream plus one for the decode in
    # flight, since acquire happens before the previous window is flushed.
    _MAX_PINNED_SLOTS = 32

    def __init__(
        self,
        model: Any,
        device: str,
        stream_chunk_size: int = 10,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        enable_batching: bool = False,
        initial_codec_chunk_frames: int = 0,
        max_batch_wait_ms: int = 0,
        batch_floor: int = 2,
        batch_ceiling: int = 8,
        enable_output_overlap: bool = True,
        enable_cuda_graph: bool = False,
        _cuda_graph_runner: Code2WavCudaGraphRunner | None = None,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)
        self._cuda_graph_runner = (
            _cuda_graph_runner if bool(enable_cuda_graph) else None
        )
        super().__init__(
            None,
            sample_rate=sample_rate,
            stream_source_hint="Qwen3-Omni code2wav",
        )
        # Note (wenyao): batching fields set after super().__init__ — the base
        # scheduler assigns its own _max_batch_wait_s and would clobber ours.
        self._enable_batching = bool(enable_batching)
        self._max_batch_wait_s = max(int(max_batch_wait_ms), 0) / 1000.0
        self._batch_floor = max(int(batch_floor), 1)
        self._initial_codec_chunk_frames = min(
            max(int(initial_codec_chunk_frames), 0), int(stream_chunk_size)
        )
        self._batch_ceiling = min(max(int(batch_ceiling), 1), _DECOMPOSE_SIZES[0])
        self._drain_mode = False
        self._last_fire_reason: str | None = None
        self._last_oldest_wait_ms: float = 0.0
        self._last_due_bucket_count: int = 0
        self._pending_step_failures: list[str] = []
        self._can_batch_stream_chunks = self._enable_batching
        if self._enable_batching:
            self._stream_chunk_batch_max = self._batch_ceiling
        # Note (edwardzh): serial path only — run_step is #1237's and
        # its window shapes are not the ones this pipeline reasons about.
        self._enable_output_overlap = bool(enable_output_overlap)
        self._eos_lazy_scan = self._enable_output_overlap and not self._enable_batching
        self._pipeline_active = self._eos_lazy_scan and self._device.type == "cuda"
        self._default_slot_samples = self._stream_chunk_size * self._total_upsample
        # Note (jiannan-17): Each slot only grows its buffer and reuses one
        # event, avoiding steady-state allocations.
        self._pinned_free: list[PinnedTransferSlot] = []
        self._pinned_created = 0
        # Note (wenyao): a slot must have exactly one owner at any time.
        # Quarantined slots are never reused but still count against
        # _MAX_PINNED_SLOTS, so an event failure cannot inflate the cap.
        self._pinned_retired: list[PinnedTransferSlot] = []
        self._pinned_quarantined: list[PinnedTransferSlot] = []

    @property
    def _chunk_aligned_dispatch(self) -> bool:
        """Chunk alignment only pays off while graphs are live: uniform windows
        keep every step inside the captured key set, so it follows the runner's
        current published keys rather than the startup flags."""
        if not self._enable_batching or self._cuda_graph_runner is None:
            return False
        steady_window = self._left_context_size + self._stream_chunk_size
        return bool(self._cuda_graph_runner.available_batch_sizes(steady_window))

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def create_stream_state(self, request_id: str) -> Code2WavStreamState:
        del request_id
        return Code2WavStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: Code2WavStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin != "stream metadata":
            return
        if state.stream_enabled is None:
            state.stream_enabled = bool(source["stream"])

    def validate_chunk(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        return codes.to(device=self._device, dtype=torch.long)

    def ingest(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if self._eos_lazy_scan:
            # Note (edwardzh): one frame per message, so checking here
            # costs one sync per frame; should_decode batches them per window.
            state.chunks.append(codes)
            return
        if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
            return
        state.chunks.append(codes)

    def should_decode(self, state: Code2WavStreamState, *, is_final: bool) -> bool:
        del is_final
        if (
            self._eos_lazy_scan
            and state.checked < len(state.chunks)
            and self._ready(state) >= self._stream_chunk_size
        ):
            # Note (edwardzh): scan before gating — a staged EOS would
            # inflate the count and fire a short window, missing the graph key.
            self._scan_unchecked(state)
        return self._ready(state) >= self._stream_chunk_size

    def _scan_unchecked(self, state: Code2WavStreamState) -> None:
        """Batched EOS scan over frames staged by the lazy-ingest path.

        Replays the eager per-frame check exactly: every staged frame whose
        leading code equals the codec EOS id is dropped and every other frame
        keeps its arrival order (0-dim frames bypass the check, as in the
        eager path). One host sync per scan instead of one per frame.

        The producer emits 1-D ``[num_quantizers]`` frames; any other rank
        falls back to the per-frame path rather than stacking into a nested
        truthy list that would silently drop malformed chunks.
        """
        chunks = state.chunks
        start = state.checked
        if start >= len(chunks):
            return
        unchecked = chunks[start:]
        if all(codes.ndim == 1 for codes in unchecked):
            heads = torch.stack([codes[0] for codes in unchecked])
            is_eos = (heads == self._codec_eos_token_id).tolist()
        else:
            is_eos = [
                codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id
                for codes in unchecked
            ]
        if any(is_eos):
            chunks[start:] = [codes for codes, eos in zip(unchecked, is_eos) if not eos]
        state.checked = len(chunks)

    def decode_delta(
        self, request_id: str, state: Code2WavStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if self._eos_lazy_scan and is_final:
            # Note (edwardzh): the stream-done flush never goes through
            # should_decode, so the tail is still unscanned here.
            self._scan_unchecked(state)
        start, end = state.emitted, len(state.chunks)
        if start >= end:
            return None
        context = min(self._left_context_size, start)
        profile_metadata: dict[str, Any] | None = None
        if _get_event_recorder().is_active():
            profile_metadata = {
                "trigger": "stream_done" if is_final else "threshold",
                "start_frame": start,
                "end_frame": end,
                "new_frames": end - start,
                "context_frames": context,
                "window_frames": end - start + context,
                "active_request_count": len(self._stream_states),
                "threshold_ready_request_count": sum(
                    self._ready(ready_state) >= self._stream_chunk_size
                    for _, ready_state in self._stream_state_items()
                ),
                "inbox_depth": self.inbox.qsize(),
                "pending_message_depth": len(self._pending_messages),
            }
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_start",
                metadata=profile_metadata,
            )
        window = torch.stack(state.chunks[start - context : end], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)
        wav, execution_metadata = self._forward_codes(
            codes,
            graph_eligible=not is_final,
        )
        wav = wav[..., -(end - start) * self._total_upsample :]
        samples = int(wav.numel())
        prev_wait_ns = 0
        prev_waveform: torch.Tensor | None = None
        slot: PinnedTransferSlot | None = None
        # Note (edwardzh): the first window stays synchronous so
        # time-to-first-audio is unchanged.
        if self._pipeline_active and not is_final and start > 0 and samples > 0:
            slot = self._acquire_slot(samples)
            if slot is None and state.pending is not None:
                # Note (edwardzh): freeing this request's own pending
                # window keeps its emission order; the retry cannot miss since
                # only this thread touches the pool.
                prev_wait_ns, prev_waveform = self._flush_pending(request_id, state)
                slot = self._acquire_slot(samples)
        if slot is None:
            audio = wav.reshape(-1).detach().cpu().float().numpy().copy()
            if profile_metadata is not None:
                extra = (
                    {"pipelined": False, "d2h_wait_ns": prev_wait_ns}
                    if self._pipeline_active
                    else {}
                )
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_decode_end",
                    metadata={
                        **profile_metadata,
                        "audio_samples": int(audio.shape[0]),
                        **execution_metadata,
                        **extra,
                    },
                )
            state.emitted = end
            state.due_since = None
            if audio.size == 0:
                return prev_waveform
            if not state.audio_parts:
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_first_audio",
                    metadata={"samples": int(audio.shape[0])},
                )
            state.audio_parts.append(audio)
            if not state.stream_enabled:
                return prev_waveform
            return torch.from_numpy(audio)
        event_recorded = False
        try:
            # Note (edwardzh): same stream is what makes this safe — the
            # copy is ordered before any later replay, so the borrowed static
            # output cannot be overwritten while it drains.
            slot.view(samples).copy_(
                wav.reshape(-1).to(torch.float32), non_blocking=True
            )
            slot.record(torch.cuda.current_stream(self._device))
            event_recorded = True
            if profile_metadata is not None:
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_decode_launched",
                    metadata={
                        **execution_metadata,
                        "window_frames": end - start + context,
                        "new_frames": end - start,
                    },
                )
            if state.pending is not None:
                # Note (edwardzh): flushed after the launch above so the
                # host work overlaps the GPU computing this window.
                prev_wait_ns, prev_waveform = self._flush_pending(request_id, state)
        except Exception:
            if event_recorded:
                self._retire_slot(slot)
            else:
                # Note (wenyao): a failed copy or record leaves no trustworthy
                # completion marker, so this buffer must never go back into
                # rotation even though it was never handed to a request.
                # Note (jiannan-17): The slot tracks record failures, but not
                # copy failures before record(); quarantine covers both cases.
                self._quarantine_slot(slot)
            raise
        state.pending = _PendingWindow(slot=slot, samples=samples)
        state.emitted = end
        state.due_since = None
        if profile_metadata is not None:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_end",
                metadata={
                    **profile_metadata,
                    "audio_samples": samples,
                    **execution_metadata,
                    "pipelined": True,
                    "d2h_wait_ns": prev_wait_ns,
                },
            )
        return prev_waveform

    def _decode_and_emit(
        self, request_id: str, state: Code2WavStreamState
    ) -> list[OutgoingMessage]:
        # Note (wenyao): frames arrive per talker decode step but windows only
        # every stream_chunk_size of them, so waiting for the next dispatch
        # would hold drained audio for a whole window arrival.
        messages: list[OutgoingMessage] = []
        pending = state.pending
        if pending is not None and pending.slot.query():
            _, waveform = self._flush_pending(request_id, state)
            if waveform is not None:
                self._mark_stream_emitted(request_id)
                messages.append(self._stream_chunk_message(request_id, waveform))
        messages.extend(super()._decode_and_emit(request_id, state))
        return messages

    def _flush_pending(
        self, request_id: str, state: Code2WavStreamState
    ) -> tuple[int, torch.Tensor | None]:
        """Materialize the pending window: wait for its D2H copy, append the
        audio, and return (wait_ns, waveform-or-None). The slot returns to the
        pool only after the audio is copied into owned memory."""
        pending = state.pending
        if pending is None:
            return 0, None
        slot = pending.slot
        wait_start = time.monotonic_ns()
        slot.synchronize()
        wait_ns = time.monotonic_ns() - wait_start
        audio = slot.view(pending.samples).numpy().copy()

        # Note (wenyao): every operation above may raise, so the request must
        # keep owning the slot until synchronization and the host copy both
        # succeed; releasing first would hand a live D2H target to the pool.
        state.pending = None
        self._release_slot(slot)
        if audio.size == 0:
            return wait_ns, None
        if not state.audio_parts:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(audio.shape[0])},
            )
        state.audio_parts.append(audio)
        if not state.stream_enabled:
            return wait_ns, None
        return wait_ns, torch.from_numpy(audio)

    def _acquire_slot(self, samples: int) -> PinnedTransferSlot | None:
        self._reap_retired_slots()
        if self._pinned_free:
            slot = self._pinned_free.pop()
            try:
                slot.ensure_capacity(samples)
            except Exception:
                # Note (wenyao): the popped slot is not in flight, and a failed
                # growth keeps its original smaller buffer, so returning it to
                # the pool is safe.
                self._release_slot(slot)
                raise
            return slot
        if self._pinned_created < self._MAX_PINNED_SLOTS:
            slot = PinnedTransferSlot(
                self._device,
                torch.float32,
                initial_capacity=max(samples, self._default_slot_samples),
            )
            self._pinned_created += 1
            return slot
        return None

    def _release_slot(self, slot: PinnedTransferSlot) -> None:
        self._pinned_free.append(slot)

    def _retire_slot(self, slot: PinnedTransferSlot) -> None:
        self._pinned_retired.append(slot)

    def _quarantine_slot(self, slot: PinnedTransferSlot) -> None:
        self._pinned_quarantined.append(slot)
        self._pipeline_active = False

    def _reap_retired_slots(self) -> None:
        """Non-blockingly return completed retired slots to the free pool.

        Callers hold ``_state_lock``. An event-query error leaves the buffer
        owned by the scheduler but permanently unavailable for reuse.
        """
        if not self._pinned_retired:
            return
        still_retired: list[PinnedTransferSlot] = []
        for slot in self._pinned_retired:
            try:
                complete = slot.query()
            except Exception:
                logger.exception("code2wav failed to query a retired D2H copy")
                self._quarantine_slot(slot)
                continue
            if complete:
                self._release_slot(slot)
            else:
                still_retired.append(slot)
        self._pinned_retired = still_retired

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: Code2WavStreamState
    ) -> dict[str, Any]:
        del payload
        if not state.audio_parts:
            raise RuntimeError(f"code2wav produced no audio for {request_id!r}")
        if state.stream_enabled:
            return {"modality": "audio", "sample_rate": self._sample_rate}
        full = np.concatenate(state.audio_parts).astype(np.float32, copy=False)
        return audio_waveform_payload(
            full,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Qwen3-Omni code2wav",
        )

    def _forward_codes(
        self,
        codes: torch.Tensor,
        *,
        graph_eligible: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            if self._device.type != "cpu":
                torch.get_device_module(self._device).set_device(self._device)
            if self._cuda_graph_runner is None:
                result = Code2WavRunResult(
                    output=self._model(codes),
                    execution_mode="eager",
                    key=None,
                    fallback_reason=None,
                )
            else:
                result = self._cuda_graph_runner.run(
                    codes,
                    eligible=graph_eligible,
                )

        graph_key = None
        if result.key is not None:
            graph_key = {
                "batch_size": int(result.key.batch_size),
                "frames": int(result.key.frames),
            }
        return result.output, {
            "execution_mode": str(result.execution_mode),
            "graph_key": graph_key,
            "fallback_reason": (
                None if result.fallback_reason is None else str(result.fallback_reason)
            ),
        }

    def _batch_deadline(self) -> float | None:
        with self._state_lock:
            due = [
                state.due_since
                for _, state in self._stream_state_items()
                if state.due_since is not None
            ]
        if not due:
            return None
        return min(due) + self._max_batch_wait_s

    def _drain_inbox(self):
        while True:
            try:
                yield self.inbox.get_nowait()
            except queue.Empty:
                return

    def _next_message(self):
        # Note (wenyao): ``Event.query()`` is non-blocking, so reaping on this
        # loop cannot recreate the abort stall it replaces.
        with self._state_lock:
            self._reap_retired_slots()
        if self._can_batch_stream_chunks:
            first_chunks: list = []
            for msg in self._drain_inbox():
                if (
                    msg.type == "stream_chunk"
                    and msg.request_id not in self._stream_states
                    and not self._is_aborted(msg.request_id)
                ):
                    first_chunks.append(msg)
                else:
                    self._pending_messages.append(msg)
            if first_chunks:
                self._handle_stream_chunk_batch(first_chunks)
            if (
                self._pending_messages
                and self._pending_messages[0].type == "stream_chunk"
            ):
                run: list = []
                while (
                    self._pending_messages
                    and self._pending_messages[0].type == "stream_chunk"
                ):
                    run.append(self._pending_messages.popleft())
                self._handle_stream_chunk_batch(run)
                return None
        if self._pending_messages:
            return self._pending_messages.popleft()
        deadline = self._batch_deadline()
        timeout = 0.1
        if deadline is not None:
            timeout = min(timeout, max(deadline - time.monotonic(), 0.0))
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            if deadline is not None and time.monotonic() >= deadline:
                self._pump_due_streams()
            return None

    def _pump_due_streams(self) -> None:
        with self._state_lock:
            failed = self._pump_streams()
        for request_id in failed:
            self._cleanup_aborted_request(request_id)

    def _pump_streams(self) -> list[str]:
        # Note (ruoyu): a step that fails partway aborts inside run_step instead
        # of raising, so its request ids surface here — the caller still owes
        # them the abort callback it runs off ``_state_lock``.
        failed = super()._pump_streams()
        if self._pending_step_failures:
            failed = failed + self._pending_step_failures
            self._pending_step_failures = []
        return failed

    def _ready(self, state: Code2WavStreamState) -> int:
        return len(state.chunks) - state.emitted

    def _step_frames(self, state: Code2WavStreamState) -> int:
        """New frames the next step consumes; capped at one chunk so a backlog
        cannot push the window outside the captured key set."""
        ready = self._ready(state)
        if state.emitted == 0 and self._initial_codec_chunk_frames:
            ready = min(ready, self._initial_codec_chunk_frames)
        if self._chunk_aligned_dispatch:
            return min(ready, self._stream_chunk_size)
        return ready

    def _bucket(self, state: Code2WavStreamState) -> tuple[int, int]:
        context = min(self._left_context_size, state.emitted)
        return (context, context + self._step_frames(state))

    @staticmethod
    def _decompose_batch(n: int, sizes: tuple[int, ...] = (8, 4, 2, 1)) -> list[int]:
        plan: list[int] = []
        for size in sizes:
            while n >= size:
                plan.append(size)
                n -= size
        if n:
            # No graph covers the remainder; keep it one eager forward instead
            # of shattering it into per-request calls.
            plan.append(n)
        return plan

    def stop(self) -> None:
        self._drain_mode = True
        super().stop()

    def on_stream_done(self, request_id: str):
        state = self._stream_states.get(request_id)
        prev_drain = self._drain_mode
        if state is not None and state.due_since is not None:
            self._drain_mode = True
        try:
            messages: list[OutgoingMessage] = []
            if state is not None and state.pending is not None:
                # Note (edwardzh): emitted as its own message, not
                # merged into the tail, so message boundaries match the
                # synchronous path; must precede the base's terminal result.
                _, waveform = self._flush_pending(request_id, state)
                if waveform is not None:
                    self._mark_stream_emitted(request_id)
                    messages.append(self._stream_chunk_message(request_id, waveform))
            messages.extend(super().on_stream_done(request_id))
            return messages
        finally:
            self._drain_mode = prev_drain

    def release_stream_resources(
        self, request_id: str, state: Code2WavStreamState
    ) -> None:
        del request_id
        pending = state.pending
        if pending is None:
            return
        # Note (wenyao): abort arrives on the stage event-loop thread, so this
        # must not block on CUDA; running under ``_state_lock`` makes the
        # handoff to the scheduler thread atomic instead.
        self._retire_slot(pending.slot)
        state.pending = None

    def on_serving_stop(self) -> None:
        """Drain retired slots at shutdown, when blocking costs no latency."""
        retired = self._pinned_retired
        self._pinned_retired = []
        for slot in retired:
            try:
                slot.synchronize()
            except Exception:
                logger.exception(
                    "code2wav failed to synchronize a retired D2H copy on shutdown"
                )
                self._pinned_quarantined.append(slot)
            else:
                self._release_slot(slot)

    def select_step_participants(self) -> list[tuple[str, Code2WavStreamState]]:
        now = time.monotonic()
        first_ready: list[tuple[str, Code2WavStreamState]] = []
        due: dict[tuple[int, int], list[tuple[str, Code2WavStreamState]]] = {}
        for rid, state in self._stream_state_items():
            ready = self._ready(state)
            if state.emitted == 0 and ready >= (
                self._initial_codec_chunk_frames or self._stream_chunk_size
            ):
                first_ready.append((rid, state))
                continue
            if state.emitted > 0 and ready >= self._stream_chunk_size:
                if state.due_since is None:
                    state.due_since = now
                due.setdefault(self._bucket(state), []).append((rid, state))
        if first_ready:
            # Note (wenyao): same bucket ⇒ one trim scalar holds for the batch.
            key = self._bucket(first_ready[0][1])
            same_bucket = [p for p in first_ready if self._bucket(p[1]) == key]
            self._last_fire_reason = "first"
            self._last_oldest_wait_ms = 0.0
            self._last_due_bucket_count = len(due)
            return same_bucket[: self._batch_ceiling]
        if not due:
            return []
        anchor_key = min(due, key=lambda k: min(s.due_since for _, s in due[k]))
        anchor = sorted(due[anchor_key], key=lambda p: p[1].due_since)
        oldest_wait = now - anchor[0][1].due_since
        fire = (
            len(anchor) >= self._batch_floor
            or oldest_wait >= self._max_batch_wait_s
            or self._drain_mode
        )
        if not fire:
            return []
        if len(anchor) >= self._batch_floor:
            reason = "floor"
        elif oldest_wait >= self._max_batch_wait_s:
            reason = "deadline"
        else:
            reason = "drain"
        self._last_fire_reason = reason
        self._last_oldest_wait_ms = oldest_wait * 1000.0
        self._last_due_bucket_count = len(due)
        return anchor[: self._batch_ceiling]

    def build_step_plan(
        self, participants: list[tuple[str, Code2WavStreamState]]
    ) -> list[int]:
        if not self._chunk_aligned_dispatch:
            return [len(participants)]
        # Decompose against the batch sizes that actually hold a published
        # graph for this window length — under memory pressure the runner may
        # have skipped the larger batched graphs, and a sub-batch without a
        # graph would replay nothing and eat the dispatch overhead twice.
        window_frames = self._bucket(participants[0][1])[1]
        sizes = tuple(
            size
            for size in self._cuda_graph_runner.available_batch_sizes(window_frames)
            if size > 1
        )
        if not sizes:
            # Note (ruoyu): batched graphs may be absent because their eager
            # warmup OOMed, so retrying that batch as eager is unsafe even
            # when it benchmarks faster than serial graph replays.
            return [1] * len(participants)
        return self._decompose_batch(len(participants), sizes)

    def run_step(
        self,
        participants: list[tuple[str, Code2WavStreamState]],
        plan: list[int],
    ) -> dict[str, torch.Tensor]:
        decoded: dict[str, torch.Tensor] = {}
        profile_metadata: dict[str, Any] | None = None
        if _get_recorder().is_active():
            first_state = participants[0][1]
            bucket = self._bucket(first_state)
            profile_metadata = {
                "batch_size": len(participants),
                "bucket": list(bucket),
                "new_frames": self._step_frames(first_state),
                "window_frames": bucket[1],
                "active_request_count": len(self._stream_states),
                "inbox_depth": self.inbox.qsize(),
                "oldest_wait_ms": self._last_oldest_wait_ms,
                "fire_reason": self._last_fire_reason,
                "due_bucket_count": self._last_due_bucket_count,
                "subbatch_decomposition": list(plan),
            }
            _emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_batch_start",
                metadata=profile_metadata,
            )
        execution_metadata = {
            "execution_mode": "eager",
            "graph_key": None,
            "fallback_reason": None,
        }
        sub_batch_execution: list[dict[str, Any]] = []
        audio_samples = 0
        cursor = 0
        for sub in plan:
            group = participants[cursor : cursor + sub]
            try:
                samples, execution_metadata = self._run_sub_batch(group, decoded)
            except Exception as exc:
                # Note (ruoyu): the sub-batches already decoded advanced their
                # cursors, so aborting every participant would drop audio the
                # client is owed; fail only the ones that did not decode.
                self._pending_step_failures.extend(
                    self.on_step_failure(participants[cursor:], exc)
                )
                break
            cursor += sub
            audio_samples += samples
            if profile_metadata is not None:
                sub_batch_execution.append(
                    {"batch_size": len(group), **execution_metadata}
                )
        if profile_metadata is not None:
            modes = {entry["execution_mode"] for entry in sub_batch_execution}
            _emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_batch_end",
                metadata={
                    **profile_metadata,
                    "audio_samples": audio_samples,
                    # Note (ruoyu): flat last-sub-batch fields kept so existing
                    # single-forward event consumers keep parsing; the
                    # per-sub-batch list is the authoritative record.
                    **execution_metadata,
                    # Note (jiaqi): a heterogeneous plan reports "mixed" rather
                    # than letting the last sub-batch speak for the whole step.
                    "execution_mode": (
                        modes.pop()
                        if len(modes) == 1
                        else "mixed" if modes else "eager"
                    ),
                    "sub_batch_execution": sub_batch_execution,
                },
            )
        return decoded

    def _run_sub_batch(
        self,
        group: list[tuple[str, Code2WavStreamState]],
        decoded: dict[str, torch.Tensor],
    ) -> tuple[int, dict[str, Any]]:
        """Decode one sub-batch and advance its participants; returns the audio
        sample count and the execution metadata of the forward."""
        rows = []
        window_ends: list[int] = []
        for _, state in group:
            start = state.emitted
            end = start + self._step_frames(state)
            window_ends.append(end)
            context = min(self._left_context_size, start)
            rows.append(
                torch.stack(state.chunks[start - context : end], dim=0).transpose(0, 1)
            )
        window_frames = rows[0].shape[-1]
        for row in rows[1:]:
            if row.shape[-1] != window_frames:
                raise RuntimeError(
                    f"code2wav bucket mismatch: window {row.shape[-1]} vs "
                    f"{window_frames}"
                )
        codes = torch.stack(rows, dim=0)
        wav, execution_metadata = self._forward_codes(
            codes,
            graph_eligible=self._chunk_aligned_dispatch,
        )
        if wav.shape[0] != len(group):
            raise RuntimeError(
                f"code2wav step returned {wav.shape[0]} rows for "
                f"{len(group)} requests"
            )
        context = min(self._left_context_size, group[0][1].emitted)
        wav = wav[..., -(window_frames - context) * self._total_upsample :]
        host = wav.detach().cpu().float()
        audio_samples = 0
        for i, (rid, state) in enumerate(group):
            audio = host[i].reshape(-1).numpy().copy()
            state.emitted = window_ends[i]
            state.due_since = None
            if audio.size == 0:
                continue
            audio_samples += int(audio.size)
            if not state.audio_parts:
                _emit_event(
                    request_id=rid,
                    stage=None,
                    event_name="code2wav_first_audio",
                    metadata={"samples": int(audio.shape[0])},
                )
            state.audio_parts.append(audio)
            if state.stream_enabled:
                decoded[rid] = torch.from_numpy(audio)
        return audio_samples, execution_metadata


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
    enable_batching: bool = False,
    initial_codec_chunk_frames: int = 0,
    max_batch_wait_ms: int = 0,
    batch_floor: int = 2,
    batch_ceiling: int = 8,
    enable_output_overlap: bool = True,
    enable_cuda_graph: bool = False,
    total_gpu_memory_fraction: float | None = None,
):
    """Factory: returns Code2WavScheduler."""
    if enable_cuda_graph and total_gpu_memory_fraction is None:
        raise ValueError(
            "Code2Wav CUDA graph requires "
            "runtime.resources.total_gpu_memory_fraction"
        )
    concrete_device = torch.device(resolve_device_spec(device, gpu_id))
    if concrete_device.type != "cpu" and concrete_device.index is None:
        concrete_device = current_platform.get_device(
            torch.get_device_module().current_device()
        )
    device = str(concrete_device)
    stream_chunk_size = max(int(stream_chunk_size), 1)
    left_context_size = max(int(left_context_size), 0)
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    cuda_graph_runner = None
    if enable_cuda_graph:
        if enable_batching:
            graph_keys = _batched_graph_keys(
                stream_chunk_size,
                left_context_size,
                min(max(int(batch_ceiling), 1), 8),
            )
        else:
            graph_keys = _serial_threshold_graph_keys(
                stream_chunk_size,
                left_context_size,
            )
        cuda_graph_runner = Code2WavCudaGraphRunner.build(
            model,
            device=concrete_device,
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            graph_keys=graph_keys,
        )
        startup_stats = cuda_graph_runner.stats()
        if enable_batching and not startup_stats["enabled"]:
            # Note (ruoyu): the runner shrinks the batched key set on its own,
            # so a fully disabled runner means even serial capture failed.
            # Batching without graph replay measured worse than serial eager
            # on H100, so it is dropped along with the graphs.
            logger.warning(
                "Code2Wav graph capture disabled (%s); disabling batching",
                startup_stats["disable_reason"],
            )
            enable_batching = False
        logger.info(
            "Code2Wav CUDA graph startup stats=%s",
            json.dumps(
                cuda_graph_runner.stats(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        enable_batching=enable_batching,
        initial_codec_chunk_frames=initial_codec_chunk_frames,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_floor=batch_floor,
        batch_ceiling=batch_ceiling,
        enable_output_overlap=enable_output_overlap,
        enable_cuda_graph=enable_cuda_graph,
        _cuda_graph_runner=cuda_graph_runner,
    )
