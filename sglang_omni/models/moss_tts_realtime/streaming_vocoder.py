# SPDX-License-Identifier: Apache-2.0
"""Fixed-slot streaming vocoder for MOSS-TTS-Realtime."""

from __future__ import annotations

import contextlib
import logging
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts_realtime.observability import (
    emit_realtime_event as _emit_event,
)
from sglang_omni.models.moss_tts_realtime.observability import (
    realtime_events_active,
    realtime_identity_metadata,
)
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.vocoder_decoder import (
    moss_tts_realtime_vocoder_decoder_dtype,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_SOURCE_HINT = "MOSS-TTS-Realtime"
_CHUNK_RAMP = (1, 2, 3)
_STEADY_CHUNK_FRAMES = 3
_DEFAULT_CUDA_GRAPH_MAX_FRAMES = 12
_MAX_CUDA_GRAPH_FRAMES = 25


class _LegacyCodecStreamingStateAdapter:
    """Access legacy codec states through one shared per-step exec mask."""

    def __init__(self, codec: Any, *, device: torch.device) -> None:
        self._states: list[Any] = []

        def collect(module: Any) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is None:
                return
            if not callable(getattr(state, "reset", None)):
                raise RuntimeError(
                    f"codec streaming state {type(state).__name__} lacks reset()"
                )
            if getattr(state, "device", None) is None:
                raise RuntimeError(
                    f"codec streaming state {type(state).__name__} has no device"
                )
            self._states.append(state)

        codec.apply(collect)
        if not self._states:
            raise RuntimeError("MOSS-TTS-Realtime codec has no active streaming state")
        self._device = torch.device(device)
        exec_masks: list[torch.Tensor] = []
        for state in self._states:
            exec_mask = getattr(state, "exec_mask", None)
            if not isinstance(exec_mask, torch.Tensor):
                raise RuntimeError(
                    f"codec streaming state {type(state).__name__} has no "
                    "Tensor exec_mask"
                )
            exec_masks.append(exec_mask)

        state_devices = {torch.device(state.device) for state in self._states}
        mask_devices = {exec_mask.device for exec_mask in exec_masks}
        mask_shapes = {tuple(exec_mask.shape) for exec_mask in exec_masks}
        mask_dtypes = {exec_mask.dtype for exec_mask in exec_masks}
        if (
            state_devices != {self._device}
            or mask_devices != {self._device}
            or len(mask_shapes) != 1
            or mask_dtypes != {torch.bool}
        ):
            raise RuntimeError(
                "MOSS-TTS-Realtime codec streaming states cannot share exec_mask: "
                f"state_devices={state_devices}, mask_devices={mask_devices}, "
                f"mask_shapes={mask_shapes}, mask_dtypes={mask_dtypes}"
            )
        (expected_shape,) = mask_shapes
        if len(expected_shape) != 1:
            raise RuntimeError(
                "MOSS-TTS-Realtime codec streaming states cannot share "
                f"exec_mask with non-vector shape {expected_shape}"
            )

        # Every state consumes the same active-slot mask. Alias it before CUDA
        # Graph capture so each live step needs one device copy instead of one
        # copy for every codec module.
        self._shared_exec_mask = exec_masks[0].clone()
        for state in self._states:
            state.exec_mask = self._shared_exec_mask

    @property
    def count(self) -> int:
        return len(self._states)

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        if exec_mask.shape != self._shared_exec_mask.shape:
            raise ValueError(
                "MOSS-TTS-Realtime codec exec_mask must have shape "
                f"{tuple(self._shared_exec_mask.shape)}, got {tuple(exec_mask.shape)}"
            )
        self._shared_exec_mask.copy_(
            exec_mask.to(device=self._device, dtype=torch.bool)
        )

    def reset_slots(self, slots: list[int], *, batch_size: int) -> None:
        if not slots:
            return
        reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        reset_mask[slots] = True
        for state in self._states:
            state.reset(reset_mask.to(device=state.device))


class _CodecStreamSession:
    """One persistent ``codec.streaming()`` context and its slot ownership."""

    def __init__(
        self,
        codec: Any,
        *,
        stream_slots: int,
        n_vq: int,
        samples_per_frame: int,
    ) -> None:
        self._codec = codec
        self._batch_size = int(stream_slots)
        self._n_vq = int(n_vq)
        self._samples_per_frame = int(samples_per_frame)
        try:
            self._device = next(codec.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("MOSS-TTS-Realtime codec has no parameters") from exc
        self._free_slots = list(range(self._batch_size - 1, -1, -1))
        self._leased_slots: set[int] = set()
        self._quarantined_slots: set[int] = set()
        self._closed = False
        self._cg_runner: Any | None = None
        self.warmup_attempted = False
        self._cg_graph_frames: Counter[int] = Counter()
        self._cg_eager_frames: Counter[int] = Counter()
        self._cg_total_steps = 0
        self._exit_stack = contextlib.ExitStack()
        try:
            with torch.no_grad():
                self._exit_stack.enter_context(codec.streaming(self._batch_size))
            self._state_adapter = _LegacyCodecStreamingStateAdapter(
                codec,
                device=self._device,
            )
        except BaseException:
            self._exit_stack.close()
            raise

    @property
    def active_leases(self) -> int:
        return len(self._leased_slots)

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def streaming_state_count(self) -> int:
        return self._state_adapter.count

    @property
    def quarantined_slots(self) -> int:
        return len(self._quarantined_slots)

    def warmup_cuda_graph(
        self,
        frames: list[int],
        *,
        min_free_gb: float = 3.0,
    ) -> list[int]:
        if self.warmup_attempted:
            return self.captured_frames()
        self.warmup_attempted = True
        if self._closed or not frames:
            return []

        from sglang_omni.models.moss_tts_realtime.vocoder_cuda_graph import (
            MossTTSRealtimeVocoderCudaGraphRunner,
        )

        self._cg_runner = MossTTSRealtimeVocoderCudaGraphRunner(
            self._codec,
            self._state_adapter,
            batch_size=self._batch_size,
            n_vq=self._n_vq,
            max_frames=max(frames),
            min_free_gb=min_free_gb,
        )
        try:
            try:
                self._cg_runner.warmup(frames)
            finally:
                self._state_adapter.reset_slots(
                    list(range(self._batch_size)),
                    batch_size=self._batch_size,
                )
        except Exception:
            self._cg_runner = None
            raise
        captured = self._cg_runner.captured_frames()
        if not captured:
            self._cg_runner = None
        return captured

    def has_cuda_graph_runner(self) -> bool:
        return bool(self._cg_runner and self._cg_runner.captured_frames())

    def captured_frames(self) -> list[int]:
        if self._cg_runner is None:
            return []
        return self._cg_runner.captured_frames()

    def acquire(self) -> int:
        if self._closed:
            raise RuntimeError("MOSS-TTS-Realtime codec session is closed")
        if not self._free_slots:
            raise RuntimeError(
                "MOSS-TTS-Realtime codec stream slots are exhausted "
                "(slots stay leased to a realtime session until session close, "
                "idle TTL, abort, or failure); increase stream_slots"
            )
        slot = self._free_slots.pop()
        self._leased_slots.add(slot)
        return slot

    def release(self, slot: int) -> None:
        if self._closed:
            return
        if slot not in self._leased_slots:
            raise RuntimeError(
                f"MOSS-TTS-Realtime codec slot {slot} is not currently leased"
            )
        try:
            self._state_adapter.reset_slots([slot], batch_size=self._batch_size)
        except BaseException:
            self._leased_slots.remove(slot)
            self._quarantined_slots.add(slot)
            raise
        self._leased_slots.remove(slot)
        self._free_slots.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        self._log_cuda_graph_stats()
        with torch.no_grad():
            self._exit_stack.close()
        self._closed = True

    def _log_cuda_graph_stats(self) -> None:
        graphed = sum(self._cg_graph_frames.values())
        eager = sum(self._cg_eager_frames.values())
        total = graphed + eager
        if total == 0:
            return
        logger.info(
            "MOSS-TTS-Realtime vocoder CUDA graph stats: %d/%d steps "
            "graphed (%.1f%%); graph T=%s eager T=%s",
            graphed,
            total,
            100.0 * graphed / total,
            dict(sorted(self._cg_graph_frames.items())),
            dict(sorted(self._cg_eager_frames.items())),
        )

    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if not slot_codes:
            return {}
        for slot in slot_codes:
            if slot not in self._leased_slots:
                raise RuntimeError(f"MOSS-TTS-Realtime codec slot {slot} is not leased")
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                "MOSS-TTS-Realtime codec step requires one uniform frame length, "
                f"got {sorted(step_lengths)}"
            )
        (step_frames,) = step_lengths
        if step_frames < 1:
            raise ValueError("MOSS-TTS-Realtime codec step must contain frames")
        for codes in slot_codes.values():
            if codes.ndim != 2 or tuple(codes.shape) != (
                self._n_vq,
                step_frames,
            ):
                raise ValueError(
                    "MOSS-TTS-Realtime codec slot codes must have shape "
                    f"[{self._n_vq}, T], got {tuple(codes.shape)}"
                )

        codes_batch = torch.zeros(
            self._n_vq,
            self._batch_size,
            step_frames,
            dtype=torch.long,
            device=self._device,
        )
        codes_lengths = torch.zeros(
            self._batch_size,
            dtype=torch.long,
            device=self._device,
        )
        exec_mask = torch.zeros(
            self._batch_size,
            dtype=torch.bool,
            device=self._device,
        )
        for slot, codes in slot_codes.items():
            codes_batch[:, slot, :] = codes.to(device=self._device, dtype=torch.long)
            codes_lengths[slot] = step_frames
            exec_mask[slot] = True

        slots = list(slot_codes)
        graphed: tuple[torch.Tensor, torch.Tensor] | None = None
        graph_failed = False
        try:
            with torch.no_grad():
                if self._cg_runner is not None:
                    try:
                        graphed = self._cg_runner.decode_step(codes_batch, exec_mask)
                    except Exception:
                        graph_failed = True
                        raise
                if graphed is not None:
                    audio, audio_lengths = graphed
                else:
                    self._state_adapter.set_exec_mask(exec_mask)
                    result = self._codec._decode_frame(codes_batch, codes_lengths)
                    audio = getattr(result, "audio", None)
                    audio_lengths = getattr(result, "audio_lengths", None)
            if not isinstance(audio, torch.Tensor) or not isinstance(
                audio_lengths, torch.Tensor
            ):
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec did not return audio/audio_lengths"
                )
            if audio.ndim != 3 or int(audio.shape[0]) != self._batch_size:
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec audio must have shape "
                    f"[{self._batch_size}, channels, samples]"
                )
            if int(audio.shape[1]) != 1:
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec must emit mono audio, got "
                    f"{audio.shape[1]} channels"
                )
            if (
                audio_lengths.ndim != 1
                or int(audio_lengths.shape[0]) != self._batch_size
            ):
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec audio_lengths must match fixed slots"
                )
            audio_cpu = (
                audio[slots]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )
            lengths_cpu = audio_lengths[slots].detach().to(device="cpu")
            expected_samples = step_frames * self._samples_per_frame
            decoded: dict[int, torch.Tensor] = {}
            for index, slot in enumerate(slots):
                samples = int(lengths_cpu[index])
                if samples != expected_samples:
                    raise RuntimeError(
                        "MOSS-TTS-Realtime codec returned an unexpected active "
                        f"length: slot={slot} samples={samples} "
                        f"expected={expected_samples}"
                    )
                decoded[slot] = audio_cpu[index, :, :samples].contiguous()
        except Exception:
            if self._cg_runner is not None and (graph_failed or graphed is not None):
                logger.exception(
                    "MOSS-TTS-Realtime vocoder CUDA graph replay failed; "
                    "disabling graphs for this codec session"
                )
                self._cg_runner = None
            raise

        if self._cg_runner is not None:
            if graphed is None:
                self._cg_eager_frames[step_frames] += 1
            else:
                self._cg_graph_frames[step_frames] += 1
            self._cg_total_steps += 1
            if self._cg_total_steps % 2000 == 0:
                self._log_cuda_graph_stats()
        return decoded

    def decode_borrowed(
        self,
        codes_list: list[torch.Tensor],
        *,
        max_step_frames: int,
        max_batch_size: int,
    ) -> list[torch.Tensor]:
        """Decode offline utterances through currently free fixed slots."""
        if not codes_list:
            return []
        borrow_count = min(
            len(codes_list),
            max(int(max_batch_size), 1),
            self.free_slots,
        )
        if borrow_count < 1:
            raise RuntimeError(
                "MOSS-TTS-Realtime codec has no free slot for offline decode"
            )
        borrowed = [self.acquire() for _ in range(borrow_count)]
        wavs: list[torch.Tensor] = []
        decode_succeeded = False
        try:
            for wave_start in range(0, len(codes_list), borrow_count):
                wave = codes_list[wave_start : wave_start + borrow_count]
                slots = borrowed[: len(wave)]
                self._state_adapter.reset_slots(slots, batch_size=self._batch_size)
                cursors = [0] * len(wave)
                chunks: list[list[torch.Tensor]] = [[] for _ in wave]
                while True:
                    remaining = [
                        int(codes.shape[1]) - cursor
                        for codes, cursor in zip(wave, cursors)
                    ]
                    positive = [value for value in remaining if value > 0]
                    if not positive:
                        break
                    if any(value >= max_step_frames for value in positive):
                        step_frames = max_step_frames
                    else:
                        step_frames = min(positive)
                    plan = {
                        slots[index]: codes[
                            :, cursors[index] : cursors[index] + step_frames
                        ]
                        for index, codes in enumerate(wave)
                        if remaining[index] >= step_frames
                    }
                    decoded = self.step(plan)
                    for index, slot in enumerate(slots):
                        if slot not in plan:
                            continue
                        chunks[index].append(decoded[slot])
                        cursors[index] += step_frames
                wavs.extend(torch.cat(item, dim=-1) for item in chunks)
            decode_succeeded = True
        finally:
            release_error: BaseException | None = None
            for slot in borrowed:
                try:
                    self.release(slot)
                except BaseException as exc:
                    logger.exception(
                        "MOSS-TTS-Realtime failed to reset borrowed codec slot %d; "
                        "the slot is quarantined",
                        slot,
                    )
                    release_error = release_error or exc
            if decode_succeeded and release_error is not None:
                raise release_error
        return wavs


@dataclass
class _RealtimeStreamState:
    slot: int | None
    pending: list[torch.Tensor] = field(default_factory=list)
    ramp_index: int = 0
    next_chunk_frames: int = _CHUNK_RAMP[0]
    session_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    identity_latched: bool = False


@dataclass
class _CodecSessionEntry:
    """Session-scoped codec slot ownership.

    The leased slot is keyed by ``session_id`` and survives per-turn request
    teardown: the causal codec state continues across a session's turns. The
    lease ends with release+reset only on session close, idle TTL sweep, abort,
    or decode failure. ``closing`` marks a session whose close arrived while a
    turn was still draining; the release lands at that request's teardown.
    """

    slot: int
    live_request_ids: set[str] = field(default_factory=set)
    closing: bool = False
    last_active_at: float = 0.0


@dataclass(frozen=True)
class _CoalescedStepPlan:
    step_frames: int
    slot_codes: dict[int, torch.Tensor]


class MossTTSRealtimeStreamingVocoderScheduler(
    StreamingVocoderBase[_RealtimeStreamState, _CoalescedStepPlan]
):
    """Decode generated 16-codebook frames in persistent request-owned slots."""

    _can_batch_stream_chunks = True
    _stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        codec: Any,
        *,
        n_vq: int,
        stream_slots: int = 16,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        cuda_graph: bool = True,
        cuda_graph_frames: list[int] | None = None,
        cuda_graph_min_free_gb: float = 3.0,
        session_idle_ttl_s: float = 300.0,
    ) -> None:
        if stream_slots < 1:
            raise ValueError(f"stream_slots must be >= 1, got {stream_slots}")
        if float(session_idle_ttl_s) <= 0:
            raise ValueError(
                f"session_idle_ttl_s must be positive, got {session_idle_ttl_s}"
            )
        missing = [
            name
            for name in ("streaming", "_decode_frame", "batch_decode", "apply")
            if not callable(getattr(codec, name, None))
        ]
        if missing:
            raise RuntimeError(
                f"MOSS-TTS-Realtime codec is incompatible; missing {sorted(missing)}"
            )
        config = getattr(codec, "config", None)
        if config is None:
            raise RuntimeError("MOSS-TTS-Realtime codec has no config")
        sample_rate = int(
            getattr(config, "sampling_rate", 0) or getattr(config, "sample_rate", 0)
        )
        downsample_rate = int(getattr(config, "downsample_rate", 0) or 0)
        quantizer_kwargs = getattr(config, "quantizer_kwargs", {}) or {}
        num_quantizers = int(quantizer_kwargs.get("num_quantizers", 0) or 0)
        codebook_size = int(quantizer_kwargs.get("codebook_size", 0) or 0)
        if sample_rate < 1 or downsample_rate < 1:
            raise ValueError("MOSS-TTS-Realtime codec timing config must be positive")
        self._n_vq = int(n_vq)
        if self._n_vq < 1:
            raise ValueError("MOSS-TTS-Realtime n_vq must be positive")
        if num_quantizers < self._n_vq:
            raise ValueError(
                f"MOSS-TTS-Realtime codec must expose at least {self._n_vq} quantizers"
            )
        if codebook_size < 1:
            raise ValueError("MOSS-TTS-Realtime codec codebook_size must be positive")

        self._codec = codec
        self._sample_rate = sample_rate
        self._samples_per_frame = downsample_rate
        self._codebook_size = codebook_size
        self._stream_slots = int(stream_slots)
        self._stream_chunk_batch_max = self._stream_slots
        self._max_batch_size = max(int(max_batch_size), 1)
        self._session: _CodecStreamSession | None = None
        self._cuda_graph = bool(cuda_graph)
        self._cuda_graph_frames = (
            [int(frame) for frame in cuda_graph_frames]
            if cuda_graph_frames is not None
            else None
        )
        self._cuda_graph_min_free_gb = float(cuda_graph_min_free_gb)
        if self._cuda_graph_min_free_gb < 0:
            raise ValueError("cuda_graph_min_free_gb must be non-negative")
        if self._cuda_graph_frames is not None:
            if not self._cuda_graph_frames:
                raise ValueError("cuda_graph_frames must not be empty")
            invalid = [
                frame
                for frame in self._cuda_graph_frames
                if not 1 <= frame <= _MAX_CUDA_GRAPH_FRAMES
            ]
            if invalid:
                raise ValueError(
                    "cuda_graph_frames must be within the realtime codec step "
                    f"range [1, {_MAX_CUDA_GRAPH_FRAMES}], got {invalid}"
                )
        self._resource_totals: Counter[str] = Counter()
        self._active_slots_high_water = 0
        self._pending_frames_high_water = 0
        self._session_idle_ttl_s = float(session_idle_ttl_s)
        # Session-scoped codec slot leases: a session's causal codec state
        # survives its turns. Slot pool changes only through the helpers below.
        self._codec_sessions: dict[str, _CodecSessionEntry] = {}
        super().__init__(
            self._vocode,
            batch_compute_fn=self._vocode_batch,
            sample_rate=self._sample_rate,
            stream_source_hint=_SOURCE_HINT,
            max_batch_size=self._max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def _ensure_session(self) -> _CodecStreamSession:
        if self._session is None:
            self._session = _CodecStreamSession(
                self._codec,
                stream_slots=self._stream_slots,
                n_vq=self._n_vq,
                samples_per_frame=self._samples_per_frame,
            )
            logger.info(
                "MOSS-TTS-Realtime codec session opened: slots=%d states=%d",
                self._stream_slots,
                self._session.streaming_state_count,
            )
            self._resource_totals["codec_session_open_total"] += 1
        return self._session

    def _cuda_graph_capture_frames(self) -> list[int]:
        if self._cuda_graph_frames is not None:
            return sorted(set(self._cuda_graph_frames))
        return list(range(1, _DEFAULT_CUDA_GRAPH_MAX_FRAMES + 1))

    def _codec_on_cuda(self) -> bool:
        try:
            return next(self._codec.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _ensure_session_graphed(self) -> _CodecStreamSession:
        with self._state_lock:
            session = self._ensure_session()
            if (
                self._cuda_graph
                and not session.warmup_attempted
                and self._codec_on_cuda()
            ):
                try:
                    session.warmup_cuda_graph(
                        self._cuda_graph_capture_frames(),
                        min_free_gb=self._cuda_graph_min_free_gb,
                    )
                except Exception:
                    logger.exception(
                        "MOSS-TTS-Realtime vocoder CUDA graph capture failed; "
                        "serving eager from this codec session"
                    )
            return session

    def warmup_now(self) -> None:
        """Capture codec graphs before the vocoder stage reports ready."""
        if not self._cuda_graph or not self._codec_on_cuda():
            return
        session = self._ensure_session_graphed()
        if session.has_cuda_graph_runner():
            logger.info(
                "MOSS-TTS-Realtime vocoder CUDA graphs captured at startup: T=%s",
                session.captured_frames(),
            )
        else:
            logger.warning(
                "MOSS-TTS-Realtime vocoder CUDA graph startup capture produced no "
                "graphs; serving eager"
            )

    def _emit_codec_event(
        self,
        request_id: str,
        event_name: str,
        **metadata: Any,
    ) -> None:
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name=f"moss_tts_realtime_codec_{event_name}",
            metadata=metadata,
        )

    def resource_snapshot(self) -> dict[str, Any]:
        session = self._session
        active_slots = session.active_leases if session is not None else 0
        quarantined_slots = session.quarantined_slots if session is not None else 0
        free_slots = session.free_slots if session is not None else self._stream_slots
        pending_frames = sum(
            len(state.pending) for state in self._stream_states.values()
        )
        self._active_slots_high_water = max(
            self._active_slots_high_water,
            active_slots,
        )
        self._pending_frames_high_water = max(
            self._pending_frames_high_water,
            pending_frames,
        )
        totals = dict(sorted(self._resource_totals.items()))
        return {
            "codec_slot_capacity": self._stream_slots,
            "codec_active_slots": active_slots,
            "codec_free_slots": free_slots,
            "codec_quarantined_slots": quarantined_slots,
            "codec_active_slots_high_water": self._active_slots_high_water,
            "codec_live_stream_states": len(self._stream_states),
            "codec_pending_frames": pending_frames,
            "codec_pending_frames_high_water": self._pending_frames_high_water,
            "codec_held_sessions": len(self._codec_sessions),
            "codec_closing_sessions": sum(
                1 for entry in self._codec_sessions.values() if entry.closing
            ),
            "codec_session_idle_ttl_s": self._session_idle_ttl_s,
            "codec_streaming_state_count": (
                session.streaming_state_count if session is not None else 0
            ),
            "codec_cuda_graph_enabled": self._cuda_graph,
            "codec_cuda_graph_warmup_attempted": (
                session.warmup_attempted if session is not None else False
            ),
            "codec_cuda_graph_captured_frames": (
                session.captured_frames() if session is not None else []
            ),
            "codec_cuda_graph_default_max_frames": _DEFAULT_CUDA_GRAPH_MAX_FRAMES,
            "codec_decoder_dtype": str(
                moss_tts_realtime_vocoder_decoder_dtype(self._codec)
            ).removeprefix("torch."),
            "codec_resource_totals": totals,
            "codec_slot_acquire_total": totals.get("codec_slot_acquire_total", 0),
            "codec_slot_release_total": totals.get("codec_slot_release_total", 0),
            "codec_slot_reset_error_total": totals.get(
                "codec_slot_reset_error_total",
                0,
            ),
            "codec_slot_exhaustion_total": totals.get(
                "codec_slot_exhaustion_total",
                0,
            ),
        }

    def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action == "close_realtime_session":
            payload = payload or {}
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                return {
                    "success": False,
                    "message": "close_realtime_session requires session_id",
                    "error": "session_id must be a non-empty string",
                    "data": {},
                }
            with self._state_lock:
                entry = self._codec_sessions.get(session_id)
                released = False
                if entry is not None:
                    entry.closing = True
                    if not entry.live_request_ids:
                        self._release_codec_session_locked(
                            session_id,
                            entry,
                            reason="session_close",
                        )
                        released = True
                return {
                    "success": True,
                    "message": "ok",
                    "data": {
                        "session_id": session_id,
                        "held": entry is not None,
                        "released": released,
                    },
                }
        del payload
        if action != "model_info":
            return {
                "success": True,
                "message": f"unsupported admin action: {action}",
                "data": {"skipped": True, "unsupported": True},
            }
        with self._state_lock:
            return {
                "success": True,
                "message": "ok",
                "data": self.resource_snapshot(),
            }

    def on_serving_start(self) -> None:
        self._ensure_session_graphed()

    def on_serving_stop(self) -> None:
        # The request stream states were cleared by the caller already; drop
        # the session lease bookkeeping before tearing down the slot pool.
        self._codec_sessions.clear()
        if self._session is not None:
            self._session.close()
            self._session = None
            self._resource_totals["codec_session_close_total"] += 1
            self._emit_codec_event("codec-session", "session_close")

    def create_stream_state(self, request_id: str) -> _RealtimeStreamState:
        # Slot acquisition is session-keyed and happens lazily at contract
        # latch: the request may start from pre-payload chunks, and the session
        # id arrives in their metadata. The idle sweep runs here as well so slot
        # reclamation piggybacks on new demand.
        self._sweep_idle_codec_sessions_locked(time.monotonic())
        return _RealtimeStreamState(slot=None)

    def _bind_session_slot_locked(
        self,
        request_id: str,
        state: _RealtimeStreamState,
    ) -> None:
        """Acquire-or-reuse the codec slot owned by the state session."""
        session_id = state.session_id
        if session_id is None:
            raise RuntimeError(
                "MOSS-TTS-Realtime stream chunk has no session_id in its "
                "metadata; the engine must stamp stream_metadata with the "
                "session identity before the first codec frame"
            )
        entry = self._codec_sessions.get(session_id)
        now = time.monotonic()
        if entry is not None:
            state.slot = entry.slot
            entry.live_request_ids.add(request_id)
            entry.last_active_at = now
            if len(entry.live_request_ids) > 1:
                logger.warning(
                    "MOSS-TTS-Realtime session %s has %d concurrent codec "
                    "requests sharing slot %d",
                    session_id,
                    len(entry.live_request_ids),
                    entry.slot,
                )
            return
        try:
            session = self._ensure_session_graphed()
            slot = session.acquire()
        except Exception as exc:
            self._resource_totals["codec_slot_acquire_error_total"] += 1
            if "exhausted" in str(exc).lower():
                self._resource_totals["codec_slot_exhaustion_total"] += 1
            self._emit_codec_event(
                request_id,
                "slot_acquire_error",
                session_id=session_id,
                error=str(exc),
            )
            raise
        self._codec_sessions[session_id] = _CodecSessionEntry(
            slot=slot,
            live_request_ids={request_id},
            last_active_at=now,
        )
        state.slot = slot
        self._resource_totals["codec_slot_acquire_total"] += 1
        self._active_slots_high_water = max(
            self._active_slots_high_water,
            session.active_leases,
        )
        self._emit_codec_event(
            request_id,
            "slot_acquire",
            slot=slot,
            session_id=session_id,
        )

    def _sweep_idle_codec_sessions_locked(self, now: float) -> None:
        """Reap sessions whose slots outlived the engine-side session TTL.

        Engine-side KV reaping frees a session at ``session_idle_ttl_s``; the
        codec lease must outlive that (an in-flight close must not strand the
        slot, and a later sweep would only mask real leaks), so the sweep
        threshold stays strictly above the engine TTL. Runs opportunistically
        under ``_state_lock`` where demand already serializes.
        """
        idle_cutoff = self._session_idle_ttl_s * 2
        for session_id, entry in list(self._codec_sessions.items()):
            if entry.live_request_ids or entry.closing:
                continue
            if now - entry.last_active_at < idle_cutoff:
                continue
            self._release_codec_session_locked(session_id, entry, reason="idle_ttl")

    def _release_codec_session_locked(
        self,
        session_id: str,
        entry: _CodecSessionEntry,
        *,
        reason: str,
    ) -> None:
        """Release a session's slot to the pool with a fresh causal reset.

        Requires ``_state_lock`` (held by every caller path: chunk pump,
        completion/abort teardown, admin, and serving stop).
        """
        slot = entry.slot
        self._codec_sessions.pop(session_id, None)
        self._return_slot_to_pool_locked(
            next(iter(entry.live_request_ids), session_id),
            slot,
            reason=f"session_{reason}",
        )
        self._resource_totals[f"codec_session_release_{reason}_total"] += 1
        self._emit_codec_event(
            next(iter(entry.live_request_ids), session_id),
            "session_slot_release",
            slot=slot,
            session_id=session_id,
            reason=reason,
        )

    def _handle_session_control_item(
        self,
        request_id: str,
        control: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        if control != "close":
            raise RuntimeError(
                f"MOSS-TTS-Realtime codec session control must be 'close', "
                f"got {control!r} for {request_id!r}"
            )
        session_id = metadata.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError(
                "MOSS-TTS-Realtime codec session control is missing session_id"
            )
        entry = self._codec_sessions.get(session_id)
        if entry is None:
            # Idempotent: the session may never have decoded (or is gone).
            return
        entry.closing = True
        if not entry.live_request_ids:
            self._release_codec_session_locked(
                session_id,
                entry,
                reason="session_close",
            )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _RealtimeStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        identity_source = source.data if origin == "payload" else source
        if not state.identity_latched and isinstance(identity_source, Mapping):
            for name in ("session_id", "turn_id"):
                value = identity_source.get(name)
                if (
                    getattr(state, name) is None
                    and isinstance(value, str)
                    and value.strip()
                ):
                    setattr(state, name, value)
            turn_index = identity_source.get("turn_index")
            if (
                state.turn_index is None
                and not isinstance(turn_index, bool)
                and isinstance(turn_index, int)
                and turn_index >= 0
            ):
                state.turn_index = turn_index
            state.identity_latched = True
        # The codec slot lease is session-keyed: bind it once the session id
        # is known. Only chunk arrivals bind -- a terminal-payload latch for a
        # request that never carried an audio chunk must not hold a slot (the
        # offline fallback path decodes without one).
        if state.slot is None and state.session_id is not None and origin != "payload":
            self._bind_session_slot_locked(request_id, state)
        if origin == "payload":
            return
        metadata: Mapping[str, Any] = source
        n_vq = metadata.get("n_vq")
        if n_vq is not None and int(n_vq) != self._n_vq:
            raise ValueError(
                f"MOSS-TTS-Realtime stream n_vq for {request_id!r} must be "
                f"{self._n_vq}, got {n_vq}"
            )
        sample_rate = metadata.get("sample_rate")
        if sample_rate is not None and int(sample_rate) != self._sample_rate:
            raise ValueError(
                f"MOSS-TTS-Realtime stream sample_rate for {request_id!r} must be "
                f"{self._sample_rate}, got {sample_rate}"
            )

    def validate_chunk(
        self,
        request_id: str,
        state: _RealtimeStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if (
            codes.dtype == torch.bool
            or torch.is_floating_point(codes)
            or torch.is_complex(codes)
        ):
            raise TypeError("MOSS-TTS-Realtime stream codes must use an integer dtype")
        if codes.ndim != 1 or int(codes.shape[0]) != self._n_vq:
            raise ValueError(
                "MOSS-TTS-Realtime stream chunk must have shape "
                f"[{self._n_vq}], got {tuple(codes.shape)}"
            )
        codes = codes.detach().to(device="cpu", dtype=torch.long).contiguous()
        if torch.any(codes < 0) or torch.any(codes >= self._codebook_size):
            raise ValueError(
                f"MOSS-TTS-Realtime stream codes must be in [0, {self._codebook_size})"
            )
        return codes

    def ingest(
        self,
        request_id: str,
        state: _RealtimeStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        if state.slot is None:
            raise RuntimeError("MOSS-TTS-Realtime stream has no codec slot")
        state.pending.append(codes)
        pending_frames = sum(
            len(stream_state.pending) for stream_state in self._stream_states.values()
        )
        self._pending_frames_high_water = max(
            self._pending_frames_high_water,
            pending_frames,
        )

    def select_step_participants(self) -> list[tuple[str, _RealtimeStreamState]]:
        due = [
            (request_id, state)
            for request_id, state in self._stream_state_items()
            if state.slot is not None and len(state.pending) >= state.next_chunk_frames
        ]
        if not due:
            return []
        step_frames = min(state.next_chunk_frames for _, state in due)
        return [
            (request_id, state)
            for request_id, state in due
            if state.next_chunk_frames == step_frames
        ]

    def _coalesced_step_frames(
        self,
        participants: list[tuple[str, _RealtimeStreamState]],
    ) -> int:
        threshold = participants[0][1].next_chunk_frames
        if any(state.next_chunk_frames != threshold for _, state in participants):
            raise RuntimeError(
                "MOSS-TTS-Realtime coalesced participants have different thresholds"
            )
        if threshold != _STEADY_CHUNK_FRAMES:
            return threshold

        common_pending = min(len(state.pending) for _, state in participants)
        session = self._session
        captured = session.captured_frames() if session is not None else []
        candidates = [
            frame
            for frame in captured
            if _STEADY_CHUNK_FRAMES <= frame <= common_pending
        ]
        return max(candidates, default=_STEADY_CHUNK_FRAMES)

    def build_step_plan(
        self,
        participants: list[tuple[str, _RealtimeStreamState]],
    ) -> _CoalescedStepPlan:
        if not participants:
            raise ValueError("MOSS-TTS-Realtime codec step has no participants")
        step_frames = self._coalesced_step_frames(participants)
        slot_codes: dict[int, torch.Tensor] = {}
        for _, state in participants:
            if state.slot is None:
                raise RuntimeError("MOSS-TTS-Realtime participant lost its codec slot")
            slot_codes[state.slot] = torch.stack(
                state.pending[:step_frames],
                dim=1,
            )
        return _CoalescedStepPlan(
            step_frames=step_frames,
            slot_codes=slot_codes,
        )

    def run_step(
        self,
        participants: list[tuple[str, _RealtimeStreamState]],
        plan: _CoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        session = self._ensure_session()
        first_participants: list[tuple[str, _RealtimeStreamState]] = []
        shared_metadata: dict[str, Any] = {}
        if realtime_events_active():
            first_participants = [
                (request_id, state)
                for request_id, state in participants
                if state.ramp_index == 0
            ]
            shared_metadata = {
                "step_frames": plan.step_frames,
                "participant_count": len(participants),
                "codec_slot_width": self._stream_slots,
                "codec_active_slots": session.active_leases,
                "execution_mode": (
                    "cuda_graph"
                    if plan.step_frames in session.captured_frames()
                    else "eager"
                ),
                "cuda_graph_enabled": self._cuda_graph,
            }
        for request_id, state in first_participants:
            metadata = realtime_identity_metadata(state)
            metadata.update(shared_metadata)
            metadata.update({"codec_slot": state.slot, "decode_step_index": 0})
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="vocoder_step_start",
                metadata=metadata,
            )

        decoded = session.step(plan.slot_codes)
        self._resource_totals["codec_decode_step_total"] += 1
        self._resource_totals["codec_decoded_frame_total"] += plan.step_frames * len(
            participants
        )
        if plan.step_frames > _STEADY_CHUNK_FRAMES:
            catchup_frames = plan.step_frames * len(participants)
            self._resource_totals["codec_catchup_step_total"] += 1
            self._resource_totals["codec_catchup_frame_total"] += catchup_frames
        output: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            if state.slot is None:
                raise RuntimeError("MOSS-TTS-Realtime participant lost its codec slot")
            del state.pending[: plan.step_frames]
            if state.ramp_index < len(_CHUNK_RAMP) - 1:
                state.ramp_index += 1
            state.next_chunk_frames = _CHUNK_RAMP[state.ramp_index]
            output[request_id] = decoded[state.slot]
        for request_id, state in first_participants:
            waveform = output[request_id]
            metadata = realtime_identity_metadata(state)
            metadata.update(shared_metadata)
            metadata.update(
                {
                    "codec_slot": state.slot,
                    "decode_step_index": 0,
                    "output_samples": int(waveform.shape[-1]),
                }
            )
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="vocoder_step_end",
                metadata=metadata,
            )
        return output

    def _return_slot_to_pool_locked(
        self,
        request_id: str,
        slot: int,
        *,
        reason: str,
    ) -> None:
        """Reset and free a slot at the pool level (session table unchanged)."""
        if self._session is None:
            return
        try:
            self._session.release(slot)
        except Exception as exc:
            self._resource_totals["codec_slot_release_error_total"] += 1
            self._resource_totals["codec_slot_reset_error_total"] += 1
            self._emit_codec_event(
                request_id,
                "slot_release_error",
                slot=slot,
                error=str(exc),
            )
            raise
        self._resource_totals["codec_slot_release_total"] += 1
        self._emit_codec_event(request_id, "slot_release", slot=slot, reason=reason)

    def _release_state_slot(
        self,
        request_id: str,
        state: _RealtimeStreamState,
    ) -> None:
        """Detach the per-request slot view and apply the session release policy.

        The slot itself is session-owned: a successful turn keeps the lease so
        the next turn continues the causal codec state. The lease is released
        (and its streaming state reset) only when this teardown is abort-driven
        or the session is closing; an orphaned slot (no session entry, e.g. a
        bind that failed halfway) always returns to the pool.
        """
        slot = state.slot
        if slot is None:
            return
        state.slot = None
        session_id = state.session_id
        entry = self._codec_sessions.get(session_id) if session_id is not None else None
        if entry is None:
            self._return_slot_to_pool_locked(request_id, slot, reason="orphan")
            return
        entry.live_request_ids.discard(request_id)
        entry.last_active_at = time.monotonic()
        if self._is_aborted(request_id):
            if entry.live_request_ids:
                # Not protocol-possible today (one active turn per session);
                # never cut a live sibling off the shared slot.
                logger.warning(
                    "MOSS-TTS-Realtime session %s kept slot %d: abort of %s "
                    "while %d request(s) still live",
                    session_id,
                    slot,
                    request_id,
                    len(entry.live_request_ids),
                )
                return
            self._release_codec_session_locked(session_id, entry, reason="abort")
            return
        if entry.closing and not entry.live_request_ids:
            self._release_codec_session_locked(
                session_id, entry, reason="session_close"
            )

    def decode_delta(
        self,
        request_id: str,
        state: _RealtimeStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if not is_final:
            return None
        audio_parts: list[torch.Tensor] = []
        if state.pending:
            if state.slot is None:
                raise RuntimeError("MOSS-TTS-Realtime final flush has no codec slot")
            session = self._ensure_session()
            while state.pending:
                step_frames = min(len(state.pending), _STEADY_CHUNK_FRAMES)
                codes = torch.stack(state.pending[:step_frames], dim=1)
                del state.pending[:step_frames]
                audio_parts.append(session.step({state.slot: codes})[state.slot])
        # No slot release here: the codec slot is session-scoped. Draining the
        # final PCM keeps the causal codec state alive for the session's next
        # turn; release+reset is owned by release_stream_resources on close,
        # idle TTL, abort, or failure.
        if not audio_parts:
            return None
        return torch.cat(audio_parts, dim=-1)

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=f"{_SOURCE_HINT} streaming",
        )

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _RealtimeStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_payload_codes(payload)

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _RealtimeStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(MossTTSRealtimeState.from_dict(payload.data))
        if usage is not None:
            result["usage"] = usage
        return result

    def release_stream_resources(
        self,
        request_id: str,
        state: _RealtimeStreamState,
    ) -> None:
        # Stage threads may abort without holding the serving lock; the session
        # table and the slot pool both live under _state_lock everywhere else.
        with self._state_lock:
            self._release_state_slot(request_id, state)

    def _ingest_stream_item(
        self, request_id: str, item: Any
    ) -> _RealtimeStreamState | None:
        """Divert codec session-control markers before any state handling."""
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        control = metadata.get("session_control")
        if control is None:
            return super()._ingest_stream_item(request_id, item)
        with self._state_lock:
            self._handle_session_control_item(request_id, control, metadata)
        return None

    def _prepare_codes(
        self,
        payload: StagePayload,
    ) -> tuple[MossTTSRealtimeState, torch.Tensor | None]:
        state = MossTTSRealtimeState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS-Realtime vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes)
        if codes.ndim != 2 or int(codes.shape[1]) != self._n_vq:
            raise ValueError(
                f"MOSS-TTS-Realtime audio_codes must have shape [T, {self._n_vq}]"
            )
        if (
            codes.dtype == torch.bool
            or torch.is_floating_point(codes)
            or torch.is_complex(codes)
        ):
            raise TypeError("MOSS-TTS-Realtime audio_codes must use an integer dtype")
        codes = codes.to(device="cpu", dtype=torch.long).contiguous()
        if codes.numel() == 0:
            return state, None
        if torch.any(codes < 0) or torch.any(codes >= self._codebook_size):
            raise ValueError(
                f"MOSS-TTS-Realtime audio_codes must be in [0, {self._codebook_size})"
            )
        return state, codes

    def _decode_full_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        device = next(self._codec.parameters()).device
        channels_first = [
            codes.transpose(0, 1).contiguous().to(device=device, dtype=torch.long)
            for codes in codes_list
        ]
        with torch.inference_mode():
            result = self._codec.batch_decode(
                channels_first,
                num_quantizers=self._n_vq,
            )
        audio = getattr(result, "audio", None)
        audio_lengths = getattr(result, "audio_lengths", None)
        if not isinstance(audio, torch.Tensor) or not isinstance(
            audio_lengths, torch.Tensor
        ):
            raise RuntimeError("MOSS-TTS-Realtime codec batch_decode returned no audio")
        if audio.ndim != 3 or int(audio.shape[0]) != len(codes_list):
            raise RuntimeError("MOSS-TTS-Realtime offline codec batch is misaligned")
        if int(audio.shape[1]) != 1:
            raise RuntimeError("MOSS-TTS-Realtime offline codec must emit mono audio")
        audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
        lengths_cpu = audio_lengths.detach().to(device="cpu")
        waveforms: list[torch.Tensor] = []
        for index, codes in enumerate(codes_list):
            samples = int(lengths_cpu[index])
            expected = int(codes.shape[0]) * self._samples_per_frame
            if samples != expected:
                raise RuntimeError(
                    "MOSS-TTS-Realtime offline codec length mismatch: "
                    f"samples={samples} expected={expected}"
                )
            waveforms.append(audio_cpu[index, :, :samples].contiguous())
        return waveforms

    def _decode_codes_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not codes_list:
            return []
        channels_first = [codes.transpose(0, 1).contiguous() for codes in codes_list]
        with self._state_lock:
            if self._session is not None and self._session.active_leases == 0:
                self._session.close()
                self._session = None
            if self._session is None:
                return self._decode_full_batch(codes_list)
            return self._session.decode_borrowed(
                channels_first,
                max_step_frames=_STEADY_CHUNK_FRAMES,
                max_batch_size=self._max_batch_size,
            )

    def _decode_payload_codes(self, payload: StagePayload) -> torch.Tensor | None:
        _, codes = self._prepare_codes(payload)
        if codes is None:
            return None
        return self._decode_codes_rows([codes])[0]

    def _store_result(
        self,
        payload: StagePayload,
        state: MossTTSRealtimeState,
        waveform: torch.Tensor,
    ) -> StagePayload:
        state.audio_codes = None
        state.sample_rate = self._sample_rate
        payload.data = state.to_dict()
        payload.data.update(
            audio_waveform_payload(
                waveform,
                sample_rate=self._sample_rate,
                modality="audio",
                source_hint=_SOURCE_HINT,
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [self._prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(self._decode_codes_rows(codes_list)) if codes_list else iter(())
        results: list[StagePayload] = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                state.audio_codes = None
                payload.data = state.to_dict()
                results.append(payload)
                continue
            results.append(self._store_result(payload, state, next(decoded)))
        return results

    def _vocode(self, payload: StagePayload) -> StagePayload:
        return self._vocode_batch([payload])[0]


__all__ = ["MossTTSRealtimeStreamingVocoderScheduler"]
