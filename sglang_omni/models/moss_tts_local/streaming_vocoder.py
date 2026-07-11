# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Local.

Streaming requests share one persistent batched ``codec.streaming()`` session.
Pure non-streaming traffic uses the MOSS decoder with packed SGLang FlashAttention
when no live streaming session owns the codec state.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.moss_tts_local.vocoder_decoder import MossTTSLocalVocoderDecoder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_SOURCE_HINT = "MOSS-TTS Local"


class _CodecStreamSession:
    """Persistent batched ``codec.streaming()`` session with slot bookkeeping (stream slots held by live requests; offline slots for non-streaming decodes). Scheduler-loop-thread only."""

    def __init__(
        self, codec: Any, *, stream_slots: int, offline_slots: int, n_vq: int
    ) -> None:
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._offline_slots = int(offline_slots)
        self._batch_size = self._stream_slots + self._offline_slots
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._free_stream_slots = list(range(self._stream_slots))
        self._closed = False
        self._cg_runner: Any | None = None
        # Capture is attempted at most once per session; a low-VRAM skip must not re-probe per step.
        self.warmup_attempted = False
        # Per-T graph-vs-eager step counts for capture-hit-rate reporting (host-side, no GPU sync).
        self._cg_graph_t: Counter = Counter()
        self._cg_eager_t: Counter = Counter()
        self._cg_total_steps = 0
        # Retain the streaming ExitStack so per-slot causal state lives across steps (closed in close());
        # graph replay is kept bit-identical to this stateful decode by the in-place cache patch.
        self._exit_stack = contextlib.ExitStack()
        with torch.no_grad():
            self._exit_stack.enter_context(codec.streaming(self._batch_size))

    def warmup_cuda_graph(
        self, frames: list[int], *, min_free_gb: float = 3.0
    ) -> list[int]:
        """Capture per-T graphs then reset all slots; returns the captured T list (rest fall back to
        eager). Attempted at most once per session; never captures during ``step``."""
        self.warmup_attempted = True
        if self._closed:
            return []
        from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
            MossVocoderCudaGraphRunner,
            patch_codec_attention_cache_for_cuda_graph,
        )

        # Patch the codec attention cache to an in-place write so the graph can capture it
        # (bit-identical to eager).
        patch_codec_attention_cache_for_cuda_graph(self._codec)
        if self._cg_runner is None:
            # Scheduler owns the capture shape range (max_frames = the largest T it asks for), rather
            # than the runner keeping an independent default limit.
            self._cg_runner = MossVocoderCudaGraphRunner(
                self._codec,
                batch_size=self._batch_size,
                n_vq=self._n_vq,
                max_frames=max(frames) if frames else 1,
                min_free_gb=min_free_gb,
            )
        try:
            self._cg_runner.warmup(frames)
        except Exception:
            # Drop a half-built runner on probe failure so serving stays on the eager path.
            self._cg_runner = None
            raise
        self._reset_slots(list(range(self._batch_size)))
        captured = self._cg_runner.captured_frames()
        if not captured:
            # Nothing captured (low VRAM / all failed): drop the runner so serving does not pay a
            # wasted decode_step probe every step only to fall back to eager.
            self._cg_runner = None
        return captured

    def has_cuda_graph_runner(self) -> bool:
        # True only if the runner exists AND captured at least one graph.
        return bool(self._cg_runner and self._cg_runner.captured_frames())

    def captured_frames(self) -> list[int]:
        return self._cg_runner.captured_frames() if self._cg_runner else []

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        return self._free_stream_slots.pop()

    def release(self, slot: int) -> None:
        if self._closed:
            return
        self._reset_slots([slot])
        self._free_stream_slots.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        if self._cg_runner is not None:
            self._log_cg_stats()
        with torch.no_grad():
            self._exit_stack.close()
        self._closed = True

    def _log_cg_stats(self) -> None:
        graph = sum(self._cg_graph_t.values())
        eager = sum(self._cg_eager_t.values())
        total = graph + eager
        if not total:
            return
        logger.info(
            "MOSS vocoder CG stats: %d/%d steps graphed (%.1f%%); graph T=%s eager T=%s",
            graph,
            total,
            100.0 * graph / total,
            dict(sorted(self._cg_graph_t.items())),
            dict(sorted(self._cg_eager_t.items())),
        )

    def _reset_slots(self, slots: list[int]) -> None:
        reset_mask = torch.zeros(
            self._batch_size, dtype=torch.bool, device=self._device
        )
        reset_mask[slots] = True

        def _reset(module: Any) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(reset_mask.to(state.device))

        with torch.no_grad():
            self._codec.apply(_reset)

    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Advance participating slots by one uniform-length step. ``slot_codes`` maps slot -> ``[n_vq, T]`` (same T); returns slot -> ``[channels, samples]`` float32 CPU audio."""
        if not slot_codes:
            return {}
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                f"streaming step requires a uniform length, got {sorted(step_lengths)}"
            )
        (step_t,) = step_lengths
        n_vq = int(next(iter(slot_codes.values())).shape[0])
        codes_step = torch.zeros(
            n_vq, self._batch_size, step_t, dtype=torch.long, device=self._device
        )
        codes_lengths = torch.zeros(
            self._batch_size, dtype=torch.long, device=self._device
        )
        exec_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        for slot, codes in slot_codes.items():
            codes_step[:, slot, :] = codes.to(device=self._device, dtype=torch.long)
            codes_lengths[slot] = step_t
            exec_mask[slot] = True
        slots = list(slot_codes)
        graphed = None
        graph_failed = False
        try:
            with torch.no_grad():
                if self._cg_runner is not None:
                    try:
                        graphed = self._cg_runner.decode_step(codes_step, exec_mask)
                    except Exception:
                        graph_failed = True
                        raise
                if graphed is not None:
                    audio, audio_lengths = graphed
                else:
                    self._codec._set_streaming_exec_mask(exec_mask)
                    result = self._codec._decode_frame(codes_step, codes_lengths)
                    audio, audio_lengths = result.audio, result.audio_lengths
            # One batched D2H per step. A graph replay error can surface async HERE (not in
            # decode_step), so materialization stays inside the replay guard.
            audio_cpu = audio[slots].detach().to("cpu", torch.float32)
            lengths_cpu = audio_lengths[slots].detach().to("cpu")
        except Exception:
            # Graphed step failed (in decode_step or async on the D2H): disable the runner so future
            # steps go eager; participants abort. An eager-path error does not disable it.
            if self._cg_runner is not None and (graph_failed or graphed is not None):
                logger.exception(
                    "MOSS vocoder CUDA-graph replay failed (in decode_step or on output "
                    "materialization); disabling runner, serving eager from here"
                )
                self._cg_runner = None
            raise
        if self._cg_runner is not None:
            if graphed is not None:
                self._cg_graph_t[step_t] += 1
            else:
                self._cg_eager_t[step_t] += 1
            self._cg_total_steps += 1
            if self._cg_total_steps % 2000 == 0:
                self._log_cg_stats()
        out: dict[int, torch.Tensor] = {}
        for index, slot in enumerate(slots):
            n_samples = int(lengths_cpu[index])
            out[slot] = audio_cpu[index, :, :n_samples]
        return out

    def decode_offline(
        self, codes_list: list[torch.Tensor], *, max_step_frames: int
    ) -> list[torch.Tensor]:
        """Decode complete utterances ``[n_vq, T]`` through offline slots in the
        persistent codec session."""
        wavs: list[torch.Tensor] = []
        for wave_start in range(0, len(codes_list), self._offline_slots):
            wave = codes_list[wave_start : wave_start + self._offline_slots]
            slots = [self._stream_slots + i for i in range(len(wave))]
            self._reset_slots(slots)
            cursors = [0] * len(wave)
            chunks: list[list[torch.Tensor]] = [[] for _ in wave]
            while True:
                remaining = [
                    int(codes.shape[1]) - cur for codes, cur in zip(wave, cursors)
                ]
                positive = [r for r in remaining if r > 0]
                if not positive:
                    break
                if any(r >= max_step_frames for r in positive):
                    step_t = max_step_frames
                else:
                    step_t = min(positive)
                plan = {
                    slots[i]: wave[i][:, cursors[i] : cursors[i] + step_t]
                    for i, rem in enumerate(remaining)
                    if rem >= step_t
                }
                decoded = self.step(plan)
                for i in range(len(wave)):
                    if slots[i] in plan:
                        chunks[i].append(decoded[slots[i]])
                        cursors[i] += step_t
            for item_chunks in chunks:
                wavs.append(torch.cat(item_chunks, dim=-1))
        return wavs


@dataclass
class _LocalStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    n_vq: int | None = None
    initial_chunk_frames: int = 0
    threshold: int = 0


@dataclass
class _CoalescedStepPlan:
    step_t: int
    slot_codes: dict[int, torch.Tensor]


class MossTTSLocalStreamingVocoderScheduler(
    StreamingVocoderBase[_LocalStreamState, _CoalescedStepPlan]
):
    """Decode MOSS-TTS Local codec rows incrementally on the v2 codec."""

    _can_batch_stream_chunks = True

    def __init__(
        self,
        codec: Any,
        *,
        n_vq: int,
        sample_rate: int,
        stream_slots: int = 8,
        stream_chunk_frames: int = 25,
        initial_chunk_frames: int = 5,
        coalesce_floor_frames: int = 5,
        max_step_frames: int = 100,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        cuda_graph: bool = True,
        cuda_graph_frames: list[int] | None = None,
        cuda_graph_min_free_gb: float = 3.0,
    ) -> None:
        if stream_slots < 1:
            raise ValueError(f"stream_slots must be >= 1, got {stream_slots}")
        if not 0 < stream_chunk_frames <= max_step_frames:
            raise ValueError(
                "stream_chunk_frames must be in (0, max_step_frames], got "
                f"{stream_chunk_frames} (max_step_frames={max_step_frames})"
            )
        missing = [
            name
            for name in (
                "streaming",
                "_set_streaming_exec_mask",
                "_decode_frame",
                "decode",
            )
            if not hasattr(codec, name)
        ]
        if missing:
            raise RuntimeError(
                f"MOSS-TTS Local streaming vocoder: codec is missing {missing}; "
                "the installed MOSS-Audio-Tokenizer-v2 version is incompatible"
            )
        nonstream_decoder = MossTTSLocalVocoderDecoder(codec.decoder)
        logger.info(
            f"MOSS-TTS Local non-streaming vocoder uses packed SGLang attention "
            f"stages={len(nonstream_decoder)}"
        )
        self._codec = codec
        self._nonstream_decoder = nonstream_decoder
        self._stream_slots = int(stream_slots)
        # Coalesce up to one full set of streaming lanes per pump, not the offline batch width.
        self._stream_chunk_batch_max = self._stream_slots
        self._stream_chunk_frames = int(stream_chunk_frames)
        self._default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(stream_chunk_frames))
        )
        self._coalesce_floor_frames = max(
            0, min(int(coalesce_floor_frames), int(stream_chunk_frames))
        )
        self._max_step_frames = int(max_step_frames)
        self._offline_slots = max(int(max_batch_size), 1)
        self._n_vq = int(n_vq)
        self._session: _CodecStreamSession | None = None
        self._session_used_by_streaming = False
        self._cuda_graph = bool(cuda_graph)
        self._cuda_graph_frames = (
            [int(t) for t in cuda_graph_frames] if cuda_graph_frames else None
        )
        self._cuda_graph_min_free_gb = float(cuda_graph_min_free_gb)
        if self._cuda_graph_frames is not None:
            too_large = [
                t for t in self._cuda_graph_frames if t > self._max_step_frames
            ]
            if too_large:
                raise ValueError(
                    f"cuda_graph_frames exceed max_step_frames={self._max_step_frames}: "
                    f"{too_large}"
                )
        super().__init__(
            self._vocode,
            batch_compute_fn=self._vocode_batch,
            sample_rate=sample_rate,
            stream_source_hint=_SOURCE_HINT,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def on_serving_stop(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_used_by_streaming = False

    def create_stream_state(self, request_id: str) -> _LocalStreamState:
        del request_id
        return _LocalStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _LocalStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = (
                source.request.params
                if isinstance(source.request.params, dict)
                else None
            )
            self._latch_thresholds(request_id, state, params)
            return
        metadata: Mapping[str, Any] = source
        n_vq = metadata.get("n_vq")
        if n_vq is not None:
            n_vq = int(n_vq)
            if state.n_vq is not None and state.n_vq != n_vq:
                raise ValueError(
                    f"MOSS-TTS Local stream n_vq changed for {request_id!r}: "
                    f"{state.n_vq} -> {n_vq}"
                )
            state.n_vq = n_vq
        if state.threshold == 0:
            self._latch_thresholds(request_id, state, metadata)

    def validate_chunk(
        self, request_id: str, state: _LocalStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        codes = codes.to(dtype=torch.long)
        n_vq = state.n_vq if state.n_vq is not None else self._n_vq
        if codes.ndim != 1 or int(codes.shape[0]) < n_vq + 1:
            raise ValueError(
                f"MOSS-TTS Local stream chunk must be a 1-D row with at least "
                f"{n_vq + 1} channels (text + codes), got {tuple(codes.shape)}"
            )
        # Row layout matches output_rows: [text_token, code_0, ..., code_{n_vq-1}].
        return codes[1 : 1 + n_vq]

    def ingest(
        self, request_id: str, state: _LocalStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        self._ensure_slot(state)

    def decode_delta(
        self, request_id: str, state: _LocalStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain: pending frames go through the request's session
        slot (released afterwards) or the offline lane when slot-starved;
        steady-state chunks decode through the coalesced step hooks instead."""
        del request_id, is_final
        audio_parts: list[torch.Tensor] = []
        if state.slot is None and state.pending:
            # Slot-starved: every frame is still buffered, decode offline.
            codes = torch.stack(state.pending, dim=1)
            state.pending = []
            audio_parts.extend(
                self._ensure_session_graphed().decode_offline(
                    [codes], max_step_frames=self._max_step_frames
                )
            )
        elif state.slot is not None:
            session = self._ensure_session_graphed()
            while state.pending:
                step_t = min(len(state.pending), self._max_step_frames)
                codes = torch.stack(state.pending[:step_t], dim=1)
                del state.pending[:step_t]
                audio_parts.append(session.step({state.slot: codes})[state.slot])
            session.release(state.slot)
            state.slot = None
        if not audio_parts:
            return None
        return torch.cat(audio_parts, dim=-1)

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform.detach().to("cpu", torch.float32),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=f"{_SOURCE_HINT} streaming",
            keep_channels=True,
        )

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _LocalStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_payload_codes(payload)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _LocalStreamState
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(MossTTSLocalState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        return final_data

    def release_stream_resources(
        self, request_id: str, state: _LocalStreamState
    ) -> None:
        del request_id
        if state.slot is not None and self._session is not None:
            self._session.release(state.slot)

    def select_step_participants(self) -> list[tuple[str, _LocalStreamState]]:
        """Every stream whose buffer crossed its threshold is due; due streams
        coalesce with peers above the join floor into one forward."""
        join_floor = max(
            1, min(self._coalesce_floor_frames or 5, self._stream_chunk_frames)
        )
        slotted = [
            (request_id, state)
            for request_id, state in self._stream_state_items()
            if state.slot is not None and state.threshold > 0
        ]
        due = [
            entry for entry in slotted if len(entry[1].pending) >= entry[1].threshold
        ]
        if not due:
            return []
        floor = min(
            min(len(state.pending) for _, state in due),
            join_floor,
        )
        return [
            entry
            for entry in slotted
            if self._can_join_coalesced_step(entry[0], entry[1], floor)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, _LocalStreamState]]
    ) -> _CoalescedStepPlan:
        """Uniform step capped at the steady chunk size (= CUDA-graph capture
        ceiling) so coalesced backlogs stay on the graphed fast path; the base
        pump loop re-pumps any remainder."""
        step_t = min(
            min(len(state.pending) for _, state in participants),
            self._stream_chunk_frames,
        )
        return _CoalescedStepPlan(
            step_t=step_t,
            slot_codes={
                state.slot: torch.stack(state.pending[:step_t], dim=1)
                for _, state in participants
            },
        )

    def run_step(
        self,
        participants: list[tuple[str, _LocalStreamState]],
        plan: _CoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        decoded = self._ensure_session().step(plan.slot_codes)
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            del state.pending[: plan.step_t]
            state.threshold = self._stream_chunk_frames
            out[request_id] = decoded[state.slot]
        return out

    def _can_join_coalesced_step(
        self, request_id: str, state: _LocalStreamState, floor: int
    ) -> bool:
        if len(state.pending) >= state.threshold:
            return True
        if not self._stream_has_emitted(request_id):
            return False
        return len(state.pending) >= floor

    def _ensure_session(self) -> _CodecStreamSession:
        if self._session is None:
            self._session = _CodecStreamSession(
                self._codec,
                stream_slots=self._stream_slots,
                offline_slots=self._offline_slots,
                n_vq=self._n_vq,
            )
        return self._session

    def _close_idle_startup_session_locked(self) -> None:
        if (
            self._session is not None
            and not self._session_used_by_streaming
            and not self._stream_states
            and not self._stream_payloads
        ):
            self._session.close()
            self._session = None

    def _cuda_graph_capture_frames(self) -> list[int]:
        """Step lengths T to capture. Config ``cuda_graph_frames`` overrides the default."""
        if self._cuda_graph_frames:
            # Validated at config (>= 1) and __init__ (<= max_step_frames); use as configured.
            return sorted(set(self._cuda_graph_frames))
        max_frame = min(self._stream_chunk_frames, self._max_step_frames)
        return list(range(1, max_frame + 1))

    def _codec_on_cuda(self) -> bool:
        try:
            return next(self._codec.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _ensure_session_graphed(self) -> _CodecStreamSession:
        """Live session with CUDA graphs captured (at most once). Streaming paths call this instead
        of _ensure_session so a session created after non-streaming traffic closed the graphed
        startup session is re-captured; a low-VRAM skip is remembered (no per-step re-probe).

        That first post-non-streaming streaming request pays a one-time warmup latency (the recapture
        runs synchronously here, fail-safe to eager on low VRAM); streaming-only traffic uses the
        factory session and never hits this path.
        """
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
                        "MOSS vocoder CUDA-graph capture failed; serving eager from this session"
                    )
            return session

    def warmup_now(self) -> None:
        """Capture the codec-decode graphs at factory-build time: codec loaded, GPU quiescent, and
        before the stage process is marked ready, so the serving loop never races a half-captured
        graph. No-op without a CUDA codec; best-effort, degrades to eager."""
        if not self._cuda_graph or not self._codec_on_cuda():
            return
        session = self._ensure_session_graphed()
        if session.has_cuda_graph_runner():
            logger.info(
                "MOSS vocoder CUDA graphs captured at startup: T=%s",
                session.captured_frames(),
            )
        else:
            logger.warning(
                "MOSS vocoder CUDA graphs did not seal at startup (low VRAM); eager vocoder"
            )

    def _ensure_slot(self, state: _LocalStreamState) -> None:
        if state.slot is None:
            self._session_used_by_streaming = True
            state.slot = self._ensure_session_graphed().acquire()

    def _latch_thresholds(
        self,
        request_id: str,
        state: _LocalStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        explicit = (
            isinstance(params, Mapping)
            and params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None
        )
        if explicit:
            # Explicit 0 opts out of a smaller first chunk.
            state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                params,
                steady_chunk_frames=self._stream_chunk_frames,
            )
        else:
            state.initial_chunk_frames = self._default_initial_chunk_frames
        if state.initial_chunk_frames > 0 and not self._stream_has_emitted(request_id):
            state.threshold = state.initial_chunk_frames
        else:
            state.threshold = self._stream_chunk_frames

    def _decode_payload_codes(self, payload: StagePayload) -> torch.Tensor | None:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            return None
        rows = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if rows.numel() == 0:
            return None
        codes = rows[:, : self._n_vq].transpose(0, 1).contiguous()
        self._session_used_by_streaming = True
        return self._ensure_session_graphed().decode_offline(
            [codes], max_step_frames=self._max_step_frames
        )[0]

    def _prepare_codes(
        self, payload: StagePayload
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Emit no audio: only this request fails downstream, not the batch.
            return state, None
        return state, codes

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: MossTTSLocalState,
        wav: torch.Tensor,
    ) -> StagePayload:
        # The v2 codec is natively stereo: keep [channels, samples] end to end.
        audio_payload = audio_waveform_payload(
            wav, source_hint=_SOURCE_HINT, keep_channels=True
        )
        state.audio_codes = None
        state.sample_rate = self._sample_rate
        payload.data = state.to_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_codes_rows_nonstream(
        self, codes_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        n_vq = self._n_vq
        device = next(self._codec.parameters()).device
        codes_channels_first = [
            codes[:, :n_vq]
            .transpose(0, 1)
            .contiguous()
            .to(device=device, dtype=torch.long)
            for codes in codes_list
        ]
        max_len = max(int(codes.shape[1]) for codes in codes_channels_first)
        audio_codes = torch.zeros(
            n_vq,
            len(codes_channels_first),
            max_len,
            device=device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            len(codes_channels_first), max_len, device=device, dtype=torch.bool
        )
        for index, codes in enumerate(codes_channels_first):
            length = int(codes.shape[1])
            audio_codes[:, index, :length] = codes
            padding_mask[index, :length] = True

        decoded = self._codec.decode(
            audio_codes,
            padding_mask=padding_mask,
            num_quantizers=n_vq,
            return_dict=True,
            chunk_duration=None,
        )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "audio_tokenizer.decode did not return audio/audio_lengths."
            )
        audio_cpu = audio.detach().to("cpu", torch.float32)
        lengths_cpu = audio_lengths.detach().to("cpu")
        return [
            audio_cpu[index, :, : int(lengths_cpu[index])].contiguous()
            for index in range(int(audio_cpu.shape[0]))
        ]

    def _decode_codes_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ``[T, >=n_vq]`` row tensors to fp32 CPU waveforms."""
        with self._state_lock:
            self._close_idle_startup_session_locked()
        if self._session is None:
            # The processor helper forces chunk_duration=8 and enters the
            # tokenizer streaming loop. This decoder is non-streaming, so it must
            # run through the tokenizer's full-sequence decode path.
            original_decoder = self._codec.decoder
            self._codec.decoder = self._nonstream_decoder
            try:
                return self._decode_codes_rows_nonstream(codes_list)
            finally:
                self._codec.decoder = original_decoder
        channels_first = [
            codes[:, : self._n_vq].transpose(0, 1).contiguous() for codes in codes_list
        ]
        # abort() resets slots under _state_lock from other threads; serialize
        # every session access on the same lock.
        with self._state_lock:
            wavs = self._session.decode_offline(
                channels_first, max_step_frames=self._max_step_frames
            )
        return [wav.detach().to("cpu", torch.float32).contiguous() for wav in wavs]

    def _vocode_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [self._prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(self._decode_codes_rows(codes_list)) if codes_list else iter(())
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                state.audio_codes = None
                payload.data = state.to_dict()
                results.append(payload)
                continue
            results.append(self._store_vocoder_result(payload, state, next(decoded)))
        return results

    def _vocode(self, payload: StagePayload) -> StagePayload:
        return self._vocode_batch([payload])[0]


__all__ = ["MossTTSLocalStreamingVocoderScheduler"]
