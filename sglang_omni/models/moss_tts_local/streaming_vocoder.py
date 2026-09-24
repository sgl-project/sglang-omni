# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Local.

Streaming requests share the native codec state, with SDPA attention and
compact CUDA-graph replay. Non-streaming traffic uses
the batched full-sequence decode (decode_codes_batch) without session slots.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts.attention import AUTO_ATTENTION_BACKEND
from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerVocoder,
    MossAudioTokenizerVocoderDecoder,
)
from sglang_omni.models.moss_tts.vocoder import decode_codes_batch
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_SOURCE_HINT = "MOSS-TTS Local"


class CodecStreamSession:
    """Persistent codec state session with slot bookkeeping.

    Live requests hold stream slots for their lifetime. Non-streaming work never
    enters the session; it uses the batched full-sequence decode path. All methods
    run on the scheduler-loop thread.
    """

    def __init__(
        self,
        codec: MossAudioTokenizerVocoder,
        *,
        stream_slots: int,
        n_vq: int,
    ) -> None:
        self.codec = codec
        self.stream_slots = int(stream_slots)
        self.n_vq = int(n_vq)
        self.device = next(codec.parameters()).device
        codec.initialize_decoder_state_pool(
            self.stream_slots, scratch_capacity=self.stream_slots
        )
        self.free_stream_slots = list(range(self.stream_slots))
        self.stream_slots_in_use: set[int] = set()
        self.closed = False
        self.cg_runner: Any | None = None
        # Capture is attempted at most once per session; a low-VRAM skip must not re-probe per step.
        self.warmup_attempted = False
        # Per-T graph-vs-eager step counts for capture-hit-rate reporting (host-side, no GPU sync).
        self.cg_graph_t: Counter = Counter()
        self.cg_eager_t: Counter = Counter()
        self.cg_total_steps = 0
        self.compact_batch_sizes: Counter = Counter()

    def warmup_cuda_graph(
        self, frames: list[int], *, min_free_gb: float = 3.0
    ) -> list[int]:
        """Capture native vocoder graphs once; uncaptured shapes use eager decode."""
        self.warmup_attempted = True
        if self.closed:
            return []
        else:
            pass
        from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
            MossVocoderCudaGraphRunner,
        )

        if self.cg_runner is None:
            self.cg_runner = MossVocoderCudaGraphRunner(
                self.codec,
                real_state_capacity=self.stream_slots,
                scratch_capacity=self.stream_slots,
                batch_sizes=self.graph_batch_sizes(),
                frame_sizes=frames,
                num_quantizers=self.n_vq,
                min_free_gb=min_free_gb,
            )
        else:
            pass
        try:
            self.cg_runner.warmup(frames)
        except Exception:
            self.cg_runner = None
            raise
        self.reset_slots(list(range(self.stream_slots)))
        captured = self.cg_runner.captured_frames()
        if not captured:
            self.cg_runner = None
        else:
            pass
        return captured

    def has_cuda_graph_runner(self) -> bool:
        # True only if the runner exists AND captured at least one graph.
        return bool(self.cg_runner and self.cg_runner.captured_frames())

    def captured_frames(self) -> list[int]:
        return self.cg_runner.captured_frames() if self.cg_runner else []

    def graph_batch_sizes(self) -> list[int]:
        """Return batch buckets shared by eager and CUDA graph execution."""
        buckets = [1, 2, 4, *range(8, self.stream_slots, 4), self.stream_slots]
        return sorted(
            {bucket for bucket in buckets if 0 < int(bucket) <= self.stream_slots}
        )

    def acquire(self) -> int | None:
        if not self.free_stream_slots:
            return None
        else:
            pass
        slot = self.free_stream_slots.pop()
        self.stream_slots_in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        if self.closed:
            return
        else:
            pass
        if slot not in self.stream_slots_in_use:
            raise RuntimeError(
                f"MOSS-Audio-Tokenizer vocoder stream slot {slot} is not leased"
            )
        else:
            pass
        self.reset_slots([slot])
        self.stream_slots_in_use.remove(slot)
        self.free_stream_slots.append(slot)

    def close(self) -> None:
        if self.closed:
            return
        else:
            pass
        if self.cg_runner is not None:
            self.log_cg_stats()
        else:
            pass
        if self.compact_batch_sizes:
            logger.info(
                "MOSS-Audio-Tokenizer vocoder compact streaming batches: B=%s",
                dict(sorted(self.compact_batch_sizes.items())),
            )
        else:
            pass
        try:
            self.codec.close_decoder_state_pool()
        finally:
            self.cg_runner = None
            self.closed = True

    def log_cg_stats(self) -> None:
        graph = sum(self.cg_graph_t.values())
        eager = sum(self.cg_eager_t.values())
        total = graph + eager
        if not total:
            return
        else:
            pass
        logger.info(
            "MOSS-Audio-Tokenizer vocoder CG stats: %d/%d steps graphed (%.1f%%); "
            "graph T=%s eager T=%s",
            graph,
            total,
            100.0 * graph / total,
            dict(sorted(self.cg_graph_t.items())),
            dict(sorted(self.cg_eager_t.items())),
        )

    def reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        else:
            pass
        slot_ids = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.codec.reset_decoder_state_slots(slot_ids)

    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Advance participating slots by one uniform-length step. ``slot_codes`` maps slot -> ``[n_vq, T]`` (same T); returns slot -> ``[channels, samples]`` float32 CPU audio."""
        if not slot_codes:
            return {}
        else:
            pass
        for slot, codes in slot_codes.items():
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError(f"streaming slot id must be an int, got {slot!r}")
            else:
                pass
            if slot < 0 or slot >= self.stream_slots:
                raise ValueError(
                    f"streaming slot {slot} is outside [0, {self.stream_slots})"
                )
            else:
                pass
            if int(codes.ndim) != 2:
                raise ValueError(
                    f"streaming slot {slot} codes must have shape [NQ, T], "
                    f"got {tuple(codes.shape)}"
                )
            else:
                pass
            if int(codes.shape[0]) <= 0 or int(codes.shape[1]) <= 0:
                raise ValueError(
                    f"streaming slot {slot} codes must have positive NQ and T, "
                    f"got {tuple(codes.shape)}"
                )
            else:
                pass
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                f"streaming step requires a uniform length, got {sorted(step_lengths)}"
            )
        else:
            pass
        (step_t,) = step_lengths
        n_vq = int(next(iter(slot_codes.values())).shape[0])
        if n_vq != self.n_vq:
            raise ValueError(
                f"streaming codes must use {self.n_vq} quantizers, got {n_vq}"
            )
        else:
            pass
        if any(int(codes.shape[0]) != n_vq for codes in slot_codes.values()):
            raise ValueError("all streaming slots must use the same quantizer count")
        else:
            pass
        slots = list(slot_codes)
        # Use the same batch bucket for eager and graph execution. Changing
        # GEMM shapes on a graph miss can change BF16 PCM and live KV state.
        batch_size = next(
            size for size in self.graph_batch_sizes() if size >= len(slots)
        )
        padding = batch_size - len(slots)
        rows = [
            codes.to(device=self.device, dtype=torch.long)
            for codes in slot_codes.values()
        ]
        if padding:
            rows.extend([rows[0].new_zeros(n_vq, step_t)] * padding)
        else:
            pass
        codes_step = torch.stack(rows, dim=1)
        codes_lengths = torch.tensor(
            [step_t] * len(slots) + [0] * padding,
            dtype=torch.long,
            device=self.device,
        )
        state_slot_ids = torch.as_tensor(
            slots + list(range(self.stream_slots, self.stream_slots + padding)),
            dtype=torch.long,
            device=self.device,
        )
        exec_mask = torch.tensor(
            [True] * len(slots) + [False] * padding,
            dtype=torch.bool,
            device=self.device,
        )
        self.compact_batch_sizes[batch_size] += 1
        graphed = None
        graph_failed = False
        try:
            with torch.no_grad():
                if self.cg_runner is not None:
                    try:
                        graphed = self.cg_runner.decode_step(
                            codes_step,
                            state_slot_ids,
                            exec_mask,
                        )
                    except Exception:
                        graph_failed = True
                        raise
                else:
                    pass
                if graphed is not None:
                    audio, audio_lengths = graphed
                else:
                    audio, audio_lengths = self.codec.decode_streaming_tensors(
                        codes_step,
                        codes_lengths,
                        state_slot_ids,
                        exec_mask,
                    )
            # One batched D2H per step. A graph replay error can surface async HERE (not in
            # decode_step), so materialization stays inside the replay guard.
            audio_cpu = audio.detach().to("cpu", torch.float32)
            lengths_cpu = audio_lengths.detach().to("cpu")
        except Exception:
            # Graphed step failed (in decode_step or async on the D2H): disable the runner so future
            # steps go eager; participants abort. An eager-path error does not disable it.
            if self.cg_runner is not None and (graph_failed or graphed is not None):
                logger.exception(
                    "MOSS-Audio-Tokenizer vocoder CUDA-graph replay failed "
                    "(in decode_step or on output "
                    "materialization); disabling runner, serving eager from here"
                )
                self.cg_runner = None
            else:
                pass
            raise
        if self.cg_runner is not None:
            if graphed is not None:
                self.cg_graph_t[step_t] += 1
            else:
                self.cg_eager_t[step_t] += 1
            self.cg_total_steps += 1
            if self.cg_total_steps % 2000 == 0:
                self.log_cg_stats()
            else:
                pass
        else:
            pass
        out: dict[int, torch.Tensor] = {}
        for index, slot in enumerate(slots):
            n_samples = int(lengths_cpu[index])
            out[slot] = audio_cpu[index, :, :n_samples]
        return out


@dataclass
class LocalStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    n_vq: int | None = None
    initial_chunk_frames: int = 0
    threshold: int = 0


@dataclass
class CoalescedStepPlan:
    step_t: int
    slot_codes: dict[int, torch.Tensor]


class MossTTSLocalStreamingVocoderScheduler(
    StreamingVocoderBase[LocalStreamState, CoalescedStepPlan]
):
    """Decode MOSS-TTS Local codec rows incrementally on the v2 codec."""

    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        codec: MossAudioTokenizerVocoder,
        *,
        n_vq: int,
        sample_rate: int,
        stream_slots: int = 16,
        stream_chunk_frames: int = 25,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        initial_chunk_frames: int = 5,
        coalesce_floor_frames: int = 5,
        max_step_frames: int = 100,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        vocoder_cuda_graph: bool = True,
        vocoder_cuda_graph_frames: list[int] | None = None,
        vocoder_cuda_graph_min_free_gb: float = 3.0,
    ) -> None:
        if stream_slots < 1:
            raise ValueError(f"stream_slots must be >= 1, got {stream_slots}")
        else:
            pass
        if not 0 < stream_chunk_frames <= max_step_frames:
            raise ValueError(
                "stream_chunk_frames must be in (0, max_step_frames], got "
                f"{stream_chunk_frames} (max_step_frames={max_step_frames})"
            )
        else:
            pass
        # Always build a separate execution view: from_module returns a native
        # decoder unchanged, which would expose live streaming offsets and KV
        # to offline requests. The new wrappers still share all model weights.
        nonstream_decoder = MossAudioTokenizerVocoderDecoder(
            source_decoder=codec.decoder,
            attention_backend=attention_backend,
        )
        logger.info(
            "MOSS-TTS Local non-streaming vocoder uses configured attention "
            "backend=%s stages=%d",
            attention_backend,
            len(nonstream_decoder),
        )
        self.codec = codec
        self.nonstream_decoder = nonstream_decoder
        quantizer = getattr(codec, "quantizer", None)
        if quantizer is None or not callable(getattr(quantizer, "decode_codes", None)):
            raise RuntimeError(
                "MOSS-TTS Local audio tokenizer has no quantizer.decode_codes; "
                "the batched non-streaming decode path is unavailable"
            )
        else:
            pass
        self.quantizer_decode = quantizer.decode_codes
        self.compute_dtype = getattr(codec, "compute_dtype", None)
        # note (Zhang Yiyang): matches the codec's _restore_channels_from_codec:
        # stereo v2 decoders interleave channels into the sample axis, so the
        # interleaved layout must be restored before slicing.
        number_channels = int(getattr(codec, "number_channels", 1) or 1)
        self.interleaved_channels = (
            number_channels
            if number_channels > 1
            and bool(getattr(codec, "enable_channel_interleave", False))
            else 1
        )
        self.attention_backend = attention_backend
        self.stream_slots = int(stream_slots)
        # Coalesce up to one full set of streaming lanes per pump, not the offline batch width.
        self.stream_chunk_batch_max = self.stream_slots
        self.stream_chunk_frames = int(stream_chunk_frames)
        self.default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(stream_chunk_frames))
        )
        self.coalesce_floor_frames = max(
            0, min(int(coalesce_floor_frames), int(stream_chunk_frames))
        )
        self.max_step_frames = int(max_step_frames)
        self.n_vq = int(n_vq)
        self.session: CodecStreamSession | None = None
        self.vocoder_cuda_graph = bool(vocoder_cuda_graph)
        self.vocoder_cuda_graph_frames = (
            [int(t) for t in vocoder_cuda_graph_frames]
            if vocoder_cuda_graph_frames
            else None
        )
        self.vocoder_cuda_graph_min_free_gb = float(vocoder_cuda_graph_min_free_gb)
        if self.vocoder_cuda_graph_frames is not None:
            too_large = [
                t for t in self.vocoder_cuda_graph_frames if t > self.max_step_frames
            ]
            if too_large:
                raise ValueError(
                    f"vocoder_cuda_graph_frames exceed max_step_frames={self.max_step_frames}: "
                    f"{too_large}"
                )
            else:
                pass
        else:
            pass
        super().__init__(
            self.vocode,
            batch_compute_fn=self.vocode_batch,
            sample_rate=sample_rate,
            stream_source_hint=_SOURCE_HINT,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def on_serving_stop(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        else:
            pass

    def create_stream_state(self, request_id: str) -> LocalStreamState:
        del request_id
        return LocalStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: LocalStreamState,
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
            self.latch_thresholds(request_id, state, params)
            return
        else:
            pass
        metadata: Mapping[str, Any] = source
        n_vq = metadata.get("n_vq")
        if n_vq is not None:
            n_vq = int(n_vq)
            if state.n_vq is not None and state.n_vq != n_vq:
                raise ValueError(
                    f"MOSS-TTS Local stream n_vq changed for {request_id!r}: "
                    f"{state.n_vq} -> {n_vq}"
                )
            else:
                pass
            state.n_vq = n_vq
        else:
            pass
        if state.threshold == 0:
            self.latch_thresholds(request_id, state, metadata)
        else:
            pass

    def validate_chunk(
        self, request_id: str, state: LocalStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        codes = codes.to(dtype=torch.long)
        n_vq = state.n_vq if state.n_vq is not None else self.n_vq
        if codes.ndim == 1 and int(codes.shape[0]) >= n_vq + 1:
            return codes[1 : 1 + n_vq]
        else:
            pass
        if codes.ndim == 2 and int(codes.shape[1]) >= n_vq + 1:
            return codes[:, 1 : 1 + n_vq]
        else:
            pass
        if codes.ndim not in (1, 2):
            shape_contract = "[channels] or [frames, channels]"
        else:
            shape_contract = f"at least {n_vq + 1} channels"
        raise ValueError(
            f"MOSS-TTS Local stream chunk must be {shape_contract}, "
            f"got {tuple(codes.shape)}"
        )

    def ingest(
        self, request_id: str, state: LocalStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if codes.ndim == 1:
            state.pending.append(codes)
        elif codes.ndim == 2:
            state.pending.extend(codes.unbind(0))
        else:
            raise ValueError(
                f"MOSS-TTS Local validated stream codes must be 1-D or 2-D, "
                f"got {tuple(codes.shape)}"
            )
        self.ensure_slot(state)

    def decode_delta(
        self, request_id: str, state: LocalStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain: pending frames go through the request's session
        slot (released afterwards) or the batched non-streaming path when
        slot-starved; steady-state chunks decode through the coalesced step
        hooks instead."""
        del request_id, is_final
        audio_parts: list[torch.Tensor] = []
        if state.slot is None and state.pending:
            # note (Zhang Yiyang): slot-starved — every frame is still
            # buffered; decode batched in one full-sequence call
            # (non-streaming route).
            codes = torch.stack(state.pending, dim=1).transpose(0, 1).contiguous()
            state.pending = []
            audio_parts.append(self.decode_codes_rows([codes])[0])
        elif state.slot is not None:
            session = self.ensure_session_graphed()
            while state.pending:
                step_t = min(len(state.pending), self.max_step_frames)
                codes = torch.stack(state.pending[:step_t], dim=1)
                del state.pending[:step_t]
                audio_parts.append(session.step({state.slot: codes})[state.slot])
            session.release(state.slot)
            state.slot = None
        else:
            pass
        if not audio_parts:
            return None
        else:
            pass
        return torch.cat(audio_parts, dim=-1)

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform.detach().to("cpu", torch.float32),
            sample_rate=self.sample_rate,
            modality="audio",
            source_hint=f"{_SOURCE_HINT} streaming",
            keep_channels=True,
        )

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: LocalStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        return self.decode_payload_codes(payload)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: LocalStreamState
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        usage = build_usage(MossTTSLocalState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        else:
            pass
        return final_data

    def release_stream_resources(
        self, request_id: str, state: LocalStreamState
    ) -> None:
        del request_id
        if state.slot is not None and self.session is not None:
            self.session.release(state.slot)
        else:
            pass

    def select_step_participants(self) -> list[tuple[str, LocalStreamState]]:
        """Every stream whose buffer crossed its threshold is due; due streams
        coalesce with peers above the join floor into one forward."""
        join_floor = max(
            1, min(self.coalesce_floor_frames or 5, self.stream_chunk_frames)
        )
        slotted = [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if state.slot is not None and state.threshold > 0
        ]
        due = [
            entry for entry in slotted if len(entry[1].pending) >= entry[1].threshold
        ]
        if not due:
            return []
        else:
            pass
        floor = min(
            min(len(state.pending) for _, state in due),
            join_floor,
        )
        return [
            entry
            for entry in slotted
            if self.can_join_coalesced_step(entry[0], entry[1], floor)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, LocalStreamState]]
    ) -> CoalescedStepPlan:
        """Uniform step capped at the steady chunk size and any un-emitted
        participant's first-chunk threshold; the base pump re-pumps remainder."""
        step_t = min(
            min(len(state.pending) for _, state in participants),
            self.stream_chunk_frames,
        )
        for request_id, state in participants:
            if not self.stream_has_emitted(request_id):
                step_t = min(step_t, state.threshold)
            else:
                pass
        return CoalescedStepPlan(
            step_t=step_t,
            slot_codes={
                state.slot: torch.stack(state.pending[:step_t], dim=1)
                for _, state in participants
            },
        )

    def run_step(
        self,
        participants: list[tuple[str, LocalStreamState]],
        plan: CoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        decoded = self.ensure_session().step(plan.slot_codes)
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            del state.pending[: plan.step_t]
            state.threshold = self.stream_chunk_frames
            out[request_id] = decoded[state.slot]
        return out

    def can_join_coalesced_step(
        self, request_id: str, state: LocalStreamState, floor: int
    ) -> bool:
        if len(state.pending) >= state.threshold:
            return True
        else:
            pass
        if not self.stream_has_emitted(request_id):
            return False
        else:
            pass
        return len(state.pending) >= floor

    def ensure_session(self) -> CodecStreamSession:
        if self.session is None:
            self.session = CodecStreamSession(
                self.codec,
                stream_slots=self.stream_slots,
                n_vq=self.n_vq,
            )
        else:
            pass
        return self.session

    def vocoder_cuda_graph_capture_frames(self) -> list[int]:
        """Step lengths T to capture. Config ``vocoder_cuda_graph_frames`` overrides the default."""
        if self.vocoder_cuda_graph_frames:
            # Validated at config (>= 1) and __init__ (<= max_step_frames); use as configured.
            return sorted(set(self.vocoder_cuda_graph_frames))
        else:
            pass
        # Note (Zhang Yiyang): Capture every emitted remainder length because
        # frame padding advances causal state; explicit frames may narrow it.
        max_frame = min(self.stream_chunk_frames, self.max_step_frames)
        return list(range(1, max_frame + 1))

    def codec_on_cuda(self) -> bool:
        try:
            return next(self.codec.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def ensure_session_graphed(self) -> CodecStreamSession:
        """Live session with CUDA graphs captured (at most once per session). Streaming paths
        call this instead of _ensure_session so a lazily created session (factory warmup
        skipped, e.g. non-CUDA codec) still gets its one capture attempt here, synchronously,
        fail-safe to eager on low VRAM; a low-VRAM skip is remembered (no per-step re-probe).
        """
        with self.state_lock:
            session = self.ensure_session()
            if (
                self.vocoder_cuda_graph
                and not session.warmup_attempted
                and self.codec_on_cuda()
            ):
                try:
                    session.warmup_cuda_graph(
                        self.vocoder_cuda_graph_capture_frames(),
                        min_free_gb=self.vocoder_cuda_graph_min_free_gb,
                    )
                except Exception:
                    logger.exception(
                        "MOSS-Audio-Tokenizer vocoder CUDA-graph capture failed; "
                        "serving eager from this session"
                    )
            else:
                pass
            return session

    def warmup_now(self) -> None:
        """Capture the codec-decode graphs at factory-build time: codec loaded, GPU quiescent, and
        before the stage process is marked ready, so the serving loop never races a half-captured
        graph. No-op without a CUDA codec; best-effort, degrades to eager."""
        if not self.vocoder_cuda_graph or not self.codec_on_cuda():
            return
        else:
            pass
        session = self.ensure_session_graphed()
        if session.has_cuda_graph_runner():
            logger.info(
                "MOSS-Audio-Tokenizer vocoder CUDA graphs captured at startup: T=%s",
                session.captured_frames(),
            )
        else:
            logger.warning(
                "MOSS-Audio-Tokenizer vocoder CUDA graphs did not seal at startup "
                "(low VRAM); eager vocoder"
            )

    def ensure_slot(self, state: LocalStreamState) -> None:
        if state.slot is None:
            state.slot = self.ensure_session_graphed().acquire()
        else:
            pass

    def latch_thresholds(
        self,
        request_id: str,
        state: LocalStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=self.stream_chunk_frames,
            default_frames=self.default_initial_chunk_frames,
        )
        if state.initial_chunk_frames > 0 and not self.stream_has_emitted(request_id):
            state.threshold = state.initial_chunk_frames
        else:
            state.threshold = self.stream_chunk_frames

    def decode_payload_codes(self, payload: StagePayload) -> torch.Tensor | None:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            return None
        else:
            pass
        rows = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if rows.numel() == 0:
            return None
        else:
            pass
        return self.decode_codes_rows([rows])[0]

    def prepare_codes(
        self, payload: StagePayload
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        else:
            pass
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Emit no audio: only this request fails downstream, not the batch.
            return state, None
        else:
            pass
        return state, codes

    def store_vocoder_result(
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
        state.sample_rate = self.sample_rate
        payload.data = state.to_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        return payload

    def decode_codes_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ``[T, >=n_vq]`` row tensors to fp32 CPU waveforms via the
        batched full-sequence path shared with MOSS-TTS Delay (decode_codes_batch);
        non-streaming work never enters the streaming session and touches no
        session-owned state."""
        rows = [codes[:, : self.n_vq] for codes in codes_list]
        return decode_codes_batch(
            rows,
            quantizer_decode=self.quantizer_decode,
            decoder=self.nonstream_decoder,
            device=next(self.codec.parameters()).device,
            compute_dtype=self.compute_dtype,
            max_batch_size=self.max_batch_size,
            interleaved_channels=self.interleaved_channels,
        )

    def vocode_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [self.prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(self.decode_codes_rows(codes_list)) if codes_list else iter(())
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                state.audio_codes = None
                payload.data = state.to_dict()
                results.append(payload)
                continue
            else:
                pass
            results.append(self.store_vocoder_result(payload, state, next(decoded)))
        return results

    def vocode(self, payload: StagePayload) -> StagePayload:
        return self.vocode_batch([payload])[0]


__all__ = ["MossTTSLocalStreamingVocoderScheduler"]
