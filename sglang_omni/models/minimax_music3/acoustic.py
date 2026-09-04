# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed DIT/DAV and the non-streaming acoustic scheduler."""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

from .checkpoint import load_torch_state, resolve_checkpoint
from .chunking import ChunkWindow, crop_sample_bounds, overlap_mel_length
from .constants import (
    AR_CHUNK_FRAMES,
    AR_CHUNK_HOP_FRAMES,
    AR_HIDDEN_SIZE,
    DAV_SAMPLE_RATE,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
    OUTPUT_SAMPLE_RATE,
    SUPPORTED_ACOUSTIC_DTYPES,
)
from .dav import MiniMaxMusic3DAV, remove_weight_norm, select_decoder_state
from .dit import MiniMaxMusic3DIT
from .payload_types import MiniMaxMusic3State

logger = logging.getLogger(__name__)


def _derive_seed(seed: int, *parts: object) -> int:
    """Stable 64-bit seed derivation; never use Python's randomized hash()."""
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        raw = str(part).encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def resample_waveform(waveform: Tensor) -> Tensor:
    import torchaudio.functional as AF

    return AF.resample(waveform.float(), DAV_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"MiniMax Music 3 {name} must be a positive integer")
    return value


def _non_negative_number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"MiniMax Music 3 {name} must be finite and non-negative")
    return float(value)


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"MiniMax Music 3 {name} must be a boolean")
    return value


def _compile_timed(label: str, enable: Callable[[], None]) -> None:
    started = time.perf_counter()
    enable()
    logger.info(
        f"MiniMax Music 3 {label} compiled in {time.perf_counter() - started:.1f}s"
    )


def _resolve_acoustic_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        name = str(value).removeprefix("torch.")
    elif isinstance(value, str):
        name = value.removeprefix("torch.").lower()
    else:
        raise ValueError(
            "MiniMax Music 3 acoustic dtype must be 'float32' or 'bfloat16'"
        )
    if name not in SUPPORTED_ACOUSTIC_DTYPES:
        raise ValueError(
            "MiniMax Music 3 acoustic dtype must be one of: "
            + ", ".join(sorted(SUPPORTED_ACOUSTIC_DTYPES))
        )
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


class MiniMaxMusic3AcousticDecoder:
    """Flow-matching DIT + DAC decoder."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:1",
        dtype: str | torch.dtype = "float32",
        dit_steps: int = DEFAULT_DIT_STEPS,
        dit_cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
        attention_backend: str = "torch_sdpa",
        cache_dit: bool = False,
        compile_acoustic: bool = True,
        breakable_cuda_graph: bool = False,
        breakable_cuda_graph_min_free_gb: float | None = None,
        cache_dit_fn_compute_blocks: int = 2,
        cache_dit_bn_compute_blocks: int = 2,
        cache_dit_max_warmup_steps: int = 4,
        cache_dit_residual_diff_threshold: float = 0.08,
        cache_dit_max_continuous_cached_steps: int = 1,
    ) -> None:
        if not (current_platform.is_cuda() or current_platform.is_musa()):
            raise RuntimeError(
                "MiniMax Music 3 acoustic inference requires CUDA/MUSA backend"
            )
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "musa"):
            raise RuntimeError(
                "MiniMax Music 3 acoustic inference requires a CUDA/MUSA device"
            )
        self.dtype = _resolve_acoustic_dtype(dtype)
        if self.dtype is torch.float32:
            # note (chenyang): TF32 keeps float32's range with a 10-bit mantissa
            # and measures 0.22% from the true float32 DIT solve while running
            # 4.4x faster, which is what makes float32 affordable enough to be
            # the only default. Scoped to this dtype because only float32 matmuls
            # are affected, so the bfloat16 AR backbone is untouched even when it
            # shares this process.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self.dit_steps = _positive_int("dit_steps", dit_steps)
        self.dit_cfg_scale = _non_negative_number("dit_cfg_scale", dit_cfg_scale)
        self.attention_backend = attention_backend.strip().lower()
        self.cache_dit = _boolean("cache_dit", cache_dit)
        self.compile_acoustic = _boolean("compile_acoustic", compile_acoustic)
        self.breakable_cuda_graph_requested = _boolean(
            "breakable_cuda_graph", breakable_cuda_graph
        )
        self.breakable_cuda_graph = False
        if self.cache_dit and self.breakable_cuda_graph_requested:
            raise ValueError(
                "MiniMax Music 3 cache_dit and breakable_cuda_graph cannot be enabled together"
            )
        if breakable_cuda_graph_min_free_gb is None:
            breakable_cuda_graph_min_free_gb = (
                24.0 if self.dtype is torch.float32 else 10.0
            )
        self.breakable_cuda_graph_min_free_gb = _non_negative_number(
            "breakable_cuda_graph_min_free_gb", breakable_cuda_graph_min_free_gb
        )
        if self.dtype is torch.float32 and self.attention_backend in {
            "fa",
            "sage_attn",
        }:
            raise ValueError(
                "MiniMax Music 3 fa and sage_attn backends require dtype=bfloat16"
            )

        paths = resolve_checkpoint(model_path)
        load_started = time.perf_counter()
        self._build_dit(
            paths.dit_path,
            cache_dit_fn_compute_blocks=cache_dit_fn_compute_blocks,
            cache_dit_bn_compute_blocks=cache_dit_bn_compute_blocks,
            cache_dit_max_warmup_steps=cache_dit_max_warmup_steps,
            cache_dit_residual_diff_threshold=cache_dit_residual_diff_threshold,
            cache_dit_max_continuous_cached_steps=cache_dit_max_continuous_cached_steps,
        )
        removed_weight_norms = self._build_dav(paths.dav_path)
        logger.info(
            f"MiniMax Music 3 PyTorch DIT/DAV loaded device={self.device} dtype={self.dtype} dit_parameters={sum((parameter.numel() for parameter in self.dit.parameters()))} dav_parameters={sum((parameter.numel() for parameter in self.dav.parameters()))} folded_weight_norms={removed_weight_norms} elapsed={time.perf_counter() - load_started:.1f}s"
        )
        logger.info(
            f"MiniMax Music 3 acoustic runtime dit_steps={self.dit_steps} dit_cfg_scale={self.dit_cfg_scale:.3f} attention_backend={self.attention_backend} cache_dit={self.cache_dit} compile_acoustic={self.compile_acoustic} breakable_cuda_graph={self.breakable_cuda_graph} breakable_cuda_graph_requested={self.breakable_cuda_graph_requested}"
        )

    def _build_dit(
        self,
        dit_path: str,
        *,
        cache_dit_fn_compute_blocks: int,
        cache_dit_bn_compute_blocks: int,
        cache_dit_max_warmup_steps: int,
        cache_dit_residual_diff_threshold: float,
        cache_dit_max_continuous_cached_steps: int,
    ) -> None:
        logger.info(
            f"Loading MiniMax Music 3 PyTorch DIT from {dit_path} on device={self.device} dtype={self.dtype}"
        )
        state = load_torch_state(dit_path, device=self.device)
        logger.info(
            f"MiniMax Music 3 DIT checkpoint variant=FM8/ELMo condition_hidden=32768 state_keys={len(state)}"
        )
        with torch.device("meta"):
            self.dit = MiniMaxMusic3DIT(
                compute_dtype=self.dtype,
                attention_backend=self.attention_backend,
            )
        self.dit.load_state_dict(state, strict=True, assign=True)
        del state
        self.dit = self.dit.to(dtype=self.dtype).eval()
        window = self.dit.aligned_mel_length(AR_CHUNK_FRAMES)
        if self.compile_acoustic and not (
            self.cache_dit or self.breakable_cuda_graph_requested
        ):
            _compile_timed(
                "DIT blocks",
                lambda: self.dit.enable_compiled_blocks(warmup_mel_length=window),
            )
        if self.breakable_cuda_graph_requested:
            self.breakable_cuda_graph = self.dit.enable_breakable_cuda_graph(
                mel_len=window,
                min_free_gb=self.breakable_cuda_graph_min_free_gb,
            )
        if self.cache_dit:
            self.dit.enable_cache_dit(
                num_steps=self.dit_steps,
                fn_compute_blocks=cache_dit_fn_compute_blocks,
                bn_compute_blocks=cache_dit_bn_compute_blocks,
                max_warmup_steps=cache_dit_max_warmup_steps,
                residual_diff_threshold=cache_dit_residual_diff_threshold,
                max_continuous_cached_steps=cache_dit_max_continuous_cached_steps,
            )

    def _build_dav(self, dav_path: str) -> int:
        """Load the DAV decoder and return how many weight norms were folded."""
        logger.info(f"Loading MiniMax Music 3 DAV from {dav_path}")
        state = load_torch_state(dav_path, device=self.device)
        with torch.device("meta"):
            self.dav = MiniMaxMusic3DAV()
        self.dav.load_state_dict(select_decoder_state(state), strict=True, assign=True)
        del state
        self.dav = self.dav.to(dtype=self.dtype).eval()
        removed_weight_norms = remove_weight_norm(self.dav)
        if self.compile_acoustic:
            window = self.dit.aligned_mel_length(AR_CHUNK_FRAMES)
            _compile_timed(
                "DAV decoder",
                lambda: self.dav.enable_compiled_decoder(warmup_mel_length=window),
            )
        return removed_weight_norms

    @torch.inference_mode()
    def decode_with_state(
        self,
        hidden: Tensor,
        *,
        seed: int,
        chunk_idx: int,
        initial_latent: Tensor | None = None,
        initial_condition: Tensor | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if should_abort is not None and should_abort():
            raise InterruptedError("MiniMax Music 3 acoustic generation aborted")
        hidden = hidden.unsqueeze(0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        generator = torch.Generator(device=self.device).manual_seed(
            _derive_seed(int(seed), "dit", int(chunk_idx))
        )
        align = self.dit.aligned_condition(hidden)
        latent = self.dit(
            align,
            generator=generator,
            initial_latent=initial_latent,
            initial_condition=initial_condition,
            num_steps=self.dit_steps,
            cfg_scale=self.dit_cfg_scale,
            should_abort=should_abort,
        )
        if should_abort is not None and should_abort():
            raise InterruptedError("MiniMax Music 3 acoustic generation aborted")
        waveform = self.dav(latent)
        if should_abort is not None and should_abort():
            raise InterruptedError("MiniMax Music 3 acoustic generation aborted")
        overlap = overlap_mel_length()
        start = max(0, latent.shape[-1] - 2 * overlap)
        end = max(start, latent.shape[-1] - overlap)
        latent_save = latent[..., start:end].detach()
        align_save = align[..., start:end].detach()
        return (
            waveform[0].float().cpu().clamp(-1.0, 1.0),
            latent_save,
            align_save,
        )


@dataclass
class _AcousticStreamState:
    final_state: MiniMaxMusic3State | None = None
    wave_chunks: list[Tensor] = field(default_factory=list)
    hidden_frames: int = 0
    last_latent: Tensor | None = None
    last_condition: Tensor | None = None
    started_at: float = field(default_factory=time.perf_counter)
    abort_event: threading.Event = field(default_factory=threading.Event)
    next_chunk_idx: int = 0
    next_start_frame: int = 0


class MiniMaxMusic3AcousticScheduler(StreamingSimpleScheduler):
    """Consumes internal hidden chunks while HTTP remains non-streaming."""

    def __init__(
        self,
        decoder: MiniMaxMusic3AcousticDecoder,
    ) -> None:
        self._decoder = decoder
        self._stream_states: dict[str, _AcousticStreamState] = {}
        super().__init__(compute_fn=None, max_batch_size=1)

    def is_streaming_payload(self, payload: Any) -> bool:
        if not isinstance(payload, StagePayload):
            return False
        data = payload.data if isinstance(payload.data, dict) else {}
        return bool(data.get("internal_chunk_stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._stream_states.setdefault(request_id, _AcousticStreamState())
        state.final_state = MiniMaxMusic3State.from_dict(payload.data)
        logger.info(
            f"MiniMax Music 3 acoustic request={request_id} payload_ready seed={state.final_state.seed} expected_frames={state.final_state.generated_frames}"
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if not isinstance(item.data, Tensor):
            raise ValueError(
                "MiniMax Music 3 acoustic chunk data must be a hidden tensor"
            )
        if (
            item.data.ndim != 3
            or item.data.shape[0] != 1
            or item.data.shape[2] != AR_HIDDEN_SIZE
        ):
            raise ValueError(
                f"MiniMax Music 3 acoustic hidden must be [1,T,{AR_HIDDEN_SIZE}], "
                f"got {tuple(item.data.shape)}"
            )
        hidden = item.data[0]
        state = self._stream_states.setdefault(request_id, _AcousticStreamState())
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise ValueError(
                "MiniMax Music 3 acoustic chunk metadata must be a mapping"
            )
        chunk_idx = int(metadata["chunk_idx"])
        start_frame = int(metadata["start_frame"])
        end_frame = int(metadata["end_frame"])
        seed = int(metadata["seed"])
        is_last = bool(metadata["is_final"])
        if chunk_idx != state.next_chunk_idx or start_frame != state.next_start_frame:
            raise ValueError(
                "MiniMax Music 3 acoustic chunks must be ordered: "
                f"expected idx={state.next_chunk_idx} start={state.next_start_frame}, "
                f"got idx={chunk_idx} start={start_frame}"
            )
        if end_frame <= start_frame or end_frame - start_frame != hidden.shape[0]:
            raise ValueError(
                "MiniMax Music 3 acoustic chunk frame range does not match hidden length"
            )
        logger.info(
            f"MiniMax Music 3 acoustic chunk request={request_id} idx={chunk_idx} frames={hidden.shape[0]} frame_range=[{start_frame},{end_frame}) final={is_last}"
        )
        try:
            wave, state.last_latent, state.last_condition = (
                self._decoder.decode_with_state(
                    hidden,
                    seed=seed,
                    chunk_idx=chunk_idx,
                    initial_latent=state.last_latent,
                    initial_condition=state.last_condition,
                    should_abort=state.abort_event.is_set,
                )
            )
        except InterruptedError:
            if state.abort_event.is_set():
                return []
            raise
        left_trim, right_trim = crop_sample_bounds(
            ChunkWindow(chunk_idx, 0, int(hidden.shape[0]), chunk_idx == 0, is_last),
        )
        end = wave.shape[-1] - right_trim if right_trim else wave.shape[-1]
        state.wave_chunks.append(wave[:, left_trim:end])
        state.hidden_frames += int(hidden.shape[0])
        state.next_chunk_idx += 1
        state.next_start_frame += AR_CHUNK_HOP_FRAMES
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self._stream_states.get(request_id)
        if state is None or state.final_state is None:
            raise ValueError("MiniMax Music 3 stream completed without final payload")
        if not state.wave_chunks:
            raise ValueError(
                "MiniMax Music 3 stream completed without decoded audio chunks"
            )
        waveform_44k = torch.cat(state.wave_chunks, dim=1)
        waveform_32k = resample_waveform(waveform_44k)
        final_state = state.final_state
        payload_data = audio_waveform_payload(
            waveform_32k,
            sample_rate=OUTPUT_SAMPLE_RATE,
            modality="audio",
            source_hint="MiniMax Music 3",
            keep_channels=True,
        )
        usage = build_usage(final_state)
        if usage is not None:
            payload_data["usage"] = usage
        payload_data["finish_reason"] = final_state.finish_reason or "stop"
        logger.info(
            f"MiniMax Music 3 acoustic done request={request_id} hidden_frames={state.hidden_frames} samples={waveform_44k.shape[-1]}->{waveform_32k.shape[-1]} duration={waveform_32k.shape[-1] / OUTPUT_SAMPLE_RATE:.2f}s sample_rate={OUTPUT_SAMPLE_RATE} finish_reason={payload_data['finish_reason']} elapsed={time.perf_counter() - state.started_at:.1f}s"
        )
        return [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=self._stream_payloads[request_id].request_id,
                    request=self._stream_payloads[request_id].request,
                    data=payload_data,
                ),
            )
        ]

    def abort(self, request_id: str) -> None:
        state = self._stream_states.get(request_id)
        if state is not None:
            state.abort_event.set()
        super().abort(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        state = self._stream_states.pop(request_id, None)
        if state is not None:
            state.abort_event.set()
            state.wave_chunks.clear()
            state.final_state = None
            state.last_latent = None
            state.last_condition = None


__all__ = [
    "MiniMaxMusic3AcousticDecoder",
    "MiniMaxMusic3AcousticScheduler",
    "resample_waveform",
    "_resolve_acoustic_dtype",
]
