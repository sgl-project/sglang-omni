# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 pipeline.

    preprocessing -> speaker_encode -> tts_engine -> vocoder

Each stage is a SimpleScheduler compute-fn over a Zonos2State dict carried in
``StagePayload.data``; the terminal vocoder merges an audio payload.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.models.zonos2.components.text_frontend import (
    build_prompt_rows,
    configure_tts_norm_cache_root,
)
from sglang_omni.models.zonos2.payload_types import N_CODEBOOKS, Zonos2State
from sglang_omni.models.zonos2.request_builders import (
    build_zonos2_state,
    ref_audio_to_encoder_input,
)
from sglang_omni.models.zonos2.streaming_contract import (
    DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

# Default quality conditioning: only trailing-silence (feature 5); rest None.
_QUALITY_FEATURES = [
    "lufs",
    "estimated_snr",
    "max_pause",
    "estimated_bandlimit_hz",
    "leading_silence_s",
    "trailing_silence_s",
]
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def _default_quality_list() -> list[int | None]:
    return [_DEFAULT_QUALITY_BUCKETS.get(f) for f in _QUALITY_FEATURES]


# ---- preprocessing (text frontend) ----


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 16,
    tts_norm: bool = True,
    tts_norm_cache_dir: str | None = None,
) -> SimpleScheduler:
    configure_tts_norm_cache_root(tts_norm_cache_dir)

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_zonos2_state(payload)
        rows = build_prompt_rows(
            state.text,
            language=state.language,
            quality_buckets=_default_quality_list(),
            normalize=tts_norm,
        )
        state.input_ids = rows.to(torch.long)
        return store_state(payload, state)

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


# ---- speaker encode (Qwen3 voice embedding) ----


def create_speaker_encode_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    speaker_cache_max_items: int = 256,
    max_concurrency: int = 4,
    spk_compile: bool = False,
) -> SimpleScheduler:
    from sglang_omni.models.zonos2.components.speaker_encoder import SpeakerEncoder

    encoder = SpeakerEncoder(
        device=_device(gpu_id),
        cache_max_items=speaker_cache_max_items,
        compile_forward=spk_compile,
    )

    def _speaker(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        if state.ref_audio is not None:
            ref = ref_audio_to_encoder_input(state.ref_audio)
            state.speaker_emb, state.speaker_fingerprint = (
                encoder.encode_with_fingerprint(ref)
            )
        return store_state(payload, state)

    return SimpleScheduler(_speaker, max_concurrency=max_concurrency)


# ---- vocoder (DAC 44.1 kHz, terminal) ----


def create_vocoder_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dac_batch: bool = False,
    vocoder_warmup: bool = False,
) -> Any:
    from sglang_omni.models.zonos2.components.streaming_vocoder import (
        Zonos2StreamingVocoderScheduler,
        decode_batch,
        decode_to_pcm,
    )

    device = _device(gpu_id)

    def _result_payload(
        payload: StagePayload, state: Zonos2State, pcm: Any
    ) -> StagePayload:
        pcm_np = (
            pcm.detach().cpu().numpy()
            if isinstance(pcm, torch.Tensor)
            else np.asarray(pcm, dtype=np.float32)
        ).reshape(-1)
        # Terminal payload is msgpack'd back to the server: emit only
        # serializable values, never the upstream state tensors.
        data: dict[str, Any] = dict(
            audio_waveform_payload(pcm_np, source_hint="ZONOS2")
        )
        data["sample_rate"] = int(state.sample_rate)
        data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=data
        )

    def _coerce_codes(state: Zonos2State) -> torch.Tensor:
        codes = state.audio_codes
        if isinstance(codes, torch.Tensor):
            return codes
        if codes is None:
            return torch.empty((0, 9), dtype=torch.long)
        return torch.as_tensor(codes, dtype=torch.long)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        codes = state.audio_codes
        if codes is None or (isinstance(codes, torch.Tensor) and codes.numel() == 0):
            raise ValueError("ZONOS2 generated no audio codes")
        if not isinstance(codes, torch.Tensor):
            codes = torch.tensor(codes, dtype=torch.long)
        pcm = decode_to_pcm(codes, state.eos_frame, device=device)
        return _result_payload(payload, state, pcm)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        states = [Zonos2State.from_dict(p.data) for p in payloads]
        pcms = decode_batch(
            [_coerce_codes(s) for s in states],
            [s.eos_frame for s in states],
            device=device,
        )
        return [_result_payload(p, s, pcm) for p, s, pcm in zip(payloads, states, pcms)]

    def _request_cost(payload: StagePayload) -> int:
        codes = Zonos2State.from_dict(payload.data).audio_codes
        if codes is None:
            return 0
        try:
            return int(codes.shape[0])
        except (AttributeError, IndexError):
            return int(len(codes))

    # note (Yue Yin): batched DAC decode coalesces non-stream requests, but joint
    # right-padding lets ConvTranspose bleed across items and changes the gate
    # output vs the single decode. Keep it opt-in (default off) until GPU
    # allclose-vs-single parity is confirmed; default path stays single-decode.
    batch_enabled = dac_batch
    scheduler = Zonos2StreamingVocoderScheduler(
        device=device,
        compute_fn=_vocode,
        batch_compute_fn=_vocode_batch if batch_enabled else None,
        max_batch_size=16 if batch_enabled else 1,
        max_batch_wait_ms=10 if batch_enabled else 0,
        request_cost_fn=_request_cost if batch_enabled else None,
        max_batch_cost=32768 if batch_enabled else None,
    )
    # note (Yue Yin): opt-in warmup moves the one-time DAC load + conv autotune
    # off the first request's critical path to startup. N_CODEBOOKS+1 rows keep
    # 2 aligned frames after the de-shear so a real decode warms the kernels.
    # Never fatal -- the lazy load still happens on first decode if this fails.
    if vocoder_warmup:
        try:
            decode_to_pcm(
                torch.zeros((N_CODEBOOKS + 1, N_CODEBOOKS), dtype=torch.long),
                device=device,
            )
            logger.info("ZONOS2 DAC vocoder warmed up at startup")
        except Exception:  # noqa: BLE001 - warmup must never block server start
            logger.warning("ZONOS2 vocoder warmup failed", exc_info=True)
    return scheduler


def _device(gpu_id: int | None) -> str:
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


# ---- AR engine stage (OmniScheduler-backed ZONOS2 backbone) ----


def create_sglang_omni_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.5,
    fp8: bool = False,
    frame_graph: bool = False,
    compile_sampler: bool = False,
    async_decode: bool = False,
    stream_emit_chunk_frames: int = 1,
    stream_emit_first_chunk_frames: int = DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    server_args_overrides: dict | None = None,
) -> Any:
    from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder

    return Zonos2EngineBuilder(
        fp8=fp8,
        frame_graph=frame_graph,
        compile_sampler=compile_sampler,
        async_decode=async_decode,
        stream_emit_chunk_frames=stream_emit_chunk_frames,
        stream_emit_first_chunk_frames=stream_emit_first_chunk_frames,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        mem_fraction_static=mem_fraction_static,
    ).build(
        model_path,
        # CUDA-only model: keep the pre-existing device rather than resolving through
        # the ambient platform.
        device="cuda:0",
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
