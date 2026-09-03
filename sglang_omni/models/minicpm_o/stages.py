# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the MiniCPM-o pipeline (text path)."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple stages — return SimpleScheduler
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    speech_enabled: bool = False,
):
    from sglang_omni.models.minicpm_o.components.preprocessor import (
        MiniCPMOPreprocessor,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = MiniCPMOPreprocessor(model_path, speech_enabled=speech_enabled)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


ENCODER_CACHE_MAX_ENTRIES = 64
ENCODER_CACHE_MAX_BYTES = 4 * 1024**3

_ENCODER_MODALITY = {"image_encoder": "image", "audio_encoder": "audio"}


def _encoder_item_count(model_inputs: dict[str, Any]) -> int | None:
    """Items in one encoder payload: image slices or audio mel chunks."""
    pixel_values = model_inputs.get("pixel_values")
    if pixel_values is not None:
        try:
            return len(pixel_values)
        except TypeError:
            return None
    audio_features = model_inputs.get("audio_features")
    shape = getattr(audio_features, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return None


def _run_single_encoder_payload(
    payload: StagePayload,
    *,
    stage_name: str,
    model: Any,
    cache: Any,
) -> StagePayload:
    import torch

    from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
    from sglang_omni.models.minicpm_o.request_builders import (
        apply_encoder_result,
        build_encoder_request,
    )

    state = MiniCPMOPipelineState.from_dict(payload.data)
    request = build_encoder_request(state, stage_name=stage_name)
    if request.skip_result is not None:
        # Note (ruoyu): skip runs emit no encoder events so the profiler's
        # encoder intervals only cover real encode/cache work.
        result = request.skip_result
    else:
        modality = _ENCODER_MODALITY.get(stage_name, stage_name)
        # Note (ruoyu): batch_size counts payloads per dispatch (qwen3_omni
        # parity); num_items counts slices/chunks inside this payload.
        start_metadata: dict[str, Any] = {"modality": modality, "batch_size": 1}
        num_items = _encoder_item_count(request.model_inputs)
        if num_items is not None:
            start_metadata["num_items"] = num_items
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata=start_metadata,
        )
        cacheable = cache is not None and request.cache_key is not None
        cache_hit = False
        status = "error"
        try:
            result = None
            if cacheable:
                result = cache.get(request.cache_key)
                cache_hit = result is not None
            if result is None:
                with torch.no_grad():
                    result = model(**request.model_inputs)
                if cacheable:
                    cache.put(request.cache_key, result)
            status = "ok"
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_end",
                metadata={
                    "modality": modality,
                    "batch_size": 1,
                    "cacheable": cacheable,
                    "cache_hit": cache_hit,
                    "status": status,
                },
            )
    apply_encoder_result(state, stage_name=stage_name, result=result)
    payload.data = state.to_dict()
    return payload


def _create_encoder_executor(model: Any, *, stage_name: str):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    cache = StageOutputCache(
        max_size=ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        return _run_single_encoder_payload(
            payload, stage_name=stage_name, model=model, cache=cache
        )

    return SimpleScheduler(_encode)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
):
    from sglang_omni.models.minicpm_o.components.image_encoder import (
        MiniCPMOImageEncoder,
    )
    from sglang_omni.utils.device import resolve_device_spec

    model = MiniCPMOImageEncoder(
        model_path, device=resolve_device_spec(device), dtype=dtype
    )
    return _create_encoder_executor(model, stage_name="image_encoder")


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
):
    from sglang_omni.models.minicpm_o.components.audio_encoder import (
        MiniCPMOAudioEncoder,
    )
    from sglang_omni.utils.device import resolve_device_spec

    model = MiniCPMOAudioEncoder(
        model_path, device=resolve_device_spec(device), dtype=dtype
    )
    return _create_encoder_executor(model, stage_name="audio_encoder")


def create_sglang_talker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
):
    """Returns OmniScheduler for the native sglang MiniCPM-o talker."""
    from sglang_omni.models.minicpm_o.bootstrap import create_talker_scheduler
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args
    from sglang_omni.utils.misc import avail_gpu_mem

    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        sampling_backend="pytorch",
    )
    overrides["tp_size"] = tp_size
    # The talker shares the thinker's GPU; a fraction-based KV budget would
    # starve the thinker engine. Cap the pool at the worst case instead:
    # every running request at full context.
    overrides.setdefault("max_total_tokens", 32 * max_seq_len)
    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MiniCPM-o talker",
        server_args=server_args,
    )

    logger.info(
        f"sglang_ar_startup stage=talker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()}"
    )
    scheduler = create_talker_scheduler(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    logger.info(
        f"sglang_ar_started stage=talker gpu_id={gpu_id} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    return scheduler


def _run_code2wav_payload(payload: StagePayload, *, model: Any) -> StagePayload:
    """Vocode one utterance, framed by profiler events.

    Event names mirror qwen3_omni's code2wav events
    (``code2wav_decode_start``/``end``, ``code2wav_first_audio``) so the
    existing profiler views and analysis tooling read both models. This
    vocoder is single-shot, so ``code2wav_first_audio`` fires when the whole
    utterance is ready — that is the honest TTFA milestone for this pipeline.
    """
    from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
    from sglang_omni.models.minicpm_o.request_builders import TALKER_STAGE
    from sglang_omni.utils.audio_payload import audio_waveform_payload

    state = MiniCPMOPipelineState.from_dict(payload.data)
    talker_out = state.engine_outputs.get(TALKER_STAGE) or {}
    codec_tokens = talker_out["codec_tokens"]
    n_codec = int(codec_tokens.numel())
    _emit_event(
        request_id=payload.request_id,
        stage=None,
        event_name="code2wav_decode_start",
        metadata={"codec_tokens": n_codec},
    )
    result = None
    status = "error"
    try:
        result = model(codec_tokens=codec_tokens)
        status = "ok"
    finally:
        end_metadata: dict[str, Any] = {"codec_tokens": n_codec, "status": status}
        if result is not None:
            waveform = result["waveform"]
            sample_rate = int(result["sample_rate"])
            samples = int(waveform.shape[0])
            end_metadata["audio_samples"] = samples
            end_metadata["audio_seconds"] = samples / sample_rate
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="code2wav_decode_end",
            metadata=end_metadata,
        )
    if samples:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="code2wav_first_audio",
            metadata={"samples": samples},
        )
    # Terminal payload goes back through msgpack: keep only the audio
    # fields, no tensors from the pipeline state.
    payload.data = dict(
        audio_waveform_payload(
            waveform,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="MiniCPM-o",
        )
    )
    return payload


def create_code2wav_executor(
    model_path: str,
    *,
    device: str | None = None,
):
    from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils.device import resolve_device_spec

    model = MiniCPMOCode2Wav(model_path, device=resolve_device_spec(device))

    def _vocode(payload: StagePayload) -> StagePayload:
        return _run_code2wav_payload(payload, model=model)

    return SimpleScheduler(_vocode)


def create_decode_executor(model_path: str):
    # State keys deliberately mirror qwen3_omni, so its streaming text
    # detokenizer applies unchanged.
    from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
        create_streaming_detokenize_scheduler,
    )

    return create_streaming_detokenize_scheduler(model_path)


# ---------------------------------------------------------------------------
# AR stages — return OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    speech_enabled: bool = False,
):
    """Returns OmniScheduler for the MiniCPM-o thinker."""
    from sglang_omni.models.minicpm_o.bootstrap import create_thinker_scheduler
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args
    from sglang_omni.utils.misc import avail_gpu_mem

    overrides = build_generation_batch_overrides(
        max_running_requests=64,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        enable_mixed_chunk=True,
        chunked_prefill_size=8192,
        sampling_backend="pytorch",
    )
    overrides["tp_size"] = tp_size
    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MiniCPM-o thinker",
        server_args=server_args,
    )

    logger.info(
        f"sglang_ar_startup stage=thinker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()}"
    )
    scheduler = create_thinker_scheduler(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        speech_enabled=speech_enabled,
    )
    logger.info(
        f"sglang_ar_started stage=thinker gpu_id={gpu_id} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    return scheduler
