# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for MiniCPM-o text and speech pipelines."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any

import torch.nn as nn

from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    StreamingDetokenizeScheduler,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

# measured on A800 SeedTTS replay, c16/n50
CODE2WAV_MAX_BATCH_SIZE = 4
CODE2WAV_MAX_BATCH_WAIT_MS = 10.0
CODE2WAV_BATCH_WAIT_WHEN_IDLE = False


def create_preprocessing_executor(
    model_path: str,
    *,
    speech_enabled: bool = False,
) -> SimpleScheduler:
    from sglang_omni.models.minicpm_o.components.preprocessor import (
        MiniCPMOPreprocessor,
    )

    preprocessor = MiniCPMOPreprocessor(model_path, speech_enabled=speech_enabled)

    return SimpleScheduler(preprocessor)


ENCODER_CACHE_MAX_ENTRIES = 64
ENCODER_CACHE_MAX_BYTES = 4 * 1024**3


def _run_single_encoder_payload(
    payload: StagePayload,
    *,
    stage_name: str,
    model: nn.Module,
    cache: StageOutputCache | None,
) -> StagePayload:
    import torch

    from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
    from sglang_omni.models.minicpm_o.request_builders import build_encoder_request

    state = MiniCPMOPipelineState.from_dict(payload.data)
    request = build_encoder_request(state, stage_name=stage_name)
    if request.skip_result is not None:
        result = request.skip_result
    else:
        result = None
        if cache is not None and request.cache_key is not None:
            result = cache.get(request.cache_key)
        if result is None:
            with torch.no_grad():
                result = model(**request.model_inputs)
            if cache is not None and request.cache_key is not None:
                cache.put(request.cache_key, result)
    state.encoder_outs[stage_name] = result
    payload.data = state.to_dict()
    return payload


def _create_encoder_executor(model: nn.Module, *, stage_name: str) -> SimpleScheduler:
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
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler:
    from sglang_omni.models.minicpm_o.components.image_encoder import (
        MiniCPMOImageEncoder,
    )
    from sglang_omni.utils.device import resolve_concrete_device

    model = MiniCPMOImageEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    return _create_encoder_executor(model, stage_name="image_encoder")


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler:
    from sglang_omni.models.minicpm_o.components.audio_encoder import (
        MiniCPMOAudioEncoder,
    )
    from sglang_omni.utils.device import resolve_concrete_device

    model = MiniCPMOAudioEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
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
) -> OmniScheduler:
    """Returns OmniScheduler for the native sglang MiniCPM-o talker."""
    from sglang.srt.arg_groups.model_override_base import resolved_view

    from sglang_omni.models.minicpm_o.bootstrap import create_talker_scheduler
    from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )
    from sglang_omni.utils.misc import avail_gpu_mem

    register_minicpm_o_hf_config()
    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        sampling_backend="pytorch",
    )
    overrides.setdefault("trust_remote_code", False)
    overrides["tp_size"] = tp_size
    # note (MayDomine): cap talker KV allocation so it does not starve the thinker.
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
        f"sglang_ar_startup stage=talker gpu_id={gpu_id} "
        f"tp_rank={tp_rank}/{tp_size} context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={resolved_view(server_args).mem_fraction_static} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
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


def vocode_code2wav_payloads(
    model: MiniCPMOCode2Wav, payloads: list[StagePayload]
) -> list[StagePayload]:
    """Vocode talker payloads, grouping rows that share a speaker reference."""
    from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
    from sglang_omni.models.minicpm_o.routing import (
        TALKER_STAGE,
        code2wav_reference_audio,
    )
    from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
    from sglang_omni.utils.audio_payload import audio_waveform_payload

    codec_tokens: list[list[int]] = []
    references: list[str | bytes] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, payload in enumerate(payloads):
        state = MiniCPMOPipelineState.from_dict(payload.data)
        tokens = state.engine_outputs[TALKER_STAGE]["codec_tokens"].reshape(-1).tolist()
        reference = model.resolve_prompt_wav(code2wav_reference_audio(payload))
        codec_tokens.append(tokens)
        references.append(reference)
        if isinstance(reference, bytes):
            group_key = f"bytes:{hash_bytes(reference)}"
        else:
            group_key = reference_path_cache_key(reference) or str(reference)
        groups[group_key].append(idx)

    logger.info(
        f"minicpm_code2wav_batch size={len(payloads)} groups={len(groups)} "
        f"max_codec_tokens={max(len(tokens) for tokens in codec_tokens)}"
    )
    waveforms_by_index = {}
    for group_indices in groups.values():
        group_waveforms = model.vocode_many(
            [codec_tokens[idx] for idx in group_indices],
            references[group_indices[0]],
        )
        for idx, waveform in zip(group_indices, group_waveforms, strict=True):
            waveforms_by_index[idx] = waveform

    outputs: list[StagePayload] = []
    for idx, payload in enumerate(payloads):
        payload.data = dict(
            audio_waveform_payload(
                waveforms_by_index[idx],
                sample_rate=model.sample_rate,
                modality="audio",
                source_hint="MiniCPM-o",
            )
        )
        outputs.append(payload)
    return outputs


def create_code2wav_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    float16: bool = False,
    max_batch_size: int = CODE2WAV_MAX_BATCH_SIZE,
    max_batch_wait_ms: float = CODE2WAV_MAX_BATCH_WAIT_MS,
    batch_wait_when_idle: bool = CODE2WAV_BATCH_WAIT_WHEN_IDLE,
    max_batch_cost: int | None = None,
) -> SimpleScheduler:
    from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
    from sglang_omni.models.minicpm_o.routing import TALKER_STAGE
    from sglang_omni.utils.device import resolve_concrete_device

    model = MiniCPMOCode2Wav(
        model_path,
        device=str(resolve_concrete_device(device, gpu_id)),
        float16=float16,
    )
    compute_batch = partial(vocode_code2wav_payloads, model)

    def codec_token_cost(payload: StagePayload) -> int:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        return int(state.engine_outputs[TALKER_STAGE]["codec_tokens"].numel())

    return SimpleScheduler(
        lambda payload: compute_batch([payload])[0],
        batch_compute_fn=compute_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=batch_wait_when_idle,
        request_cost_fn=codec_token_cost,
        max_batch_cost=max_batch_cost,
    )


def create_decode_executor(model_path: str) -> StreamingDetokenizeScheduler:
    from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
        create_streaming_detokenize_scheduler,
    )

    return create_streaming_detokenize_scheduler(model_path)


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
) -> OmniScheduler:
    """Returns OmniScheduler for the MiniCPM-o thinker."""
    from sglang.srt.arg_groups.model_override_base import resolved_view

    from sglang_omni.models.minicpm_o.bootstrap import create_thinker_scheduler
    from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )
    from sglang_omni.utils.misc import avail_gpu_mem

    register_minicpm_o_hf_config()
    overrides = build_generation_batch_overrides(
        max_running_requests=64,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        enable_mixed_chunk=True,
        chunked_prefill_size=8192,
        sampling_backend="pytorch",
    )
    overrides.setdefault("trust_remote_code", False)
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
        f"sglang_ar_startup stage=thinker gpu_id={gpu_id} "
        f"tp_rank={tp_rank}/{tp_size} context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={resolved_view(server_args).mem_fraction_static} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
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
