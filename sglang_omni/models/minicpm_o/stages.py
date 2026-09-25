# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for MiniCPM-o text and speech pipelines."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from sglang.srt.arg_groups.model_override_base import resolved_view
from transformers import AutoTokenizer

from sglang_omni.models.minicpm_o.bootstrap import (
    create_talker_scheduler,
    create_thinker_scheduler,
)
from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
from sglang_omni.models.minicpm_o.merge import build_decode_result
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.models.minicpm_o.routing import TALKER_STAGE, code2wav_reference_audio
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.streaming_detokenizer import StreamingDetokenizeScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.misc import avail_gpu_mem

logger = logging.getLogger(__name__)


def create_preprocessing_executor(
    model_path: str,
    *,
    speech_enabled: bool = False,
) -> SimpleScheduler:
    preprocessor = MiniCPMOPreprocessor(model_path, speech_enabled=speech_enabled)

    return SimpleScheduler(preprocessor)


ENCODER_CACHE_MAX_ENTRIES = 64
ENCODER_CACHE_MAX_BYTES = 4 * 1024**3


def create_encoder_executor(encoder: nn.Module, *, stage_name: str) -> SimpleScheduler:
    cache = StageOutputCache(
        max_size=ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode_stage(payload: StagePayload) -> StagePayload:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=stage_name)
        cached = (
            None if request.skip_result is not None else cache.get(request.cache_key)
        )
        if request.skip_result is not None:
            encoder_out = request.skip_result
        elif cached is not None:
            encoder_out = cached
        else:
            with torch.no_grad():
                encoder_out = encoder(**request.model_inputs)
            cache.put(request.cache_key, encoder_out)
        state.encoder_outs[stage_name] = encoder_out
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode_stage)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler:
    encoder = MiniCPMOImageEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    return create_encoder_executor(encoder, stage_name="image_encoder")


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler:
    encoder = MiniCPMOAudioEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    return create_encoder_executor(encoder, stage_name="audio_encoder")


def create_sglang_talker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    """Returns OmniScheduler for the native sglang MiniCPM-o talker."""
    concrete_device = resolve_concrete_device(device, gpu_id)
    gpu_id = concrete_device.index or 0
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
    """Vocode talker payloads, one reference per row."""
    codec_tokens: list[list[int]] = []
    references: list[str | bytes] = []
    for payload in payloads:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        tokens = state.engine_outputs[TALKER_STAGE]["codec_tokens"].reshape(-1).tolist()
        codec_tokens.append(tokens)
        references.append(model.resolve_prompt_wav(code2wav_reference_audio(payload)))

    logger.info(
        f"minicpm_code2wav_batch size={len(payloads)} "
        f"max_codec_tokens={max(len(tokens) for tokens in codec_tokens)}"
    )
    waveforms = model.vocode(codec_tokens, references)

    outputs: list[StagePayload] = []
    for payload, waveform in zip(payloads, waveforms, strict=True):
        payload.data = dict(
            audio_waveform_payload(
                waveform,
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
    max_batch_size: int = 8,
    max_batch_wait_ms: float = 0.0,
    batch_wait_when_idle: bool = False,
    dtype: str | None = None,
    max_batch_cost: int | None = None,
    enable_flow_variable_length: bool = False,
    enable_dit_torch_compile: bool = True,
    enable_flow_cuda_graph: bool = True,
    flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] | None = None,
    packed_dit_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] | None = None,
) -> SimpleScheduler:
    model = MiniCPMOCode2Wav(
        model_path,
        device=str(resolve_concrete_device(device, gpu_id)),
        dtype=dtype,
        enable_flow_variable_length=enable_flow_variable_length,
        enable_dit_torch_compile=enable_dit_torch_compile,
        enable_flow_cuda_graph=enable_flow_cuda_graph,
        flow_cuda_graph_capture_shapes=flow_cuda_graph_capture_shapes,
        packed_dit_cuda_graph_capture_shapes=packed_dit_cuda_graph_capture_shapes,
    )

    def codec_token_cost(payload: StagePayload) -> int:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        return int(state.engine_outputs[TALKER_STAGE]["codec_tokens"].numel())

    return SimpleScheduler(
        lambda payload: vocode_code2wav_payloads(model, [payload])[0],
        batch_compute_fn=lambda payloads: vocode_code2wav_payloads(model, payloads),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=batch_wait_when_idle,
        request_cost_fn=codec_token_cost,
        max_batch_cost=max_batch_cost,
        shutdown_callback=model.close_reference_pool,
    )


def create_decode_executor(model_path: str) -> StreamingDetokenizeScheduler:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    return StreamingDetokenizeScheduler(
        tokenizer,
        eos_token_id,
        build_result=lambda payload, is_streaming: build_decode_result(
            payload,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            is_streaming=is_streaming,
        ),
    )


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
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
    concrete_device = resolve_concrete_device(device, gpu_id)
    gpu_id = concrete_device.index or 0
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
