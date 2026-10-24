# SPDX-License-Identifier: Apache-2.0
"""Build the FireRedTTS3 pipeline stage executors."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from sglang_omni.models.firered_tts3.config import (
    DEFAULT_MAX_RUNNING_REQUESTS,
    load_checkpoint_hf_config,
)
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.flow_graph import FlowMode
from sglang_omni.models.firered_tts3.hf_config import register_fireredtts3_hf_config
from sglang_omni.models.firered_tts3.preprocessing import (
    preprocess_firered_tts3_payload,
)
from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceBundle,
    FireRedReferenceEncoder,
    prompt_stream_message,
)
from sglang_omni.models.firered_tts3.request_builders import (
    FireRedTTS3SGLangRequestData,
)
from sglang_omni.models.firered_tts3.vocoder import FireRedTTS3StreamingVocoder
from sglang_omni.models.firered_tts3.weights import load_redae_decoder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.json import JsonValue


def create_preprocessing_executor(
    model_path: str, *, n_timesteps: int, max_patches: int, max_concurrency: int
) -> SimpleScheduler[StagePayload, StagePayload]:
    register_fireredtts3_hf_config()
    checkpoint_directory = str(resolve_checkpoint(model_path))
    checkpoint_config = load_checkpoint_hf_config(checkpoint_directory)
    if checkpoint_config is None:
        raise FileNotFoundError(
            f"FireRedTTS3 config.json not found at {checkpoint_directory}"
        )
    else:
        pass
    text_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_directory, local_files_only=True
    )

    def preprocess(payload: StagePayload) -> StagePayload:
        return preprocess_firered_tts3_payload(
            payload,
            tokenizer=text_tokenizer,
            config=checkpoint_config,
            n_timesteps=n_timesteps,
            max_patches=max_patches,
        )

    return SimpleScheduler(preprocess, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int,
    max_batch_size: int,
    max_batch_wait_ms: float,
    warmup_on_init: bool,
    reference_cache: bool,
) -> SimpleScheduler[StagePayload, StagePayload]:
    resolved_device = resolve_concrete_device(device, gpu_id)
    if resolved_device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("FireRedTTS3 reference encode requires CUDA")
    else:
        pass
    checkpoint_directory = str(resolve_checkpoint(model_path))
    checkpoint_config = load_checkpoint_hf_config(checkpoint_directory)
    if checkpoint_config is None:
        raise FileNotFoundError(
            f"FireRedTTS3 config.json not found at {checkpoint_directory}"
        )
    else:
        pass
    reference_bundle = FireRedReferenceBundle(
        checkpoint_directory, checkpoint_config, str(resolved_device)
    )
    if warmup_on_init:
        reference_bundle.warmup()
    else:
        pass
    reference_encoder = FireRedReferenceEncoder(
        reference_bundle,
        model_id=checkpoint_directory,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cache_enabled=reference_cache,
    )

    def encode_and_stream_prompt(payload: StagePayload) -> StagePayload:
        result = reference_encoder.encode_payload(payload)
        chunk = prompt_stream_message(result)
        if chunk is not None:
            reference_scheduler.outbox.put(chunk)
        else:
            pass
        return result

    reference_scheduler = SimpleScheduler(
        encode_and_stream_prompt,
        max_concurrency=max_concurrency,
        shutdown_callback=reference_encoder.close,
    )
    return reference_scheduler


def create_talker_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    n_timesteps: int,
    max_patches: int,
    flow_cuda_graph: bool,
    flow_cuda_graph_capture_sizes: list[int] | None,
    flow_cuda_graph_max_graphs: int,
    flow_batching: bool,
    control_token_fastpath: bool,
    skip_min_patch_stop: bool,
    flow_mode: FlowMode,
    warmup_on_init: bool,
    enable_async_decode: bool,
    async_decode_min_batch_size: int,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
) -> OmniScheduler[FireRedTTS3SGLangRequestData]:
    if not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 talker requires CUDA")
    else:
        pass
    return FireRedTTS3EngineBuilder(
        n_timesteps=n_timesteps,
        max_patches=max_patches,
        max_running_requests=DEFAULT_MAX_RUNNING_REQUESTS,
        flow_cuda_graph=flow_cuda_graph,
        flow_cuda_graph_capture_sizes=flow_cuda_graph_capture_sizes,
        flow_cuda_graph_max_graphs=flow_cuda_graph_max_graphs,
        flow_batching=flow_batching,
        control_token_fastpath=control_token_fastpath,
        skip_min_patch_stop=skip_min_patch_stop,
        flow_mode=flow_mode,
        warmup_on_init=warmup_on_init,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int,
    max_batch_wait_ms: float,
    stream_slots: int,
    prompt_predecode: bool,
    stream_batching: bool,
    separate_decode_stream: bool,
    incremental_cuda_graph: bool,
    cuda_graph_latent_sizes: list[int] | None,
    cuda_graph_batch_sizes: list[int] | None,
    cold_cuda_graph: bool,
    cold_cuda_graph_latent_sizes: list[int] | None,
    cold_cuda_graph_batch_sizes: list[int] | None,
    warmup_on_init: bool,
) -> FireRedTTS3StreamingVocoder:
    if not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 vocoder requires CUDA")
    else:
        pass
    resolved_device = resolve_concrete_device(device, gpu_id)
    checkpoint_directory = str(resolve_checkpoint(model_path))
    checkpoint_config = load_checkpoint_hf_config(checkpoint_directory)
    if checkpoint_config is None:
        raise FileNotFoundError(
            f"FireRedTTS3 config.json not found at {checkpoint_directory}"
        )
    else:
        pass
    audio_decoder = load_redae_decoder(
        checkpoint_directory, checkpoint_config, device=resolved_device
    )
    vocoder_scheduler = FireRedTTS3StreamingVocoder(
        audio_decoder,
        redae_scale=float(checkpoint_config.redae_scale or 1.0),
        sample_rate=int(checkpoint_config.audio_sample_rate),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_slots=stream_slots,
        prompt_predecode=prompt_predecode,
        stream_batching=stream_batching,
        separate_decode_stream=separate_decode_stream,
        incremental_cuda_graph=incremental_cuda_graph,
        cuda_graph_latent_sizes=cuda_graph_latent_sizes,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        cold_cuda_graph=cold_cuda_graph,
        cold_cuda_graph_latent_sizes=cold_cuda_graph_latent_sizes,
        cold_cuda_graph_batch_sizes=cold_cuda_graph_batch_sizes,
    )
    if warmup_on_init:
        # note (kermit): Capture must finish before colocated stages begin serving requests.
        vocoder_scheduler.warmup_now()
    else:
        pass
    return vocoder_scheduler


__all__ = [
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_talker_executor",
    "create_vocoder_executor",
]
