# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS-Nano pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import resolve_moss_audio_dtype
from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat
from sglang_omni.models.moss_tts_local.audio_tokenizer import (
    load_moss_tts_local_audio_vocoder,
)
from sglang_omni.models.moss_tts_local.config import resolve_vocoder_cuda_graph
from sglang_omni.models.moss_tts_local.stages import (
    _BatchedReferenceEncoder,
    _configure_pipeline_threads,
    _MossLocalReferenceEncoder,
    _resolve_codec_device,
    _validate_loaded_process_memory_budget,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.models.moss_tts_nano.audio_tokenizer import (
    DEFAULT_MOSS_TTS_NANO_AUDIO_TOKENIZER,
    load_moss_tts_nano_audio_tokenizer,
)
from sglang_omni.models.moss_tts_nano.payload_types import MossTTSNanoState
from sglang_omni.models.moss_tts_nano.request_builders import (
    cleanup_prepared_moss_tts_nano_request,
    preprocess_moss_tts_nano_payload,
    set_moss_tts_nano_preprocessing_context,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "MOSS-TTS-Nano support requires its custom Transformers config and "
    "SentencePiece tokenizer. Launch with trust_remote_code=True and install "
    "the project dependencies."
)


def _load_moss_tts_nano_config_and_tokenizer(model_path: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoTokenizer

        with moss_transformers_processor_compat():
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return model_config, tokenizer


def _load_moss_tts_nano_config(model_path: str) -> Any:
    try:
        from transformers import AutoConfig

        with moss_transformers_processor_compat():
            return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc


def _resolve_audio_tokenizer_model_path(
    model_config: Any,
    codec_model_path: str | None,
) -> str:
    if codec_model_path is not None:
        return codec_model_path
    return str(
        getattr(
            model_config,
            "audio_tokenizer_pretrained_name_or_path",
            DEFAULT_MOSS_TTS_NANO_AUDIO_TOKENIZER,
        )
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    compute_dtype: str | torch.dtype | None = "float32",
    attention_backend: str = "auto",
    codec_model_path: str | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    worker_count = max(int(max_concurrency), 1)
    intraop_threads = _configure_pipeline_threads(worker_count)
    logger.info(
        "MOSS-TTS-Nano pipeline uses %d preprocessing workers, %d shared "
        "intra-op threads",
        worker_count,
        intraop_threads,
    )
    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )

    device = _resolve_codec_device(device, gpu_id)
    model_config, tokenizer = _load_moss_tts_nano_config_and_tokenizer(model_path)
    resolved_compute_dtype = resolve_moss_audio_dtype(
        compute_dtype,
        name="compute_dtype",
        allow_none=True,
    )
    audio_tokenizer = load_moss_tts_nano_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(model_config, codec_model_path),
        device=device,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )
    reference_encoder: Any = _BatchedReferenceEncoder(
        audio_tokenizer,
        n_vq=int(model_config.n_vq),
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    if ref_audio_cache:
        reference_encoder = _MossLocalReferenceEncoder(
            reference_encoder,
            n_vq=int(model_config.n_vq),
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_nano_preprocessing_context(
        tokenizer=tokenizer,
        model_config=model_config,
        reference_encoder=reference_encoder,
    )
    return SimpleScheduler(
        preprocess_moss_tts_nano_payload,
        abort_callback=cleanup_prepared_moss_tts_nano_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    codec_mem_reserve: float = 0.0,
) -> Any:
    from sglang_omni.models.moss_tts_nano.engine_builder import MossTtsNanoEngineBuilder

    return MossTtsNanoEngineBuilder(
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        process_total_gpu_memory_fraction=process_total_gpu_memory_fraction,
        codec_mem_reserve=codec_mem_reserve,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | torch.dtype = "float32",
    compute_dtype: str | torch.dtype | None = "float32",
    attention_backend: str = "auto",
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_slots: int = 15,
    stream_chunk_frames: int = 25,
    initial_chunk_frames: int = 5,
    coalesce_floor_frames: int = 5,
    cuda_graph: bool | None = None,
    cuda_graph_frames: list[int] | None = None,
    cuda_graph_min_free_gb: float = 3.0,
) -> MossTTSLocalStreamingVocoderScheduler:
    cuda_graph = resolve_vocoder_cuda_graph(
        cuda_graph,
        model_name="MOSS-TTS-Nano",
    )
    device = _resolve_codec_device(device, gpu_id)
    model_config = _load_moss_tts_nano_config(model_path)
    decoder_dtype = resolve_moss_audio_dtype(dtype, name="dtype", allow_none=False)
    assert decoder_dtype is not None
    resolved_compute_dtype = resolve_moss_audio_dtype(
        compute_dtype,
        name="compute_dtype",
        allow_none=True,
    )
    audio_vocoder = load_moss_tts_local_audio_vocoder(
        _resolve_audio_tokenizer_model_path(model_config, codec_model_path),
        device=device,
        decoder_dtype=decoder_dtype,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        audio_vocoder.model,
        n_vq=int(model_config.n_vq),
        sample_rate=audio_vocoder.sample_rate,
        state_cls=MossTTSNanoState,
        source_hint="MOSS-TTS-Nano",
        attention_backend=attention_backend,
        stream_slots=stream_slots,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
        coalesce_floor_frames=coalesce_floor_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cuda_graph=cuda_graph,
        cuda_graph_frames=cuda_graph_frames,
        cuda_graph_min_free_gb=cuda_graph_min_free_gb,
    )
    scheduler.warmup_now()
    device_index = torch.device(device).index
    _validate_loaded_process_memory_budget(
        stage_name="MOSS-TTS-Nano vocoder",
        gpu_id=(
            int(device_index)
            if device_index is not None
            else (0 if gpu_id is None else int(gpu_id))
        ),
        total_gpu_memory_fraction=(
            process_total_gpu_memory_fraction
            if process_total_gpu_memory_fraction is not None
            else total_gpu_memory_fraction
        ),
    )
    return scheduler


__all__ = [
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_tts_engine_executor",
    "create_vocoder_executor",
]
