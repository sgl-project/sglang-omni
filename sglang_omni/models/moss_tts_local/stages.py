# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Local (v1.5) pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
    load_moss_audio_encoder,
    load_moss_audio_vocoder,
    resolve_moss_audio_dtype,
)
from sglang_omni.models.moss_tts.hf_loading import (
    MossLoadedProcessor,
    MossLocalReferences,
    MossProcessorConfigSource,
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts.reference_encoder import MossReferenceEncoder
from sglang_omni.models.moss_tts_local.config import resolve_vocoder_cuda_graph
from sglang_omni.models.moss_tts_local.payload_types import (
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
    cleanup_prepared_moss_tts_local_request,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.cpu import bounded_intraop_threads

if TYPE_CHECKING:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)

_MOSS_TTS_LOCAL_INSTALL_HINT = (
    "MOSS-TTS Local support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2."
)
_MAX_PIPELINE_INTRAOP_THREADS = 8

# NOTE: preprocessing and vocoder stages each load their own codec instance:
# `model.streaming()` flips codec state, so a decode on a shared instance would
# corrupt a concurrent reference encode (see streaming_vocoder.py).


@dataclass(frozen=True)
class ArMemoryBudget:
    effective_total_gpu_memory_fraction: float | None
    applied_codec_mem_reserve: float


def configure_pipeline_threads(worker_count: int) -> int:
    """Bound the shared Torch CPU pool used by the colocated pipeline process."""
    override = os.environ.get("OMP_NUM_THREADS", "").strip()
    if override.isdigit() and int(override) >= 1:
        requested = int(override)
        torch.set_num_threads(requested)
        return requested
    else:
        pass

    intraop_threads = bounded_intraop_threads(
        worker_count=max(int(worker_count), 1),
        max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
    )
    torch.set_num_threads(intraop_threads)
    return intraop_threads


def apply_colocated_ar_memory_budget(
    overrides: dict[str, object],
    *,
    total_gpu_memory_fraction: float | None,
    codec_mem_reserve: float,
) -> ArMemoryBudget:
    if total_gpu_memory_fraction is None:
        return ArMemoryBudget(
            effective_total_gpu_memory_fraction=None,
            applied_codec_mem_reserve=0.0,
        )
    else:
        pass
    if not 0.0 <= codec_mem_reserve < 1.0:
        raise ValueError("codec_mem_reserve must be in [0, 1)")
    else:
        pass

    effective_total_gpu_memory_fraction = round(
        total_gpu_memory_fraction - codec_mem_reserve,
        3,
    )
    if effective_total_gpu_memory_fraction < 0.1:
        raise ValueError(
            f"colocated total_gpu_memory_fraction {total_gpu_memory_fraction:.3f} "
            f"minus codec_mem_reserve {codec_mem_reserve:.3f} = "
            f"{effective_total_gpu_memory_fraction:.3f} is below the safe floor "
            f"0.1; lower codec_mem_reserve or increase the tts_engine stage budget."
        )
    else:
        pass

    explicit_mem_fraction = overrides.get("mem_fraction_static")
    applied_codec_mem_reserve = codec_mem_reserve
    if explicit_mem_fraction is not None:
        # Range is the schema's rule (engine.mem_fraction_static in (0, 1));
        # only the model's own budget relation is checked here.
        explicit_mem_fraction = float(explicit_mem_fraction)
        if explicit_mem_fraction > total_gpu_memory_fraction:
            raise ValueError(
                f"MOSS-TTS Local tts_engine mem_fraction_static cannot exceed "
                f"runtime.resources.total_gpu_memory_fraction: "
                f"{explicit_mem_fraction:.3f} > {total_gpu_memory_fraction:.3f}"
            )
        else:
            pass
        effective_total_gpu_memory_fraction = explicit_mem_fraction
        applied_codec_mem_reserve = round(
            total_gpu_memory_fraction - effective_total_gpu_memory_fraction,
            3,
        )
    else:
        overrides["mem_fraction_static"] = effective_total_gpu_memory_fraction

    return ArMemoryBudget(
        effective_total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
        applied_codec_mem_reserve=applied_codec_mem_reserve,
    )


def validate_loaded_process_memory_budget(
    *,
    stage_name: str,
    gpu_id: int,
    total_gpu_memory_fraction: float | None,
) -> None:
    if total_gpu_memory_fraction is None:
        return
    else:
        pass

    from sglang_omni.utils.gpu_memory import (
        format_bytes_gib,
        get_gpu_device_info,
        get_process_gpu_memory_bytes,
    )

    process_bytes = get_process_gpu_memory_bytes(gpu_id)
    total_bytes = get_gpu_device_info(gpu_id).total_memory_bytes
    if process_bytes is None or total_bytes is None:
        logger.warning(
            f"{stage_name} GPU memory budget cannot be verified: "
            f"gpu_id={gpu_id} fraction={total_gpu_memory_fraction:.3f}"
        )
        return
    else:
        pass

    budget_bytes = int(total_bytes * total_gpu_memory_fraction)
    if process_bytes > budget_bytes:
        raise RuntimeError(
            f"{stage_name} process GPU memory exceeds its configured budget: "
            f"used={format_bytes_gib(process_bytes)}, "
            f"budget={format_bytes_gib(budget_bytes)}, "
            f"fraction={total_gpu_memory_fraction:.3f}"
        )
    else:
        pass
    logger.info(
        f"{stage_name} process GPU memory: "
        f"used={format_bytes_gib(process_bytes)} "
        f"budget={format_bytes_gib(budget_bytes)} "
        f"fraction={total_gpu_memory_fraction:.3f}"
    )


def normalize_processor_config(processor: MossProcessorConfigSource) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    else:
        pass
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_local_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)
        else:
            pass


def load_moss_tts_local_processor(
    model_path: str,
) -> MossLoadedProcessor[MossLocalReferences]:
    logger.info(f"Loading MOSS-TTS Local processor from {model_path} without codec")
    try:
        from transformers import AutoConfig, AutoTokenizer

        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(model_path)
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = processor_cls(
                tokenizer=tokenizer,
                audio_tokenizer=None,
                model_config=model_config,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_LOCAL_INSTALL_HINT) from exc

    normalize_processor_config(processor)
    return processor


def resolve_audio_tokenizer_model_path(
    processor: MossProcessorConfigSource,
    codec_model_path: str | None,
) -> str:
    if codec_model_path is not None:
        return codec_model_path
    else:
        pass
    return str(
        getattr(
            processor.model_config,
            "audio_tokenizer_name_or_path",
            DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
        )
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    compute_dtype: str | torch.dtype | None = "bfloat16",
    attention_backend: str = "auto",
    codec_model_path: str | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler[StagePayload, StagePayload]:
    worker_count = max(int(max_concurrency), 1)
    intraop_threads = configure_pipeline_threads(worker_count)
    logger.info(
        f"MOSS-TTS Local pipeline uses {worker_count} preprocessing workers, "
        f"{intraop_threads} shared intra-op threads"
    )
    # MOSS_REF_AUDIO_CACHE=0 disables the cache at startup (ops kill switch / A-B
    # toggle) without a config edit; unset => kwarg/config default.
    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    else:
        pass
    from sglang_omni.utils.device import resolve_concrete_device

    device = str(resolve_concrete_device(device, gpu_id))
    processor = load_moss_tts_local_processor(model_path)
    resolved_compute_dtype = resolve_moss_audio_dtype(
        compute_dtype,
        name="compute_dtype",
        allow_none=True,
    )
    resolved_codec_model_path = resolve_audio_tokenizer_model_path(
        processor, codec_model_path
    )
    audio_tokenizer = load_moss_audio_encoder(
        resolved_codec_model_path,
        device=device,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )
    reference_encoder = MossReferenceEncoder(
        audio_tokenizer,
        codec_model_path=resolved_codec_model_path,
        n_vq=int(processor.model_config.n_vq),
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
        cache_enabled=ref_audio_cache,
        max_items=ref_audio_cache_max_items,
        max_bytes=ref_audio_cache_max_bytes,
    )
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=reference_encoder
    )
    return SimpleScheduler(
        preprocess_moss_tts_local_payload,
        abort_callback=cleanup_prepared_moss_tts_local_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: Mapping[str, object] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 1,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    codec_mem_reserve: float = 0.0,
) -> OmniScheduler[MossTTSLocalSGLangRequestData]:
    from sglang_omni.models.moss_tts_local.engine_builder import (
        MossTtsLocalEngineBuilder,
    )

    return MossTtsLocalEngineBuilder(
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
    compute_dtype: str | torch.dtype | None = "bfloat16",
    attention_backend: str = "auto",
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_slots: int = 16,
    stream_chunk_frames: int = 25,
    initial_chunk_frames: int = 5,
    coalesce_floor_frames: int = 5,
    vocoder_cuda_graph: bool | None = None,
    vocoder_cuda_graph_frames: list[int] | None = None,
    vocoder_cuda_graph_min_free_gb: float = 3.0,
) -> MossTTSLocalStreamingVocoderScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    vocoder_cuda_graph = resolve_vocoder_cuda_graph(vocoder_cuda_graph)
    device = str(resolve_concrete_device(device, gpu_id))
    processor = load_moss_tts_local_processor(model_path)
    decoder_dtype = resolve_moss_audio_dtype(
        dtype,
        name="dtype",
        allow_none=False,
    )
    assert decoder_dtype is not None
    resolved_compute_dtype = resolve_moss_audio_dtype(
        compute_dtype,
        name="compute_dtype",
        allow_none=True,
    )
    resolved_codec_path = resolve_audio_tokenizer_model_path(
        processor, codec_model_path
    )
    audio_vocoder = load_moss_audio_vocoder(
        resolved_codec_path,
        device=device,
        decoder_dtype=decoder_dtype,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        audio_vocoder.model,
        n_vq=int(processor.model_config.n_vq),
        sample_rate=audio_vocoder.sample_rate,
        attention_backend=attention_backend,
        stream_slots=stream_slots,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
        coalesce_floor_frames=coalesce_floor_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        vocoder_cuda_graph=vocoder_cuda_graph,
        vocoder_cuda_graph_frames=vocoder_cuda_graph_frames,
        vocoder_cuda_graph_min_free_gb=vocoder_cuda_graph_min_free_gb,
    )
    # Capture graphs in the factory: it runs before the process is marked ready, so serving never
    # races a half-captured graph. Same-process guarantee (each colocate/split stage warms its own).
    scheduler.warmup_now()
    device_index = torch.device(device).index
    # Direct factory calls have no process aggregate; preserve their standalone stage budget.
    validate_loaded_process_memory_budget(
        stage_name="MOSS-TTS Local vocoder",
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
