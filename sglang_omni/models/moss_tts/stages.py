# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    load_moss_audio_encoder,
    load_moss_audio_vocoder,
    resolve_moss_audio_dtype,
)
from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.hf_loading import (
    MossDelayReferences,
    MossLoadedProcessor,
    MossProcessorConfigSource,
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts.payload_types import moss_tts_special_token_defaults
from sglang_omni.models.moss_tts.reference_encoder import MossReferenceEncoder
from sglang_omni.models.moss_tts.request_builders import (
    MossTTSSGLangRequestData,
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
from sglang_omni.models.moss_tts.vocoder import MossTTSVocoder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

if TYPE_CHECKING:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)


def resolve_compute_dtype(
    dtype: str | torch.dtype | None,
) -> torch.dtype | None:
    return resolve_moss_audio_dtype(
        dtype,
        name="compute_dtype",
        allow_none=True,
    )


def normalize_moss_processor_config(
    processor: MossProcessorConfigSource | None,
) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    else:
        pass
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)
        else:
            pass


def audio_tokenizer_model_path_from_processor_dict(
    processor_dict: Mapping[str, object],
) -> str | None:
    model_path = processor_dict.get("audio_tokenizer_name_or_path")
    audio_tokenizer_dict = processor_dict.get("audio_tokenizer")
    if isinstance(audio_tokenizer_dict, dict):
        model_path = (
            audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or model_path
        )
    else:
        pass
    return str(model_path) if model_path else None


def load_moss_processor(
    model_path: str,
) -> MossLoadedProcessor[MossDelayReferences]:
    logger.info(f"Loading MOSS-TTS processor from {model_path} without codec")
    try:
        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(model_path)
            processor_dict, _ = processor_cls.get_processor_dict(model_path)
            audio_tokenizer_model_path = audio_tokenizer_model_path_from_processor_dict(
                processor_dict
            )
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if audio_tokenizer_model_path:
                # processor_cls.from_pretrained normally resolves this metadata
                # before loading the codec. Preserve the same selection while
                # constructing the processor without a codec instance.
                model_config.audio_tokenizer_name_or_path = audio_tokenizer_model_path
            else:
                pass
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
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    normalize_moss_processor_config(processor)
    return processor


def resolve_audio_tokenizer_model_path(
    processor: MossProcessorConfigSource,
    codec_model_path: str | None,
) -> str:
    return str(
        codec_model_path
        or getattr(processor.model_config, "audio_tokenizer_name_or_path", None)
        or DEFAULT_MOSS_TTS_AUDIO_TOKENIZER
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
    for name, value in (
        ("ref_audio_cache_max_items", ref_audio_cache_max_items),
        ("ref_audio_cache_max_bytes", ref_audio_cache_max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")
        else:
            pass

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
    processor = load_moss_processor(model_path)
    resolved_codec_model_path = resolve_audio_tokenizer_model_path(
        processor,
        codec_model_path,
    )
    resolved_compute_dtype = resolve_compute_dtype(compute_dtype)
    audio_encoder = load_moss_audio_encoder(
        resolved_codec_model_path,
        device=device,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )
    reference_encoder = MossReferenceEncoder(
        audio_encoder,
        codec_model_path=resolved_codec_model_path,
        n_vq=int(processor.model_config.n_vq),
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
        cache_enabled=ref_audio_cache,
        max_items=ref_audio_cache_max_items,
        max_bytes=ref_audio_cache_max_bytes,
    )
    set_moss_tts_preprocessing_context(
        processor=processor,
        reference_encoder=reference_encoder,
    )
    # note (Zhang Yiyang): Every device uses the same batch queue; there is no
    # device-specific fallback.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[MossTTSSGLangRequestData]:
    overrides = dict(server_args_overrides or {})
    # Note (Jiaxin Deng): a declared stage fraction only reserves the card on paper, so
    # the AR engine has to be told about it or it profiles against the whole GPU and the
    # vocoder process has nothing left to claim. The process total is the right
    # denominator when the group hosts more than this one GPU stage.
    engine_fraction = (
        process_total_gpu_memory_fraction
        if process_total_gpu_memory_fraction is not None
        else total_gpu_memory_fraction
    )
    if engine_fraction is not None and "mem_fraction_static" not in overrides:
        overrides["mem_fraction_static"] = float(
            total_gpu_memory_fraction or engine_fraction
        )
    else:
        pass
    return MossTtsEngineBuilder(total_gpu_memory_fraction=engine_fraction).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=overrides or None,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 8,
    stream_followup_stride: int = 8,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 1,
    initial_chunk_frames: int = 0,
    compute_dtype: str | torch.dtype | None = "bfloat16",
    attention_backend: str = "auto",
) -> MossStreamingVocoderScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    # note (lennox): device is the policy override; gpu_id is the placement fallback.
    device = str(resolve_concrete_device(device, gpu_id))
    resolved_compute_dtype = resolve_compute_dtype(compute_dtype)
    processor = load_moss_processor(model_path)
    decoder_dtype = resolve_moss_audio_dtype(
        dtype,
        name="dtype",
        allow_none=False,
    )
    assert decoder_dtype is not None
    audio_vocoder = load_moss_audio_vocoder(
        resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
        decoder_dtype=decoder_dtype,
        compute_dtype=resolved_compute_dtype,
        attention_backend=attention_backend,
    )

    vocoder = MossTTSVocoder(
        processor,
        audio_vocoder,
        device,
        compute_dtype=resolved_compute_dtype,
        max_segment_batch_size=max_batch_size,
    )
    return MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
