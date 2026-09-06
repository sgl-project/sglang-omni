# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Qwen3-TTS Base pipeline."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.request_builders import (
    cleanup_prepared_qwen3_tts_request,
    preprocess_qwen3_tts_payload,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES,
    DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_QWEN3_TTS_STREAM_STRIDE,
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec

logger = logging.getLogger(__name__)

_QWEN_TTS_INSTALL_HINT = (
    "Qwen3-TTS support requires the official `qwen-tts` package:\n"
    "    apt-get update && apt-get install -y sox\n"
    "    uv pip install --no-deps sox einops\n"
    "    uv pip install --no-deps qwen-tts==0.1.1\n"
    "`--no-deps` is required on both lines: qwen-tts pins Transformers 4.57.3, "
    "and resolving sox lifts numpy past the numba==0.65.1 ceiling. See "
    "docs/cookbook/qwen3_tts.md."
)


def _load_qwen3_tts_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    apply_qwen_tts_transformers_compatibility_patches()
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    tokenizer_path = os.path.join(checkpoint_dir, "speech_tokenizer")
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    logger.info(f"Loading Qwen3-TTS speech tokenizer from {tokenizer_path} on {device}")
    return Qwen3TTSTokenizer.from_pretrained(tokenizer_path, **kwargs)


def _register_qwen3_tts_hf_config() -> None:
    apply_qwen_tts_transformers_compatibility_patches()
    try:
        from qwen_tts.core.models import Qwen3TTSConfig
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc
    if not hasattr(Qwen3TTSConfig, "_sglang_omni_patched"):
        original_init = Qwen3TTSConfig.__init__

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            talker_config = getattr(self, "talker_config", None)
            if talker_config is not None:
                self.text_config = talker_config

        Qwen3TTSConfig.__init__ = _patched_init
        Qwen3TTSConfig._sglang_omni_patched = True
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    except ValueError:
        pass


def _load_qwen3_tts_generate_defaults(checkpoint_dir: str) -> dict[str, Any]:
    import json

    path = os.path.join(checkpoint_dir, "generation_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    stream_codec_output: bool = True,
    load_frontend: bool = False,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> ThreadedSimpleScheduler:
    if load_frontend:
        _load_standalone_preprocessing_context(
            model_path,
            device=device,
            gpu_id=gpu_id,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    # note (luojiaxuan): preprocessing must admit several requests at once. A
    # serial executor keeps at most one reference-code request in flight, so
    # the speech-tokenizer batcher would only ever see batches of one; the
    # default matches the batcher's max_batch_size.
    return ThreadedSimpleScheduler(
        functools.partial(
            preprocess_qwen3_tts_payload,
            default_stream_codec_output=stream_codec_output,
        ),
        max_concurrency=max_concurrency,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )


def _load_standalone_preprocessing_context(
    model_path: str,
    *,
    device: str | None,
    gpu_id: int | None,
    dtype: str,
    attn_implementation: str | None,
) -> None:
    """Load the prompt frontend for a preprocessing stage outside the engine process."""
    from transformers import AutoProcessor

    from sglang_omni.models.qwen3_tts import request_builders
    from sglang_omni.models.qwen3_tts.prompt_frontend import (
        load_qwen3_tts_prompt_frontend,
    )

    _register_qwen3_tts_hf_config()
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    device = resolve_device_spec(device, gpu_id)
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    logger.info(f"Loading Qwen3-TTS prompt frontend from {checkpoint_dir} on {device}")
    frontend = load_qwen3_tts_prompt_frontend(
        checkpoint_dir, device=device, dtype=torch_dtype
    )
    frontend.load_speech_tokenizer(
        _load_qwen3_tts_tokenizer(
            checkpoint_dir,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    )
    processor = AutoProcessor.from_pretrained(checkpoint_dir, fix_mistral_regex=True)
    wrapper = Qwen3TTSModel(
        model=frontend,
        processor=processor,
        generate_defaults=_load_qwen3_tts_generate_defaults(checkpoint_dir),
    )
    request_builders.set_qwen3_tts_preprocessing_context(
        model=frontend, wrapper=wrapper, standalone=True
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    return Qwen3TtsEngineBuilder(
        attn_implementation=attn_implementation,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
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
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = DEFAULT_QWEN3_TTS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE,
    stream_initial_followup_stride: int | None = None,
    initial_chunk_frames: int | None = None,
    stream_chunk_ramp: tuple[int, ...] | list[int] | None = None,
    stream_left_context_frames: int = DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES,
    initial_max_batch_size: int = 32,
    initial_batch_wait_ms: int = 2,
    followup_max_batch_size: int = 8,
    followup_batch_wait_ms: int = 1,
    followup_worker_count: int = 2,
    initial_cuda_graph: bool = True,
    enable_deterministic_inference: bool = False,
    followup_cuda_graph: bool = True,
    fused_snake_activation: bool = False,
    enable_stateful_codec_decoder: bool = False,
    suppress_bootstrap_silence: bool = True,
    suppress_bootstrap_max_streams: int = 24,
) -> SimpleScheduler:
    device = resolve_device_spec(device, gpu_id)
    tokenizer = _load_qwen3_tts_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device=device,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_initial_followup_stride=stream_initial_followup_stride,
        initial_chunk_frames=initial_chunk_frames,
        stream_chunk_ramp=stream_chunk_ramp,
        stream_left_context_frames=stream_left_context_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        initial_max_batch_size=initial_max_batch_size,
        initial_batch_wait_ms=initial_batch_wait_ms,
        followup_max_batch_size=followup_max_batch_size,
        followup_batch_wait_ms=followup_batch_wait_ms,
        followup_worker_count=followup_worker_count,
        initial_cuda_graph=initial_cuda_graph,
        enable_deterministic_inference=enable_deterministic_inference,
        followup_cuda_graph=followup_cuda_graph,
        fused_snake_activation=fused_snake_activation,
        enable_stateful_codec_decoder=enable_stateful_codec_decoder,
        suppress_bootstrap_silence=suppress_bootstrap_silence,
        suppress_bootstrap_max_streams=suppress_bootstrap_max_streams,
    )
    # note (ratish): Factory construction completes before the stage process
    # publishes readiness, so CUDA capture cannot overlap request-time GPU work
    # from colocated stages.
    scheduler.warmup_now()
    return scheduler
