# SPDX-License-Identifier: Apache-2.0
"""Stage factories for MOSS-TTS-Realtime."""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_tts_realtime.audio_tokenizer import (
    DEFAULT_CODEC_MODEL,
    encode_reference_audio,
    load_moss_realtime_audio_tokenizer,
)
from sglang_omni.models.moss_tts_realtime.processor import (
    MossTTSRealtimePromptProcessor,
)
from sglang_omni.models.moss_tts_realtime.request_builders import (
    cleanup_prepared_request,
    preprocess_payload,
    set_preprocessing_context,
)
from sglang_omni.models.moss_tts_realtime.vocoder import MossTTSRealtimeVocoder
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeKey,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


@dataclass(frozen=True)
class _ReferenceInput:
    source: Any
    kind: str


class _ReferenceHook(TensorReferenceEncodeHook[_ReferenceInput]):
    model_id = DEFAULT_CODEC_MODEL
    model_revision = "main"
    encoder_id = "moss_audio_tokenizer"
    artifact_kind = "moss_tts_realtime_reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(self, codec: Any) -> None:
        self._codec = codec
        self._lock = threading.Lock()
        self.encoder_config_hash = hash_bytes(b"sample_rate:24000;n_vq:16")

    def normalize_input(self, raw_input: Any) -> _ReferenceInput:
        if isinstance(raw_input, str):
            kind = "data_uri" if raw_input.startswith("data:audio/") else "path"
        elif isinstance(raw_input, (bytes, bytearray)):
            kind = "bytes"
        elif isinstance(raw_input, torch.Tensor):
            kind = "tensor"
        else:
            kind = "array"
        return _ReferenceInput(raw_input, kind)

    def input_key(self, item: _ReferenceInput) -> str | None:
        if item.kind == "path":
            return reference_path_cache_key(str(item.source), trust_stat=False)
        if item.kind == "data_uri":
            encoded = str(item.source).split(",", 1)[-1]
            return f"bytes:{hash_bytes(base64.b64decode(encoded))}"
        if item.kind == "bytes":
            return f"bytes:{hash_bytes(bytes(item.source))}"
        if item.kind == "tensor":
            values = item.source.detach().to(device="cpu").contiguous()
            return f"tensor:{hash_bytes(values.numpy().tobytes())}"
        return None

    def encode_one(self, item: _ReferenceInput) -> torch.Tensor:
        with self._lock:
            return encode_reference_audio(self._codec, item.source)

    def revalidate(self, item: _ReferenceInput, key: ReferenceEncodeKey) -> bool:
        return item.kind != "path" or self.input_key(item) == key.input_key


def _device(device: str | None, gpu_id: int | None) -> str:
    if gpu_id is not None:
        return f"cuda:{gpu_id}"
    return device or "cuda:0"


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str = DEFAULT_CODEC_MODEL,
    cache_max_items: int = 256,
    cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    from transformers import AutoTokenizer

    resolved_device = _device(device, gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = MossTTSRealtimePromptProcessor(tokenizer)
    codec = load_moss_realtime_audio_tokenizer(
        codec_model_path,
        device=resolved_device,
    )
    service = ReferenceEncodeService(
        _ReferenceHook(codec),
        max_items=cache_max_items,
        max_bytes=cache_max_bytes,
        log_prefix="MOSS-TTS-Realtime reference cache",
    )
    set_preprocessing_context(
        processor=processor,
        reference_encoder=lambda reference: service.get_or_encode(
            reference,
            desc="MOSS-TTS-Realtime reference",
        ),
    )
    return SimpleScheduler(
        preprocess_payload,
        abort_callback=cleanup_prepared_request,
        max_concurrency=1,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    codec_mem_reserve: float = 0.22,
) -> Any:
    from sglang_omni.models.moss_tts_realtime.engine_builder import (
        MossTTSRealtimeEngineBuilder,
    )

    return MossTTSRealtimeEngineBuilder(
        enable_async_decode=False,
        async_decode_min_batch_size=2,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        codec_mem_reserve=codec_mem_reserve,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str = DEFAULT_CODEC_MODEL,
    stream_chunk_frames: int = 6,
    initial_chunk_frames: int = 1,
) -> MossTTSRealtimeVocoder:
    del model_path
    codec = load_moss_realtime_audio_tokenizer(
        codec_model_path,
        device=_device(device, gpu_id),
    )
    return MossTTSRealtimeVocoder(
        codec,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
    )


__all__ = [
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
