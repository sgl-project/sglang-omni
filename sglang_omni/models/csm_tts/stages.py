# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the CSM TTS pipeline."""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as F_audio

from sglang_omni.models.csm_tts.payload_types import (
    DEFAULT_MAX_NEW_TOKENS,
    CsmTtsState,
    resolve_csm_generation_params,
)
from sglang_omni.models.csm_tts.text_tokenizer import CsmTokenizerAdapter
from sglang_omni.models.csm_tts.utils import (
    FRAME_RATE,
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_F32,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import (
    DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_CSM_STREAM_STRIDE,
    CsmStreamingVocoderScheduler,
)
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

DEFAULT_MAX_CONCURRENCY = 4

MAX_CONTEXT_AUDIO_SECONDS = 60
_MAX_CONTEXT_FRAMES = int(MAX_CONTEXT_AUDIO_SECONDS * FRAME_RATE)
_MAX_TEXT_TOKENS = 1500
_CTX_CODE_CACHE_MAX_ITEMS = 256
_CTX_CODE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CTX_WAVEFORM_CACHE_MAX_ITEMS = 256
_CTX_WAVEFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024
_CONSUMED_CONTEXT_INPUT_KEYS = frozenset(
    {
        "context",
        "references",
        "reference_audio",
        "reference_text",
        "reference_codes",
        "reference_speaker",
    }
)


def context_audio_cache_key(context_audio: Any) -> str | None:
    """Stable cache key for one context-audio input."""
    if isinstance(context_audio, (str, Path)):
        return reference_path_cache_key(context_audio)
    if not isinstance(context_audio, dict):
        return None
    path = context_audio.get("audio_path") or context_audio.get("path")
    if path:
        return reference_path_cache_key(path)
    if "bytes" in context_audio:
        data = context_audio["bytes"]
        if isinstance(data, str):
            data = data.encode()
        return hash_media_item(data)
    encoded = context_audio.get("base64") or context_audio.get("data")
    if encoded is None:
        return None
    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)


def without_consumed_context_media(inputs: Any) -> Any:
    """Return inputs with context / reference media already consumed."""
    if not isinstance(inputs, dict):
        return inputs
    return {
        key: value
        for key, value in inputs.items()
        if key not in _CONSUMED_CONTEXT_INPUT_KEYS
    }


def normalize_context_inputs(
    inputs: dict[str, Any], default_speaker: int
) -> list[dict[str, Any]]:
    """Fold references / reference_audio into context segments."""
    context = inputs.get("context")
    if context:
        if not isinstance(context, list):
            raise TypeError(f"context must be a list, got {type(context).__name__}")
        return [dict(seg) for seg in context]

    segments: list[dict[str, Any]] = []
    refs = inputs.get("references")
    if refs and isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            seg: dict[str, Any] = {
                "speaker": ref.get("speaker", default_speaker),
                "text": ref.get("text") or "",
            }
            if "bytes" in ref or "base64" in ref or "data" in ref:
                seg["audio"] = ref
            else:
                seg["audio"] = ref.get("audio_path") or ref.get("path")
            segments.append(seg)
        return segments

    reference_audio = inputs.get("reference_audio")
    if reference_audio is not None:
        segments.append(
            {
                "speaker": inputs.get("reference_speaker", default_speaker),
                "text": inputs.get("reference_text") or "",
                "audio": reference_audio,
            }
        )
    return segments


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ThreadedSimpleScheduler:
    """CPU preprocessing: parse inputs, tokenize, route the audio path."""
    checkpoint_dir = resolve_checkpoint(model_path)
    adapter = CsmTokenizerAdapter(checkpoint_dir)
    waveform_cache = StageOutputCache(
        max_size=_CTX_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=_CTX_WAVEFORM_CACHE_MAX_BYTES,
    )
    waveform_cache_lock = threading.Lock()

    def _load_context_waveform(audio: Any) -> tuple[torch.Tensor, str | None]:
        cache_key = context_audio_cache_key(audio)
        with waveform_cache_lock:
            cached = waveform_cache.get(cache_key)
        if cached is not None:
            return cached.clone(), cache_key
        waveform_np, sample_rate = load_audio_to_24k(audio)
        wav = torch.from_numpy(waveform_np)
        if sample_rate != SAMPLE_RATE:
            wav = F_audio.resample(wav, sample_rate, SAMPLE_RATE)
        if wav.shape[-1] > MAX_CONTEXT_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError(
                f"context audio is too long ({wav.shape[-1] / SAMPLE_RATE:.1f}s); "
                f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s."
            )
        wav = wav.reshape(1, 1, -1).contiguous().float()
        with waveform_cache_lock:
            waveform_cache.put(cache_key, wav.clone())
        return wav, cache_key

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        if not str(text).strip():
            raise ValueError("CSM TTS requires non-empty input text")
        speaker_id = int(inputs.get("speaker", inputs.get("speaker_id", 0)) or 0)
        text_tokens = adapter.tokenize_segment_text(speaker_id, text)
        if len(text_tokens) > _MAX_TEXT_TOKENS:
            raise ValueError(
                f"input text is too long ({len(text_tokens)} tokens); cap at "
                f"{_MAX_TEXT_TOKENS} tokens after reserving frame budget."
            )

        context_entries: list[dict[str, Any]] = []
        context_codes: list[torch.Tensor | None] = []
        has_raw_audio = False
        for seg in normalize_context_inputs(inputs, speaker_id):
            seg_speaker = int(
                seg.get("speaker", seg.get("speaker_id", speaker_id)) or 0
            )
            seg_text = seg.get("text") or ""
            entry: dict[str, Any] = {"speaker_id": seg_speaker, "text": seg_text}
            codes = to_codes_F32(seg.get("codes"))
            if codes is not None:
                if codes.shape[0] > _MAX_CONTEXT_FRAMES:
                    raise ValueError(
                        f"context codes are too long ({int(codes.shape[0])} frames); "
                        f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s "
                        f"(~{_MAX_CONTEXT_FRAMES} frames at {FRAME_RATE} Hz)."
                    )
                context_codes.append(codes.to(torch.int32).cpu())
            else:
                audio = seg.get("audio")
                if audio is None:
                    raise ValueError(
                        "each context segment needs either pre-encoded 'codes' "
                        "or raw 'audio'"
                    )
                wav, cache_key = _load_context_waveform(audio)
                entry["waveform"] = wav
                entry["cache_key"] = cache_key
                context_codes.append(None)
                has_raw_audio = True
            context_entries.append(entry)

        if has_raw_audio:
            prompt_ids: list[int] = []
        else:
            prompt_ids = adapter.build_prompt(
                text=text,
                speaker_id=speaker_id,
                context=context_entries,
                context_frame_counts=[
                    int(c.shape[0]) for c in context_codes if c is not None
                ],
            )

        tts_params = {}
        metadata = payload.request.metadata or {}
        if isinstance(metadata, dict):
            raw_tts = metadata.get("tts_params")
            if isinstance(raw_tts, dict):
                tts_params = raw_tts
        gen = resolve_csm_generation_params(params, tts_params)
        state = CsmTtsState(
            target_text=text,
            speaker_id=speaker_id,
            context=context_entries,
            prompt_token_ids=prompt_ids,
            context_codes=context_codes if context_entries else None,
            max_new_tokens=int(gen["max_new_tokens"]),
            temperature=float(gen["temperature"]),
            top_k=int(gen["top_k"]),
            top_p=gen["top_p"],
            depth_temperature=float(gen["depth_temperature"]),
            depth_top_k=int(gen["depth_top_k"]),
            seed=gen["seed"],
        )
        payload.data = state.to_dict()
        payload.request.inputs = without_consumed_context_media(inputs)
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    """Mimi encode stage. No-op when the prompt is already assembled."""
    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = resolve_checkpoint(model_path)
    adapter = CsmTokenizerAdapter(checkpoint_dir)
    codec = get_or_load_codec(checkpoint_dir, device, "float32")
    codec.encode_reference(torch.zeros(1, 1, codec.SAMPLE_RATE), codec.SAMPLE_RATE)
    code_cache = StageOutputCache(
        max_size=_CTX_CODE_CACHE_MAX_ITEMS,
        max_bytes=_CTX_CODE_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        state = CsmTtsState.from_dict(payload.data)
        if state.prompt_token_ids:
            return payload

        context_codes = list(state.context_codes or [None] * len(state.context))
        for i, entry in enumerate(state.context):
            if context_codes[i] is not None:
                continue
            waveform = entry.get("waveform")
            if waveform is None:
                raise ValueError(
                    f"context segment {i} has neither codes nor a loaded waveform"
                )
            cache_key = entry.get("cache_key")
            codes = code_cache.get(cache_key)
            if codes is None:
                wav = torch.as_tensor(waveform, dtype=torch.float32)
                codes_F32 = codec.encode_reference(wav, codec.SAMPLE_RATE)
                if codes_F32.ndim != 2 or codes_F32.shape[1] != NUM_CODEBOOKS:
                    raise ValueError(
                        f"Mimi encode must return [F, {NUM_CODEBOOKS}], got "
                        f"{tuple(codes_F32.shape)}"
                    )
                codes = codes_F32.to(torch.int32).cpu()
                code_cache.put(cache_key, codes)
            context_codes[i] = codes
            entry.pop("waveform", None)
            entry.pop("cache_key", None)

        prompt_ids = adapter.build_prompt(
            text=state.target_text or "",
            speaker_id=state.speaker_id,
            context=state.context,
            context_frame_counts=[int(c.shape[0]) for c in context_codes],
        )
        state.prompt_token_ids = prompt_ids
        state.context_codes = context_codes
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS,
    max_running_requests: int = DEFAULT_MAX_CONCURRENCY,
    cuda_graph_max_bs: int = DEFAULT_MAX_CONCURRENCY,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    total_gpu_memory_fraction: float | None = None,
):
    """sglang-backed AR engine for CSM TTS."""
    from sglang_omni.models.csm_tts.engine_builder import CsmTtsEngineBuilder

    return CsmTtsEngineBuilder(
        max_new_tokens=max_new_tokens,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
    stream_stride: int = DEFAULT_CSM_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE,
    stream_overlap_frames: int = 4,
    stream_holdback_frames: int = 2,
) -> CsmStreamingVocoderScheduler:
    """Decode CSM frames to a mono 24 kHz waveform."""
    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)

    return CsmStreamingVocoderScheduler(
        codec,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_frames=stream_overlap_frames,
        stream_holdback_frames=stream_holdback_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONTEXT_AUDIO_SECONDS",
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
