# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the CSM TTS pipeline."""

from __future__ import annotations

import threading
from collections.abc import Mapping

import torch
import torchaudio.functional as F_audio

from sglang_omni.models.csm_tts.audio_codec import load_cached_codec
from sglang_omni.models.csm_tts.config import DEFAULT_MAX_CONCURRENCY
from sglang_omni.models.csm_tts.context_audio import (
    as_audio_source,
    context_audio_cache_key,
    drop_consumed_context_inputs,
    normalize_context_inputs,
)
from sglang_omni.models.csm_tts.engine_builder import CsmTtsEngineBuilder
from sglang_omni.models.csm_tts.payload_types import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SPEAKER_ID,
    CsmTtsState,
    resolve_csm_generation_params,
)
from sglang_omni.models.csm_tts.text_tokenizer import CsmTokenizerAdapter
from sglang_omni.models.csm_tts.utils import (
    FRAME_RATE,
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    AudioSource,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_F32,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import (
    DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_CSM_STREAM_STRIDE,
    CsmStreamingVocoderScheduler,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

MAX_CONTEXT_AUDIO_SECONDS = 60
MAX_CONTEXT_FRAMES = int(MAX_CONTEXT_AUDIO_SECONDS * FRAME_RATE)
MAX_TEXT_TOKENS = 1500
CONTEXT_CODE_CACHE_MAX_ITEMS = 256
CONTEXT_CODE_CACHE_MAX_BYTES = 256 * 1024 * 1024
CONTEXT_WAVEFORM_CACHE_MAX_ITEMS = 256
CONTEXT_WAVEFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ThreadedSimpleScheduler:
    """CPU preprocessing: parse inputs, tokenize, route the audio path."""
    checkpoint_dir = resolve_checkpoint(model_path)
    adapter = CsmTokenizerAdapter(checkpoint_dir)
    waveform_cache = StageOutputCache(
        max_size=CONTEXT_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=CONTEXT_WAVEFORM_CACHE_MAX_BYTES,
    )
    waveform_cache_lock = threading.Lock()

    def load_context_waveform(audio: AudioSource) -> tuple[torch.Tensor, str | None]:
        cache_key = context_audio_cache_key(audio)
        with waveform_cache_lock:
            cached = waveform_cache.get(cache_key)
        if cached is not None:
            return cached.clone(), cache_key
        else:
            pass
        waveform_np, sample_rate = load_audio_to_24k(audio)
        waveform = torch.from_numpy(waveform_np)
        if sample_rate != SAMPLE_RATE:
            waveform = F_audio.resample(waveform, sample_rate, SAMPLE_RATE)
        else:
            pass
        if waveform.shape[-1] > MAX_CONTEXT_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError(
                f"context audio is too long "
                f"({waveform.shape[-1] / SAMPLE_RATE:.1f}s); "
                f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s."
            )
        else:
            pass
        waveform = waveform.reshape(1, 1, -1).contiguous().float()
        with waveform_cache_lock:
            waveform_cache.put(cache_key, waveform.clone())
        return waveform, cache_key

    def preprocess(payload: StagePayload) -> StagePayload:
        raw_inputs = payload.request.inputs or {}
        raw_params = payload.request.params
        params: Mapping[str, object] = (
            raw_params if isinstance(raw_params, Mapping) else {}
        )
        if isinstance(raw_inputs, str):
            inputs: Mapping[str, object] = {"text": raw_inputs}
        elif isinstance(raw_inputs, Mapping):
            inputs = raw_inputs
        else:
            raise TypeError(
                f"CSM TTS inputs must be text or a mapping, got "
                f"{type(raw_inputs).__name__}"
            )

        text = inputs.get("input") or inputs.get("text") or ""
        if not str(text).strip():
            raise ValueError("CSM TTS requires non-empty input text")
        else:
            pass
        speaker_id = DEFAULT_SPEAKER_ID
        text_tokens = adapter.tokenize_segment_text(speaker_id, text)
        if len(text_tokens) > MAX_TEXT_TOKENS:
            raise ValueError(
                f"input text is too long ({len(text_tokens)} tokens); cap at "
                f"{MAX_TEXT_TOKENS} tokens after reserving frame budget."
            )
        else:
            pass

        context_entries: list[dict[str, object]] = []
        context_codes: list[torch.Tensor | None] = []
        has_raw_audio = False
        for segment in normalize_context_inputs(inputs, speaker_id):
            segment_speaker = int(
                segment.get("speaker", segment.get("speaker_id", speaker_id)) or 0
            )
            segment_text = segment.get("text") or ""
            entry: dict[str, object] = {
                "speaker_id": segment_speaker,
                "text": segment_text,
            }
            codes = to_codes_F32(segment.get("codes"))
            if codes is not None:
                if codes.shape[0] > MAX_CONTEXT_FRAMES:
                    raise ValueError(
                        f"context codes are too long ({int(codes.shape[0])} frames); "
                        f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s "
                        f"(~{MAX_CONTEXT_FRAMES} frames at {FRAME_RATE} Hz)."
                    )
                else:
                    pass
                context_codes.append(codes.to(torch.int32).cpu())
            else:
                audio = segment.get("audio")
                if audio is None:
                    raise ValueError(
                        "each context segment needs either pre-encoded 'codes' "
                        "or raw 'audio'"
                    )
                else:
                    pass
                waveform, cache_key = load_context_waveform(as_audio_source(audio))
                entry["waveform"] = waveform
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
                    int(codes.shape[0]) for codes in context_codes if codes is not None
                ],
            )

        metadata = payload.request.metadata
        if isinstance(metadata, Mapping):
            raw_tts_params = metadata.get("tts_params")
        else:
            raw_tts_params = None
        tts_params: Mapping[str, object] = (
            raw_tts_params if isinstance(raw_tts_params, Mapping) else {}
        )
        generation = resolve_csm_generation_params(params, tts_params)
        state = CsmTtsState(
            target_text=text,
            speaker_id=speaker_id,
            context=context_entries,
            prompt_token_ids=prompt_ids,
            context_codes=context_codes if context_entries else None,
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]),
            top_p=generation["top_p"],
            depth_temperature=float(generation["depth_temperature"]),
            depth_top_k=int(generation["depth_top_k"]),
            seed=generation["seed"],
        )
        payload.data = state.to_dict()
        payload.request.inputs = drop_consumed_context_inputs(inputs)
        return payload

    return ThreadedSimpleScheduler(preprocess, max_concurrency=max_concurrency)


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
    codec = load_cached_codec(checkpoint_dir, device, "float32")
    codec.encode_reference(torch.zeros(1, 1, SAMPLE_RATE), SAMPLE_RATE)
    code_cache = StageOutputCache(
        max_size=CONTEXT_CODE_CACHE_MAX_ITEMS,
        max_bytes=CONTEXT_CODE_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def encode(payload: StagePayload) -> StagePayload:
        state = CsmTtsState.from_dict(payload.data)
        if state.prompt_token_ids:
            return payload
        else:
            pass

        context_codes = list(state.context_codes or [None] * len(state.context))
        for i, entry in enumerate(state.context):
            if context_codes[i] is not None:
                continue
            else:
                pass
            waveform = entry.get("waveform")
            if waveform is None:
                raise ValueError(
                    f"context segment {i} has neither codes nor a loaded waveform"
                )
            else:
                pass
            cache_key = entry.get("cache_key")
            codes = code_cache.get(cache_key)
            if codes is None:
                waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32)
                codes_F32 = codec.encode_reference(waveform_tensor, SAMPLE_RATE)
                if codes_F32.ndim != 2 or codes_F32.shape[1] != NUM_CODEBOOKS:
                    raise ValueError(
                        f"Mimi encode must return [F, {NUM_CODEBOOKS}], got "
                        f"{tuple(codes_F32.shape)}"
                    )
                else:
                    pass
                codes = codes_F32.to(torch.int32).cpu()
                code_cache.put(cache_key, codes)
            else:
                pass
            context_codes[i] = codes
            entry.pop("waveform", None)
            entry.pop("cache_key", None)

        prompt_ids = adapter.build_prompt(
            text=state.target_text or "",
            speaker_id=state.speaker_id,
            context=state.context,
            context_frame_counts=[int(codes.shape[0]) for codes in context_codes],
        )
        state.prompt_token_ids = prompt_ids
        state.context_codes = context_codes
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        encode,
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
    server_args_overrides: dict[str, object] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    total_gpu_memory_fraction: float | None = None,
):
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
    codec = load_cached_codec(checkpoint_dir, device, dtype)

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
