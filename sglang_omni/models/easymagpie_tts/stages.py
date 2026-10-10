# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the EasyMagpie pipeline.

    preprocessing -> tts_engine -> vocoder
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from sglang_omni.models.easymagpie_tts.request_builders import build_easymagpie_state
from sglang_omni.models.easymagpie_tts.speakers import load_speaker_embeddings
from sglang_omni.models.easymagpie_tts.streaming_vocoder import (
    DEFAULT_STARTUP_CHUNK_FRAMES,
    DEFAULT_STEADY_CHUNK_FRAMES,
    EasyMagpieStreamingVocoder,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 64,
    lead_in_frame: bool = True,
) -> ThreadedSimpleScheduler:
    """``lead_in_frame`` also streams the frame of the step before the speech
    delay, which reaches the vocoder one decode step earlier."""
    # note (Yashwant Hayaran): CPU-only stage declaring gpu only to share the
    # pipeline process; it does not touch the device.
    del device, gpu_id
    checkpoint = Path(resolve_checkpoint(model_path))
    config = json.loads((checkpoint / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=False, fix_mistral_regex=True
    )
    text_eos_id = int(config["text_eos_id"])
    phoneme_delay = int(config["streaming_phonemes_delay"])
    speech_delay = int(config["streaming_speech_delay"])
    if speech_delay <= phoneme_delay:
        raise ValueError(
            "EasyMagpie needs streaming_speech_delay greater than "
            "streaming_phonemes_delay"
        )
    else:
        pass
    # The engine holds the voice rows on the GPU; requests carry their length.
    voice_frames = {
        name: int(embedding.shape[0])
        for name, embedding in load_speaker_embeddings(
            checkpoint, int(config["embedding_dim"])
        ).items()
    }

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_easymagpie_state(payload)
        if state.voice not in voice_frames:
            raise ValueError(
                f"Unknown EasyMagpie voice {state.voice!r}; "
                f"available voices: {sorted(voice_frames)}"
            )
        else:
            pass
        state.phoneme_delay = phoneme_delay
        state.speech_delay = speech_delay
        # Every row up to and including the phoneme BOS is known before any
        # prediction exists.
        state.text_prefill_num = phoneme_delay + 1
        if lead_in_frame:
            state.audio_emit_delay = max(speech_delay - 1, state.text_prefill_num)
        else:
            state.audio_emit_delay = speech_delay
        state.text_token_ids = [
            *tokenizer.encode(state.text, add_special_tokens=False),
            text_eos_id,
        ]
        state.context_token_ids = list(
            tokenizer.encode(state.context_text, add_special_tokens=False)
        )
        state.speaker_frames = voice_frames[state.voice]
        return store_state(payload, state)

    # Tokenization is blocking CPU work; asyncio's default executor would cap
    # the thread count below max_concurrency.
    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float16",
    max_running_requests: int = 64,
    mem_fraction_static: float = 0.72,
    cuda_graph: bool = True,
    torch_compile: bool = True,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    server_args_overrides: dict | None = None,
) -> Any:
    from sglang_omni.models.easymagpie_tts.engine_builder import (
        EasyMagpieTTSEngineBuilder,
    )

    return EasyMagpieTTSEngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
        cuda_graph=cuda_graph,
        torch_compile=torch_compile,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
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
    max_batch_size: int = 64,
    max_batch_wait_ms: float = 5,
    startup_chunk_frames: list[int] | None = None,
    steady_chunk_frames: int | None = None,
    cuda_graph: bool = True,
    freeze_gc: bool = True,
) -> EasyMagpieStreamingVocoder:
    from sglang_omni.models.easymagpie_tts.codec import load_codec
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = str(resolve_concrete_device(device, gpu_id))
    vocoder = EasyMagpieStreamingVocoder(
        load_codec(resolve_checkpoint(model_path), concrete_device),
        startup_chunk_frames=(
            DEFAULT_STARTUP_CHUNK_FRAMES
            if startup_chunk_frames is None
            else startup_chunk_frames
        ),
        steady_chunk_frames=steady_chunk_frames or DEFAULT_STEADY_CHUNK_FRAMES,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cuda_graph=cuda_graph,
        freeze_gc=freeze_gc,
    )
    # Capture before the stage reports ready, so no colocated stage runs GPU
    # work during capture.
    vocoder.warmup_now()
    return vocoder
