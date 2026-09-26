# SPDX-License-Identifier: Apache-2.0
"""Stage factories: preprocessing, Mimi encode, the LM engine, decode and code2wav."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.personaplex.architecture import MIMI_WEIGHTS_GLOB, SAMPLE_RATE
from sglang_omni.models.personaplex.code2wav_stream import (
    PersonaPlexCode2WavScheduler,
    trim_to_caller,
)
from sglang_omni.models.personaplex.components.mimi import (
    MimiCodec,
    load_mimi_codec,
    resolve_mimi_weights,
)
from sglang_omni.models.personaplex.config import PREPROCESSING_STAGE
from sglang_omni.models.personaplex.engine_builder import PersonaPlexEngineBuilder
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.prompts import (
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VOICE,
    VoicePrompt,
    decode_text,
    load_text_tokenizer,
    load_voice_prompt,
    pad_to_whole_frames,
    resolve_voice_path,
    tokenize_text_prompt,
)
from sglang_omni.models.personaplex.request_builders import stage_request_params
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


def load_channels(source: str | bytes, *, source_name: str) -> np.ndarray:
    """Any sample rate in, [channels, samples] float32 at 24 kHz out."""
    return load_audio(
        source, source_name=source_name, target_sample_rate=SAMPLE_RATE, mono=False
    )


def caller_audio_source(payload: StagePayload) -> str | bytes:
    """The caller recording: an audio_path-style input, or the one entry of
    audios that chat completions sends."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("audios"):
        audios = inputs["audios"]
        if len(audios) != 1:
            raise ValueError(
                f"PersonaPlex takes one caller recording, got {len(audios)} audios"
            )
        else:
            pass
        return audios[0]
    else:
        pass
    return resolve_audio_source(payload)


def request_text_prompt(params: dict) -> str | None:
    for key in ("text_prompt", "instructions"):
        if key in params:
            return params[key]
        else:
            pass
    return DEFAULT_TEXT_PROMPT


def create_preprocessing_executor(model_path: str, **_) -> SimpleScheduler:
    model_dir = Path(resolve_model_path(model_path))
    tokenizer = load_text_tokenizer(model_dir)
    voice_cache: dict[Path, VoicePrompt] = {}

    def preprocess(payload: StagePayload) -> StagePayload:
        params = stage_request_params(payload.request.params, PREPROCESSING_STAGE)
        # Note (wilsonzheng0327): Channel 0, not a downmix: in a two-party recording the
        # agent is on channel 1.
        channels = load_channels(
            caller_audio_source(payload), source_name="PersonaPlex"
        )
        caller = torch.as_tensor(channels[0], dtype=torch.float32)

        state = PersonaPlexState.from_dict(payload.data)
        state.num_samples = int(caller.shape[-1])
        state.waveform = pad_to_whole_frames(caller)
        state.text_prompt_ids = tokenize_text_prompt(
            tokenizer, request_text_prompt(params)
        )

        voice = params.get("voice", DEFAULT_VOICE)
        if voice:
            path = resolve_voice_path(model_dir, str(voice))
            prompt = voice_cache.get(path)
            if prompt is None:
                prompt = load_voice_prompt(
                    path,
                    load_audio=lambda p: load_channels(
                        p, source_name="PersonaPlex voice"
                    ),
                )
                voice_cache[path] = prompt
            else:
                pass
            state.voice_frames = prompt.frames
            state.voice_embeddings = prompt.embeddings
            state.voice_tail_codes = prompt.tail_codes
            state.voice_waveform = prompt.waveform
        else:
            pass
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(preprocess)


def load_codec(
    model_path: str, *, device: str | None, gpu_id: int | None
) -> tuple[MimiCodec, torch.device]:
    device = resolve_concrete_device(device, gpu_id)
    weights = resolve_mimi_weights(resolve_model_path(model_path), MIMI_WEIGHTS_GLOB)
    return load_mimi_codec(weights, device=device), device


def create_mimi_encode_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None, **_
) -> SimpleScheduler:
    codec, device = load_codec(model_path, device=device, gpu_id=gpu_id)

    def encode_waveform(waveform: torch.Tensor) -> torch.Tensor:
        codes = codec.encode(
            waveform.to(device=device, dtype=torch.float32).view(1, 1, -1)
        )
        return codes[0].T.cpu()

    def encode(payload: StagePayload) -> StagePayload:
        state = PersonaPlexState.from_dict(payload.data)
        if state.waveform is not None:
            state.user_codes = encode_waveform(state.waveform)
        else:
            pass
        if state.voice_waveform is not None:
            state.voice_codes = encode_waveform(state.voice_waveform)
        else:
            pass
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(encode)


def create_lm_executor(
    model_path: str,
    *,
    dtype: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
    context_length: int | None = None,
    server_args_overrides: dict[str, object] | None = None,
    **overrides: object,
):
    server_args_overrides = {**overrides, **(server_args_overrides or {})}
    # Note (wilsonzheng0327): The shim config is written before the engine reads its
    # overrides, so an engine context_length must reach the builder too.
    context_length = server_args_overrides.get("context_length", context_length)
    builder = PersonaPlexEngineBuilder(
        max_running_requests=1, context_length=context_length
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype or "bfloat16",
        server_args_overrides=server_args_overrides or None,
    )


def create_decode_executor(model_path: str, **_) -> SimpleScheduler:
    """Turn the frame-locked text stream into the reply text."""
    tokenizer = load_text_tokenizer(resolve_model_path(model_path))

    def detokenize(payload: StagePayload) -> StagePayload:
        state = PersonaPlexState.from_dict(payload.data)
        payload.data = {"text": decode_text(tokenizer, state.text_ids)}
        return payload

    return SimpleScheduler(detokenize)


def create_code2wav_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None, **_
) -> PersonaPlexCode2WavScheduler:
    codec, device = load_codec(model_path, device=device, gpu_id=gpu_id)

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        """Render a whole reply at once, for a request that never streamed."""
        state = PersonaPlexState.from_dict(payload.data)
        codes = state.codes
        if codes is None or codes.shape[0] == 0:
            waveform = torch.zeros(0)
        else:
            waveform = codec.decode(codes.to(device=device, dtype=torch.long).T[None])[
                0, 0
            ].cpu()
        payload.data = audio_waveform_payload(
            trim_to_caller(waveform, state.num_samples),
            sample_rate=SAMPLE_RATE,
            modality="audio",
            source_hint="PersonaPlex",
        )
        return payload

    return PersonaPlexCode2WavScheduler(codec, compute_fn=decode)


__all__ = [
    "create_code2wav_executor",
    "create_decode_executor",
    "create_lm_executor",
    "create_mimi_encode_executor",
    "create_preprocessing_executor",
]
