# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the EasyMagpie pipeline.

    preprocessing -> tts_engine -> vocoder
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.models.easymagpie_tts.request_builders import build_easymagpie_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint

SPEAKER_SUBDIR = "speaker_embeddings"


def load_speaker_embeddings(
    checkpoint: Path, embedding_dim: int
) -> dict[str, torch.Tensor]:
    """Load every preset voice as a [frames, embedding_dim] float16 tensor."""
    voices = {}
    for path in sorted((checkpoint / SPEAKER_SUBDIR).glob("*.pt")):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            embedding = loaded["speaker_encoding"]
        else:
            embedding = loaded
        if embedding.ndim != 2 or embedding.shape[1] != embedding_dim:
            raise ValueError(
                f"EasyMagpie speaker embedding {path} must be [frames, {embedding_dim}]"
            )
        else:
            pass
        voices[path.stem] = embedding.detach().to(torch.float16)
    if not voices:
        raise ValueError(
            f"No EasyMagpie voices found under {checkpoint / SPEAKER_SUBDIR}"
        )
    else:
        pass
    return voices


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 8,
) -> SimpleScheduler:
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
    voices = load_speaker_embeddings(checkpoint, int(config["embedding_dim"]))

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_easymagpie_state(payload)
        if state.voice not in voices:
            raise ValueError(
                f"Unknown EasyMagpie voice {state.voice!r}; "
                f"available voices: {sorted(voices)}"
            )
        else:
            pass
        state.phoneme_delay = phoneme_delay
        state.speech_delay = speech_delay
        # Every row up to and including the phoneme BOS is known before any
        # prediction exists.
        state.text_prefill_num = phoneme_delay + 1
        state.text_token_ids = [
            *tokenizer.encode(state.text, add_special_tokens=False),
            text_eos_id,
        ]
        state.context_token_ids = list(
            tokenizer.encode(state.context_text, add_special_tokens=False)
        )
        state.speaker_embedding = voices[state.voice]
        return store_state(payload, state)

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float16",
    max_running_requests: int = 8,
    mem_fraction_static: float = 0.72,
    server_args_overrides: dict | None = None,
) -> Any:
    from sglang_omni.models.easymagpie_tts.engine_builder import (
        EasyMagpieTTSEngineBuilder,
    )

    return EasyMagpieTTSEngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
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
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 5,
) -> SimpleScheduler:
    from sglang_omni.models.easymagpie_tts.codec import load_codec
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = str(resolve_concrete_device(device, gpu_id))
    codec = load_codec(resolve_checkpoint(model_path), concrete_device)
    sample_rate = codec.config.output_sample_rate

    def _result_payload(
        payload: StagePayload, state: EasyMagpieTTSState, audio: torch.Tensor
    ) -> StagePayload:
        data = dict(
            audio_waveform_payload(
                audio.float().cpu().numpy(),
                sample_rate=sample_rate,
                modality="audio",
                source_hint="EasyMagpie",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=data
        )

    def _codes(state: EasyMagpieTTSState) -> torch.Tensor:
        if state.audio_codes is None or state.audio_codes.numel() == 0:
            raise ValueError("EasyMagpie generated no audio frames")
        else:
            return torch.as_tensor(state.audio_codes, dtype=torch.long)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        states = [EasyMagpieTTSState.from_dict(p.data) for p in payloads]
        audio = codec.decode_batch([_codes(state) for state in states])
        return [
            _result_payload(payload, state, waveform)
            for payload, state, waveform in zip(payloads, states, audio)
        ]

    def _vocode(payload: StagePayload) -> StagePayload:
        return _vocode_batch([payload])[0]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
