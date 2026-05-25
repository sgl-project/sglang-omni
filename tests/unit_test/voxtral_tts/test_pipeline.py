# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline import stages
from sglang_omni.models.voxtral_tts.request_builders import build_sglang_voxtral_request
from sglang_omni.proto import OmniRequest, StagePayload


def test_voxtral_tts_config_uses_current_stage_schema() -> None:
    config = VoxtralTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_generation",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_generation": 0, "vocoder": 0}
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("VoxtralTTSForConditionalGeneration")
        is VoxtralTTSPipelineConfig
    )


def test_voxtral_radix_cache_is_namespaced_by_voice() -> None:
    """Different voice embeddings must not share a placeholder-token cache prefix."""
    model = SimpleNamespace(
        audio_token_id=24,
        voxtral_config=SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=32000),
        ),
    )
    voice_embeddings = {
        "cheerful_female": torch.ones(4, 8),
        "neutral_female": torch.ones(4, 8),
    }

    def make_payload(request_id: str, voice: str) -> StagePayload:
        state = VoxtralTTSState(
            input_ids=[1, 25, 24, 24, 24, 36, 100, 25],
            voice=voice,
        )
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="", params={}),
            data=state.to_dict(),
        )

    cheerful = build_sglang_voxtral_request(
        make_payload("r1", "cheerful_female"),
        model=model,
        voice_embeddings=voice_embeddings,
    )
    neutral = build_sglang_voxtral_request(
        make_payload("r2", "neutral_female"),
        model=model,
        voice_embeddings=voice_embeddings,
    )

    assert cheerful.req.origin_input_ids == neutral.req.origin_input_ids
    assert cheerful.req.extra_key != neutral.req.extra_key
    assert cheerful.req.extra_key.startswith("voxtral_voice:")


def test_voxtral_speech_validation_accepts_supported_fields() -> None:
    stages._validate_voxtral_speech_params(
        inputs="hello",
        params={
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
            "stream": True,
        },
        tts_params={
            "voice": "cheerful_female",
            "response_format": "wav",
            "speed": 1.0,
            "explicit_generation_params": ["max_new_tokens"],
        },
    )


@pytest.mark.parametrize(
    ("params", "tts_params", "inputs", "field"),
    [
        ({"temperature": 0.2}, {}, "hello", "temperature"),
        ({}, {"explicit_generation_params": ["seed"], "seed": 7}, "hello", "seed"),
        ({}, {"language": "en"}, "hello", "language"),
        ({}, {"ref_audio": "ref.wav"}, "hello", "ref_audio"),
        (
            {},
            {},
            {"text": "hello", "references": [{"audio_path": "ref.wav"}]},
            "references",
        ),
        ({"stage_params": {"tts_generation": {"x": 1}}}, {}, "hello", "stage_params"),
    ],
)
def test_voxtral_speech_validation_rejects_ignored_fields(
    params: dict,
    tts_params: dict,
    inputs,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        stages._validate_voxtral_speech_params(
            inputs=inputs,
            params=params,
            tts_params=tts_params,
        )


@pytest.mark.parametrize("audio_codes", [None, torch.empty((0, 0), dtype=torch.long)])
def test_voxtral_vocoder_rejects_empty_audio_codes(audio_codes) -> None:
    with pytest.raises(ValueError, match="generated no audio codes"):
        stages._ensure_non_empty_audio_codes(audio_codes)
