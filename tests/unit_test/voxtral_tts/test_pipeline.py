# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
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
