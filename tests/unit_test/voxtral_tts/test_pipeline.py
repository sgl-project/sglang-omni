# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline import stages
from sglang_omni.models.voxtral_tts.request_builders import build_sglang_voxtral_request
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.utils.audio_payload import audio_waveform_payload


def test_voxtral_tts_config_uses_current_stage_schema() -> None:
    config = VoxtralTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_generation",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_generation": 0, "vocoder": 0}
    assert "device" not in config.stages[1].factory_args
    assert "device" not in config.stages[2].factory_args
    assert config.stages[1].factory_args["gpu_id"] == 0
    assert config.stages[2].factory_args["gpu_id"] == 0
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
    assert cheerful.voice_embedding is voice_embeddings["cheerful_female"]


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
        (
            {"temperature": 0.2},
            {"explicit_generation_params": ["temperature"]},
            "hello",
            "temperature",
        ),
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


def test_voxtral_audio_waveform_payload_is_compact() -> None:
    payload = audio_waveform_payload(
        torch.tensor([0.0, 0.5, -0.5]),
        source_hint="Voxtral TTS",
    )

    audio = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == [0.0, 0.5, -0.5]
    assert payload["audio_waveform_shape"] == [3]
    assert payload["audio_waveform_dtype"] == "float32"


def test_voxtral_collect_audio_step_reuses_output_tokens_for_eos_filter() -> None:
    from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner._pending_audio_codes = None
    runner._pending_audio_embeds = None
    runner.model = SimpleNamespace(
        acoustic_transformer=lambda hidden: torch.tensor(
            [[11, 12, 13], [eos_id, 21, 22]], dtype=torch.long
        ),
        audio_token_embedding=lambda codes: codes.to(torch.float32).unsqueeze(-1),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones((2, 4))),
        next_token_ids=None,
    )
    schedule_batch = SimpleNamespace(output_ids=None)
    requests = [
        SimpleNamespace(
            request_id="active",
            data=SimpleNamespace(
                output_codes=[],
                pending_feedback_queue=[],
            ),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(
                output_codes=[],
                pending_feedback_queue=[],
            ),
        ),
    ]

    runner._collect_audio_step(result, schedule_batch, requests)

    assert result.next_token_ids.tolist() == [11, eos_id]
    assert schedule_batch.output_ids.tolist() == [11, eos_id]
    assert requests[0].data.output_codes == []
    assert requests[1].data.output_codes == []

    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=11),
            "eos": RequestOutput("eos", data=eos_id),
        },
    )

    assert [chunk.tolist() for chunk in requests[0].data.output_codes] == [[11, 12, 13]]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_codes == []
    assert requests[1].data.pending_feedback_queue == []

    runner.post_process_outputs(result, SimpleNamespace(requests=requests), {})
    assert len(requests[0].data.output_codes) == 1
