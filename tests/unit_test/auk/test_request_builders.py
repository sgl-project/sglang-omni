# SPDX-License-Identifier: Apache-2.0
"""Request normalization for AuK (no GPU, no weights)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio

from sglang_omni.models.auk.constants import (
    MAX_SECONDS,
    SAMPLE_RATE,
    VAE_DOWNSAMPLE_RATE,
)
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    build_auk_state,
    clear_auk_preprocessing_context,
    preprocess_auk_payload,
    set_auk_preprocessing_context,
)
from sglang_omni.proto import OmniRequest, StagePayload

FRAME_RATE = SAMPLE_RATE // VAE_DOWNSAMPLE_RATE


@pytest.fixture
def context():
    config = AuKRuntimeConfig(model_path="unused")
    set_auk_preprocessing_context(AuKPreprocessingContext(config=config))
    yield config
    clear_auk_preprocessing_context()


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref.wav"
    torchaudio.save(str(path), torch.zeros(1, 24240), SAMPLE_RATE)
    return str(path)


def make_payload(inputs, params=None) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=inputs, params=params or {}),
        data={},
    )


@pytest.mark.parametrize(
    "seconds, frames",
    [(None, 250), (2.01, 101), (0.001, 1), (999, MAX_SECONDS * FRAME_RATE)],
)
def test_generation_duration_and_state_round_trip(context, seconds, frames):
    payload = make_payload("Say hello", {"gen_seconds": seconds, "seed": 42})
    result = preprocess_auk_payload(payload)
    state = AuKState.from_dict(result.data)
    assert result.request_id == payload.request_id
    assert state.instruction == "Say hello"
    assert state.gen_frames == frames
    assert state.seed == 42
    assert state.ref_audio is None


def test_sampling_recipe_cannot_be_overridden_per_request(context):
    with pytest.raises(ValueError, match="server-level"):
        build_auk_state(make_payload("Hello", {"nfe": 1}), context)


def test_multiple_references_are_rejected(context):
    payload = make_payload(
        {
            "text": "Hello",
            "references": [{"audio_path": "a.wav"}, {"audio_path": "b.wav"}],
        }
    )
    with pytest.raises(ValueError, match="at most one"):
        build_auk_state(payload, context)


def speech_payload(**kwargs):
    from sglang_omni.client.client import Client
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import SpeechRequestValidator

    request = CreateSpeechRequest(**kwargs)
    service = SpeechRequestValidator(default_model="tencent/AuK")
    generated = service.build_generate_request(request)
    return StagePayload(
        request_id="speech", request=Client._build_omni_request(generated), data={}
    )


@pytest.mark.parametrize("description", ["warm, relaxed female voice", None])
def test_speech_instruct_tts_combines_instructions_and_input(context, description):
    payload = speech_payload(
        input="Welcome home.",
        instructions=description,
        stage_params={"auk_engine": {"gen_seconds": 3.01}},
    )
    state = AuKState.from_dict(preprocess_auk_payload(payload).data)
    voice = description or "A clear, natural voice."
    assert state.instruction == (
        f'Generate speech based on the following description: "{voice}". '
        'The content to speak is: "Welcome home.".'
    )
    assert state.gen_frames == 151


@pytest.mark.parametrize(
    "structured", [False, True], ids=["legacy-reference", "structured-reference"]
)
def test_reference_speech_duration(context, reference, structured):
    kwargs = (
        {"references": [{"audio_path": reference, "text": "你好"}]}
        if structured
        else {"ref_audio": reference, "ref_text": "你好"}
    )
    state = build_auk_state(speech_payload(input="Hello world!", **kwargs), context)
    assert state.instruction == 'Say the following with the same voice: "Hello world!"'
    assert state.gen_frames == 101
    assert state.ref_seconds == pytest.approx(1.01)
    assert state.ref_audio.dtype == np.float32

    payload = speech_payload(
        input="Hello world!", **kwargs, stage_params={"auk_engine": {"gen_seconds": 3}}
    )
    assert build_auk_state(payload, context).gen_frames == 150


def test_reference_speech_requires_duration_or_transcript(context, reference):
    with pytest.raises(ValueError, match="gen_seconds"):
        preprocess_auk_payload(
            speech_payload(input="Welcome home.", ref_audio=reference)
        )


def test_generate_preserves_raw_editing_instruction(context, reference):
    from sglang_omni.client.client import Client
    from sglang_omni.serve.openai_api import _build_rollout_generate_request
    from sglang_omni.serve.protocol import RolloutGenerateRequest

    request = RolloutGenerateRequest(
        prompt="Remove the background noise.",
        metadata={"tts_params": {"ref_audio": reference}},
        output_modalities=["audio"],
    )
    generated = _build_rollout_generate_request(request)
    payload = StagePayload(
        request_id="editing", request=Client._build_omni_request(generated), data={}
    )
    state = AuKState.from_dict(preprocess_auk_payload(payload).data)
    assert state.instruction == request.prompt
    assert state.gen_frames == 50
