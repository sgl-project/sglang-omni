# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from sglang_omni.models.breeze_tts.frontend import text_segments
from sglang_omni.models.breeze_tts.request import parse_request
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_service import _build_sampling_params, _build_tts_params


def make_payload(*, text="Hello", tts=None, params=None):
    return StagePayload(
        request_id="test",
        data={},
        request=OmniRequest(
            inputs=text,
            params=params or {},
            metadata={"tts_params": {"instructions": "A gentle voice", **(tts or {})}},
        ),
    )


def test_endpoint_defaults_do_not_replace_model_defaults():
    public = CreateSpeechRequest(
        input="Hello", instructions="Warm voice", cfg_scale=4, seed=42
    )
    tts = _build_tts_params(public)
    sampling = _build_sampling_params(public)
    request = parse_request(
        make_payload(
            tts=tts,
            params={
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "repetition_penalty": sampling.repetition_penalty,
            },
        )
    )
    assert request.sampling.temperature == 0.9  # not Fish's 0.8
    assert request.sampling.top_p == 1.0
    assert request.sampling.top_k == 50
    assert request.sampling.cfg_scale == 4
    assert request.sampling.seed == 42


def test_explicit_sampling_and_generation_limit_survive_endpoint():
    public = CreateSpeechRequest(
        input="Hello",
        instructions="Warm voice",
        temperature=0,
        top_k=1,
        top_p=0.5,
        max_new_tokens=4,
    )
    request = parse_request(
        make_payload(
            tts=_build_tts_params(public),
            params={
                "temperature": 0,
                "top_k": 1,
                "top_p": 0.5,
                "max_new_tokens": 4,
            },
        )
    )
    assert request.sampling.temperature == 0
    assert request.sampling.top_k == 1
    assert request.sampling.top_p == 0.5
    assert request.sampling.max_new_tokens == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("cfg_scale", -1),
        ("cfg_scale", float("nan")),
        ("cfg_scale", float("inf")),
        ("seed", -1),
        ("seed", True),
        ("seed", 2**64),
        ("temperature", -0.1),
        ("top_p", 0),
        ("repetition_penalty", 0),
        ("max_new_tokens", 0),
        ("max_new_tokens", 751),
        ("top_k", 2049),
    ],
)
def test_invalid_generation_controls_fail_before_encoding(field, value):
    with pytest.raises(ValueError, match="Breeze"):
        parse_request(
            make_payload(
                tts={field: value, "explicit_generation_params": [field]},
                params={field: value},
            )
        )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_cfg_scale_http_validation(value):
    with pytest.raises(ValidationError):
        CreateSpeechRequest(input="Hello", cfg_scale=value)


def test_cfg_branches_keep_reference_but_remove_only_instruction():
    request = parse_request(
        make_payload(
            tts={
                "ref_audio": "reference.wav",
                "ref_text": "Reference text",
                "instructions": "Whisper",
            }
        )
    )
    assert text_segments(request) == [
        "[S0]Reference text",
        "[S0]<ins_bos>Whisper<ins_eos>Hello",
    ]
    assert text_segments(request, negative=True) == ["[S0]Reference text", "[S0]Hello"]


@pytest.mark.parametrize(
    "tts,inputs",
    [
        ({"ref_audio": "reference.wav"}, "Hello"),
        ({"ref_text": "orphan"}, "Hello"),
        ({"instructions": ""}, "Hello"),
        (
            {},
            {
                "text": "Hello",
                "references": [
                    {"audio_path": "a", "text": "a"},
                    {"audio_path": "b", "text": "b"},
                ],
            },
        ),
        ({}, {"text": "Hello", "references": [{"vq_codes": [[1]], "text": "a"}]}),
    ],
)
def test_invalid_reference_contract(tts, inputs):
    with pytest.raises(ValueError):
        parse_request(make_payload(text=inputs, tts=tts))


def test_clone_without_instructions_and_inline_reference():
    payload = make_payload(
        text={
            "text": "你好",
            "references": [{"data": "YWJj", "media_type": "audio/wav", "text": "参考"}],
        },
        tts={"instructions": None},
    )
    request = parse_request(payload)
    assert request.ref_audio == "data:audio/wav;base64,YWJj"
    assert text_segments(request) == ["[S0]参考", "[S0]你好"]
