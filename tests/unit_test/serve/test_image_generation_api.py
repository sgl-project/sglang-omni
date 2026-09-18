# SPDX-License-Identifier: Apache-2.0
"""Image API validation, metadata forwarding, and PR3 response format."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.client import Client
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest, ImageGenerationParams


class RecordingCoordinator:
    def __init__(self):
        self.requests = []

    async def submit(self, request_id, request):
        self.requests.append(request)
        return {
            "decode": {
                "text": "A red square",
                "finish_reason": "length",
                "prompt_tokens": 3,
                "completion_tokens": 2,
            },
            "image_decode": {"image": "cG5n"},
        }

    async def stream(self, request_id, request):
        raise AssertionError("Image requests must be rejected before streaming")
        yield


@pytest.fixture
def api():
    coordinator = RecordingCoordinator()
    app = create_app(Client(coordinator), model_name="llada2-uni")
    with TestClient(app) as client:
        yield client, coordinator


@pytest.mark.parametrize(
    "field", ["cfg_scale", "cfg_text_scale", "cfg_image_scale", "cfg_rescale"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cfg_must_be_finite(field, value):
    with pytest.raises(ValidationError):
        ImageGenerationParams(**{field: value})


@pytest.mark.parametrize(
    "field", ["cfg_scale", "cfg_text_scale", "cfg_image_scale", "cfg_rescale"]
)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_http_rejects_nonfinite_cfg_without_dispatch(api, field, value):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "image_generation": {field: value},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "image_generation", field]
    assert coordinator.requests == []


@pytest.mark.parametrize(
    "params",
    [
        {"cfg_scale": 0.9},
        {"cfg_text_scale": -0.1},
        {"cfg_image_scale": -0.1},
        {"cfg_rescale": -0.1},
        {"cfg_rescale": 1.1},
        {"decoder_steps": 0},
        {"dllm_steps": 0},
        {"image_h": 31},
        {"image_w": 33},
        {"mode": "interleaved"},
        {"decode_mode": "unknown"},
    ],
)
def test_invalid_image_parameters(params):
    with pytest.raises(ValidationError):
        ImageGenerationParams(**params)


def test_source_image_tokens_are_validated_and_forwarded():
    config = {
        "source_image_tokens": {
            "token_ids": [1, 2, 3, 4],
            "grid_thw": [1, 2, 2],
        }
    }
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Make it red"}],
        image_generation=config,
    )

    assert _build_chat_generate_request(request).metadata["image_generation"] == {
        "source_image_tokens": {
            "token_ids": [1, 2, 3, 4],
            "grid_thw": (1, 2, 2),
        }
    }
    with pytest.raises(ValidationError):
        ImageGenerationParams(
            source_image_tokens={
                "token_ids": [1, 2, 3],
                "grid_thw": [1, 2, 2],
            }
        )


@pytest.mark.parametrize(
    "modalities", [None, [], ["text"], ["image"], ["text", "image"]]
)
@pytest.mark.parametrize(
    "image_config", [{}, {"cfg_scale": 1.0}, {"cfg_text_scale": 0.0}]
)
def test_image_config_and_modalities_reach_omni_request(modalities, image_config):
    req = ChatCompletionRequest(
        model="llada2-uni",
        messages=[{"role": "user", "content": "Make it red"}],
        images=["input.png"],
        image_generation=image_config,
        modalities=modalities,
        temperature=1.0,
        max_completion_tokens=64,
        stage_sampling={"thinker": {"temperature": 0.5, "max_new_tokens": 32}},
        stage_params={"thinker": {"return_logprob": True}},
    )
    generate = _build_chat_generate_request(req)
    omni = Client._build_omni_request(generate)
    expected_modalities = ["text"] if modalities is None else modalities
    assert generate.output_modalities == expected_modalities
    assert omni.metadata["output_modalities"] == expected_modalities
    assert omni.metadata["image_generation"] == image_config
    assert omni.metadata["model"] == "llada2-uni"
    assert omni.inputs["images"] == ["input.png"]
    assert omni.inputs["messages"] == [{"role": "user", "content": "Make it red"}]
    assert omni.params["stream"] is False
    assert omni.params["max_new_tokens"] == 64
    assert omni.params["stage_sampling"]["thinker"]["temperature"] == 0.5
    assert omni.params["stage_sampling"]["thinker"]["max_new_tokens"] == 32
    assert omni.params["stage_params"] == req.stage_params
    assert "temperature" in omni.metadata[EXPLICIT_GENERATION_PARAMS_KEY]


def test_all_image_controls_preserve_explicit_values():
    config = {
        "mode": "thinking",
        "decode_mode": "decoder-turbo",
        "decoder_steps": 1,
        "seed": 0,
        "cfg_scale": 1.0,
        "cfg_text_scale": 0.0,
        "cfg_image_scale": 0.0,
        "cfg_rescale": 0.0,
        "image_h": 32,
        "image_w": 1024,
        "dllm_steps": 1,
    }
    req = ChatCompletionRequest(messages=[], image_generation=config)
    assert _build_chat_generate_request(req).metadata["image_generation"] == config
    req = ChatCompletionRequest(messages=[], image_generation={"cfg_text_scale": None})
    assert _build_chat_generate_request(req).metadata["image_generation"] == {}
    assert (
        "image_generation"
        not in _build_chat_generate_request(ChatCompletionRequest(messages=[])).metadata
    )


@pytest.mark.parametrize("modalities", [None, ["text"], ["image"], ["text", "image"]])
def test_chat_image_response_keeps_pr3_format_and_modality_choices(api, modalities):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw a square"}],
            "image_generation": {},
            "modalities": modalities,
        },
    )
    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]
    if modalities and "image" in modalities:
        assert message["image"] == {"data": "cG5n", "format": "png"}
    else:
        assert "image" not in message
    if modalities == ["image"]:
        assert "content" not in message
    else:
        assert message["content"] == "A red square"
    assert "audio" not in message
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert len(coordinator.requests) == 1
    assert coordinator.requests[0].metadata["image_generation"] == {}


@pytest.mark.parametrize(
    "extra",
    [
        {"image_generation": {}},
        {"modalities": ["image"]},
        {"modalities": ["text", "image"]},
        {"image_generation": {}, "modalities": ["text"]},
    ],
)
def test_image_streaming_is_rejected_before_dispatch(api, extra):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "stream": True,
            **extra,
        },
    )
    assert response.status_code == 400
    assert "stream=false" in response.json()["detail"]
    assert response.headers["content-type"] == "application/json"
    assert coordinator.requests == []


@pytest.mark.parametrize(
    "image_part",
    [
        {"type": "image_url", "image_url": {"url": "input.png"}},
        {"type": "image_url", "image_url": "input.png"},
        {"type": "image", "image": "input.png"},
        None,
    ],
)
@pytest.mark.parametrize("instruction", ["", "Make it red"])
def test_edit_requires_latest_user_instruction(api, image_part, instruction):
    client, coordinator = api
    body = {
        "messages": [
            {"role": "user", "content": "Earlier instruction must not count"},
            {"role": "assistant", "content": "Assistant text must not count"},
            {
                "role": "user",
                "content": [
                    *([image_part] if image_part else []),
                    {"type": "text", "text": instruction},
                ],
            },
        ],
        "image_generation": {},
        "modalities": ["image"],
    }
    if image_part is None:
        body["images"] = ["input.png"]
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == (200 if instruction else 400)
    assert len(coordinator.requests) == bool(instruction)
    if not instruction:
        assert (
            response.json()["detail"]
            == "Image editing requires a non-empty instruction"
        )


@pytest.mark.parametrize("content", ["  ", None, [{"text": None}], [{"text": 123}]])
def test_empty_or_malformed_edit_text_is_not_a_server_error(api, content):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": content}],
            "images": ["input.png"],
            "image_generation": {},
        },
    )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_image_input_without_generation_config_remains_chat(api):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": ""}], "images": ["input.png"]},
    )
    assert response.status_code == 200
    assert "image_generation" not in coordinator.requests[0].metadata
