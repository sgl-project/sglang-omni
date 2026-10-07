# SPDX-License-Identifier: Apache-2.0
"""Image API validation, metadata forwarding, and response format."""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import ClientError
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import build_chat_generate_request
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
    app = create_app(
        Client(coordinator), model_name="llada2-uni", supports_image_api=True
    )
    with TestClient(app) as client:
        yield client, coordinator


def test_non_image_app_does_not_import_diffusion() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoDiffusion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang.multimodal_gen" or fullname.startswith("sglang.multimodal_gen."):
            raise ModuleNotFoundError("Diffusion is not installed", name=fullname)
        else:
            return None

sys.meta_path.insert(0, NoDiffusion())
from fastapi.testclient import TestClient
from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app

app = create_app(Client(None), model_name="test", supports_image_api=False)
with TestClient(app) as client:
    assert client.get("/v1/models").status_code == 200
    assert client.post("/v1/images/generations", json={"prompt": "Draw"}).status_code == 404
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    generate = build_chat_generate_request(req)
    omni = Client.build_omni_request(generate)
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
    assert build_chat_generate_request(req).metadata["image_generation"] == config
    req = ChatCompletionRequest(messages=[], image_generation={"cfg_text_scale": None})
    assert build_chat_generate_request(req).metadata["image_generation"] == {}
    assert (
        "image_generation"
        not in build_chat_generate_request(ChatCompletionRequest(messages=[])).metadata
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


@pytest.mark.parametrize("editing", [False, True])
def test_native_image_routes_preserve_generation_controls(api, editing):
    client, coordinator = api
    fields = {
        "prompt": "Make it red",
        "seed": 0,
        "guidance_scale": 4.0,
        "num_inference_steps": 8,
        "dllm_steps": 32,
        "decode_mode": "decoder-turbo",
        "response_format": "b64_json",
    }
    if editing:
        response = client.post(
            "/v1/images/edits",
            data=fields,
            files={"image[]": ("source.png", b"png", "image/png")},
        )
    else:
        response = client.post(
            "/v1/images/generations", json={**fields, "size": "1024x768"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["b64_json"] == "cG5n"
    request = coordinator.requests[0]
    options = request.metadata["image_generation"]
    assert options["seed"] == 0
    assert options["cfg_scale"] == 4.0
    assert options["decoder_steps"] == 8
    assert options["dllm_steps"] == 32
    if editing:
        assert request.inputs["images"] == ["data:image/png;base64,cG5n"]
        assert "image_w" not in options
    else:
        assert (options["image_w"], options["image_h"]) == (1024, 768)


def test_native_image_rejects_unsupported_controls_and_multiple_sources(api):
    client, coordinator = api
    for options in ({"n": 2}, {"width": 0, "height": 1024}, {"mask": "mask.png"}):
        response = client.post(
            "/v1/images/generations", json={"prompt": "Draw", **options}
        )
        assert response.status_code == 400
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "Edit"},
        files=[
            ("image", ("a.png", b"png", "image/png")),
            ("image[]", ("b.png", b"png", "image/png")),
        ],
    )
    assert response.status_code == 400
    assert coordinator.requests == []


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("client_error", [False, True])
@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("Requested token count exceeds the model's maximum context length", 400),
        ("Request requires more tokens than the thinker KV cache can hold", 400),
        ("Image decoder failed", 500),
        (QueueFullError.MESSAGE, 503),
    ],
)
def test_native_image_pipeline_errors(
    api, monkeypatch, editing, client_error, message, status
):
    client, coordinator = api

    async def submit(request_id, request):
        if client_error:
            raise ClientError(message)
        else:
            raise QueueFullError.from_message(message)

    monkeypatch.setattr(coordinator, "submit", submit)
    if editing:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "Make it red"},
            files={"image": ("source.png", b"png", "image/png")},
        )
    else:
        response = client.post("/v1/images/generations", json={"prompt": "Draw"})
    assert response.status_code == status
    assert response.json()["detail"] == message


def test_interleaved_chat_exposes_cosmos_segments():
    class InterleavedCoordinator(RecordingCoordinator):
        async def submit(self, request_id, request):
            return {
                "modality": "interleaved",
                "finish_reason": "stop",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "image_ref", "image_id": "frame-0"},
                    {"type": "text", "text": "Last"},
                ],
                "images": [
                    {
                        "id": "frame-0",
                        "data": "cG5n",
                        "format": "png",
                        "width": 32,
                        "height": 32,
                    }
                ],
            }

    with TestClient(create_app(Client(InterleavedCoordinator()))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Draw a story"}],
                "modalities": ["text", "image"],
                "image_generation": {"mode": "interleaved"},
            },
        )
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "FirstLast"
    assert "images" not in message
    segments = message["segments"]
    assert [segment["segment_index"] for segment in segments] == [0, 1, 2]
    assert len({segment["session_id"] for segment in segments}) == 1
    assert [segment["kind"] for segment in segments] == ["text", "image", "text"]
    media = segments[1]["data"]
    payload = base64.b64decode(media["url"].split(",", 1)[1])
    assert media["mime_type"] == "image/png"
    assert media["size_bytes"] == len(payload)
    assert media["sha256"] == hashlib.sha256(payload).hexdigest()
