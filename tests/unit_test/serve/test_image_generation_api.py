# SPDX-License-Identifier: Apache-2.0
"""Image API validation, metadata forwarding, and response format."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Literal, NoReturn, TypedDict

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import ClientError, GenerateRequest
from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    LLaDA2ImageDecoderStageConfig,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.protocol import ImageGenerationParams


class DecodePayload(TypedDict, total=False):
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class ImageDecodePayload(TypedDict):
    image: str


class PipelineResult(TypedDict, total=False):
    decode: DecodePayload
    image_decode: ImageDecodePayload
    image: str


class RecordingCoordinator:
    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []

    async def submit(self, request_id: str, request: GenerateRequest) -> PipelineResult:
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

    async def stream(
        self, request_id: str, request: GenerateRequest
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        raise AssertionError("Image requests must be rejected before streaming")
        yield


@pytest.fixture
def api() -> Iterator[tuple[TestClient, RecordingCoordinator]]:
    coordinator = RecordingCoordinator()
    app = create_app(
        Client(coordinator),
        model_name="served-model",
        supports_image_api=True,
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
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType

class NoDiffusion(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
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
        {"mode": "thinking"},
        {"decode_mode": "unknown"},
        {"decode_mode": "normal", "decoder_steps": 1},
    ],
)
def test_invalid_image_parameters(params: dict[str, int | float | str]) -> None:
    with pytest.raises(ValidationError):
        ImageGenerationParams(**params)


def test_image_input_without_generation_config_remains_chat(
    api: tuple[TestClient, RecordingCoordinator],
) -> None:
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": ""}], "images": ["input.png"]},
    )
    assert response.status_code == 200
    assert "image_generation" not in coordinator.requests[0].metadata


@pytest.mark.parametrize("editing", [False, True])
def test_native_image_routes_preserve_generation_controls(
    api: tuple[TestClient, RecordingCoordinator], editing: bool
) -> None:
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
    assert request.metadata["output_modalities"] == ["image"]
    assert request.params["stream"] is False
    assert "choices" not in response.json()
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


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize(
    "factory_mode,factory_steps,request_mode,steps,status",
    [
        ("normal", 50, None, 1, 400),
        ("decoder-turbo", 8, None, 1, 200),
        ("normal", 50, "decoder-turbo", 1, 200),
        ("decoder-turbo", 8, "normal", 1, 400),
        ("decoder-turbo", 8, None, None, 200),
        ("normal", 50, "decoder-turbo", None, 200),
        ("decoder-turbo", 1, "normal", None, 400),
        ("decoder-turbo", 1, "normal", 2, 200),
        ("decoder-turbo", 1, None, None, 200),
    ],
)
def test_image_steps_follow_factory_mode(
    editing: bool,
    factory_mode: Literal["normal", "decoder-turbo"],
    factory_steps: int,
    request_mode: Literal["normal", "decoder-turbo"] | None,
    steps: int | None,
    status: int,
) -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="test")
    stage = config.stage_named(IMAGE_DECODE_STAGE)
    assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
    stage.factory.decode_mode = factory_mode
    stage.factory.num_steps = factory_steps
    coordinator = RecordingCoordinator()
    app = create_app(
        Client(coordinator),
        supports_image_api=config.supports_image_api,
        image_decode_mode=config.image_decode_mode(),
        image_decode_steps=config.image_decode_steps(),
    )
    fields: dict[str, str | int] = {"prompt": "Make it red"}
    if request_mode is not None:
        fields["decode_mode"] = request_mode
    if steps is not None:
        fields["num_inference_steps"] = steps
    with TestClient(app) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data=fields,
                files={"image": ("source.png", b"png", "image/png")},
            )
        else:
            response = client.post("/v1/images/generations", json=fields)
    assert response.status_code == status, response.text
    if status == 400:
        assert not coordinator.requests
    else:
        options = coordinator.requests[0].metadata["image_generation"]
        assert options.get("decode_mode") == request_mode
        assert ("decode_mode" in options) == (request_mode is not None)
        assert options.get("decoder_steps") == steps
        assert ("decoder_steps" in options) == (steps is not None)


def test_native_image_rejects_unsupported_controls_and_multiple_sources(
    api: tuple[TestClient, RecordingCoordinator],
) -> None:
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


@pytest.mark.parametrize(
    ("editing", "client_error", "message", "status"),
    [
        (
            False,
            True,
            "Requested token count exceeds the model's maximum context length",
            400,
        ),
        (
            True,
            False,
            "Request requires more tokens than the thinker KV cache can hold",
            400,
        ),
        (False, False, "Image decoder failed", 500),
        (True, True, "Image decoder failed", 500),
        (False, True, QueueFullError.MESSAGE, 503),
        (True, False, QueueFullError.MESSAGE, 503),
    ],
)
def test_native_image_pipeline_errors(
    api: tuple[TestClient, RecordingCoordinator],
    monkeypatch: pytest.MonkeyPatch,
    editing: bool,
    client_error: bool,
    message: str,
    status: int,
) -> None:
    client, coordinator = api

    async def submit(request_id: str, request: GenerateRequest) -> NoReturn:
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
