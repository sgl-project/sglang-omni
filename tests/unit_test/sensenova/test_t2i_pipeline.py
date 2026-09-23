# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from sglang_omni.models.sensenova_u1.stages import generate_image
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import create_app


def test_t2i_stage_encodes_real_png_and_passes_source_arguments():
    model = Mock()
    model.t2i_generate.return_value = torch.zeros((1, 3, 32, 64))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs="a blue cat",
            params={"width": 64, "height": 32, "seed": 7},
        ),
        data=None,
    )

    result = generate_image(payload, model, tokenizer=object())

    kwargs = model.t2i_generate.call_args.kwargs
    assert kwargs["image_size"] == (64, 32)
    assert kwargs["seed"] == 7
    assert kwargs["think_mode"] is False
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_t2i_stage_rejects_invalid_output():
    model = Mock()
    model.t2i_generate.return_value = torch.zeros((1, 3, 16, 16))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(inputs="a cat", params={"width": 32, "height": 32}),
        data=None,
    )
    try:
        generate_image(payload, model, tokenizer=object())
    except ValueError as exc:
        assert "invalid image tensor" in str(exc)
    else:
        raise AssertionError("Invalid model output was accepted")


def test_i2i_stage_decodes_reference_and_passes_source_arguments():
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    model = Mock()
    model.it2i_generate.return_value = torch.zeros((1, 3, 32, 64))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "make the background blue",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            params={
                "width": 64,
                "height": 32,
                "num_inference_steps": 12,
                "guidance_scale": 2.5,
                "img_cfg_scale": 1.5,
                "seed": 7,
            },
        ),
        data=None,
    )

    result = generate_image(payload, model, tokenizer=object())

    args = model.it2i_generate.call_args.args
    kwargs = model.it2i_generate.call_args.kwargs
    assert args[1] == "make the background blue"
    assert len(args[2]) == 1
    assert args[2][0].size == (16, 8)
    assert kwargs["image_size"] == (64, 32)
    assert kwargs["num_steps"] == 12
    assert kwargs["cfg_scale"] == 2.5
    assert kwargs["img_cfg_scale"] == 1.5
    assert kwargs["seed"] == 7
    assert kwargs["think_mode"] is False
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_i2i_stage_rejects_invalid_reference_image():
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": "not-base64",
            }
        ),
        data=None,
    )

    try:
        generate_image(payload, Mock(), tokenizer=object())
    except ValueError as exc:
        assert "invalid reference image" in str(exc)
    else:
        raise AssertionError("Invalid reference image was accepted")


class _ImageClient:
    async def generate(self, request, request_id):
        assert request.extra_params["width"] == 32
        assert request.output_modalities == ["image"]
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (0, 0, 255)).save(buffer, format="PNG")
        yield GenerateChunk(
            request_id=request_id,
            image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )


class _ImageEditClient:
    async def generate(self, request, request_id):
        assert request.prompt["task"] == "image_edit"
        assert request.prompt["prompt"] == "make the background blue"
        reference = Image.open(
            io.BytesIO(base64.b64decode(request.prompt["image_b64"]))
        )
        assert reference.size == (16, 8)
        assert request.extra_params == {
            "width": 64,
            "height": 32,
            "num_inference_steps": 12,
            "guidance_scale": 2.5,
            "img_cfg_scale": 1.5,
            "seed": 7,
            "n": 1,
        }
        assert request.output_modalities == ["image"]
        buffer = io.BytesIO()
        Image.new("RGB", (64, 32), (0, 0, 255)).save(buffer, format="PNG")
        yield GenerateChunk(
            request_id=request_id,
            image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )


def test_image_api_exposes_image_only_for_sensenova():
    app = create_app(
        _ImageClient(), model_name="sensenova", architectures=["NEOChatModel"]
    )
    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        json={"model": "sensenova", "prompt": "a cat", "size": "32x32"},
    )
    assert response.status_code == 200
    image_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
    image = Image.open(io.BytesIO(image_bytes))
    assert image.format == "PNG"
    assert image.size == (32, 32)
    invalid = client.post(
        "/v1/images/generations", json={"prompt": "a cat", "size": "31x32"}
    )
    assert invalid.status_code == 400

    other = TestClient(create_app(_ImageClient(), architectures=["OtherModel"]))
    assert (
        other.post("/v1/images/generations", json={"prompt": "a cat"}).status_code
        == 404
    )
    assert (
        other.post(
            "/v1/images/edits",
            files={"image": ("reference.png", b"not used", "image/png")},
            data={"prompt": "edit this"},
        ).status_code
        == 404
    )


def test_image_edit_api_accepts_one_reference_image():
    app = create_app(
        _ImageEditClient(), model_name="sensenova", architectures=["NEOChatModel"]
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")

    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={
            "model": "sensenova",
            "prompt": "make the background blue",
            "size": "64x32",
            "seed": "7",
            "num_inference_steps": "12",
            "guidance_scale": "2.5",
            "img_cfg_scale": "1.5",
        },
    )

    assert response.status_code == 200
    image_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
    image = Image.open(io.BytesIO(image_bytes))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_image_edit_api_rejects_unsupported_or_invalid_inputs():
    app = create_app(
        _ImageEditClient(), model_name="sensenova", architectures=["NEOChatModel"]
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    image_file = ("reference.png", reference.getvalue(), "image/png")

    bad_image = client.post(
        "/v1/images/edits",
        files={"image": ("bad.png", b"not an image", "image/png")},
        data={"prompt": "edit this"},
    )
    assert bad_image.status_code == 400
    invalid_size = client.post(
        "/v1/images/edits",
        files={"image": image_file},
        data={"prompt": "edit this", "size": "31x32"},
    )
    assert invalid_size.status_code == 400
    with_mask = client.post(
        "/v1/images/edits",
        files={"image": image_file, "mask": image_file},
        data={"prompt": "edit this"},
    )
    assert with_mask.status_code == 400
    multiple_images = client.post(
        "/v1/images/edits",
        files=[("image", image_file), ("image", image_file)],
        data={"prompt": "edit this"},
    )
    assert multiple_images.status_code == 400


def test_config_registers_checkpoint_architecture(tmp_path):
    assert SenseNovaU1PipelineConfig.architecture == "NEOChatModel"
    assert SenseNovaU1PipelineConfig(model_path="/model/sensenova").stages[0].terminal
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "neo_chat", "architectures": ["NEOChatModel"]})
    )
    assert resolve_config_cls_for_model_path(str(tmp_path)) is SenseNovaU1PipelineConfig


def test_client_preserves_generated_image():
    chunk = Client._default_result_builder(
        "test", {"image_b64": "cG5n", "modality": "image"}
    )
    assert chunk.image_b64 == "cG5n"
    assert chunk.modality == "image"
