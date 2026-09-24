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
from sglang_omni.config import FactoryArgs
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from sglang_omni.models.sensenova_u1.stages import (
    generate_image,
    generate_images,
    image_generation_batch_key,
    image_generation_request_cost,
)
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


def _decode_result_pixel(payload):
    image = Image.open(io.BytesIO(base64.b64decode(payload.data["image_b64"])))
    return image.getpixel((0, 0))


def test_t2i_stage_batches_compatible_requests_with_per_request_seeds():
    model = Mock()
    model.t2i_generate.return_value = torch.stack(
        [torch.full((3, 32, 64), -1.0), torch.full((3, 32, 64), 1.0)]
    )
    payloads = [
        StagePayload(
            request_id=f"test-{index}",
            request=OmniRequest(
                inputs=prompt,
                params={
                    "width": 64,
                    "height": 32,
                    "num_inference_steps": 12,
                    "guidance_scale": 2.5,
                    "seed": seed,
                },
            ),
            data=None,
        )
        for index, (prompt, seed) in enumerate([("a blue cat", 7), ("a red dog", 19)])
    ]

    results = generate_images(payloads, model, tokenizer=object())

    args = model.t2i_generate.call_args.args
    kwargs = model.t2i_generate.call_args.kwargs
    assert args[1] == ["a blue cat", "a red dog"]
    assert kwargs["batch_size"] == 2
    assert kwargs["seed"] == [7, 19]
    assert kwargs["image_size"] == (64, 32)
    assert kwargs["num_steps"] == 12
    assert [_decode_result_pixel(result) for result in results] == [
        (0, 0, 0),
        (255, 255, 255),
    ]


def test_t2i_stage_splits_incompatible_requests_and_restores_order():
    def t2i_generate(_tokenizer, prompt, *, image_size, batch_size, **_kwargs):
        if isinstance(prompt, list):
            assert prompt == ["first", "third"]
            values = (-1.0, 0.0)
        else:
            assert prompt == "second"
            values = (1.0,)
        width, height = image_size
        assert len(values) == batch_size
        return torch.stack([torch.full((3, height, width), value) for value in values])

    model = Mock()
    model.t2i_generate.side_effect = t2i_generate
    payloads = [
        StagePayload(
            request_id=f"test-{index}",
            request=OmniRequest(
                inputs=prompt,
                params={"width": width, "height": 32, "seed": index},
            ),
            data=None,
        )
        for index, (prompt, width) in enumerate(
            [("first", 64), ("second", 32), ("third", 64)]
        )
    ]

    results = generate_images(payloads, model, tokenizer=object())

    assert model.t2i_generate.call_count == 2
    assert [result.request_id for result in results] == [
        "test-0",
        "test-1",
        "test-2",
    ]
    assert [_decode_result_pixel(result) for result in results] == [
        (0, 0, 0),
        (255, 255, 255),
        (127, 127, 127),
    ]


def test_image_generation_request_cost_counts_pixels_steps_and_cfg():
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs="a cat",
            params={
                "width": 64,
                "height": 32,
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
            },
        ),
        data=None,
    )

    assert image_generation_request_cost(payload) == 64 * 32 * 12 * 2


def test_image_generation_batch_key_separates_parameters_and_image_edits():
    def payload(request_id, inputs, **params):
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=inputs, params=params),
            data=None,
        )

    first = payload("first", "a cat", width=64, height=32, seed=7)
    same = payload("same", "a dog", width=64, height=32, seed=19)
    different = payload("different", "a dog", width=32, height=32, seed=19)
    edit = payload("edit", {"task": "image_edit"})

    assert image_generation_batch_key(first) == image_generation_batch_key(same)
    assert image_generation_batch_key(first) != image_generation_batch_key(different)
    assert image_generation_batch_key(edit) == ("image_edit", "edit")


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
                "do_resize": False,
                "size_explicit": True,
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
    assert kwargs["timestep_shift"] == 3.0
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


def test_i2i_stage_preserves_reference_aspect_ratio_without_explicit_size():
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    model = Mock()

    def generate(*_args, **kwargs):
        width, height = kwargs["image_size"]
        return torch.zeros((1, 3, height, width))

    model.it2i_generate.side_effect = generate
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": [base64.b64encode(reference.getvalue()).decode("ascii")],
            },
            params={
                "width": 256,
                "height": 256,
                "do_resize": False,
                "size_explicit": False,
            },
        ),
        data=None,
    )

    result = generate_image(payload, model, tokenizer=object())

    assert model.it2i_generate.call_args.kwargs["image_size"] == (384, 192)
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.size == (384, 192)


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
            io.BytesIO(base64.b64decode(request.prompt["image_b64"][0]))
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
            "size_explicit": True,
        }
        assert request.output_modalities == ["image"]
        buffer = io.BytesIO()
        Image.new("RGB", (64, 32), (0, 0, 255)).save(buffer, format="PNG")
        yield GenerateChunk(
            request_id=request_id,
            image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )


class _DefaultImageEditClient:
    def __init__(self, expected_images=1):
        self.expected_images = expected_images

    async def generate(self, request, request_id):
        assert len(request.prompt["image_b64"]) == self.expected_images
        assert request.extra_params == {
            "width": 2048,
            "height": 2048,
            "num_inference_steps": 50,
            "guidance_scale": 4.0,
            "img_cfg_scale": 1.0,
            "seed": 42,
            "n": 1,
            "size_explicit": False,
        }
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (0, 0, 255)).save(buffer, format="PNG")
        yield GenerateChunk(
            request_id=request_id,
            image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )


class _MultipleOutputImageEditClient:
    def __init__(self):
        self.seeds = []

    async def generate(self, request, request_id):
        assert request.extra_params["n"] == 1
        self.seeds.append(request.extra_params["seed"])
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (self.seeds[-1], 0, 0)).save(buffer, format="PNG")
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


def test_image_edit_api_uses_sglang_defaults_when_omitted():
    app = create_app(
        _DefaultImageEditClient(),
        model_name="sensenova",
        architectures=["NEOChatModel"],
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")

    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={"prompt": "edit this"},
    )

    assert response.status_code == 200


def test_image_edit_api_accepts_multiple_reference_images():
    app = create_app(
        _DefaultImageEditClient(expected_images=2),
        model_name="sensenova",
        architectures=["NEOChatModel"],
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    image_file = ("reference.png", reference.getvalue(), "image/png")

    response = client.post(
        "/v1/images/edits",
        files=[("image", image_file), ("image", image_file)],
        data={"prompt": "edit this"},
    )

    assert response.status_code == 200


def test_image_edit_api_expands_n_with_sequential_seeds():
    image_client = _MultipleOutputImageEditClient()
    app = create_app(
        image_client,
        model_name="sensenova",
        architectures=["NEOChatModel"],
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")

    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={"prompt": "edit this", "n": 3, "seed": 7},
    )

    assert response.status_code == 200
    assert len(response.json()["data"]) == 3
    assert image_client.seeds == [7, 8, 9]


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


def test_config_registers_checkpoint_architecture(tmp_path):
    assert SenseNovaU1PipelineConfig.architecture == "NEOChatModel"
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    assert config.stages[0].terminal
    factory = FactoryArgs(max_batch_size=2, max_batch_cost=41943040)
    assert factory.max_batch_cost == 41943040
    assert factory.model_extra in (None, {})
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
