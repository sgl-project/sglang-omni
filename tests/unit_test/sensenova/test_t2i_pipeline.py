# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from transformers import PreTrainedTokenizerBase

import sglang_omni.models.sensenova_u1.generation as generation_module
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import (
    GenerateChunk,
    GenerateRequest,
    UMMGenerateChunk,
    UMMSegment,
)
from sglang_omni.config import FactoryArgs, ProcessConfig
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from sglang_omni.models.sensenova_u1.generation import (
    generate_image,
    generate_images,
    text_to_image_options,
)
from sglang_omni.models.sensenova_u1.image_processing import (
    decode_reference_images,
    prepare_reference_images,
    resolve_edit_output_size,
    resolve_reference_output_size,
)
from sglang_omni.models.sensenova_u1.images_api import register_images_api
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling
from sglang_omni.models.sensenova_u1.stages import (
    image_generation_batch_key,
    image_generation_request_cost,
    validate_generation_request,
)
from sglang_omni.models.sensenova_u1.tensor_parallel import (
    SenseNovaTensorParallelScheduler,
)
from sglang_omni.models.sensenova_u1.tensor_parallel_scheduler import (
    TensorParallelScheduler,
)
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.serve.openai_api import create_app, image_result_with_disconnect_watch
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    RecordingStageControlPlane,
)


def umm_image_chunk(request_id: str, encoded_images: list[str]) -> UMMGenerateChunk:
    """The final UMM result the SenseNova pipeline returns for these images."""
    return UMMGenerateChunk(
        request_id=request_id,
        segments=[
            UMMSegment(
                "execution", index, "image", InlineImage.from_png(image).model_dump()
            )
            for index, image in enumerate(encoded_images)
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [True, False])
async def test_image_result_cancellation_aborts_backend_and_closes_stream(disconnect):
    entered = asyncio.Event()
    closed = asyncio.Event()
    aborted = []

    class Request:
        async def is_disconnected(self):
            return disconnect and entered.is_set()

    class ImageClient:
        async def generate(self, request, request_id):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield None
            finally:
                closed.set()

        async def abort(self, request_id):
            aborted.append(request_id)

    task = asyncio.create_task(
        image_result_with_disconnect_watch(Request(), ImageClient(), object(), "req")
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    if not disconnect:
        task.cancel()
    else:
        pass
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert aborted == ["req"]
    assert closed.is_set()


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return create_autospec(
        PreTrainedTokenizerBase, instance=True, return_value={"input_ids": [1, 2, 3]}
    )


def test_t2i_api_returns_all_outputs_in_one_request():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    class MultiClient:
        calls = 0

        async def generate(self, request, request_id):
            self.calls += 1
            assert request.extra_params["n"] == 3
            yield umm_image_chunk(request_id, [encoded] * 3)

    backend = MultiClient()
    client = TestClient(
        create_app(
            backend, model_name="sensenova", image_api_registrar=register_images_api
        )
    )
    response = client.post("/v1/images/generations", json={"prompt": "a cat", "n": 3})
    assert response.status_code == 200
    assert response.json()["data"] == [{"b64_json": encoded}] * 3
    assert backend.calls == 1
    assert (
        client.post(
            "/v1/images/generations", json={"prompt": "a cat", "n": 11}
        ).status_code
        == 422
    )


def test_t2i_stage_encodes_real_png_and_passes_source_arguments(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
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

    result = generate_image(payload, model, tokenizer=tokenizer)

    kwargs = model.t2i_generate.call_args.kwargs
    assert kwargs["image_size"] == (64, 32)
    assert kwargs["seed"] == 7
    assert kwargs["think_mode"] is False
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_t2i_stage_rejects_invalid_output(tokenizer: PreTrainedTokenizerBase) -> None:
    model = Mock()
    model.t2i_generate.return_value = torch.zeros((1, 3, 16, 16))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(inputs="a cat", params={"width": 32, "height": 32}),
        data=None,
    )
    try:
        generate_image(payload, model, tokenizer=tokenizer)
    except ValueError as exc:
        assert "invalid image tensor" in str(exc)
    else:
        raise AssertionError("Invalid model output was accepted")


def decode_result_pixel(payload):
    image = Image.open(io.BytesIO(base64.b64decode(payload.data["image_b64"])))
    return image.getpixel((0, 0))


def test_t2i_stage_batches_compatible_requests_with_per_request_seeds(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    tokenizer.side_effect = [{"input_ids": [1, 2, 3]}, {"input_ids": [1, 2, 3, 4, 5]}]
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
        for index, (prompt, seed) in enumerate(
            [("a blue cat", 7), ("a red dog on grass", 19)]
        )
    ]

    results = generate_images(payloads, model, tokenizer=tokenizer)

    args = model.t2i_generate.call_args.args
    kwargs = model.t2i_generate.call_args.kwargs
    assert args[1] == ["a blue cat", "a red dog on grass"]
    assert kwargs["batch_size"] == 2
    assert kwargs["seed"] == [7, 19]
    assert kwargs["image_size"] == (64, 32)
    assert kwargs["num_steps"] == 12
    assert [decode_result_pixel(result) for result in results] == [
        (0, 0, 0),
        (255, 255, 255),
    ]


def test_t2i_stage_splits_incompatible_requests_and_restores_order(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
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

    results = generate_images(payloads, model, tokenizer=tokenizer)

    assert model.t2i_generate.call_count == 2
    assert [result.request_id for result in results] == [
        "test-0",
        "test-1",
        "test-2",
    ]
    assert [decode_result_pixel(result) for result in results] == [
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


def test_i2i_stage_decodes_reference_and_passes_source_arguments(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    model = Mock()

    def edit_image(
        model_tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        references: list[Image.Image],
        *,
        batch_size: int,
        seed: int,
        image_size: tuple[int, int],
        cfg_scale: float,
        img_cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        enable_timestep_shift: bool,
        cfg_interval: tuple[float, float],
        num_steps: int,
        t_eps: float,
        think_mode: bool,
    ) -> torch.Tensor:
        return torch.zeros((1, 3, 32, 64))

    model.it2i_generate.side_effect = edit_image
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

    result = generate_image(payload, model, tokenizer=tokenizer)

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


def test_i2i_stage_rejects_invalid_reference_image(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
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
        generate_image(payload, Mock(), tokenizer=tokenizer)
    except ValueError as exc:
        assert "invalid reference image" in str(exc)
    else:
        raise AssertionError("Invalid reference image was accepted")


def test_i2i_stage_preserves_reference_aspect_ratio_without_explicit_size(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
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

    result = generate_image(payload, model, tokenizer=tokenizer)

    assert model.it2i_generate.call_args.kwargs["image_size"] == (384, 192)
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.size == (384, 192)


class ImageClient:
    async def generate(self, request, request_id):
        assert request.extra_params["width"] == 32
        assert request.output_modalities == ["image"]
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (0, 0, 255)).save(buffer, format="PNG")
        yield umm_image_chunk(
            request_id, [base64.b64encode(buffer.getvalue()).decode("ascii")]
        )


class ImageEditClient:
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
        yield umm_image_chunk(
            request_id, [base64.b64encode(buffer.getvalue()).decode("ascii")]
        )


class DefaultImageEditClient:
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
        yield umm_image_chunk(
            request_id, [base64.b64encode(buffer.getvalue()).decode("ascii")]
        )


class MultipleOutputImageEditClient:
    def __init__(self):
        self.seeds = []

    async def generate(self, request, request_id):
        assert request.extra_params["n"] == 1
        self.seeds.append(request.extra_params["seed"])
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (self.seeds[-1], 0, 0)).save(buffer, format="PNG")
        yield umm_image_chunk(
            request_id, [base64.b64encode(buffer.getvalue()).decode("ascii")]
        )


def test_image_api_exposes_image_only_for_sensenova():
    app = create_app(
        ImageClient(),
        model_name="sensenova",
        image_api_registrar=import_string(
            SenseNovaU1PipelineConfig.image_api_registrar
        ),
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

    other = TestClient(create_app(ImageClient(), architectures=["OtherModel"]))
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
        ImageEditClient(),
        model_name="sensenova",
        image_api_registrar=register_images_api,
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


@pytest.mark.parametrize(
    ("stage_error", "expected_status"),
    [
        (
            "SenseNova-U1 I2I with 3 reference images exceeds GPU memory; "
            "reduce references or lower generate.factory.dit_layerwise_resident_layers",
            400,
        ),
        (
            "SenseNova-U1 I2I with 3 reference images exceeds NPU memory; "
            "reduce references or lower input_max_pixels",
            400,
        ),
        ("SenseNova-U1 generation crashed", 500),
    ],
)
def test_image_edit_api_maps_stage_errors(stage_error: str, expected_status: int):
    """I2I references that exhausted GPU memory returned 500 once stage errors arrived as RuntimeError."""

    class FailingClient:
        async def generate(self, request, request_id):
            raise RuntimeError(stage_error)
            yield

    client = TestClient(
        create_app(
            FailingClient(),
            model_name="sensenova",
            image_api_registrar=register_images_api,
        )
    )
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")
    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={"prompt": "make the background blue"},
    )
    assert response.status_code == expected_status
    assert response.json()["detail"] == stage_error


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_image_edit_api_parses_cache_flag(raw: str, expected: bool) -> None:
    class CacheFlagClient:
        async def generate(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncIterator[GenerateChunk]:
            assert request.extra_params["enable_cache_dit"] is expected
            buffer = io.BytesIO()
            Image.new("RGB", (32, 32), (0, 0, 255)).save(buffer, format="PNG")
            yield umm_image_chunk(
                request_id, [base64.b64encode(buffer.getvalue()).decode("ascii")]
            )

    app = create_app(
        CacheFlagClient(),
        model_name="sensenova",
        image_api_registrar=register_images_api,
    )
    client = TestClient(app)
    reference = io.BytesIO()
    Image.new("RGB", (16, 8), (255, 0, 0)).save(reference, format="PNG")

    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={"prompt": "edit this", "enable_cache_dit": raw},
    )

    assert response.status_code == 200


def test_image_edit_api_uses_sglang_defaults_when_omitted():
    app = create_app(
        DefaultImageEditClient(),
        model_name="sensenova",
        image_api_registrar=register_images_api,
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


@pytest.mark.parametrize("boolean_name", ["do_resize", "enable_cache_dit"])
@pytest.mark.parametrize(
    "pixel_budget, boolean_text, expected_boolean, expected_status",
    [
        (262144, "false", False, 200),
        (262144, "true", True, 200),
        (None, "false", False, 200),
        (262143, "true", True, 400),
        (262144, "invalid", False, 400),
        (262144, "", False, 400),
        (262144, " ", False, 400),
    ],
)
def test_image_edit_api_forwards_boolean_controls(
    boolean_name: Literal["do_resize", "enable_cache_dit"],
    pixel_budget: int | None,
    boolean_text: str,
    expected_boolean: bool,
    expected_status: int,
) -> None:
    """Explicit empty boolean fields must not silently use model defaults."""

    class ResizeClient:
        calls: int = 0

        async def generate(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncIterator[GenerateChunk]:
            self.calls += 1
            assert request.extra_params[boolean_name] is expected_boolean
            assert request.extra_params.get("input_max_pixels") == pixel_budget
            yield umm_image_chunk(request_id, [encoded])

    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    encoded = base64.b64encode(reference.getvalue()).decode("ascii")
    backend = ResizeClient()
    client = TestClient(
        create_app(
            backend, model_name="sensenova", image_api_registrar=register_images_api
        )
    )
    parameters = {"prompt": "edit this", boolean_name: boolean_text}
    if pixel_budget is not None:
        parameters["input_max_pixels"] = str(pixel_budget)
    else:
        pass
    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data=parameters,
    )
    assert response.status_code == expected_status
    assert backend.calls == (1 if expected_status == 200 else 0)


@pytest.mark.parametrize("do_resize", [False, True])
@pytest.mark.parametrize("device_type", ["cuda", "npu", "cpu"])
def test_image_edit_oom_uses_model_device_backend(
    device_type: Literal["cuda", "npu", "cpu"],
    do_resize: bool,
    tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU I2I failures used the CUDA allocator and GPU-only advice."""
    backend = SimpleNamespace(empty_cache=Mock())
    get_device_module = Mock(return_value=backend)
    monkeypatch.setattr(torch, "get_device_module", get_device_module)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", Mock(side_effect=AssertionError("wrong backend"))
    )
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    model = Mock()
    model.device = SimpleNamespace(type=device_type)
    model.it2i_generate.side_effect = torch.OutOfMemoryError("device exhausted")
    payload = StagePayload(
        "oom",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            {
                "width": 32,
                "height": 32,
                "size_explicit": True,
                "do_resize": do_resize,
                "input_max_pixels": 512 * 512,
            },
        ),
        None,
    )
    error_type = torch.OutOfMemoryError if device_type == "cpu" else ValueError
    error_message = (
        "device exhausted" if device_type == "cpu" else f"{device_type.upper()} memory"
    )
    with pytest.raises(error_type, match=error_message) as raised:
        generate_image(payload, model, tokenizer)
    if device_type == "cpu":
        get_device_module.assert_not_called()
        backend.empty_cache.assert_not_called()
    else:
        get_device_module.assert_called_once_with(model.device)
        backend.empty_cache.assert_called_once_with()
        assert "output size" in str(raised.value)
        assert ("input_max_pixels" in str(raised.value)) is do_resize
        assert ("dit_layerwise_resident_layers" in str(raised.value)) is (
            device_type == "cuda"
        )


def test_multi_output_failure_cleans_cache_controller(
    tokenizer: PreTrainedTokenizerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The n>1 branch bypassed cache cleanup after a failed generation."""
    cleanup = Mock(wraps=generation_module.unmount_cache_dit_after_failure)
    monkeypatch.setattr(generation_module, "unmount_cache_dit_after_failure", cleanup)
    model = Mock()
    model.t2i_generate.side_effect = RuntimeError("generation failed")
    payload = StagePayload(
        "multi", OmniRequest("a cat", {"n": 2, "width": 32, "height": 32}), None
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        generate_images([payload], model, tokenizer)
    cleanup.assert_called_once_with(model)


@pytest.mark.parametrize("size_explicit", [False, True])
@pytest.mark.parametrize(
    "guidance_scale, img_cfg_scale, branch_count",
    [(1.0, 1.0, 1), (4.0, 4.0, 2), (0.5, 1.0, 2)],
)
def test_i2i_stage_overrides_match_cost_and_tp_validation(
    tokenizer: PreTrainedTokenizerBase,
    guidance_scale: float,
    img_cfg_scale: float,
    branch_count: int,
    size_explicit: bool,
) -> None:
    """I2I cost and TP validation used options different from model execution."""
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    payload = StagePayload(
        "edit",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            {
                "width": 64,
                "height": 32,
                "size_explicit": size_explicit,
                "do_resize": False,
                "stage_params": {
                    "generate": {
                        "width": 96,
                        "height": 64,
                        "num_inference_steps": 4,
                        "guidance_scale": guidance_scale,
                        "img_cfg_scale": img_cfg_scale,
                    }
                },
            },
        ),
        None,
    )
    model = Mock()
    model.sensenova_cache_dit.enabled_by_default = True
    output_width_pixels, output_height_pixels = (96, 64) if size_explicit else (96, 96)
    model.it2i_generate.return_value = torch.zeros(
        (1, 3, output_height_pixels, output_width_pixels)
    )
    with patch.object(Image, "open", wraps=Image.open) as open_image:
        assert image_generation_request_cost(payload) == 96 * 64 * 4 * branch_count
    open_image.assert_not_called()
    validate_generation_request(payload, model)
    assert model.sensenova_cache_dit.validate_request.call_args.kwargs["steps"] == 4
    result = generate_image(payload, model, tokenizer)
    assert model.it2i_generate.call_args.kwargs["num_steps"] == 4
    assert model.it2i_generate.call_args.kwargs["image_size"] == (
        output_width_pixels,
        output_height_pixels,
    )
    assert model.it2i_generate.call_args.kwargs["cfg_scale"] == guidance_scale
    assert model.it2i_generate.call_args.kwargs["img_cfg_scale"] == img_cfg_scale
    assert Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"]))).size == (
        output_width_pixels,
        output_height_pixels,
    )


@pytest.mark.parametrize("reference_count", [1, 3])
@pytest.mark.parametrize("do_resize", [False, True])
@pytest.mark.parametrize("reference_size_pixels", [(64, 32), (77, 39)])
def test_i2i_validation_decodes_references_without_resizing(
    reference_count: int,
    do_resize: bool,
    reference_size_pixels: tuple[int, int],
) -> None:
    """Validation must decode pixels but leave resizing to model preprocessing."""
    image_buffer = io.BytesIO()
    Image.new("RGB", reference_size_pixels).save(image_buffer, format="PNG")
    references = [
        base64.b64encode(image_buffer.getvalue()).decode("ascii")
    ] * reference_count
    parameters = {"width": 96, "height": 64, "do_resize": do_resize}
    options = SenseNovaU1ImageEditSampling.from_params(parameters)
    prepared_references = prepare_reference_images(references, options)
    output_width_pixels, output_height_pixels = resolve_edit_output_size(
        prepared_references[0].size, options
    )
    payload = StagePayload(
        "edit",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": references},
            parameters,
        ),
        None,
    )
    model = Mock()
    model.sensenova_cache_dit.enabled_by_default = False
    with (
        patch.object(
            Image.Image, "load", autospec=True, side_effect=Image.Image.load
        ) as load,
        patch.object(
            Image.Image, "resize", autospec=True, side_effect=Image.Image.resize
        ) as resize,
    ):
        assert resolve_reference_output_size(references, options) == (
            output_width_pixels,
            output_height_pixels,
        )
        validate_generation_request(payload, model)
    load.assert_called()
    resize.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_reference_index", [0, 1])
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
@pytest.mark.parametrize(
    "scheduler_class", [TensorParallelScheduler, SenseNovaTensorParallelScheduler]
)
async def test_truncated_image_is_rejected_before_tp_fanout(
    tokenizer: PreTrainedTokenizerBase,
    invalid_reference_index: int,
    image_format: Literal["JPEG", "PNG"],
    scheduler_class: type[TensorParallelScheduler],
) -> None:
    """A truncated reference killed TP execution instead of failing its request."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (80, 100, 130)).save(image_buffer, format=image_format)
    image_bytes = image_buffer.getvalue()
    truncated_image_bytes = (
        image_bytes[:-2] if image_format == "JPEG" else image_bytes[:45]
    )
    with Image.open(io.BytesIO(truncated_image_bytes)) as image:
        if image_format == "JPEG":
            image.verify()
        else:
            pass
    reference = base64.b64encode(image_bytes).decode("ascii")
    references = [reference, reference]
    references[invalid_reference_index] = base64.b64encode(
        truncated_image_bytes
    ).decode("ascii")
    invalid = StagePayload(
        "invalid",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": references},
            {"width": 64, "height": 32, "size_explicit": True, "do_resize": False},
        ),
        None,
    )
    valid = StagePayload(
        "valid",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": reference},
            invalid.request.params,
        ),
        None,
    )
    model = Mock()
    model.sensenova_cache_dit.enabled_by_default = False
    model.it2i_generate.return_value = torch.zeros(1, 3, 32, 64)
    scheduler = scheduler_class(
        lambda payload: generate_image(payload, model, tokenizer),
        validate_request=lambda payload: validate_generation_request(payload, model),
    )
    fanout = Mock()
    leader = Stage(
        name="generate",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        tp_fanout=fanout,
    )
    leader.send_failure = AsyncMock()

    await leader.execute(invalid)
    leader.send_failure.assert_awaited_once_with(
        "invalid", "SenseNova-U1 received an invalid reference image"
    )
    fanout.fanout_work.assert_not_called()
    assert scheduler.inbox.empty()
    model.it2i_generate.assert_not_called()

    await leader.execute(valid)
    fanout.fanout_work.assert_called_once_with(valid)
    scheduler.run_single(scheduler.inbox.get_nowait(), asyncio.get_running_loop())
    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.request_id == "valid"
    assert scheduler.compute_error is None
    model.it2i_generate.assert_called_once()
    with Image.open(
        io.BytesIO(base64.b64decode(result.data.data["image_b64"]))
    ) as image:
        assert image.size == (64, 32)


def test_invalid_edit_seed_is_rejected_before_tp_execution(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """An invalid seed must fail validation before TP ranks execute model work."""
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    payload = StagePayload(
        "invalid",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "edit",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            {"seed": 2**64, "do_resize": False},
        ),
        None,
    )
    model = Mock()
    model.sensenova_cache_dit.enabled_by_default = False
    with pytest.raises(ValueError, match="seed"):
        validate_generation_request(payload, model)
    with pytest.raises(ValueError, match="seed"):
        generate_image(payload, model, tokenizer)
    model.it2i_generate.assert_not_called()


@pytest.mark.parametrize(
    "seed, image_count, expected_status",
    [(2**64 - 1, 1, 200), (2**64 - 2, 2, 200), (2**64 - 1, 2, 400), (2**64, 1, 400)],
)
def test_image_edit_api_validates_all_output_seeds(
    seed: int, image_count: int, expected_status: int
) -> None:
    """The edits loop accepted a base seed whose later outputs overflowed the RNG."""
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    encoded = base64.b64encode(reference.getvalue()).decode("ascii")

    class SeedClient:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        async def generate(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncIterator[GenerateChunk]:
            self.seeds.append(request.extra_params["seed"])
            yield umm_image_chunk(request_id, [encoded])

    backend = SeedClient()
    client = TestClient(
        create_app(
            backend, model_name="sensenova", image_api_registrar=register_images_api
        )
    )
    response = client.post(
        "/v1/images/edits",
        files={"image": ("reference.png", reference.getvalue(), "image/png")},
        data={"prompt": "edit", "seed": str(seed), "n": str(image_count)},
    )
    assert response.status_code == expected_status
    assert backend.seeds == (
        [seed + index for index in range(image_count)] if expected_status == 200 else []
    )


@pytest.mark.parametrize("failure", [None, "later_reference", "aspect_ratio"])
def test_reference_images_are_closed_after_validation(
    failure: Literal["later_reference", "aspect_ratio"] | None,
) -> None:
    """Validation must close every decoded reference on success or failure."""
    image_buffer = io.BytesIO()
    reference_size_pixels = (32, 7000) if failure == "aspect_ratio" else (64, 32)
    Image.new("RGB", reference_size_pixels).save(image_buffer, format="PNG")
    encoded_reference = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    references = [
        encoded_reference,
        "invalid" if failure == "later_reference" else encoded_reference,
    ]
    options = SenseNovaU1ImageEditSampling.from_params(
        {"width": 96, "height": 64, "input_max_pixels": 262144}
    )
    with patch.object(
        Image.Image, "close", autospec=True, side_effect=Image.Image.close
    ) as close_image:
        if failure is None:
            assert resolve_reference_output_size(references, options) == (96, 32)
        else:
            error_message = (
                "invalid reference image"
                if failure == "later_reference"
                else "aspect ratio"
            )
            with pytest.raises(ValueError, match=error_message):
                resolve_reference_output_size(references, options)
    assert close_image.call_count == (1 if failure == "later_reference" else 2)
    assert all(call.args[0].fp is None for call in close_image.call_args_list)


def test_reference_pixel_decode_failure_closes_open_images() -> None:
    """A truncated reference must close both earlier images and its own header."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(image_buffer, format="PNG")
    encoded_reference = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    truncated_reference = base64.b64encode(image_buffer.getvalue()[:45]).decode("ascii")
    with patch.object(
        Image.Image, "close", autospec=True, side_effect=Image.Image.close
    ) as close_image:
        with pytest.raises(ValueError, match="invalid reference image"):
            decode_reference_images([encoded_reference, truncated_reference])
    assert close_image.call_count == 2
    assert all(call.args[0].fp is None for call in close_image.call_args_list)


@pytest.mark.parametrize("references", [None, [], "", "invalid", ["invalid"]])
def test_reference_parsing_rejects_invalid_inputs(
    references: str | list[str] | None,
) -> None:
    """Validation and generation must both reject malformed references."""
    options = SenseNovaU1ImageEditSampling.from_params({})
    with pytest.raises(ValueError):
        decode_reference_images(references)
    with pytest.raises(ValueError):
        resolve_reference_output_size(references, options)


def test_image_edits_stay_single_item_without_decoding_cost() -> None:
    """I2I costs must not decode references that cannot affect batch admission."""
    scheduler = SimpleScheduler(
        Mock(),
        batch_compute_fn=Mock(),
        max_batch_size=2,
        max_batch_cost=1,
        batch_key_fn=image_generation_batch_key,
        request_cost_fn=image_generation_request_cost,
    )
    messages = [
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=StagePayload(
                request_id,
                OmniRequest(
                    {"task": "image_edit", "prompt": "edit", "image_b64": "unused"},
                    {"width": 96, "height": 64},
                ),
                None,
            ),
        )
        for request_id in ("first", "second")
    ]
    scheduler.inbox.put(messages[1])
    with patch.object(Image, "open", wraps=Image.open) as open_image:
        assert scheduler.collect_batch(messages[0]) == [messages[0]]
        assert scheduler.next_message() is messages[1]
        assert scheduler.collect_batch(messages[1]) == [messages[1]]
    open_image.assert_not_called()


def test_image_edit_api_accepts_multiple_reference_images():
    app = create_app(
        DefaultImageEditClient(expected_images=2),
        model_name="sensenova",
        image_api_registrar=register_images_api,
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
    image_client = MultipleOutputImageEditClient()
    app = create_app(
        image_client,
        model_name="sensenova",
        image_api_registrar=register_images_api,
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
        ImageEditClient(),
        model_name="sensenova",
        image_api_registrar=register_images_api,
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
    malformed_size = client.post(
        "/v1/images/edits",
        files={"image": image_file},
        data={"prompt": "edit this", "size": "bad"},
    )
    assert malformed_size.status_code == 400
    assert malformed_size.json()["detail"] == "size must be WxH"
    for n in (0, -1, 11):
        invalid_n = client.post(
            "/v1/images/edits",
            files={"image": image_file},
            data={"prompt": "edit this", "n": n},
        )
        assert invalid_n.status_code == 400
    with_mask = client.post(
        "/v1/images/edits",
        files={"image": image_file, "mask": image_file},
        data={"prompt": "edit this"},
    )
    assert with_mask.status_code == 400


def test_config_registers_checkpoint_architecture(tmp_path):
    assert SenseNovaU1PipelineConfig.architecture == "NEOChatModel"
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    assert config.resolved_entry_stage == "orchestrator"
    assert not config.stage_named("generate").terminal
    assert config.stage_named("output").terminal
    factory = FactoryArgs(max_batch_size=2, max_batch_cost=41943040)
    assert factory.max_batch_cost == 41943040
    assert factory.model_extra in (None, {})
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "neo_chat", "architectures": ["NEOChatModel"]})
    )
    assert resolve_config_cls_for_model_path(str(tmp_path)) is SenseNovaU1PipelineConfig


def test_sensenova_dp_expands_to_distinct_gpu_replicas():
    config = SenseNovaU1PipelineConfig(
        model_path="/model/sensenova",
        processes={
            "sensenova_generate": ProcessConfig(num_replicas=2, replica_devices=[0, 1])
        },
    )

    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)

    assert [
        (stage.name, stage.gpu)
        for stage in expanded
        if stage.name.startswith("generate")
    ] == [
        ("generate@r0", 0),
        ("generate@r1", 1),
    ]
    assert topology.instances("generate") == ("generate@r0", "generate@r1")


def test_sensenova_accepts_tp_and_rejects_same_gpu_dp():
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    tp_config = config.model_dump()
    generate_config = next(
        stage for stage in tp_config["stages"] if stage["name"] == "generate"
    )
    generate_config["tp_size"] = 2
    generate_config["gpu"] = [0, 1]
    validated = SenseNovaU1PipelineConfig.model_validate(tp_config)
    assert validated.stage_named("generate").tp_size == 2

    with pytest.raises(ValueError, match="distinct GPU"):
        SenseNovaU1PipelineConfig(
            model_path="/model/sensenova",
            processes={
                "sensenova_generate": ProcessConfig(
                    num_replicas=2, replica_devices=[0, 0]
                )
            },
        )


def test_generate_stage_params_layer_onto_t2i_options():
    payload = StagePayload(
        "request",
        OmniRequest(
            "A blue cat",
            {
                "stage_params": {
                    "generate": {
                        "width": 512,
                        "height": 384,
                        "num_inference_steps": 2,
                        "guidance_scale": 3.0,
                        "seed": 7,
                        "n": 1,
                    }
                }
            },
        ),
        None,
    )
    options = text_to_image_options(payload)
    assert (options.width, options.height) == (512, 384)
    assert options.num_inference_steps == 2
    assert options.guidance_scale == 3.0
    assert options.seed == 7
