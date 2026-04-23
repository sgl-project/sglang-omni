# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the image generation module.

Tests the pipeline wiring (config, routing, executor, API) without requiring
actual diffusion model weights. Uses mocks for the DiffusionBackend.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. Test DiffusionBackend abstraction & ImageGenParams
# ---------------------------------------------------------------------------


def test_image_gen_params_defaults():
    from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams

    p = ImageGenParams()
    assert p.width == 1024
    assert p.height == 1024
    assert p.num_inference_steps == 28
    assert p.guidance_scale == 7.0
    assert p.seed is None
    assert p.negative_prompt == ""


def test_image_gen_params_custom():
    from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams

    p = ImageGenParams(width=512, height=768, num_inference_steps=50, seed=42)
    assert p.width == 512
    assert p.height == 768
    assert p.num_inference_steps == 50
    assert p.seed == 42


def test_diffusion_backend_is_abstract():
    from sglang_omni.models.ming_omni.diffusion.backend import DiffusionBackend

    with pytest.raises(TypeError):
        DiffusionBackend()


# ---------------------------------------------------------------------------
# 2. Test pipeline routing (next_stage.py)
# ---------------------------------------------------------------------------


def test_image_gen_stage_constant():
    from sglang_omni.models.ming_omni.pipeline.next_stage import IMAGE_GEN_STAGE

    assert IMAGE_GEN_STAGE == "img_gen"


def test_thinker_next_image():
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        DECODE_STAGE,
        IMAGE_GEN_STAGE,
        thinker_next_image,
    )

    result = thinker_next_image("req-1", None)
    assert result == [DECODE_STAGE, IMAGE_GEN_STAGE]


def test_thinker_next_full():
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        DECODE_STAGE,
        IMAGE_GEN_STAGE,
        TALKER_STAGE,
        thinker_next_full,
    )

    result = thinker_next_full("req-1", None)
    assert result == [DECODE_STAGE, TALKER_STAGE, IMAGE_GEN_STAGE]


def test_image_gen_next_is_terminal():
    from sglang_omni.models.ming_omni.pipeline.next_stage import image_gen_next

    assert image_gen_next("req-1", None) is None


# ---------------------------------------------------------------------------
# 3. Test OmniEventType includes image_final
# ---------------------------------------------------------------------------


def test_omni_event_type_has_image_final():
    from sglang_omni.models.ming_omni.io import OmniEventType

    assert "image_final" in OmniEventType.__args__


def test_omni_event_image_final():
    from sglang_omni.models.ming_omni.io import OmniEvent

    event = OmniEvent(
        type="image_final",
        modality="image",
        payload={"images": [{"b64_data": "abc"}]},
        is_final=True,
    )
    assert event.type == "image_final"
    assert event.modality == "image"
    assert event.is_final is True


# ---------------------------------------------------------------------------
# 4. Test pipeline configs
# ---------------------------------------------------------------------------


def test_image_pipeline_config_stages():
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    config = MingOmniImagePipelineConfig(
        model_path="/fake/path",
        dit_type="zimage",
        dit_model_path="/fake/dit",
    )
    stage_names = [s.name for s in config.stages]
    assert "img_gen" in stage_names
    assert "decode" in stage_names
    assert config.terminal_stages == ["decode", "img_gen"]


def test_image_pipeline_config_gpu_placement():
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    config = MingOmniImagePipelineConfig(
        model_path="/fake/path",
        dit_model_path="/fake/dit",
        gpu_placement={"thinker": 0, "img_gen": 2},
    )
    assert config.gpu_placement["img_gen"] == 2


def test_full_pipeline_config_stages():
    from sglang_omni.models.ming_omni.config import MingOmniFullPipelineConfig

    config = MingOmniFullPipelineConfig(
        model_path="/fake/path",
        dit_type="sd3",
        dit_model_path="/fake/dit",
    )
    stage_names = [s.name for s in config.stages]
    assert "img_gen" in stage_names
    assert "talker" in stage_names
    assert "decode" in stage_names
    assert config.terminal_stages == ["decode", "talker", "img_gen"]


def test_image_config_injects_dit_model_path():
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    config = MingOmniImagePipelineConfig(
        model_path="/fake/path",
        dit_type="sd3",
        dit_model_path="/my/sd3/weights",
    )
    img_stage = [s for s in config.stages if s.name == "img_gen"][0]
    assert img_stage.executor.args["dit_model_path"] == "/my/sd3/weights"
    assert img_stage.executor.args["dit_type"] == "sd3"


# ---------------------------------------------------------------------------
# 5. Test MingImageGenExecutor (with mocked backend)
# ---------------------------------------------------------------------------


def _make_fake_payload(text: str = "a cat", image_params: dict | None = None):
    """Create a minimal StagePayload-like object for testing."""
    from sglang_omni.proto import StagePayload

    req = MagicMock()
    req.metadata = {"image_generation": image_params or {}}
    return StagePayload(
        request_id="test-req-1",
        request=req,
        data={
            "thinker_out": {"output_ids": [], "step": 1, "is_final": True},
            "generated_text": text,
        },
    )


@pytest.mark.asyncio
async def test_executor_extract_input_text():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake",
        dit_type="zimage",
        dit_model_path="/fake/dit",
    )
    payload = _make_fake_payload("a beautiful sunset")
    text, params = executor._extract_input(payload)
    assert text == "a beautiful sunset"
    assert params.width == 1024
    assert params.height == 1024


@pytest.mark.asyncio
async def test_executor_extract_input_params():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake",
        dit_type="zimage",
        dit_model_path="/fake/dit",
    )
    payload = _make_fake_payload(
        "cat",
        image_params={
            "size": "512x768",
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "seed": 123,
            "negative_prompt": "blurry",
        },
    )
    # Put image_generation in raw_inputs too
    payload.data["raw_inputs"] = {
        "image_generation": {
            "size": "512x768",
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "seed": 123,
            "negative_prompt": "blurry",
        }
    }
    text, params = executor._extract_input(payload)
    assert params.width == 512
    assert params.height == 768
    assert params.num_inference_steps == 50
    assert params.guidance_scale == 5.0
    assert params.seed == 123
    assert params.negative_prompt == "blurry"


@pytest.mark.asyncio
async def test_executor_add_request_empty_text():
    """Empty text should produce a result with image_data=None."""
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake",
        dit_type="zimage",
        dit_model_path="/fake/dit",
    )
    payload = _make_fake_payload("")
    await executor.add_request(payload)
    result = await executor.get_result()
    assert result.request_id == "test-req-1"
    assert result.data["image_data"] is None


@pytest.mark.asyncio
async def test_executor_add_request_with_mock_backend():
    """Full flow with a mocked DiffusionBackend."""
    from PIL import Image

    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    # Create a tiny test image
    test_image = Image.new("RGB", (64, 64), color="red")

    executor = MingImageGenExecutor(
        model_path="/fake",
        dit_type="zimage",
        dit_model_path="/fake/dit",
    )
    # Mock the backend
    mock_backend = MagicMock()
    mock_backend.generate.return_value = test_image
    executor._backend = mock_backend
    executor._thinker_tokenizer = None  # will use fallback text

    payload = _make_fake_payload("a red square")
    await executor.add_request(payload)
    result = await executor.get_result()

    assert result.request_id == "test-req-1"
    assert result.data["image_data"] is not None
    assert result.data["image_format"] == "png"
    assert result.data["image_width"] == 64
    assert result.data["image_height"] == 64
    assert result.data["modality"] == "image"

    # Verify the base64 data decodes to a valid PNG
    decoded = base64.b64decode(result.data["image_data"])
    assert decoded[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_executor_abort():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake", dit_type="zimage", dit_model_path="/fake/dit"
    )
    await executor.abort("req-to-abort")
    assert "req-to-abort" in executor._aborted


# ---------------------------------------------------------------------------
# 6. Test client types
# ---------------------------------------------------------------------------


def test_generate_chunk_image_fields():
    from sglang_omni.client.types import GenerateChunk

    chunk = GenerateChunk(
        request_id="r1",
        modality="image",
        image_data="base64str",
        image_format="png",
        image_width=1024,
        image_height=1024,
    )
    d = chunk.to_dict()
    assert d["image_data"] == "base64str"
    assert d["image_format"] == "png"
    assert d["image_width"] == 1024
    assert d["modality"] == "image"


def test_completion_image_dataclass():
    from sglang_omni.client.types import CompletionImage

    img = CompletionImage(b64_json="abc", format="png", width=512, height=512)
    assert img.b64_json == "abc"
    assert img.format == "png"


def test_completion_result_with_images():
    from sglang_omni.client.types import CompletionImage, CompletionResult

    img = CompletionImage(b64_json="abc", format="png", width=1024, height=1024)
    result = CompletionResult(
        request_id="r1",
        text="here is your image",
        images=[img],
    )
    assert result.images is not None
    assert len(result.images) == 1


def test_completion_stream_chunk_image_fields():
    from sglang_omni.client.types import CompletionStreamChunk

    chunk = CompletionStreamChunk(
        request_id="r1",
        modality="image",
        image_b64="base64data",
        image_format="png",
        image_width=1024,
        image_height=1024,
    )
    assert chunk.image_b64 == "base64data"


# ---------------------------------------------------------------------------
# 7. Test protocol (ChatCompletionRequest with image_generation)
# ---------------------------------------------------------------------------


def test_protocol_image_generation_field():
    from sglang_omni.serve.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "draw a cat"}],
        modalities=["text", "image"],
        image_generation={
            "model": "zimage",
            "size": "1024x1024",
            "num_inference_steps": 28,
        },
    )
    assert req.image_generation is not None
    assert req.image_generation["model"] == "zimage"
    assert "image" in req.modalities


def test_protocol_image_data_model():
    from sglang_omni.serve.protocol import ChatCompletionImageData

    img = ChatCompletionImageData(
        b64_json="abc123", format="png", width=1024, height=1024
    )
    d = img.model_dump()
    assert d["b64_json"] == "abc123"
    assert d["format"] == "png"


def test_stream_delta_with_images():
    from sglang_omni.serve.protocol import (
        ChatCompletionImageData,
        ChatCompletionStreamDelta,
    )

    delta = ChatCompletionStreamDelta(
        content="here is the image",
        images=[ChatCompletionImageData(b64_json="x", format="png")],
    )
    d = delta.model_dump(exclude_none=True)
    assert "images" in d
    assert len(d["images"]) == 1


# ---------------------------------------------------------------------------
# 8. Test _create_backend factory
# ---------------------------------------------------------------------------


def test_create_backend_zimage():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        _create_backend,
    )
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = _create_backend("zimage")
    assert isinstance(backend, ZImageBackend)


def test_create_backend_sd3():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        _create_backend,
    )
    from sglang_omni.models.ming_omni.diffusion.sd3_backend import SD3Backend

    backend = _create_backend("sd3")
    assert isinstance(backend, SD3Backend)


def test_create_backend_unknown_raises():
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        _create_backend,
    )

    with pytest.raises(ValueError, match="Unknown dit_type"):
        _create_backend("flux")
