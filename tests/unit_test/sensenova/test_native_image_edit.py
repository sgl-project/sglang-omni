"""Exercise reference preprocessing, native editing and sequential outputs."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("sglang")

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as image_api_utils
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.sensenova_u1.native import generation
from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.image_edit import SenseNovaU1InputStage
from sglang_omni.models.sensenova_u1.native.pipeline_config import (
    SenseNovaU1PipelineConfig,
)
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams
from sglang_omni.vendor.sensenova_u1.configuration_neo_chat import NEOChatConfig
from sglang_omni.vendor.sensenova_u1.image_edit import it2i_generate
from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import NEOChatModel


@pytest.mark.parametrize("explicit_size", [False, True])
def test_reference_preparation_preserves_alpha_and_size_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_size: bool
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    reference_path = tmp_path / "reference.png"
    reference = Image.new("RGBA", (1024, 512), (0, 0, 0, 0))
    reference.putpixel((0, 0), (255, 0, 0, 255))
    reference.save(reference_path)
    sampling = SenseNovaU1SamplingParams(
        prompt="change the background",
        image_path=str(reference_path),
        **({"width": 64, "height": 64} if explicit_size else {}),
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.extra["explicit_fields"] = ["width", "height"] if explicit_size else []
    stage = SenseNovaU1InputStage()
    request.condition_image = stage.load_condition_image(str(reference_path))
    stage.preprocess_condition_image(request, None, 64, 32)
    prepared = request.condition_image[0]
    assert prepared.size == (1024, 512)
    assert prepared.getpixel((0, 0)) == (255, 0, 0)
    assert prepared.getpixel((1, 0)) == (255, 255, 255)
    assert (request.width, request.height) == (
        (64, 64) if explicit_size else (2048, 2048)
    )


def test_reference_order_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling = SenseNovaU1SamplingParams(
        prompt="combine the images",
        image_path=["red.png", "blue.png"],
        width=64,
        height=64,
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.extra["explicit_fields"] = ["width", "height"]
    request.condition_image = [
        Image.new("RGB", (64, 64), color) for color in ["red", "blue"]
    ]
    SenseNovaU1InputStage().preprocess_condition_image(request, None, 64, 64)
    assert [image.size for image in request.condition_image] == [(512, 512)] * 2
    assert [image.getpixel((0, 0)) for image in request.condition_image] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert (request.width, request.height) == (64, 64)


@pytest.mark.parametrize(
    "reference_count,reference_size_pixels", [(60, (32, 64)), (65, (64, 64))]
)
def test_reference_pixel_overflow_is_rejected_before_resize(
    monkeypatch: pytest.MonkeyPatch,
    reference_count: int,
    reference_size_pixels: tuple[int, int],
) -> None:
    """Aligned reference sizes could exceed the total budget before generation."""
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling = SenseNovaU1SamplingParams(
        prompt="combine the images", image_path=["reference.png"] * reference_count
    )
    request = Req(sampling_params=sampling)
    references = [
        Image.new("RGB", reference_size_pixels) for _ in range(reference_count)
    ]
    request.condition_image = references
    with patch.object(
        Image.Image, "resize", autospec=True, side_effect=Image.Image.resize
    ) as resize:
        with pytest.raises(ValueError, match="exceeding the total limit"):
            SenseNovaU1InputStage().preprocess_condition_image(request, None, 64, 64)
        resize.assert_not_called()
    assert request.condition_image is references


def test_reference_pixel_budget_allows_exact_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling = SenseNovaU1SamplingParams(
        prompt="combine the images", image_path=["reference.png"] * 64
    )
    request = Req(sampling_params=sampling)
    request.condition_image = [Image.new("RGB", (64, 64)) for _ in range(64)]
    SenseNovaU1InputStage().preprocess_condition_image(request, None, 64, 64)
    assert len(request.condition_image) == 64
    assert all(image.size == (512, 512) for image in request.condition_image)
    assert (
        sum(image.width * image.height for image in request.condition_image) == 4096**2
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"img_cfg_scale": -1},
        {"img_cfg_scale": float("nan")},
    ],
)
def test_invalid_edit_parameters_are_rejected(
    parameters: dict[str, float | int | str]
) -> None:
    with pytest.raises(ValidationError):
        SenseNovaU1SamplingParams(
            prompt="edit", image_path="reference.png", **parameters
        )


@pytest.mark.parametrize("image_path", ["reference.png", ["first.png", "second.png"]])
def test_edit_rejects_excess_image_placeholders(image_path: str | list[str]) -> None:
    with pytest.raises(
        ValueError, match="more image placeholders than reference images"
    ):
        SenseNovaU1SamplingParams(prompt="<image><image><image>", image_path=image_path)


@pytest.mark.parametrize("prompt", ["edit", "<image> edit", "<image><image> edit"])
def test_edit_accepts_implicit_and_explicit_references(prompt: str) -> None:
    SenseNovaU1SamplingParams(prompt=prompt, image_path=["first.png", "second.png"])


def test_edit_api_rejects_excess_placeholders_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excess placeholders used to fail in the worker and surface as HTTP 500."""
    server_args = SimpleNamespace(
        model_path=str(tmp_path),
        pipeline_class_name="SenseNovaU1Pipeline",
        pipeline_config=SenseNovaU1PipelineConfig(),
        input_save_path=None,
        output_path=None,
    )
    monkeypatch.setattr(image_api, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(image_api_utils, "get_global_server_args", lambda: server_args)
    dispatch = AsyncMock()
    monkeypatch.setattr(image_api.async_scheduler_client, "forward", dispatch)
    image_bytes = BytesIO()
    Image.new("RGB", (32, 32)).save(image_bytes, format="PNG")
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "<image><image> edit", "generator_device": "cpu"},
            files={"image": ("reference.png", image_bytes.getvalue(), "image/png")},
        )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Prompt has more image placeholders than reference images"
    )
    dispatch.assert_not_called()


def test_t2i_does_not_validate_unused_edit_guidance() -> None:
    sampling = SenseNovaU1SamplingParams(prompt="a landscape", img_cfg_scale=-1)
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    assert request.image_path is None


@pytest.mark.parametrize("output_count", [1, 2])
def test_edit_parameters_reach_each_sequential_output(
    monkeypatch: pytest.MonkeyPatch, output_count: int
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling = SenseNovaU1SamplingParams(
        prompt="edit",
        image_path="reference.png",
        width=64,
        height=32,
        img_cfg_scale="1.5",
        seed=7,
        num_outputs_per_prompt=output_count,
        generator_device="cpu",
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.condition_image = [Image.new("RGB", (64, 32))]
    edit = Mock(return_value=torch.zeros(1, 3, 32, 64))
    monkeypatch.setattr(generation, "it2i_generate", edit)
    model = Mock()
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    generation_stage = SenseNovaU1GenerationStage(model, Mock())
    expanded_requests = list(
        SenseNovaU1InputStage().iter_sequential_requests(request, server_args)
    )
    assert len(expanded_requests) == output_count
    for output_index, expanded_request in enumerate(expanded_requests):
        assert expanded_request.num_outputs_per_prompt == 1
        assert expanded_request.image_path == "reference.png"
        output = generation_stage.forward(expanded_request, server_args)
        assert edit.call_args.args[3] is request.condition_image
        assert edit.call_args.kwargs["img_cfg_scale"] == 1.5
        assert edit.call_args.kwargs["image_size"] == (64, 32)
        assert edit.call_args.kwargs["seed"] == 7 + output_index
        assert output.output[0].shape == (3, 32, 64)
        assert torch.equal(output.output[0], torch.full((3, 32, 64), 0.5))
    assert edit.call_count == output_count
    model.t2i_generate.assert_not_called()


def test_edit_requests_are_not_dynamically_batched() -> None:
    config = SenseNovaU1PipelineConfig()
    sampling = SenseNovaU1SamplingParams(prompt="edit", image_path="reference.png")
    request = Req(sampling_params=sampling)
    assert not config.supports_dynamic_batching_for_request(request)
    assert config.resolve_task_type(has_image=True).name == "TI2I"
    assert config.resolve_task_type(has_image=False).name == "T2I"


@pytest.mark.parametrize(
    "cfg_scale,img_cfg_scale", [(1.0, 1.0), (4.0, 1.0), (2.0, 2.0), (4.0, 1.5)]
)
def test_edit_denoising_uses_references_and_is_repeatable(
    cfg_scale: float,
    img_cfg_scale: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual reference encoding, prefix attention and CFG on a small CPU model."""
    torch.manual_seed(0)
    config = NEOChatConfig(
        vision_config={
            "hidden_size": 32,
            "llm_hidden_size": 32,
            "patch_size": 16,
            "max_position_embeddings_vision": 64,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 2048,
            "max_position_embeddings_hw": 64,
        },
        template="neo1_0",
        use_pixel_head=False,
        fm_head_layers=2,
        concat_time_token_num=0,
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=1,
        add_noise_scale_embedding=True,
        noise_scale_max_value=10.0,
    )
    model = NEOChatModel(config).eval()
    extract_feature = Mock(wraps=model.extract_feature)
    monkeypatch.setattr(model, "extract_feature", extract_feature)
    timestep_forward = Mock(wraps=model.fm_modules["timestep_embedder"].forward)
    noise_forward = Mock(wraps=model.fm_modules["noise_scale_embedder"].forward)
    monkeypatch.setattr(
        model.fm_modules["timestep_embedder"], "forward", timestep_forward
    )
    monkeypatch.setattr(
        model.fm_modules["noise_scale_embedder"], "forward", noise_forward
    )
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "<img>": 2, "</img>": 3, "<IMG_CONTEXT>": 4}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<img>", "</img>", "<IMG_CONTEXT>"],
    )
    references = [Image.new("RGB", (32, 32), color) for color in ["red", "blue"]]
    parameters = dict(
        image_size=(64, 64),
        num_steps=2,
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
        seed=7,
        timestep_shift=3.0,
        t_eps=0.02,
    )
    first = it2i_generate(model, tokenizer, "edit", references, **parameters)
    repeated = it2i_generate(model, tokenizer, "edit", references, **parameters)
    changed = it2i_generate(
        model, tokenizer, "edit", list(reversed(references)), **parameters
    )
    assert first.shape == (1, 3, 64, 64)
    # note (Wes): Native CUDA embeds four image tokens per step, rather than broadcasting a precomputed embedding.
    expected_steps = [0.0, 0.25] * 3
    assert timestep_forward.call_count == len(expected_steps)
    assert noise_forward.call_count == len(expected_steps)
    for call, timestep in zip(timestep_forward.call_args_list, expected_steps):
        torch.testing.assert_close(call.args[0], torch.full((4,), timestep))
    for call in noise_forward.call_args_list:
        torch.testing.assert_close(call.args[0], torch.full((4,), 0.1))
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    changed_guidance = it2i_generate(
        model,
        tokenizer,
        "edit",
        references,
        **(parameters | {"img_cfg_scale": img_cfg_scale + 0.5}),
    )
    assert not torch.equal(first, changed_guidance)
    reference_encoding_calls = [
        call
        for call in extract_feature.call_args_list
        if call.kwargs["gen_model"] is False
    ]
    assert len(reference_encoding_calls) == 4


@pytest.mark.parametrize(
    "reference_size,reference_count,expected_size",
    [
        ((512, 384), 1, (608, 448)),
        ((1024, 768), 2, (1024, 768)),
        ((2560, 2048), 5, (2016, 1632)),
    ],
)
def test_reference_pixels_match_native_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    reference_size: tuple[int, int],
    reference_count: int,
    expected_size: tuple[int, int],
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    reference = Image.linear_gradient("L").resize(reference_size).convert("RGB")
    sampling = SenseNovaU1SamplingParams(
        prompt="edit",
        image_path=["reference.png"] * reference_count,
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.extra["explicit_fields"] = ["width", "height"]
    request.condition_image = [reference] * reference_count
    SenseNovaU1InputStage().preprocess_condition_image(request, None, *reference_size)
    # note (Wes): Sizes were checked against native SGLang baa9befa; native resize uses Pillow's default interpolation.
    expected = reference.resize(expected_size)
    for prepared in request.condition_image:
        assert prepared.size == expected.size
        assert prepared.tobytes() == expected.tobytes()
