"""Exercise reference preprocessing, native editing and sequential outputs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("sglang")

import torch
from PIL import Image
from pydantic import ValidationError
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.sensenova_u1.native import generation
from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.image_edit import (
    SenseNovaInputValidationStage,
)
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
    reference = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    reference.putpixel((0, 0), (255, 0, 0, 255))
    reference.save(reference_path)
    sampling = SenseNovaU1SamplingParams(
        prompt="change the background",
        image_path=str(reference_path),
        width=64,
        height=64,
        do_resize=False,
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.extra["explicit_fields"] = ["width", "height"] if explicit_size else []
    stage = SenseNovaInputValidationStage()
    request.condition_image = stage.load_condition_image(str(reference_path))
    stage.preprocess_condition_image(request, None, 64, 32)
    prepared = request.condition_image[0]
    assert prepared.size == (64, 32)
    assert prepared.getpixel((0, 0)) == (255, 0, 0)
    assert prepared.getpixel((1, 0)) == (255, 255, 255)
    assert (request.width, request.height) == ((64, 64) if explicit_size else (96, 64))


@pytest.mark.parametrize(
    "input_max_pixels,reference_side_pixels", [(None, 512), (1024 * 1024, 1024)]
)
def test_reference_budget_and_order_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    input_max_pixels: int | None,
    reference_side_pixels: int,
) -> None:
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
        input_max_pixels=input_max_pixels,
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.extra["explicit_fields"] = ["width", "height"]
    request.condition_image = [
        Image.new("RGB", (64, 64), color) for color in ["red", "blue"]
    ]
    SenseNovaInputValidationStage().preprocess_condition_image(request, None, 64, 64)
    assert [image.size for image in request.condition_image] == [
        (reference_side_pixels, reference_side_pixels)
    ] * 2
    assert [image.getpixel((0, 0)) for image in request.condition_image] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert (request.width, request.height) == (64, 64)


def test_unresized_reference_requires_patch_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling = SenseNovaU1SamplingParams(
        prompt="edit", image_path="reference.png", do_resize=False
    )
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.condition_image = Image.new("RGB", (33, 32))
    with pytest.raises(ValueError, match="divisible"):
        SenseNovaInputValidationStage().preprocess_condition_image(
            request, None, 33, 32
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"img_cfg_scale": -1},
        {"img_cfg_scale": float("nan")},
        {"input_max_pixels": 1},
        {"do_resize": "invalid"},
    ],
)
def test_invalid_edit_parameters_are_rejected(
    parameters: dict[str, float | int | str]
) -> None:
    with pytest.raises(ValidationError):
        SenseNovaU1SamplingParams(
            prompt="edit", image_path="reference.png", **parameters
        )


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
        input_max_pixels="262144",
        do_resize="false",
        seed=7,
        num_outputs_per_prompt=output_count,
        generator_device="cpu",
    )
    assert sampling.do_resize is False
    assert sampling.input_max_pixels == 262144
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    request.condition_image = [Image.new("RGB", (64, 32))]
    edit = Mock(return_value=torch.zeros(1, 3, 32, 64))
    monkeypatch.setattr(generation, "it2i_generate", edit)
    model = Mock()
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    generation_stage = SenseNovaU1GenerationStage(model, Mock())
    expanded_requests = list(
        SenseNovaInputValidationStage().iter_sequential_requests(request, server_args)
    )
    assert len(expanded_requests) == output_count
    for output_index, expanded_request in enumerate(expanded_requests):
        assert expanded_request.num_outputs_per_prompt == 1
        assert expanded_request.image_path == "reference.png"
        assert expanded_request.input_max_pixels == 262144
        assert expanded_request.do_resize is False
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
        add_noise_scale_embedding=False,
        noise_scale_max_value=10.0,
    )
    model = NEOChatModel(config).eval()
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
        image_size=(32, 32),
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
    assert first.shape == (1, 3, 32, 32)
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
