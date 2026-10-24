# SPDX-License-Identifier: Apache-2.0
"""Check standalone native modules without loading accelerator dependencies."""

import base64
import importlib
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pytest
import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.image_processing import (
    prepare_reference_images,
    reference_image_size,
)
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "P_alpha", "RGB", "L"])
def test_transparent_references_are_composited_on_white(mode: str) -> None:
    """Palette and grayscale PNG transparency was discarded instead of composited."""
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (64, 64, 64, 255))
    image.putpixel((1, 1), (128, 128, 128, 128))
    if mode in ("P", "P_alpha"):
        image = image.convert("RGB").quantize(colors=3)
        if mode == "P_alpha":
            alpha_values = bytearray([255] * 256)
            alpha_values[image.getpixel((1, 0))] = 0
            alpha_values[image.getpixel((1, 1))] = 128
            image.info["transparency"] = bytes(alpha_values)
        else:
            image.info["transparency"] = image.getpixel((1, 0))
    elif mode in ("RGB", "L"):
        image = image.convert(mode)
        image.info["transparency"] = (0, 0, 0) if mode == "RGB" else 0
    else:
        image = image.convert(mode)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    prepared = prepare_reference_images(
        base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        SenseNovaU1ImageEditSampling.from_params({"do_resize": False}),
    )
    try:
        assert prepared[0].mode == "RGB"
        assert prepared[0].getpixel((1, 0)) == (255, 255, 255)
        assert prepared[0].getpixel((0, 0)) == (64, 64, 64)
        expected_gray = 191 if mode in ("RGBA", "LA", "P_alpha") else 128
        assert prepared[0].getpixel((1, 1)) == (expected_gray,) * 3
    finally:
        prepared[0].close()


class NativeModuleLoader(Protocol):
    def __call__(self, module: str) -> ModuleType:
        """Load a native module without accelerator package initialization."""


@pytest.fixture
def native_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeModuleLoader]:
    name = "sglang_omni.models.sensenova_u1.neo_unify"
    package = ModuleType(name)
    package.__path__ = [
        str(
            Path(__file__).resolve().parents[3]
            / "sglang_omni/models/sensenova_u1/neo_unify"
        )
    ]
    monkeypatch.setitem(sys.modules, name, package)
    try:
        yield lambda module: importlib.import_module(f"{name}.{module}")
    finally:
        for key in list(sys.modules):
            if key.startswith(f"{name}."):
                sys.modules.pop(key)
            else:
                pass


@pytest.mark.parametrize("is_portrait", [False, True])
def test_reference_axis_bound_matches_vision_positions(
    native_modules: NativeModuleLoader, is_portrait: bool
) -> None:
    """The last admitted patch must fit the model's actual rotary position cache."""
    vision = native_modules("modeling_neo_vit")
    configuration = vision.NEOVisionConfig(hidden_size=8, llm_hidden_size=8)
    embeddings = vision.NEOVisionEmbeddings(configuration)
    embeddings.ensure_rotary_embedding_cache(torch.device("cpu"))
    long_edge_pixels = (
        configuration.patch_size * configuration.max_position_embeddings_vision
    )
    width, height = (32, long_edge_pixels) if is_portrait else (long_edge_pixels, 32)
    sampling = SenseNovaU1ImageEditSampling.from_params({"do_resize": False})
    assert reference_image_size(width, height, 1, sampling) == (width, height)
    grid = torch.tensor(
        [[height // configuration.patch_size, width // configuration.patch_size]]
    )
    patch_embeddings = torch.ones(int(grid.prod()), configuration.hidden_size)
    outputs = embeddings.apply_2d_rotary_position_embeddings(patch_embeddings, grid)
    assert torch.isfinite(outputs).all()
    with pytest.raises(ValueError, match="pixels per axis"):
        reference_image_size(width * 2, height * 2, 1, sampling)


def test_moe_config_detects_expert_count_without_model_type(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("configuration_neo_chat")
    config = module.build_llm_config(
        {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32}
    )
    assert isinstance(config, module.NEOMoELLMConfig)
    assert config.num_experts == 4
    assert not module.is_moe_llm_config({"num_experts": 1})


def test_deep_flow_head_constructs_and_accepts_timesteps(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("flow_matching")
    head = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    with torch.inference_mode():
        outputs = head(torch.randn(5, 8), torch.linspace(0, 1, 5))
    assert outputs.shape == (5, 12)
    assert torch.isfinite(outputs).all()
    assert "net.time_embed.mlp.0.weight" in head.state_dict()
    assert "net.final_layer.adaLN_modulation.1.weight" in head.state_dict()


@pytest.mark.parametrize("widths", [2048, [2048], [[2048]], [1024, 2048]])
def test_vision_config_normalizes_checkpoint_dimensions(
    native_modules: NativeModuleLoader, widths: int | list[int] | list[list[int]]
) -> None:
    module = native_modules("configuration_neo_vit")
    config = module.NEOVisionConfig(llm_hidden_size=widths, downsample_ratio=[0.5])
    expected = (1024, 2048) if widths == [1024, 2048] else (2048,)
    assert config.llm_hidden_size == expected
    assert config.downsample_ratio == (0.5,)
    restored = module.NEOVisionConfig.from_dict(config.to_dict())
    assert restored.llm_hidden_size == expected
    assert restored.downsample_ratio == (0.5,)


def test_flow_head_state_dict_round_trip(native_modules: NativeModuleLoader) -> None:
    module = native_modules("flow_matching")
    original = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    restored = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    hidden_states = torch.randn(5, 8)
    timesteps = torch.linspace(0, 1, 5)
    with torch.inference_mode():
        torch.testing.assert_close(
            original(hidden_states, timesteps), restored(hidden_states, timesteps)
        )


def test_sdpa_grouped_query_matches_explicit_head_repetition(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("attention_backend")
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, 3, 4, 8, generator=generator)
    key = torch.randn(2, 5, 2, 8, generator=generator)
    value = torch.randn(2, 5, 2, 8, generator=generator)
    actual = module.sdpa_attn_func(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.repeat_interleave(2, dim=2).transpose(1, 2),
        value.repeat_interleave(2, dim=2).transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_seeded_generation_preserves_global_rng(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("generation_math")
    previous_state = torch.get_rng_state().clone()
    actual = module.randn_with_seed(
        (2, 3, 4), device="cpu", dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(seed))
            for seed in (7, 19)
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.get_rng_state(), previous_state)


@pytest.mark.parametrize(
    "template_name,expected_prompt",
    [
        (
            "Hermes-2",
            "<|im_start|>system\nsystem<|im_end|>"
            "<|im_start|>user\nrequest-specific prompt<|im_end|>"
            "<|im_start|>assistant\n",
        ),
        (
            "internlm2-chat",
            "<|im_start|>system\nsystem<|im_end|>"
            "<|im_start|>user\nrequest-specific prompt<|im_end|>"
            "<|im_start|>assistant\n",
        ),
        (
            "phi3-chat",
            "<|system|>\nsystem<|end|>"
            "<|user|>\nrequest-specific prompt<|end|>"
            "<|assistant|>\n",
        ),
        (
            "internvl2_5",
            "<|im_start|>system\nsystem<|im_end|>\n"
            "<|im_start|>user\nrequest-specific prompt<|im_end|>\n"
            "<|im_start|>assistant\n",
        ),
        (
            "neo1_0",
            "<|im_start|>system\nsystem<|im_end|>\n"
            "<|im_start|>user\nrequest-specific prompt<|im_end|>\n"
            "<|im_start|>assistant\n",
        ),
    ],
)
def test_conversation_templates_keep_request_histories_independent(
    native_modules: NativeModuleLoader, template_name: str, expected_prompt: str
) -> None:
    conversations = native_modules("conversation")
    first = conversations.get_conv_template(template_name)
    second = conversations.get_conv_template(template_name)
    original_prompt = second.get_prompt()
    first.set_system_message("system")
    first.append_message(first.roles[0], "request-specific prompt")
    first.append_message(first.roles[1], None)
    assert first.get_prompt() == expected_prompt
    assert second.get_prompt() == original_prompt
    assert (
        conversations.get_conv_template(template_name).get_prompt() == original_prompt
    )


def test_conversation_preserves_image_message_and_registration_contract(
    native_modules: NativeModuleLoader,
) -> None:
    conversations = native_modules("conversation")
    template = conversations.get_conv_template("neo1_0")
    template.name = "custom-review-template"
    template.append_message(template.roles[0], ("image question", [], "Default"))
    conversations.register_conv_template(template)
    assert conversations.get_conv_template(template.name).get_prompt() == (
        "<|im_start|>user\nimage question<|im_end|>\n"
    )
    with pytest.raises(AssertionError, match="has been registered"):
        conversations.register_conv_template(template)
    replacement = template.copy()
    replacement.set_system_message("replacement system prompt")
    conversations.register_conv_template(replacement, override=True)
    assert conversations.get_conv_template(template.name).get_prompt() == (
        "<|im_start|>system\nreplacement system prompt<|im_end|>\n"
        "<|im_start|>user\nimage question<|im_end|>\n"
    )


@pytest.mark.parametrize("ratio", [0.5, [0.5], [[0.5]], [[[0.5]]], (0.5,)])
def test_vision_config_preserves_nested_ratio_round_trip(
    native_modules: NativeModuleLoader,
    ratio: (
        float | list[float] | list[list[float]] | list[list[list[float]]] | tuple[float]
    ),
) -> None:
    configuration = native_modules("configuration_neo_vit")
    config = configuration.NEOVisionConfig(downsample_ratio=ratio)
    assert config.downsample_ratio == (0.5,)
    restored = configuration.NEOVisionConfig.from_dict(config.to_dict())
    assert restored.downsample_ratio == (0.5,)


@pytest.mark.parametrize(
    "reference_size,resize_enabled,pixel_budget",
    [((256, 256), False, None), ((768, 512), False, None), ((512, 512), True, 262144)],
)
def test_reference_patch_tensors_preserve_the_admitted_resize(
    native_modules: NativeModuleLoader,
    reference_size: tuple[int, int],
    resize_enabled: bool,
    pixel_budget: int | None,
) -> None:
    """The native model resized references again after serving applied its policy."""
    image_buffer = io.BytesIO()
    Image.new("RGB", reference_size, (64, 128, 192)).save(image_buffer, format="PNG")
    references = prepare_reference_images(
        base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        SenseNovaU1ImageEditSampling.from_params(
            {"do_resize": resize_enabled, "input_max_pixels": pixel_budget}
        ),
    )
    native_images = native_modules("utils")
    pixels, grid = native_images.load_image_native(references[0], 16, 0.5)
    assert (int(grid[0, 1]) * 16, int(grid[0, 0]) * 16) == references[0].size
    reconstructed = (
        pixels.view(int(grid[0, 0]), int(grid[0, 1]), 3, 16, 16)
        .permute(2, 0, 3, 1, 4)
        .reshape(3, references[0].height, references[0].width)
    )
    expected_rgb = torch.tensor([64, 128, 192], dtype=torch.float32) / 255
    expected_rgb = (
        expected_rgb - torch.tensor(native_images.IMAGENET_MEAN)
    ) / torch.tensor(native_images.IMAGENET_STD)
    torch.testing.assert_close(
        reconstructed, expected_rgb[:, None, None].expand_as(reconstructed)
    )
