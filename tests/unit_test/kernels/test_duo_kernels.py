"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import copy

import pytest
import torch
from accelerate import dispatch_model
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

from sglang_omni.models.duo.kernels import SGLangQwenMLP, use_intent_kernels


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 64])
def test_native_swiglu_preserves_loaded_projection_order(tokens: int) -> None:
    torch.manual_seed(42)
    reference_intent = Qwen3VLTextMLP(
        Qwen3VLTextConfig(hidden_size=128, intermediate_size=256)
    ).to("cuda", torch.bfloat16)
    native_intent = copy.deepcopy(reference_intent)
    native_intent.__class__ = SGLangQwenMLP
    hidden = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(
            native_intent(hidden), reference_intent(hidden), rtol=0, atol=0
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_intent_native_activation_replaces_dispatched_forward() -> None:
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 128,
        },
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "out_hidden_size": 128,
            "depth": 1,
            "num_heads": 4,
        },
    )
    model = Qwen3VLForConditionalGeneration(config).to("cuda", torch.bfloat16)
    dispatch_model(model, device_map={"": "cuda:0"}, force_hooks=True)
    original = model.model.language_model.layers[0].mlp
    hidden = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(hidden)
        use_intent_kernels(model)
        replacement = model.model.language_model.layers[0].mlp
        assert replacement.gate_proj is original.gate_proj
        assert replacement.up_proj is original.up_proj
        assert replacement.down_proj is original.down_proj
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            actual = replacement(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any(
        event.key == "sglang::_run_activation_with_rounding_inplace"
        for event in profile.key_averages()
    )
