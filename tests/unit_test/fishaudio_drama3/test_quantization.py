# SPDX-License-Identifier: Apache-2.0
"""bsgemm-moe: block-FP8 experts only, and only the checkpoint's exact format."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from sglang_omni.models.fishaudio_drama3.quantization import ExpertFp8Config

CHECKPOINT_QUANTIZATION = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}


class ExpertLayer(FusedMoE):
    """FusedMoE for isinstance dispatch; its own __init__ needs an SGLang runtime."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


class ProjectionLayer(LinearBase):
    """LinearBase for isinstance dispatch, without building weights."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


def test_only_experts_get_block_fp8() -> None:
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANTIZATION)

    expert_method = config.get_quant_method(ExpertLayer(), "model.layers.0.mlp.experts")
    projection_method = config.get_quant_method(
        ProjectionLayer(), "model.layers.0.mlp.shared_expert.gate_up_proj"
    )
    other_method = config.get_quant_method(torch.nn.Module(), "model.embed_tokens")

    assert isinstance(expert_method, Fp8MoEMethod)
    assert expert_method.block_quant is True
    assert expert_method.weight_block_size == [128, 128]
    assert expert_method.quant_config.activation_scheme == "dynamic"
    assert expert_method.quant_config.is_checkpoint_fp8_serialized is True
    assert isinstance(projection_method, UnquantizedLinearMethod)
    assert other_method is None
    assert config.get_name() == "bsgemm-moe"
    assert config.get_supported_act_dtypes() == [torch.bfloat16, torch.half]
    assert ExpertFp8Config.get_min_capability() == 80
    assert ExpertFp8Config.get_config_filenames() == []
    assert config.get_scaled_act_names() == []


@pytest.mark.parametrize(
    ("config_field_name", "config_field_value"),
    [
        ("quant_method", "fp8"),
        ("activation_scheme", "static"),
        ("weight_block_size", None),
        ("weight_block_size", [0, 128]),
        ("weight_block_size", [-1, 128]),
        ("weight_block_size", [64, 128]),
        ("weight_block_size", [128, 128, 1]),
    ],
)
def test_other_formats_are_rejected(
    config_field_name: str, config_field_value: str | list[int] | None
) -> None:
    quantization = dict(CHECKPOINT_QUANTIZATION)
    quantization[config_field_name] = config_field_value

    with pytest.raises(ValueError, match=config_field_name):
        ExpertFp8Config.from_config(quantization)


def test_missing_block_size_is_rejected() -> None:
    quantization = dict(CHECKPOINT_QUANTIZATION)
    quantization.pop("weight_block_size")

    with pytest.raises(ValueError, match="weight_block_size"):
        ExpertFp8Config.from_config(quantization)
