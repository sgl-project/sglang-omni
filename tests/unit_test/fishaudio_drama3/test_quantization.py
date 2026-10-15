# SPDX-License-Identifier: Apache-2.0
"""bsgemm-moe and drama3-int4: which layers each quantizes, and only their exact formats."""

from __future__ import annotations

import copy

import pytest
import torch
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinLinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from sglang_omni.models.fishaudio_drama3.quantization import (
    ExpertDenseInt4Config,
    ExpertFp8Config,
    ExpertMarlinMoEMethod,
)

CHECKPOINT_QUANTIZATION = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}
INT4_CHECKPOINT_QUANTIZATION = {
    "quant_method": "drama3-int4",
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "desc_act": False,
    "experts": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True},
    "dense": {
        "quant_method": "gptq_marlin",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "targets": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ],
    },
}
INT4_DENSE_PREFIXES = (
    "model.layers.3.self_attn.qkv_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.0.linear_attn.in_proj_qkvz",
    "model.layers.0.linear_attn.out_proj",
)
BF16_DENSE_PREFIXES = (
    "model.layers.0.linear_attn.in_proj_ba",
    "model.layers.0.mlp.gate",
    "model.layers.0.mlp.shared_expert_gate",
    "model.layers.0.mlp.shared_expert.gate_up_proj",
    "model.layers.0.mlp.shared_expert.down_proj",
)


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


def test_int4_quantizes_experts_and_the_listed_dense_linears() -> None:
    config = ExpertDenseInt4Config.from_config(INT4_CHECKPOINT_QUANTIZATION)

    expert_method = config.get_quant_method(ExpertLayer(), "model.layers.0.mlp.experts")
    other_method = config.get_quant_method(torch.nn.Module(), "model.embed_tokens")

    assert isinstance(expert_method, ExpertMarlinMoEMethod)
    assert expert_method.quant_config.weight_bits == 4
    assert expert_method.quant_config.group_size == 128
    assert expert_method.quant_config.is_sym is True
    assert expert_method.quant_config.desc_act is False
    for prefix in INT4_DENSE_PREFIXES:
        method = config.get_quant_method(ProjectionLayer(), prefix)
        assert isinstance(method, GPTQMarlinLinearMethod), prefix
        assert method.quant_config.weight_bits == 4, prefix
        assert method.quant_config.group_size == 128, prefix
    for prefix in BF16_DENSE_PREFIXES:
        method = config.get_quant_method(ProjectionLayer(), prefix)
        assert isinstance(method, UnquantizedLinearMethod), prefix
    assert other_method is None
    assert config.get_name() == "drama3-int4"
    assert config.get_supported_act_dtypes() == [torch.bfloat16, torch.half]
    assert ExpertDenseInt4Config.get_min_capability() == 80
    assert ExpertDenseInt4Config.get_config_filenames() == []
    assert config.get_scaled_act_names() == []


def test_int4_without_dense_targets_keeps_every_linear_unquantized() -> None:
    quantization = copy.deepcopy(INT4_CHECKPOINT_QUANTIZATION)
    quantization["dense"]["targets"] = []
    config = ExpertDenseInt4Config.from_config(quantization)

    for prefix in INT4_DENSE_PREFIXES + BF16_DENSE_PREFIXES:
        method = config.get_quant_method(ProjectionLayer(), prefix)
        assert isinstance(method, UnquantizedLinearMethod), prefix


@pytest.mark.parametrize(
    ("section_name", "field_name", "field_value", "message"),
    [
        (None, "quant_method", "gptq", "quant_method"),
        (None, "experts", None, "experts"),
        ("experts", "quant_method", "awq", "experts"),
        (None, "dense", None, "dense"),
        ("dense", "quant_method", "gptq", "dense"),
        ("dense", "targets", "self_attn.q_proj", "dense.targets"),
        ("dense", "targets", ["self_attn.q_proj"], "whole fused"),
        ("dense", "targets", ["mlp.shared_expert.gate_proj"], "whole fused"),
    ],
)
def test_int4_other_formats_are_rejected(
    section_name: str | None,
    field_name: str,
    field_value: str | list[str] | None,
    message: str,
) -> None:
    quantization = copy.deepcopy(INT4_CHECKPOINT_QUANTIZATION)
    if section_name is None:
        quantization[field_name] = field_value
    else:
        quantization[section_name][field_name] = field_value

    with pytest.raises(ValueError, match=message):
        ExpertDenseInt4Config.from_config(quantization)
