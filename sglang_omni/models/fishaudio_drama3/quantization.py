# SPDX-License-Identifier: Apache-2.0
"""bsgemm-moe quantization: block-FP8 routed experts, every other layer unquantized."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

QUANTIZATION_METHOD = "bsgemm-moe"
EXPERT_WEIGHT_BLOCK_SIZE = [128, 128]
EXPERT_ACTIVATION_SCHEME = "dynamic"


class ExpertFp8Config(QuantizationConfig):
    """FusedMoE experts are FP8 ([128, 128] weight blocks, dynamic activations)."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=EXPERT_ACTIVATION_SCHEME,
            weight_block_size=list(EXPERT_WEIGHT_BLOCK_SIZE),
        )

    def get_name(self) -> str:
        return QUANTIZATION_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Same floor as Fp8Config, which the experts run under.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ExpertFp8Config:
        # note (SunskyXH): Fp8Config accepts a null, zero or negative block size,
        # so the checkpoint format is checked here instead of delegated.
        if config.get("quant_method") != QUANTIZATION_METHOD:
            raise ValueError(
                f"quant_method={config.get('quant_method')!r}; expected "
                f"{QUANTIZATION_METHOD!r}"
            )
        elif config.get("activation_scheme") != EXPERT_ACTIVATION_SCHEME:
            raise ValueError(
                f"{QUANTIZATION_METHOD} needs activation_scheme="
                f"{EXPERT_ACTIVATION_SCHEME!r}; got {config.get('activation_scheme')!r}"
            )
        elif config.get("weight_block_size") != EXPERT_WEIGHT_BLOCK_SIZE:
            raise ValueError(
                f"{QUANTIZATION_METHOD} needs weight_block_size="
                f"{EXPERT_WEIGHT_BLOCK_SIZE}; got {config.get('weight_block_size')!r}"
            )
        else:
            return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # The interface passes the layer prefix; the choice here depends on the
        # layer type alone.
        if isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self.expert_fp8_config)
        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        else:
            return None

    def get_scaled_act_names(self) -> list[str]:
        return []
