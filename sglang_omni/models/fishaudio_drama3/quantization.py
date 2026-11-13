# SPDX-License-Identifier: Apache-2.0
"""Drama 3 checkpoint quantizations.

bsgemm-moe: block-FP8 routed experts, every other layer unquantized.
drama3-int4: W4A16 GPTQ-Marlin routed experts and the dense linears the
checkpoint lists, every other layer unquantized.
"""

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
from sglang.srt.layers.quantization.gptq.gptq import (
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
    GPTQMarlinMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import replace_parameter

QUANTIZATION_METHOD = "bsgemm-moe"
EXPERT_WEIGHT_BLOCK_SIZE = [128, 128]
EXPERT_ACTIVATION_SCHEME = "dynamic"

INT4_QUANTIZATION_METHOD = "drama3-int4"
INT4_EXPERT_METHOD = "gptq"
INT4_DENSE_METHOD = "gptq_marlin"
# The engine's fused linears, by their last two prefix parts, and the checkpoint
# projections each one loads.
INT4_DENSE_MODULE_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "self_attn.o_proj": ("self_attn.o_proj",),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    "linear_attn.out_proj": ("linear_attn.out_proj",),
}


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


class ExpertMarlinMoEMethod(GPTQMarlinMoEMethod):
    """GPTQ-Marlin routed experts with their scales in the activation dtype."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        assert isinstance(layer, FusedMoE)
        # The scheme loads scales as fp16; the Marlin MoE kernel needs them in the
        # activation dtype.
        replace_parameter(layer, "w13_scales", layer.w13_scales.to(layer.params_dtype))
        replace_parameter(layer, "w2_scales", layer.w2_scales.to(layer.params_dtype))


class ExpertDenseInt4Config(QuantizationConfig):
    """FusedMoE experts and the listed dense linears run GPTQ-Marlin."""

    def __init__(
        self,
        *,
        expert_config: GPTQMarlinConfig,
        dense_config: GPTQMarlinConfig,
        quantized_dense_modules: frozenset[str],
    ) -> None:
        super().__init__()
        self.expert_config = expert_config
        self.dense_config = dense_config
        self.quantized_dense_modules = quantized_dense_modules

    def get_name(self) -> str:
        return INT4_QUANTIZATION_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # GPTQ-Marlin, which runs the experts and the dense linears, needs sm_80.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ExpertDenseInt4Config:
        expert_quantization = config.get("experts")
        dense_quantization = config.get("dense")
        if config.get("quant_method") != INT4_QUANTIZATION_METHOD:
            raise ValueError(
                f"quant_method={config.get('quant_method')!r}; expected "
                f"{INT4_QUANTIZATION_METHOD!r}"
            )
        elif (
            not isinstance(expert_quantization, dict)
            or expert_quantization.get("quant_method") != INT4_EXPERT_METHOD
        ):
            raise ValueError(
                f"{INT4_QUANTIZATION_METHOD} needs an experts config with "
                f"quant_method={INT4_EXPERT_METHOD!r}; got {expert_quantization!r}"
            )
        elif (
            not isinstance(dense_quantization, dict)
            or dense_quantization.get("quant_method") != INT4_DENSE_METHOD
        ):
            raise ValueError(
                f"{INT4_QUANTIZATION_METHOD} needs a dense config with "
                f"quant_method={INT4_DENSE_METHOD!r}; got {dense_quantization!r}"
            )
        else:
            pass
        targets = dense_quantization.get("targets")
        if not isinstance(targets, list) or not all(
            isinstance(target, str) for target in targets
        ):
            raise ValueError(
                f"{INT4_QUANTIZATION_METHOD} dense.targets must be a list of "
                f"projection names; got {targets!r}"
            )
        else:
            pass
        quantized_dense_modules = frozenset(
            module_name
            for module_name, projections in INT4_DENSE_MODULE_PROJECTIONS.items()
            if set(projections) <= set(targets)
        )
        covered_targets = {
            projection
            for module_name in quantized_dense_modules
            for projection in INT4_DENSE_MODULE_PROJECTIONS[module_name]
        }
        # A target outside a whole fused linear would leave its int4 tensors unloaded.
        if covered_targets != set(targets):
            raise ValueError(
                f"{INT4_QUANTIZATION_METHOD} dense.targets "
                f"{sorted(set(targets) - covered_targets)} do not cover whole fused "
                f"linears; supported groups are "
                f"{list(INT4_DENSE_MODULE_PROJECTIONS.values())}"
            )
        else:
            # The experts block omits desc_act: its tensors carry no g_idx, so there
            # is no activation order.
            return cls(
                expert_config=GPTQMarlinConfig.from_config(
                    {"desc_act": False, **expert_quantization}
                ),
                dense_config=GPTQMarlinConfig.from_config(dense_quantization),
                quantized_dense_modules=quantized_dense_modules,
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        module_name = ".".join(prefix.split(".")[-2:])
        if isinstance(layer, FusedMoE):
            return ExpertMarlinMoEMethod(self.expert_config)
        elif (
            isinstance(layer, LinearBase)
            and module_name in self.quantized_dense_modules
        ):
            return GPTQMarlinLinearMethod(self.dense_config)
        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        else:
            return None

    def get_scaled_act_names(self) -> list[str]:
        return []
