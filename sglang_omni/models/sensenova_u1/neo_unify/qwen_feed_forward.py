"""Qwen feed forward."""

import torch
import torch._dynamo
import torch.nn.functional as F
from sglang.kernels.fused_op import KernelBackend
from sglang.srt.layers.layernorm import RMSNorm
from torch import nn
from transformers import Qwen3Config
from transformers.activations import ACT2FN

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    npu_swiglu_available,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class SenseNovaRMSNorm(RMSNorm):
    """Use shared accelerator dispatch and native math for arbitrary CPU inputs."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_residual_addition: torch.Tensor | None = None,
        quant_linear: nn.Module | None = None,
        *,
        backend: KernelBackend | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.device.type == "cpu":
            return self.forward_native(
                x,
                residual,
                post_residual_addition,
            )
        else:
            return super().forward(
                x,
                residual,
                post_residual_addition,
                quant_linear,
                backend=backend,
            )


def make_qwen3_rms_norm(hidden_size: int, eps: float) -> RMSNorm:
    return SenseNovaRMSNorm(
        hidden_size,
        eps=eps,
        cast_x_before_out_mul=True,
    )


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config: Qwen3Config = config
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size
        self.gate_proj: nn.Linear = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj: nn.Linear = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn: nn.Module = ACT2FN[config.hidden_act]
        self.npu_gate_up_weight: torch.Tensor | None = None

    def pack_npu_gate_up_weights(
        self,
    ) -> torch.Tensor:
        reference = self.gate_proj.weight
        packed = self.npu_gate_up_weight
        if (
            packed is not None
            and packed.device == reference.device
            and packed.dtype == reference.dtype
            and packed.untyped_storage().data_ptr()
            == reference.untyped_storage().data_ptr()
        ):
            return packed
        else:
            pass

        with torch.no_grad():
            packed = torch.cat(
                [self.gate_proj.weight, self.up_proj.weight], dim=0
            ).contiguous()
            self.gate_proj.weight.set_(packed[: self.intermediate_size])
            self.up_proj.weight.set_(packed[self.intermediate_size :])
        self.npu_gate_up_weight = packed
        return packed

    def use_npu_fused_mlp(self, x: torch.Tensor) -> bool:
        if (
            x.device.type != "npu"
            or self.training
            or torch.is_grad_enabled()
            or x.dtype != torch.bfloat16
            or self.gate_proj.weight.dtype != x.dtype
            or self.config.hidden_act != "silu"
        ):
            return False
        else:
            pass
        return npu_swiglu_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_npu_fused_mlp(x):
            gate_up = F.linear(x, self.pack_npu_gate_up_weights())
            return self.down_proj(torch.ops.npu.npu_swiglu(gate_up))
        else:
            pass
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
