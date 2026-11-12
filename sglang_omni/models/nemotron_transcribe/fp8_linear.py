# SPDX-License-Identifier: Apache-2.0
"""Static per-tensor FP8 linears for the speech encoder of ModelOpt FP8 checkpoints."""

from __future__ import annotations

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_GEMM_CAPABILITY = (8, 9)


class Fp8Linear(nn.Module):
    """FP8 weight with per-tensor weight and input scales, named as in the checkpoint."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dense_dtype = dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device),
        )
        self.register_buffer(
            "weight_scale", torch.empty((), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "input_scale", torch.empty((), dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def replacing(cls, linear: nn.Linear) -> Fp8Linear:
        return cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, self.in_features)
        quantized = (flat / self.input_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        output = torch._scaled_mm(  # noqa: leading-underscore - PyTorch operator name
            quantized,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=hidden.dtype,
        )
        return output.view(*hidden.shape[:-1], self.out_features)

    def to_linear(self) -> nn.Linear:
        """Dense linear with the weight scale folded into the weight."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            dtype=self.dense_dtype,
            device=self.weight.device,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight.float() * self.weight_scale)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
            else:
                pass
        return linear


def attach_fp8_linears(
    module: nn.Module, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Swap each linear the checkpoint stores in FP8 for an Fp8Linear.

    Returns the weights with their scales shaped like the Fp8Linear buffers.
    """
    quantized = sorted(
        name[: -len(".weight_scale")]
        for name in weights
        if name.endswith(".weight_scale")
    )
    loadable = dict(weights)
    for name in quantized:
        linear = module.get_submodule(name)
        weight = weights.get(f"{name}.weight")
        weight_scale = weights[f"{name}.weight_scale"]
        input_scale = weights.get(f"{name}.input_scale")
        if (
            not isinstance(linear, nn.Linear)
            or weight is None
            or weight.dtype != FP8_DTYPE
            or weight_scale.numel() != 1
            or input_scale is None
            or input_scale.numel() != 1
        ):
            raise ValueError(
                f"unsupported FP8 perception linear {name}: expected a "
                "float8_e4m3fn weight with one weight_scale and one input_scale"
            )
        else:
            pass
        loadable[f"{name}.weight_scale"] = weight_scale.reshape(())
        loadable[f"{name}.input_scale"] = input_scale.reshape(())
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent), child, Fp8Linear.replacing(linear))
    attached = set(quantized)
    unscaled = [
        name
        for name, tensor in weights.items()
        if is_float8(tensor) and name.rpartition(".")[0] not in attached
    ]
    if unscaled:
        raise ValueError(
            f"FP8 perception tensors without a weight_scale: {unscaled[:8]}"
        )
    else:
        pass
    return loadable


def dequantize_fp8_linears(module: nn.Module) -> int:
    """Replace every Fp8Linear with a dense linear and return how many were replaced."""
    targets = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, Fp8Linear)
    ]
    for name, fp8_linear in targets:
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent), child, fp8_linear.to_linear())
    return len(targets)


def count_fp8_linears(module: nn.Module) -> int:
    return sum(isinstance(child, Fp8Linear) for child in module.modules())


def fp8_gemm_supported(device: torch.device) -> bool:
    """CUDA GPUs from compute capability 8.9 run float8_e4m3fn GEMMs."""
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    else:
        return torch.cuda.get_device_capability(device) >= FP8_GEMM_CAPABILITY


def is_float8(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() and tensor.element_size() == 1


__all__ = [
    "Fp8Linear",
    "attach_fp8_linears",
    "count_fp8_linears",
    "dequantize_fp8_linears",
    "fp8_gemm_supported",
]
