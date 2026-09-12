# SPDX-License-Identifier: Apache-2.0
"""Eager native Metal INT4 execution of already-calibrated HF weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
GROUP_SIZES = (32, 64, 128, 256)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Read unsigned, least-significant-first nibbles along the last dimension."""
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("Packed HF INT4 weights must be a matrix of int32 words")
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & 15).flatten(1).to(torch.uint8)


class MpsQuantizedLinear(nn.Module):
    """Keep weights packed throughout inference; no dense weight cache."""

    @staticmethod
    def _validate_target(bits, dtype, device):
        if bits != 4 or dtype not in _DTYPES or torch.device(device).type != "mps":
            raise ValueError(
                "HF quantized inference requires INT4 and floating MPS activations"
            )
        for operator in ("_convert_weight_to_int4pack", "_weight_int4pack_mm"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                f"aten::{operator}", "MPS"
            ):
                raise RuntimeError(
                    f"Native MPS INT4 requires aten::{operator}; no CPU fallback"
                )

    def __init__(self, codes, scales, zeros, group_size, *, bias=None, dtype, device):
        super().__init__()
        self._validate_target(4, dtype, device)
        if (
            codes.ndim != 2
            or codes.dtype != torch.uint8
            or min(codes.shape) < 1
            or codes.device.type != "cpu"
            or torch.any(codes > 15)
            or group_size not in GROUP_SIZES
            or codes.shape[1] % group_size
        ):
            raise ValueError("Invalid INT4 codes or group size")
        self.out_features, self.in_features = codes.shape
        self.group_size = group_size
        expected = (self.out_features, self.in_features // group_size)
        # AutoRound may use signed scales; preserve the export's calibrated values.
        if (
            tuple(scales.shape) != expected
            or tuple(zeros.shape) != expected
            or not scales.is_floating_point()
            or not torch.isfinite(scales).all()
            or not torch.isfinite(zeros).all()
        ):
            raise ValueError("Invalid INT4 scale/zero shape or values")
        if bias is not None and tuple(bias.shape) != (self.out_features,):
            raise ValueError("Invalid INT4 linear bias shape")
        self.padded_in_features = (
            (self.in_features + max(128, group_size) - 1) // max(128, group_size)
        ) * max(128, group_size)
        padded_rows = (self.out_features + 7) // 8 * 8
        codes = F.pad(
            codes,
            (
                0,
                self.padded_in_features - self.in_features,
                0,
                padded_rows - self.out_features,
            ),
        )
        pairs = ((codes[:, ::2] << 4) | codes[:, 1::2]).contiguous().to(device)
        self.register_buffer(
            "packed_weight", torch.ops.aten._convert_weight_to_int4pack(pairs, 8)
        )
        padding = (
            0,
            self.padded_in_features // group_size - expected[1],
            0,
            padded_rows - expected[0],
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.stack(
                (F.pad(scales.float(), padding), F.pad(zeros.float(), padding)), dim=-1
            )
            .transpose(0, 1)
            .contiguous()
            .to(device),
        )
        self.register_buffer(
            "bias", None if bias is None else bias.to(device=device, dtype=dtype)
        )

    def forward(self, hidden_states):
        if (
            hidden_states.device.type != "mps"
            or hidden_states.dtype not in _DTYPES
            or hidden_states.ndim < 1
            or hidden_states.shape[-1] != self.in_features
        ):
            raise ValueError(
                f"Expected floating MPS activations with width {self.in_features}"
            )
        shape = (*hidden_states.shape[:-1], self.out_features)
        if hidden_states.numel() == 0:
            return hidden_states.new_empty(shape)
        # ATen interprets qparams in the activation dtype. Use FP32 for both.
        x = F.pad(
            hidden_states.reshape(-1, self.in_features).float(),
            (0, self.padded_in_features - self.in_features),
        ).contiguous()
        result = torch.ops.aten._weight_int4pack_mm(
            x, self.packed_weight, self.group_size, self.scales_and_zeros
        )[:, : self.out_features]
        if self.bias is not None:
            result = result + self.bias.float()
        return result.to(hidden_states.dtype).reshape(shape)


class MpsQuantizedExperts(nn.Module):
    """Per-expert projections avoid materializing Transformers' dense MoE stacks."""

    def __init__(self, gate_projs, up_projs, down_projs, act_fn):
        super().__init__()
        if not (len(gate_projs) == len(up_projs) == len(down_projs)):
            raise ValueError("Expert projection counts must agree")
        self.num_experts = len(gate_projs)
        self.gate_projs = nn.ModuleList(gate_projs)
        self.up_projs = nn.ModuleList(up_projs)
        self.down_projs = nn.ModuleList(down_projs)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if (
            top_k_index.shape != top_k_weights.shape
            or top_k_index.ndim != 2
            or top_k_index.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "Expert routing must provide matching indices/weights per token"
            )
        output = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index).tolist():
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"Invalid routed expert index {expert}")
            token, slot = torch.where(top_k_index == expert)
            selected = hidden_states[token]
            intermediate = self.act_fn(self.gate_projs[expert](selected))
            current = self.down_projs[expert](
                intermediate * self.up_projs[expert](selected)
            )
            output.index_add_(
                0, token, (current * top_k_weights[token, slot, None]).to(output.dtype)
            )
        return output
