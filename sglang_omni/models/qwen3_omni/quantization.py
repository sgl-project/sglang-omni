# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni quantization compatibility helpers."""

from __future__ import annotations

import torch


def convert_fp8_weight_scale_inv_for_sglang(
    target_name: str,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    """Convert Qwen3-Omni inverse FP8 scales to SGLang runtime scales."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight
    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")

    return torch.reciprocal(loaded_weight)
