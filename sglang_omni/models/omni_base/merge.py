# SPDX-License-Identifier: Apache-2.0
"""Merge strategies and utilities for omni pipelines."""

from __future__ import annotations

from typing import Any, Callable

import torch

from sglang_omni.proto import StagePayload

# Tensor utilities


def as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    """Convert value to tensor, avoiding unnecessary dtype conversions."""
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if dtype is not None and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value

    if isinstance(value, (list, tuple, int, float, bool)):
        return torch.as_tensor(value, dtype=dtype)

    if hasattr(value, "__array__"):
        try:
            return torch.as_tensor(value, dtype=dtype)
        except (TypeError, ValueError):
            return None

    return None


def as_tensor_list(value: Any) -> list[torch.Tensor] | None:
    """Convert value to list of tensors."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = [v for v in value if isinstance(v, torch.Tensor)]
        return tensors or None
    return None


def is_non_empty(tensor: torch.Tensor | None) -> bool:
    """Check if tensor is non-None and has elements."""
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


# Merge strategies


def create_standard_merge_fn(
    *,
    source_stages: list[str],
    build_thinker_inputs_fn: Callable[[StagePayload, dict[str, Any]], dict[str, Any]],
    prune_fn: Callable[[StagePayload, dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, StagePayload]], StagePayload]:
    """Create a standard merge function for encoder outputs.

    Args:
        source_stages: List of stage names to merge (e.g., ["preprocessing", "image_encoder", "audio_encoder"])
        build_thinker_inputs_fn: Model-specific function to build thinker inputs from encoder outputs
        prune_fn: Optional function to prune unnecessary data before sending to thinker

    Returns:
        A merge function that can be used in pipeline config
    """

    def merge_fn(payloads: dict[str, StagePayload]) -> StagePayload:
        # Use preprocessing as base, or first available payload
        base = payloads.get(source_stages[0]) or next(iter(payloads.values()))

        # Collect encoder outputs from all stages
        encoder_outs: dict[str, Any] = {}

        # Start with base encoder outputs
        if base.encoder_outs:
            encoder_outs.update(base.encoder_outs)

        # Merge encoder outputs from other stages
        for stage_name, payload in payloads.items():
            stage_encoder_outs = payload.encoder_outs or {}
            stage_engine_outputs = payload.engine_outputs or {}

            if stage_name in stage_encoder_outs:
                encoder_outs[stage_name] = stage_encoder_outs[stage_name]
            elif stage_name in stage_engine_outputs:
                encoder_outs[stage_name] = stage_engine_outputs[stage_name]

        # Build thinker inputs using model-specific logic
        thinker_inputs = build_thinker_inputs_fn(base, encoder_outs)

        # Update payload
        base.encoder_outs = encoder_outs
        base.thinker_inputs = thinker_inputs
        base.encoder_inputs = {}  # Clear encoder inputs (no longer needed)

        # Prune unnecessary data if provided
        if prune_fn is not None:
            prune_fn(base, encoder_outs)

        return base

    return merge_fn
