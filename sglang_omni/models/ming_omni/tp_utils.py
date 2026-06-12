# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_MING_TP_STAGES = frozenset({"thinker", "image_encoder"})


def validate_attention_tp_config(
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    tp_size: int,
    context: str,
) -> None:
    """Validate Ming attention head partitioning before SGLang layers assert."""
    if tp_size < 1:
        raise ValueError(f"{context}: tp_size must be >= 1, got {tp_size}")
    if num_attention_heads < 1:
        raise ValueError(
            f"{context}: num_attention_heads must be >= 1, got {num_attention_heads}"
        )
    if num_key_value_heads < 1:
        raise ValueError(
            f"{context}: num_key_value_heads must be >= 1, got {num_key_value_heads}"
        )
    if num_attention_heads % tp_size != 0:
        raise ValueError(
            f"{context}: num_attention_heads={num_attention_heads} must be "
            f"divisible by tp_size={tp_size}"
        )
    if num_key_value_heads >= tp_size:
        if num_key_value_heads % tp_size != 0:
            raise ValueError(
                f"{context}: num_key_value_heads={num_key_value_heads} must be "
                f"divisible by tp_size={tp_size} when KV heads are sharded"
            )
        return
    if tp_size % num_key_value_heads != 0:
        raise ValueError(
            f"{context}: tp_size={tp_size} must be divisible by "
            f"num_key_value_heads={num_key_value_heads} when KV heads are replicated"
        )


def validate_stage_tp_support(*, stage_name: str, tp_size: int) -> None:
    """Reject TP for Ming stages that do not implement SGLang tensor parallelism."""
    if tp_size < 1:
        raise ValueError(f"Stage {stage_name!r}: tp_size must be >= 1, got {tp_size}")
    if tp_size == 1:
        return
    if stage_name not in _MING_TP_STAGES:
        raise ValueError(
            f"Stage {stage_name!r} does not support TP in Ming-Omni V1. "
            f"SGLang-TP is supported for: {', '.join(sorted(_MING_TP_STAGES))}."
        )
