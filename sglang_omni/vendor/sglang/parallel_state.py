"""Compatibility helpers for SGLang's ``ParallelState`` wrapper."""

from __future__ import annotations

import dataclasses
from typing import Any


def create_parallel_state(
    parallel_state_cls: type,
    *,
    tp_rank: int,
    dcp_size: int,
    **fields: Any,
):
    """Construct ``ParallelState`` across the DCP field-layout transition."""
    field_names = {field.name for field in dataclasses.fields(parallel_state_cls)}
    if "dcp_size" in field_names:
        dcp_fields = {"dcp_size": dcp_size}
    elif {"attn_dcp_rank", "attn_dcp_size"} <= field_names:
        dcp_fields = {
            "attn_dcp_rank": tp_rank % dcp_size,
            "attn_dcp_size": dcp_size,
        }
    else:
        raise TypeError(
            "Unsupported SGLang ParallelState DCP fields: " f"{sorted(field_names)}"
        )

    return parallel_state_cls(tp_rank=tp_rank, **dcp_fields, **fields)


__all__ = ["create_parallel_state"]
