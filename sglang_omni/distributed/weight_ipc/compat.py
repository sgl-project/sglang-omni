# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for same-GPU CUDA weight IPC."""

from __future__ import annotations

from collections.abc import Sequence

from sglang_omni.distributed.weight_ipc.types import WeightIpcRole

SUPPORTED_WEIGHT_IPC_ARCHITECTURES = frozenset(
    {"HiggsMultimodalQwen3ForConditionalGeneration"}
)


def validate_weight_ipc_compatibility(
    *,
    role: WeightIpcRole,
    architectures: Sequence[str] | None,
    tp_size: int,
    pp_size: int,
) -> None:
    """Fail fast when the requested IPC configuration is unsupported."""
    if role == "off":
        return

    raw_architectures = tuple(architectures or ())
    if (
        len(raw_architectures) != 1
        or not isinstance(raw_architectures[0], str)
        or not raw_architectures[0].strip()
    ):
        raise ValueError(
            "weight IPC requires exactly one model architecture, got "
            f"{list(raw_architectures)!r}"
        )

    architecture = raw_architectures[0].strip()
    if architecture not in SUPPORTED_WEIGHT_IPC_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_WEIGHT_IPC_ARCHITECTURES))
        raise ValueError(
            f"weight IPC is unsupported for architecture {architecture!r}; "
            f"supported architectures: {supported}"
        )
    if tp_size != 1:
        raise ValueError(f"weight IPC requires tp_size=1, got {tp_size}")
    if pp_size != 1:
        raise ValueError(f"weight IPC requires pp_size=1, got {pp_size}")
