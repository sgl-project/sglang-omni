# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni hardware compatibility shims.

Forces ``moe_runner_backend='triton'`` on SM 80 (A100) hosts only, where
flashinfer's cutlass fused_moe kernel raises ``ValueError: Invalid
backend: 80`` during CUDA graph capture.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Escape hatch: force the SM 80 workaround on any hardware.
_FORCE_SM80_MOE_TRITON = "1" == os.environ.get(
    "SGLANG_OMNI_FORCE_SM80_MOE_TRITON", "0"
)


@lru_cache(maxsize=8)
def _device_capability(gpu_id: int) -> tuple[int, int] | None:
    if not isinstance(gpu_id, int) or gpu_id < 0:
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        if gpu_id >= torch.cuda.device_count():
            return None
        return torch.cuda.get_device_capability(gpu_id)
    except Exception:  # noqa: BLE001
        return None


def _is_sm80(gpu_id: int) -> bool:
    if _FORCE_SM80_MOE_TRITON:
        return True
    return _device_capability(gpu_id) == (8, 0)


def _sm80_moe_runner_triton_override(gpu_id: int) -> dict[str, str]:
    if not _is_sm80(gpu_id):
        return {}
    logger.info(
        "sglang_omni_compat: cuda:%d is SM 80 (A100); forcing moe_runner_backend='triton'",
        gpu_id,
    )
    return {"moe_runner_backend": "triton"}


def get_qwen3_omni_compat_overrides(gpu_id: int | None = None) -> dict[str, str]:
    if gpu_id is None:
        return {}
    merged: dict[str, str] = {}
    for src in (_sm80_moe_runner_triton_override,):
        try:
            merged.update(src(gpu_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sglang_omni_compat: %s raised %s; skipping", src.__name__, exc
            )
    return merged
