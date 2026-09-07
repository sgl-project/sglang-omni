# SPDX-License-Identifier: Apache-2.0
"""Native MLX backend primitives for Qwen3-Omni.

Configuration and checkpoint helpers are usable without MLX installed.
Shared math primitives load lazily so Torch encoder imports do not require
the optional Apple backend.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from sglang_omni.models.qwen3_omni.mlx.config import (
    CodePredictorConfig,
    MoeTextConfig,
    QuantizationConfig,
    Qwen3OmniMlxConfig,
    TalkerConfig,
    ThinkerConfig,
)

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_omni.mlx.common import (
        SparseMoeBlock,
        apply_multimodal_rope,
        quantize_converted_module,
        sanitize_qwen3_omni_weights,
        tie_lm_head_weights,
    )

__all__ = [
    "CodePredictorConfig",
    "MoeTextConfig",
    "QuantizationConfig",
    "Qwen3OmniMlxConfig",
    "SparseMoeBlock",
    "TalkerConfig",
    "ThinkerConfig",
    "apply_multimodal_rope",
    "quantize_converted_module",
    "sanitize_qwen3_omni_weights",
    "tie_lm_head_weights",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.common"), name)
    globals()[name] = value
    return value
