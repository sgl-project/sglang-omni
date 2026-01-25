# SPDX-License-Identifier: Apache-2.0
"""Shared model registry for Qwen3-Omni."""

from __future__ import annotations

from threading import Lock
from typing import Any

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[key]


_REGISTRY: dict[tuple[str, torch.dtype | None], hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration] = {}
_LOCK = Lock()


def get_shared_thinker(
    model_id: str,
    *,
    dtype: str | torch.dtype | None = None,
) -> hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration:
    """Load (or reuse) a thinker model on CPU."""
    torch_dtype = _resolve_dtype(dtype)
    key = (model_id, torch_dtype)
    with _LOCK:
        cached = _REGISTRY.get(key)
        if cached is not None:
            return cached

        try:
            model = hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception:
            model = hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        model.eval()
        _REGISTRY[key] = model
        return model


def describe_registry() -> dict[str, Any]:
    return {
        "entries": [f"{model_id}:{dtype}" for (model_id, dtype) in _REGISTRY.keys()],
        "count": len(_REGISTRY),
    }

