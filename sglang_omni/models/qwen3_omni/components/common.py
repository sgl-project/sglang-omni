# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.utils import load_hf_config


def load_torch_component(
    model_cls: type[nn.Module],
    config: Any,
    model_path: str,
    *,
    prefix: str | tuple[str, ...],
    dtype: torch.dtype | None,
    device: str | torch.device,
    strict: bool = True,
) -> nn.Module:
    from sglang_omni.models.weight_loader import load_module
    from sglang_omni.utils import instantiate_module

    return load_module(
        instantiate_module(model_cls, config),
        model_path,
        prefix=prefix,
        dtype=dtype,
        device=device,
        strict=strict,
    )


def load_thinker_config(model_path: str) -> Any:
    cfg = load_hf_config(model_path, trust_remote_code=True, local_files_only=True)
    return deepcopy(cfg.thinker_config)


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_path: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_path(cls, model_path: str) -> "Qwen3OmniSpec":
        thinker_cfg = load_thinker_config(model_path)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_path=model_path,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )
