# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils.hub import cached_file


@lru_cache(maxsize=4)
def load_thinker_config(model_id: str) -> Any:
    try:
        config_path = cached_file(model_id, "config.json", local_files_only=True)
        cfg = AutoConfig.from_pretrained(
            str(config_path.parent),
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(cfg, "thinker_config", cfg)


def instantiate_module(module_cls: type[nn.Module], config: Any) -> nn.Module:
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(config)
        return module_cls(config)


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_id: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_id(cls, model_id: str) -> "Qwen3OmniSpec":
        thinker_cfg = load_thinker_config(model_id)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_id=model_id,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )


def log_module_stats(name: str, module: nn.Module, device: str | torch.device) -> None:
    logger = logging.getLogger(__name__)
    try:
        param_count = sum(p.numel() for p in module.parameters())
        byte_count = sum(p.numel() * p.element_size() for p in module.parameters())
    except Exception:
        param_count = 0
        byte_count = 0

    gb = byte_count / (1024**3)
    device_str = str(device)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            torch_device = torch.device(device)
            alloc = torch.cuda.memory_allocated(torch_device)
            reserved = torch.cuda.memory_reserved(torch_device)
            logger.warning(
                "Loaded %s: params=%.2fB, param_bytes=%.2fGB, cuda_alloc=%.2fGB, cuda_reserved=%.2fGB",
                name,
                param_count / 1e9,
                gb,
                alloc / (1024**3),
                reserved / (1024**3),
            )
            return
        except Exception:
            pass
    logger.warning(
        "Loaded %s: params=%.2fB, param_bytes=%.2fGB, device=%s",
        name,
        param_count / 1e9,
        gb,
        device_str,
    )
