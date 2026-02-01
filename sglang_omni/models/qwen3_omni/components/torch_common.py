# SPDX-License-Identifier: Apache-2.0
"""Torch-native config helpers for Qwen3-Omni components."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sglang_omni.models.weight_utils import load_config


def load_config_dict(
    model_path: str,
) -> dict[str, Any]:
    return load_config(model_path)


def strip_audio_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy with audio_config disabled for thinker-only use."""
    cfg = deepcopy(config)
    thinker_cfg = cfg.get("thinker_config", cfg)
    if isinstance(thinker_cfg, dict):
        thinker_cfg = dict(thinker_cfg)
        thinker_cfg["audio_config"] = None
        cfg["thinker_config"] = thinker_cfg
    else:
        cfg["audio_config"] = None
    return cfg
