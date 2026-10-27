# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

import json
import os
from typing import Literal

ClassifierFreeGuidanceNormalization = Literal[
    "none", "global", "channel", "cfg_zero_star"
]

SENSENOVA_U1_REQUEST_EXTRA_KEY: str = "sensenova_u1"

SENSENOVA_U1_MODEL_IDS: set[str] = {
    "sensenova/sensenova-u1.5-8b-mot",
}

SENSENOVA_U1_CFG_NORM_CHOICES: tuple[ClassifierFreeGuidanceNormalization, ...] = (
    "none",
    "global",
    "channel",
    "cfg_zero_star",
)
RESOLUTION_ALIGNMENT: int = 32

DEFAULT_CFG_NORM: ClassifierFreeGuidanceNormalization = "none"
DEFAULT_TIMESTEP_SHIFT: float = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT: bool = True
DEFAULT_CFG_INTERVAL: tuple[float, float] = (0.0, 1.0)
DEFAULT_T_EPS: float = 0.02
DEFAULT_THINK_MODE: bool = False


def is_sensenova_u1_model(model_path: str) -> bool:
    """Identify SenseNova-U1 Hub IDs and local base checkpoints."""
    if os.path.isdir(model_path):
        checkpoint_config_path = os.path.join(model_path, "config.json")
        try:
            with open(checkpoint_config_path) as checkpoint_config_stream:
                checkpoint_config = json.load(checkpoint_config_stream)
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(checkpoint_config, dict):
            return False
        else:
            pass
        model_architectures = checkpoint_config.get("architectures", [])
        return (
            checkpoint_config.get("model_type") == "neo_chat"
            and isinstance(model_architectures, list)
            and "NEOChatModel" in model_architectures
        )
    else:
        pass

    return model_path.rstrip("/").lower() in SENSENOVA_U1_MODEL_IDS
