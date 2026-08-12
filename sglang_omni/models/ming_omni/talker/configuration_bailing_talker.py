# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni talker configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MingOmniTalkerConfig:
    """Explicit config for the MingOmniTalker."""

    def __init__(
        self,
        *,
        # LLM backbone (Qwen2)
        llm_config: dict | None = None,
        # Flow matching (DiT)
        flowmodel: dict | None = None,
        steps: int = 10,
        # Aggregator
        aggregator: dict | None = None,
        # Audio patch sizes
        patch_size: int = 4,
        history_patch_size: int = 32,
        # Latent / speaker dims
        latent_dim: int = 64,
        spk_dim: int = 192,
        # Concurrency
        max_conc: int = 1,
    ):
        self.llm_config = llm_config or {}
        self.flowmodel = flowmodel or {}
        self.steps = steps
        self.aggregator = aggregator or {}
        self.patch_size = patch_size
        self.history_patch_size = history_patch_size
        self.latent_dim = latent_dim
        self.spk_dim = spk_dim
        self.max_conc = normalize_max_conc(max_conc)

    @classmethod
    def from_pretrained_dir(cls, model_dir: str) -> MingOmniTalkerConfig:
        """Load config from a talker checkpoint directory.

        Reads ``<model_dir>/config.json`` for the top-level talker config
        and ``<model_dir>/llm/config.json`` for the Qwen2 backbone config.
        """
        model_dir = Path(model_dir)

        # Top-level talker config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # LLM sub-config
        llm_config_path = model_dir / "llm" / "config.json"
        llm_config: dict = {}
        if llm_config_path.exists():
            with llm_config_path.open("r", encoding="utf-8") as f:
                llm_config = json.load(f)

        return cls(
            llm_config=llm_config,
            flowmodel=raw.get("flowmodel", {}),
            steps=raw.get("steps", 10),
            aggregator=raw.get("aggregator", {}),
            patch_size=raw.get("patch_size", 4),
            history_patch_size=raw.get("history_patch_size", 32),
            latent_dim=raw.get("latent_dim", 64),
            spk_dim=raw.get("spk_dim", 192),
            max_conc=raw.get("max_conc", 1),
        )


def normalize_max_conc(value: int | None, *, default: int = 1) -> int:
    """Normalize talker concurrency to an int >= 1."""
    if value is None:
        return default
    conc = int(value)
    if conc < 1:
        raise ValueError(f"max_conc must be >= 1, got {conc}")
    return conc


def resolve_talker_max_conc(
    talker_model_path: str,
    max_conc: int | None = None,
) -> int:
    """Resolve effective talker max_conc.

    Explicit overrides win. Otherwise read from the talker checkpoint config,
    falling back to 1 when the config file is missing.
    """
    if max_conc is not None:
        return normalize_max_conc(max_conc)
    try:
        return MingOmniTalkerConfig.from_pretrained_dir(talker_model_path).max_conc
    except FileNotFoundError:
        return 1
