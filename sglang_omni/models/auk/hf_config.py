# SPDX-License-Identifier: Apache-2.0
"""AuK checkpoint configuration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sglang_omni.models.auk import constants as C

logger = logging.getLogger(__name__)

CONFIG_YAML_NAMES = ("config.yaml", "config.yml")


@dataclass
class AuKRuntimeConfig:
    """Normalized AuK configuration for one checkpoint directory."""

    model_path: str
    name: str = "AuK"
    arch: dict[str, Any] = field(default_factory=dict)
    vae: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    text_encoder_path: str = C.DEFAULT_TEXT_ENCODER

    @property
    def is_flash(self) -> bool:
        return self.name == "AuK-Flash"

    @property
    def sample_rate(self) -> int:
        return int(self.vae.get("target_sample_rate", C.SAMPLE_RATE))

    @property
    def downsample_rate(self) -> int:
        return int(self.vae.get("downsample_rate", C.VAE_DOWNSAMPLE_RATE))

    @property
    def latent_dim(self) -> int:
        return int(self.vae.get("latent_dim", C.LATENT_DIM))

    @property
    def vae_init_kwargs(self) -> dict[str, Any]:
        kwargs = self.vae.get("model_init_kwargs") or {}
        return dict(kwargs)

    def seconds_to_frames(self, seconds: float) -> int:
        return max(1, math.ceil(seconds * self.sample_rate / self.downsample_rate))

    def frames_to_seconds(self, frames: int) -> float:
        return frames * self.downsample_rate / self.sample_rate


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return loaded if isinstance(loaded, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def load_auk_config(model_path: str) -> AuKRuntimeConfig:
    """Read config.yaml or config.json from a checkpoint."""
    root = Path(model_path)
    raw: dict[str, Any] = {}
    for name in CONFIG_YAML_NAMES:
        candidate = root / name
        if candidate.is_file():
            raw = _load_yaml(candidate)
            break
    else:
        config_json = root / "config.json"
        if config_json.is_file():
            raw = _load_json(config_json)

    model = raw.get("model") if isinstance(raw.get("model"), dict) else raw
    model = model or {}

    arch = dict(model.get("arch") or {})
    vae = dict(model.get("vae") or {})
    schedule = dict(model.get("schedule") or {})
    text_encoder = model.get("text_encoder") or {}
    text_encoder_path = (
        text_encoder.get("text_encoder_path")
        if isinstance(text_encoder, dict)
        else None
    )

    name = str(model.get("name") or raw.get("name") or "AuK")
    if not arch:
        logger.warning(
            "AuK: no model.arch section under %s; falling back to architecture defaults",
            model_path,
        )

    return AuKRuntimeConfig(
        model_path=str(model_path),
        name=name,
        arch=arch,
        vae=vae,
        schedule=schedule,
        text_encoder_path=(
            str(text_encoder_path) if text_encoder_path else C.DEFAULT_TEXT_ENCODER
        ),
    )


def make_runtime_config(
    model_path: str,
    *,
    text_encoder_path: str | None = None,
) -> AuKRuntimeConfig:
    """Load a config, applying command-line overrides."""
    config = load_auk_config(model_path)
    if text_encoder_path:
        config.text_encoder_path = text_encoder_path
    return config
