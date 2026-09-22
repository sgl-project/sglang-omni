# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (C) 2026 Tencent. All rights reserved.
"""AuK checkpoint configuration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf

from sglang_omni.models.auk import constants as C

logger = logging.getLogger(__name__)

CONFIG_YAML_NAMES = ("config.yaml", "config.yml")

SectionT = TypeVar("SectionT", bound="FromCheckpointSection")


class FromCheckpointSection:
    """Ignore training-only keys when reading inference config dataclasses."""

    @classmethod
    def from_dict(cls: type[SectionT], config_dict: dict[str, Any] | None) -> SectionT:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (config_dict or {}).items() if k in valid})


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


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return loaded if isinstance(loaded, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
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
            raw = load_yaml(candidate)
            break
        else:
            pass
    else:
        config_json = root / "config.json"
        if config_json.is_file():
            raw = load_json(config_json)
        else:
            pass

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
    else:
        pass

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
    else:
        pass
    return config


@dataclass
class AuKDitConfig(FromCheckpointSection):
    """Backbone hyperparameters."""

    dim: int = 1024
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    num_layers: int = 8
    num_single_layers: int = 24
    latent_dim: int = 64
    attn_mask_enabled: bool = True
    depth: int = 8


@dataclass
class AuKVAEConfig(FromCheckpointSection):
    upsample_rates: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: list[int] = field(
        default_factory=lambda: [10, 8, 6, 4, 4, 4]
    )
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    downsample_rates: list[int] = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: list[int] = field(
        default_factory=lambda: [12, 24, 48, 96, 192, 384, 768]
    )
    snake_logscale: bool = True
    latent_dim: int = 64
    use_vae: bool = True
    causal: bool = True
    flow_hidden_channels: int = 256
    act_causal: bool = True

    @property
    def hop_size(self) -> int:
        return math.prod(self.downsample_rates)
