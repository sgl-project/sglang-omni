# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.weight_loader import resolve_model_path

VOXCPM2_MODEL_TYPE = "voxcpm2"
VOXCPM2_MODEL_ARCH_OVERRIDE = "VoxCPM2SGLangModel"

_voxcpm2_hf_config_registered = False


@dataclass
class VoxCPM2RuntimeConfig:
    """Normalized VoxCPM2 configuration for one checkpoint directory."""

    model_path: str
    lm: dict[str, Any] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)
    dit: dict[str, Any] = field(default_factory=dict)
    audio_vae: dict[str, Any] = field(default_factory=dict)
    patch_size: int = C.PATCH_SIZE
    feat_dim: int = C.FEAT_DIM
    residual_lm_num_layers: int = 0
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 0
    scalar_quantization_scale: int = 0
    max_length: int = 0
    dtype: str = "bfloat16"

    @property
    def sample_rate(self) -> int:
        return int(self.audio_vae.get("sample_rate", C.SAMPLE_RATE))

    @property
    def out_sample_rate(self) -> int:
        return int(self.audio_vae.get("out_sample_rate", C.OUT_SAMPLE_RATE))

    @property
    def latent_dim(self) -> int:
        return int(self.audio_vae.get("latent_dim", self.feat_dim))

    @property
    def cfm(self) -> dict[str, Any]:
        return dict(self.dit.get("cfm_config") or {})

    @property
    def dit_mean_mode(self) -> bool:
        return bool(self.dit.get("mean_mode", False))


def load_voxcpm2_config(
    model_path: str, *, local_files_only: bool = False
) -> VoxCPM2RuntimeConfig:
    """Read ``config.json`` from a VoxCPM2 checkpoint."""
    root = Path(resolve_model_path(model_path, local_files_only=local_files_only))
    with (root / C.CONFIG_FILE).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle) or {}

    def _section(key: str) -> dict[str, Any]:
        value = raw.get(key)
        return dict(value) if isinstance(value, dict) else {}

    return VoxCPM2RuntimeConfig(
        model_path=str(model_path),
        lm=_section("lm_config"),
        encoder=_section("encoder_config"),
        dit=_section("dit_config"),
        audio_vae=_section("audio_vae_config"),
        patch_size=int(raw.get("patch_size", C.PATCH_SIZE)),
        feat_dim=int(raw.get("feat_dim", C.FEAT_DIM)),
        residual_lm_num_layers=int(raw.get("residual_lm_num_layers", 0)),
        residual_lm_no_rope=bool(raw.get("residual_lm_no_rope", False)),
        scalar_quantization_latent_dim=int(
            raw.get("scalar_quantization_latent_dim", 0)
        ),
        scalar_quantization_scale=int(raw.get("scalar_quantization_scale", 0)),
        max_length=int(raw.get("max_length", 0)),
        dtype=str(raw.get("dtype", "bfloat16")),
    )


class VoxCPM2Config(PretrainedConfig):
    """The AR stacks as one flat model, which is what SGLang sizes itself from."""

    model_type = VOXCPM2_MODEL_TYPE

    def __init__(
        self,
        lm_config: dict[str, Any] | PretrainedConfig | None = None,
        voxcpm2_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # note (Xinhao Tan): VoxCPM2 has two AR stacks, the base LM and the
        # residual acoustic LM, and sglang_model lays them out as one flat list
        # of layers sharing a single KV pool. So num_hidden_layers below is the
        # sum of both stacks, not the base stack's own depth, and this config is
        # the flattened view of the pair rather than a copy of either one.
        #
        # That sum only reaches SGLang through this top-level config: SGLang
        # picks what it sizes the pool from by looking for text_config /
        # llm_config / language_config / thinker_config, "lm_config" matches
        # none of them, so it falls back here. Renaming this to llm_config hands
        # SGLang the base stack's depth instead and leaves the pool too small
        # for the residual stack's layer ids.
        if isinstance(lm_config, dict):
            lm_config = PretrainedConfig(**lm_config)
        self.lm_config = lm_config
        self.voxcpm2_config = dict(voxcpm2_config or {})

        if lm_config is not None:
            base_layers = int(getattr(lm_config, "num_hidden_layers", 0))
            residual_layers = int(self.voxcpm2_config.get("residual_lm_num_layers", 0))
            kwargs.setdefault("num_hidden_layers", base_layers + residual_layers)
            for field_name in (
                "hidden_size",
                "intermediate_size",
                "max_position_embeddings",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "rope_theta",
                "rope_scaling",
                "vocab_size",
                "scale_emb",
                "dim_model_base",
                "scale_depth",
                "use_mup",
            ):
                value = getattr(lm_config, field_name, None)
                if value is not None:
                    kwargs.setdefault(field_name, value)
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "VoxCPM2Config":
        merged = dict(config_dict)
        merged.setdefault("voxcpm2_config", dict(config_dict))
        return super().from_dict(merged, **kwargs)


def register_voxcpm2_hf_config() -> None:
    """Register the local VoxCPM2 config before SGLang builds its ModelConfig."""
    global _voxcpm2_hf_config_registered
    if _voxcpm2_hf_config_registered:
        return

    from transformers import AutoConfig

    AutoConfig.register(VOXCPM2_MODEL_TYPE, VoxCPM2Config, exist_ok=True)
    _voxcpm2_hf_config_registered = True


__all__ = [
    "VOXCPM2_MODEL_ARCH_OVERRIDE",
    "VOXCPM2_MODEL_TYPE",
    "VoxCPM2Config",
    "VoxCPM2RuntimeConfig",
    "load_voxcpm2_config",
    "register_voxcpm2_hf_config",
]
