# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint configuration."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.weight_loader import resolve_model_path

VOXCPM2_MODEL_TYPE = "voxcpm2"
VOXCPM2_MODEL_ARCH_OVERRIDE = "VoxCPM2SGLangModel"

voxcpm2_hf_config_registered = False


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

    def section(key: str) -> dict[str, Any]:
        value = raw.get(key)
        return dict(value) if isinstance(value, dict) else {}

    return VoxCPM2RuntimeConfig(
        model_path=str(model_path),
        lm=section("lm_config"),
        encoder=section("encoder_config"),
        dit=section("dit_config"),
        audio_vae=section("audio_vae_config"),
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
            # note (Xinhao Tan): hidden_act is absent from the checkpoint, but
            # SGLang's MiniCPMMLP reads it off the config. Upstream hardcodes
            # nn.SiLU in its own MLP, so silu is what the weights were trained
            # with; this is filling in a field, not choosing an activation.
            lm_config = PretrainedConfig(**{"hidden_act": "silu", **lm_config})
        self.lm_config = lm_config
        self.voxcpm2_config = dict(voxcpm2_config or {})

        # The checkpoint names its architecture in the singular "architecture"
        # key, so PretrainedConfig leaves architectures empty and SGLang's
        # ModelConfig indexes into None while resolving the model class.
        kwargs.setdefault("architectures", [VOXCPM2_MODEL_ARCH_OVERRIDE])

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
    global voxcpm2_hf_config_registered
    if voxcpm2_hf_config_registered:
        return

    from transformers import AutoConfig

    AutoConfig.register(VOXCPM2_MODEL_TYPE, VoxCPM2Config, exist_ok=True)
    voxcpm2_hf_config_registered = True


__all__ = [
    "VOXCPM2_MODEL_ARCH_OVERRIDE",
    "VOXCPM2_MODEL_TYPE",
    "VoxCPM2Config",
    "VoxCPM2RuntimeConfig",
    "load_voxcpm2_config",
    "register_voxcpm2_hf_config",
    "stage_checkpoint_for_autoconfig",
]


def stage_checkpoint_for_autoconfig(checkpoint: str) -> str:
    """Expose the checkpoint through a directory whose config carries model_type.

    note (Xinhao Tan): VoxCPM2's config.json has no ``model_type``, and
    transformers' AutoConfig keys its registry on exactly that field, so the
    engine cannot load the checkpoint directory as published. Editing the
    snapshot in place would corrupt a cache entry every other tool shares, so
    this stages a sibling directory that symlinks the weights and carries a
    config of its own.
    """
    root = Path(checkpoint).resolve()
    raw = json.loads((root / C.CONFIG_FILE).read_text(encoding="utf-8"))
    if raw.get("model_type") == VOXCPM2_MODEL_TYPE:
        return checkpoint

    staged = root.parent / f"{root.name}-sglang-omni"
    staged.mkdir(parents=True, exist_ok=True)
    for entry in root.iterdir():
        if entry.name == C.CONFIG_FILE:
            continue
        link = staged / entry.name
        if link.is_symlink() and not link.exists():
            # note (Xinhao Tan): stale links survive the existence guard below.
            # Publish their replacement atomically so concurrent loaders never
            # observe a missing path between unlink and symlink creation.
            with tempfile.TemporaryDirectory(prefix=".link-", dir=staged) as tmp:
                replacement = Path(tmp) / entry.name
                replacement.symlink_to(entry)
                replacement.replace(link)
            continue
        # note (Xinhao Tan): exists() is false for dangling symlinks. Another
        # builder can also create the link between this check and symlink_to().
        if link.is_symlink() or link.exists():
            continue
        try:
            link.symlink_to(entry)
        except FileExistsError:
            pass

    raw["model_type"] = VOXCPM2_MODEL_TYPE
    # note (Xinhao Tan): another engine may read this directory during startup.
    # Writing the visible config in place exposes empty or incomplete JSON.
    with tempfile.TemporaryDirectory(prefix=".config-", dir=staged) as tmp:
        config = Path(tmp) / C.CONFIG_FILE
        config.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        config.replace(staged / C.CONFIG_FILE)
    return str(staged)
