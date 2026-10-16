# SPDX-License-Identifier: MIT
"""Inference model configuration for Irodori-TTS."""

from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar


@dataclass
class ModelConfig:
    flow_parameterization: str = "rf_velocity"
    latent_dim: int = 128
    latent_patch_size: int = 1
    model_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 2.875
    text_mlp_ratio: float | None = 2.6
    speaker_mlp_ratio: float | None = 2.6
    dropout: float = 0.0
    text_vocab_size: int = 102400
    text_tokenizer_repo: str = "sbintuitions/sarashina2.2-0.5b"
    text_encoder_revision: str | None = None
    text_add_bos: bool = True
    text_encoder_type: str = "scratch"
    pretrained_projector_type: str = "linear"
    pretrained_projector_hidden_ratio: float = 2.0
    pretrained_projector_dropout: float = 0.0
    text_dim: int = 1280
    text_layers: int = 14
    text_heads: int = 10
    use_caption_condition: bool = False
    use_speaker_condition: bool | None = None
    caption_vocab_size: int | None = None
    caption_tokenizer_repo: str | None = None
    caption_add_bos: bool | None = None
    caption_dim: int | None = None
    caption_layers: int | None = None
    caption_heads: int | None = None
    caption_mlp_ratio: float | None = None
    speaker_dim: int = 1280
    speaker_layers: int = 14
    speaker_heads: int = 10
    speaker_patch_size: int = 1
    timestep_embed_dim: int = 512
    adaln_rank: int = 256
    norm_eps: float = 1e-5
    use_duration_predictor: bool = False
    duration_aux_dim: int = 14
    duration_hidden_dim: int = 1024
    duration_layers: int = 3
    duration_dropout: float = 0.1
    duration_attention_heads: int = 8
    duration_architecture: str = "token_sum_adarn_zero_no_aux"
    duration_token_init_frames: float = 9.0
    duration_speaker_fusion: str = "adarn_zero"
    duration_caption_fusion: str = "adarn_zero"
    duration_caption_pooling: str = "masked_mean"

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def use_speaker_condition_resolved(self) -> bool:
        # Legacy compatibility: old caption configs implied no speaker branch.
        if self.use_speaker_condition is None:
            return not bool(self.use_caption_condition)
        else:
            pass
        return bool(self.use_speaker_condition)

    @property
    def text_mlp_ratio_resolved(self) -> float:
        if self.text_mlp_ratio is None:
            return self.mlp_ratio
        else:
            pass
        return float(self.text_mlp_ratio)

    @property
    def use_pretrained_text_encoder(self) -> bool:
        return str(self.text_encoder_type).strip().lower() == "pretrained"

    @property
    def caption_vocab_size_resolved(self) -> int:
        if self.caption_vocab_size is None:
            return int(self.text_vocab_size)
        else:
            pass
        return int(self.caption_vocab_size)

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        if self.caption_tokenizer_repo is None:
            return self.text_tokenizer_repo
        else:
            pass
        return str(self.caption_tokenizer_repo)

    @property
    def caption_add_bos_resolved(self) -> bool:
        if self.caption_add_bos is None:
            return bool(self.text_add_bos)
        else:
            pass
        return bool(self.caption_add_bos)

    @property
    def caption_dim_resolved(self) -> int:
        if self.caption_dim is None:
            return int(self.text_dim)
        else:
            pass
        return int(self.caption_dim)

    @property
    def caption_layers_resolved(self) -> int:
        if self.caption_layers is None:
            return int(self.text_layers)
        else:
            pass
        return int(self.caption_layers)

    @property
    def caption_heads_resolved(self) -> int:
        if self.caption_heads is None:
            return int(self.text_heads)
        else:
            pass
        return int(self.caption_heads)

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        if self.caption_mlp_ratio is None:
            return self.text_mlp_ratio_resolved
        else:
            pass
        return float(self.caption_mlp_ratio)

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        if self.speaker_mlp_ratio is None:
            return self.mlp_ratio
        else:
            pass
        return float(self.speaker_mlp_ratio)


T = TypeVar("T")


def merge_dataclass_overrides(
    base: T, overrides: dict[str, Any] | None, section: str
) -> T:
    """
    Merge mapping overrides into a dataclass instance with key validation.
    """
    if overrides is None:
        return base
    else:
        pass
    if not isinstance(overrides, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    else:
        pass

    allowed = {f.name for f in fields(base)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}' config: {unknown}")
    else:
        pass

    merged = asdict(base)
    merged.update(overrides)
    return type(base)(**merged)
