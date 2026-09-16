# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Prince Canuma and contributors.
# Derived from Blaizzy/mlx-audio CosyVoice3 PR #861 (commit 5272f213f8cc).
# Based on FunAudioLLM/CosyVoice (Apache-2.0, Copyright 2024-2025 Alibaba Inc).
# Modified for the non-streaming sglang-omni vocoder contract.

"""Configuration for the native MLX Fun-CosyVoice3 vocoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowConfig:
    """Speech-token to mel Flow/DiT configuration."""

    input_size: int = 80
    output_size: int = 80
    spk_embed_dim: int = 192
    vocab_size: int = 6561
    pre_lookahead_len: int = 3
    pre_lookahead_channels: int = 1024
    token_mel_ratio: int = 2
    dit_hidden_size: int = 1024
    dit_depth: int = 22
    dit_num_heads: int = 16
    dit_head_dim: int = 64
    dit_mlp_ratio: float = 2.0
    dit_mel_dim: int = 80
    dit_mu_dim: int = 80
    dit_spk_dim: int = 80
    dit_static_chunk_size: int = 50
    dit_num_decoding_left_chunks: int = -1
    n_timesteps: int = 10
    inference_cfg_rate: float = 0.7

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "FlowConfig":
        values = dict(values or {})
        dit = dict(values.pop("dit", values.pop("estimator", {})) or {})
        aliases = {
            "dim": "dit_hidden_size",
            "depth": "dit_depth",
            "heads": "dit_num_heads",
            "dim_head": "dit_head_dim",
            "ff_mult": "dit_mlp_ratio",
            "mel_dim": "dit_mel_dim",
            "mu_dim": "dit_mu_dim",
            "spk_dim": "dit_spk_dim",
            "static_chunk_size": "dit_static_chunk_size",
            "num_decoding_left_chunks": "dit_num_decoding_left_chunks",
        }
        for source, target in aliases.items():
            if source in dit:
                values[target] = dit[source]
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class HiFTConfig:
    """Mel to 24 kHz waveform HiFT configuration."""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 11, 7])
    istft_params: dict[str, int] = field(
        default_factory=lambda: {"n_fft": 16, "hop_len": 4}
    )
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    conv_pre_look_right: int = 4

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "HiFTConfig":
        values = dict(values or {})
        if "istft_params" not in values:
            values["istft_params"] = {
                "n_fft": int(values.pop("istft_n_fft", 16)),
                "hop_len": int(values.pop("istft_hop_len", 4)),
            }
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class VocoderConfig:
    """Complete native MLX vocoder configuration."""

    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: HiFTConfig = field(default_factory=HiFTConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocoderConfig":
        return cls(
            flow=FlowConfig.from_dict(values.get("flow")),
            hift=HiFTConfig.from_dict(values.get("hifigan", values.get("hift"))),
        )
