# SPDX-License-Identifier: Apache-2.0
"""AudioVAE checkpoint config adapter for Ming-Omni-TTS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig

from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)
from sglang_omni.models.ming_tts.payload_types import MING_TTS_SAMPLE_RATE


def resolve_ming_tts_audio_vae_config(
    audio_config: AudioVAEconfig | PretrainedConfig | dict[str, Any],
    *,
    attn_implementation: str,
) -> AudioVAEconfig:
    if isinstance(audio_config, AudioVAEconfig):
        config = deepcopy(audio_config)
    elif isinstance(audio_config, PretrainedConfig):
        config = AudioVAEconfig(**audio_config.to_dict())
    else:
        config = AudioVAEconfig(**deepcopy(audio_config))

    sample_rate = int(getattr(config, "sample_rate", 0))
    if sample_rate != MING_TTS_SAMPLE_RATE:
        raise ValueError(
            "Ming-Omni-TTS AudioVAE config sample_rate must be "
            f"{MING_TTS_SAMPLE_RATE}, got {sample_rate}"
        )
    if not isinstance(config.enc_kwargs, dict):
        raise ValueError("Ming-Omni-TTS AudioVAE config is missing enc_kwargs")
    if not isinstance(config.dec_kwargs, dict):
        raise ValueError("Ming-Omni-TTS AudioVAE config is missing dec_kwargs")
    if int(getattr(config, "patch_size", -1)) <= 0:
        raise ValueError("Ming-Omni-TTS AudioVAE config is missing patch_size")

    for name, stage_kwargs in (
        ("enc_kwargs", config.enc_kwargs),
        ("dec_kwargs", config.dec_kwargs),
    ):
        backbone = stage_kwargs.get("backbone")
        if not isinstance(backbone, dict):
            raise ValueError(
                f"Ming-Omni-TTS AudioVAE config {name}.backbone is missing"
            )
        backbone["_attn_implementation"] = attn_implementation
    return config


__all__ = ["AudioVAEconfig", "resolve_ming_tts_audio_vae_config"]
