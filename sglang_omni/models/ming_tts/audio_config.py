# SPDX-License-Identifier: Apache-2.0
"""AudioVAE checkpoint config adapter for Ming-Omni-TTS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig

from sglang_omni.models.ming_tts.payload_types import MING_TTS_SAMPLE_RATE


class AudioVAEconfig(PretrainedConfig):
    """Config class matching the official AudioVAE checkpoint metadata."""

    def __init__(
        self,
        sample_rate: int = 16000,
        enc_kwargs: dict[str, Any] | None = None,
        dec_kwargs: dict[str, Any] | None = None,
        hifi_gan_disc_kwargs: dict[str, Any] | None = None,
        spec_disc_kwargs: dict[str, Any] | None = None,
        lambda_disc: float = 1.0,
        lambda_mel_loss: float = 15.0,
        lambda_adv: float = 1.0,
        lambda_feat_match_loss: float = 1.0,
        semantic_module_kwargs: dict[str, Any] | None = None,
        lambda_semantic: float | None = 5.0,
        init_method: str = "normal",
        patch_size: int = -1,
        **kwargs: Any,
    ) -> None:
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs
        self.dec_kwargs = dec_kwargs
        self.hifi_gan_disc_kwargs = hifi_gan_disc_kwargs
        self.spec_disc_kwargs = spec_disc_kwargs
        self.lambda_disc = lambda_disc
        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_adv = lambda_adv
        self.lambda_feat_match_loss = lambda_feat_match_loss
        self.semantic_module_kwargs = semantic_module_kwargs
        self.lambda_semantic = lambda_semantic
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)


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
