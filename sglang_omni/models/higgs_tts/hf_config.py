# SPDX-License-Identifier: Apache-2.0
"""HF config for HiggsMultimodalQwen3 (TTS path).

The schema also carries ASR (Whisper encoder) fields; the TTS path uses
the discrete-audio-encoder branch and ignores the Whisper side.
"""

from __future__ import annotations

from typing import Any

import transformers

# Higgs Qwen3 sub-configs ship ``rope_theta=null``; transformers' default
# 10000 is wrong for Qwen3 (trained at 1e6). Patch before instantiation
# so sglang's RoPE picks up the right base.
_QWEN3_ROPE_THETA = 1_000_000


def _build_text_config(raw: Any) -> transformers.PretrainedConfig:
    """Realise a text-backbone sub-config into a concrete ``PretrainedConfig``."""
    if isinstance(raw, transformers.PretrainedConfig):
        return raw
    cfg = dict(raw or {})
    model_type = cfg.get("model_type", "qwen3")
    if model_type == "qwen3" and cfg.get("rope_theta") is None:
        cfg["rope_theta"] = _QWEN3_ROPE_THETA
    try:
        cfg_cls = transformers.CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown text backbone model_type {model_type!r}; "
            f"expected one registered in transformers.CONFIG_MAPPING."
        ) from exc
    return cfg_cls(**cfg)


class HiggsMultimodalQwen3Config(transformers.PretrainedConfig):
    """HiggsMultimodalQwen3 config.

    ``audio_encoder_config`` is a unified dict with an ``encoder_type``
    discriminator (``"discrete"`` for TTS, ``"whisper"`` for ASR);
    ``text_config`` is the Qwen3 backbone config, eagerly realised so
    consumers can access ``self.text_config.num_attention_heads`` directly.
    """

    model_type = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | transformers.PretrainedConfig | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        **kwargs,
    ):
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample
        self.audio_encoder_config = audio_encoder_config
        self.text_config = _build_text_config(text_config)
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        del decoder
        return self.text_config
