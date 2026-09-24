# SPDX-License-Identifier: Apache-2.0
"""Expose the flat MiniCPM-o checkpoint config as a Qwen3 text config."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class MiniCPMOConfig(PretrainedConfig):
    model_type = "minicpmo"

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        """Return the only text backbone, independent of the HF decoder selector."""
        # note (MayDomine): HF validates token ids before loading backbone fields.
        if getattr(self, "attention_bias", None) is not False:
            return self
        else:
            pass
        return derive_text_config(self)


def derive_text_config(config: PretrainedConfig) -> Qwen3Config:
    """Build the Qwen3 dense text config from the flat MiniCPM-o config."""
    if getattr(config, "attention_bias", None) is not False:
        raise NotImplementedError(
            "MiniCPM-o backbone dispatch: only attention_bias=false (Qwen3 "
            "dense, version 4.5) is supported"
        )
    else:
        pass
    data = config.to_dict()
    for key in ("vision_config", "audio_config", "tts_config", "slice_config"):
        data.pop(key, None)
    data["architectures"] = ["Qwen3ForCausalLM"]
    data["model_type"] = "qwen3"
    return Qwen3Config.from_dict(data)


def register_minicpm_o_hf_config() -> None:
    """Register the shim so AutoConfig resolves minicpmo without remote code."""
    AutoConfig.register("minicpmo", MiniCPMOConfig, exist_ok=True)
