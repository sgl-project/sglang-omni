# SPDX-License-Identifier: Apache-2.0
"""HF config shim for MiniCPM-o.

The MiniCPM-o checkpoints ship a flat config: the text-backbone fields
(hidden_size, num_hidden_layers, ...) live at the top level next to nested
``vision_config`` / ``audio_config`` / ``tts_config`` dicts. This shim keeps
the flat layout intact and derives a proper text config on demand so sglang
model code can consume a nested view without remote code.

Backbone dispatch follows the checkpoint convention:
- ``attention_bias`` false → Qwen3 dense (MiniCPM-o-4.5)
- otherwise → Qwen2 (older checkpoints, unsupported here)
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class MiniCPMOConfig(PretrainedConfig):
    model_type = "minicpmo"

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        del decoder
        # transformers calls this during __init__ (token-id validation) before
        # all checkpoint fields are set; the flat config is its own text config
        # for that purpose. Callers wanting the derived nested Qwen3 view use
        # derive_text_config() directly.
        if getattr(self, "attention_bias", None) is not False:
            return self
        return derive_text_config(self)


def derive_text_config(config: PretrainedConfig) -> Qwen3Config:
    """Build the Qwen3 dense text config from the flat MiniCPM-o config."""
    if getattr(config, "attention_bias", None) is not False:
        raise NotImplementedError(
            "MiniCPM-o backbone dispatch: only attention_bias=false (Qwen3 "
            "dense, version 4.5) is supported"
        )
    data = config.to_dict()
    # Drop the multimodal sub-configs; Qwen3Config ignores unknown keys but
    # keeping them around confuses config serialization.
    for key in ("vision_config", "audio_config", "tts_config", "slice_config"):
        data.pop(key, None)
    data["architectures"] = ["Qwen3ForCausalLM"]
    data["model_type"] = "qwen3"
    return Qwen3Config.from_dict(data)


def register_minicpm_o_hf_config() -> None:
    """Register the shim so AutoConfig resolves minicpmo without remote code."""
    AutoConfig.register("minicpmo", MiniCPMOConfig, exist_ok=True)


register_minicpm_o_hf_config()
