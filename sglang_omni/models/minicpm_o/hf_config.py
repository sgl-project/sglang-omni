# SPDX-License-Identifier: Apache-2.0
"""Expose the flat MiniCPM-o checkpoint config as a Qwen3 text config."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class MiniCPMOConfig(PretrainedConfig):
    model_type = "minicpmo"

    # note (justkyriecai): AutoConfig reads config.json itself and hands the directory to from_dict as name_or_path; a direct from_pretrained goes through get_config_dict instead, so both hooks expand a Realtime-Venus stub.
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        config_dict, remaining_kwargs = super().get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return (
            expand_venus_manifest(config_dict, Path(pretrained_model_name_or_path)),
            remaining_kwargs,
        )

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, object], **kwargs: object
    ) -> "MiniCPMOConfig":
        if "venus_config_ref" in config_dict:
            config_dict = expand_venus_manifest(
                config_dict, Path(str(kwargs["name_or_path"]))
            )
        else:
            pass
        return super().from_dict(config_dict, **kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        """Return the only text backbone, independent of the HF decoder selector."""
        # note (MayDomine): HF validates token ids before loading backbone fields.
        if getattr(self, "attention_bias", None) is not False:
            return self
        else:
            pass
        return derive_text_config(self)


class RealtimeVenusOmniConfig(MiniCPMOConfig):
    """Realtime-Venus-Omni is MiniCPM-o 4.5 fine-tuned under its own architecture name."""

    model_type = "realtime_venus_omni"


def expand_venus_manifest(
    config_dict: dict[str, object], model_directory: Path
) -> dict[str, object]:
    """Return the full hf_config when config.json is a Realtime-Venus stub, else the dict unchanged.

    A Venus download keeps one config.yaml beside its model directories; each config.json only carries a venus_config_ref naming that manifest and its SHA-256.
    """
    reference = config_dict.get("venus_config_ref")
    if reference is None:
        return config_dict
    else:
        pass
    manifest_path = model_directory.parent / "config.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{model_directory} is a Realtime-Venus checkpoint but {manifest_path} is missing; "
            "download it with the repository's download_models.py, which keeps config.yaml beside the model directory"
        )
    else:
        pass
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != reference["sha256"]:
        raise ValueError(
            f"{manifest_path} does not match the SHA-256 pinned by {model_directory / 'config.json'}; re-download the checkpoint"
        )
    else:
        pass
    manifest = yaml.safe_load(manifest_bytes)
    return manifest["models"][reference["model"]]["hf_config"]


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
    """Register the shim so AutoConfig resolves minicpmo and realtime-venus without remote code."""
    AutoConfig.register("minicpmo", MiniCPMOConfig, exist_ok=True)
    AutoConfig.register("realtime_venus_omni", RealtimeVenusOmniConfig, exist_ok=True)
