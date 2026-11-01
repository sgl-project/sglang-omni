# SPDX-License-Identifier: Apache-2.0
"""Ming 3 checkpoint and AudioVAE loading shared by stage factories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, JsonValue
from safetensors import safe_open

from sglang_omni.models.ming3_tts.sglang_model import VOCODER_WEIGHT_PREFIX
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE


class SpeechConfig(BaseModel):
    """The tts_config subset this pipeline implements; other contracts fail at load."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_format: Literal["legacy_antmmf"]
    with_rml: Literal[True]
    conditioning_mode: Literal["speaker_only"]
    decode_alignment: Literal["old_shift"]
    patch_size: Literal[1]
    history_size: Literal[1]
    latent_dim: int
    speaker_embedding_dim: int
    sampling_temperature: float
    stop_threshold: float
    minimum_decode_steps: int
    maximum_decode_steps: int
    audio_patch_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    speaker_start_token_id: int
    speaker_end_token_id: int


def load_checkpoint_config(checkpoint: Path) -> dict[str, JsonValue]:
    return json.loads((checkpoint / "config.json").read_text())


def is_dense_checkpoint(configuration: dict[str, JsonValue]) -> bool:
    return configuration.get("model_type") == "dense"


def audio_section(configuration: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if is_dense_checkpoint(configuration):
        return configuration["audio_tokenizer_config"]
    else:
        return configuration["audio_config"]


def load_prefixed_weights(checkpoint: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Read the weights under prefix with it stripped. Tiny checkpoints are sharded."""
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_index = json.loads(index_path.read_text())
        shard_names = {
            shard_name
            for weight_name, shard_name in weight_index["weight_map"].items()
            if weight_name.startswith(prefix)
        }
    else:
        shard_names = {"model.safetensors"}
    weights: dict[str, torch.Tensor] = {}
    for shard_name in sorted(shard_names):
        with safe_open(str(checkpoint / shard_name), framework="pt") as shard:
            for weight_name in shard.keys():
                if weight_name.startswith(prefix):
                    weights[weight_name.removeprefix(prefix)] = shard.get_tensor(
                        weight_name
                    )
                else:
                    pass
    return weights


def load_audio_vae(
    checkpoint: Path,
    device: torch.device,
    *,
    component: Literal["encoder", "decoder"] | None = None,
) -> tuple[AudioVAE, int]:
    """Load the bfloat16 AudioVAE with SDPA attention and its sample rate."""
    vae_configuration = audio_section(load_checkpoint_config(checkpoint))
    for codec_section in ("enc_kwargs", "dec_kwargs"):
        vae_configuration[codec_section]["backbone"]["_attn_implementation"] = "sdpa"
    vae = AudioVAE(AudioVAEconfig(**vae_configuration)).eval()
    vae.load_state_dict(load_prefixed_weights(checkpoint, VOCODER_WEIGHT_PREFIX))
    if component == "encoder":
        vae.decoder = None
    elif component == "decoder":
        vae.encoder = None
    else:
        pass
    vae.to(device=device, dtype=torch.bfloat16)
    return vae, int(vae_configuration["sample_rate"])
