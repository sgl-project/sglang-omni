# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native codec conversion: ``codec_native/config.json`` + ``model.safetensors``.

Adapted from NeMo ``tools/easymagpie_vllm_omni/scripts/convert_codec.py`` and
``easymagpie_vllm_omni/codec/{config,weight_conversion}.py``. Only the causal
decoder is exported; serving needs no encoder or watermark.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from math import prod
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from easymagpie_converter.talker import write_json

LOG = logging.getLogger(__name__)

CODEC_SUBDIR = "codec_native"
DECODER_TARGET = "nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder"
DECODER_PREFIX = "audio_decoder."
WEIGHT_NORM_V = ".parametrizations.weight.original1"
WEIGHT_NORM_G = ".parametrizations.weight.original0"


def read_nemo(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with tarfile.open(path) as archive:
        config_file = archive.extractfile(archive.getmember("./model_config.yaml"))
        weights_file = archive.extractfile(archive.getmember("./model_weights.ckpt"))
        if config_file is None or weights_file is None:
            raise RuntimeError(f"invalid NeMo archive: {path}")
        config = yaml.safe_load(config_file)
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as temporary:
            shutil.copyfileobj(weights_file, temporary)
            temporary.flush()
            state = torch.load(temporary.name, map_location="cpu", weights_only=True)
    return config, state


def validate_decoder(decoder: dict[str, Any]) -> None:
    if decoder.get("_target_") != DECODER_TARGET:
        raise ValueError(f"expected decoder {DECODER_TARGET}, got {decoder.get('_target_')}")
    if decoder.get("is_causal") is not True:
        raise ValueError("only the causal spectral codec decoder is supported")
    if str(decoder.get("activation", "half_snake")) != "half_snake":
        raise ValueError("only the half_snake decoder activation is supported")


def restore_decoder(decoder: dict[str, Any], state: dict[str, torch.Tensor]) -> None:
    """Strictly restore the NeMo decoder so a layout change fails here, not at serving."""
    from hydra.utils import instantiate

    module = instantiate(decoder)
    module.load_state_dict(
        {k.removeprefix(DECODER_PREFIX): v for k, v in state.items() if k.startswith(DECODER_PREFIX)},
        strict=True,
    )


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, v.dim()))
    return v * (g / torch.linalg.vector_norm(v, dim=dims, keepdim=True))


def decoder_weights(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the decoder, fold weight norm, and flatten HalfSnake parameter names."""
    decoder = {k: v for k, v in state.items() if k.startswith(DECODER_PREFIX)}
    converted = {}
    for name, value in decoder.items():
        if name.endswith(WEIGHT_NORM_G):
            continue
        if name.endswith(WEIGHT_NORM_V):
            base = name[: -len(WEIGHT_NORM_V)]
            g = decoder.get(base + WEIGHT_NORM_G)
            if g is None:
                raise KeyError(f"missing weight-norm magnitude for {base}")
            converted[base + ".weight"] = fold_weight_norm(g, value).contiguous()
            continue
        renamed = name.replace(".activation.activation.snake_act.alpha", ".activation.alpha")
        renamed = renamed.replace(
            ".output_activation.activation.snake_act.alpha", ".output_activation.alpha"
        )
        converted[renamed] = value.contiguous()
    return converted


def codec_config(
    nemo_config: dict[str, Any],
    *,
    num_codebooks: int,
    num_levels_per_group: list[int],
    frame_stacking_factor: int,
) -> dict[str, Any]:
    decoder = nemo_config["audio_decoder"]
    config = {
        "architectures": ["EasyMagpieCodecForConditionalGeneration"],
        "model_type": "easymagpie_codec",
        "dtype": "float32",
        "activation": "half_snake",
        "input_dim": int(decoder["input_dim"]),
        "input_filters": int(decoder["input_filters"]),
        "hidden_filters": int(decoder["hidden_filters"]),
        "num_hidden_layers": int(decoder["n_hidden_layers"]),
        "pre_upsample_rates": list(decoder["pre_up_sample_rates"]),
        "pre_upsample_filters": list(decoder["pre_up_sample_filters"]),
        "resblock_upsample_rates": list(decoder["resblock_up_sample_rates"]),
        "resblock_upsample_filters": list(decoder["resblock_up_sample_filters"]),
        "kernel_size": int(decoder.get("kernel_size", 3)),
        "resblock_kernel_size": int(decoder.get("resblock_up_sample_kernel_size", 7)),
        "num_codebooks": num_codebooks,
        "codebook_size": prod(num_levels_per_group),
        "num_levels_per_group": list(num_levels_per_group),
        "frame_stacking_factor": frame_stacking_factor,
        "output_sample_rate": int(nemo_config.get("output_sample_rate", nemo_config["sample_rate"])),
    }
    validate_codec_config(config)
    return config


def validate_codec_config(config: dict[str, Any]) -> None:
    levels = config["num_levels_per_group"]
    if config["input_dim"] != config["num_codebooks"] * len(levels):
        raise ValueError(
            f"decoder input_dim {config['input_dim']} must equal "
            f"num_codebooks * len(num_levels_per_group) = {config['num_codebooks'] * len(levels)}"
        )
    if any(level <= 1 for level in levels):
        raise ValueError("every num_levels_per_group value must be greater than 1")
    if len(config["pre_upsample_rates"]) != len(config["pre_upsample_filters"]):
        raise ValueError("pre_upsample rates and filters must have the same length")
    if len(config["resblock_upsample_rates"]) != len(config["resblock_upsample_filters"]):
        raise ValueError("resblock_upsample rates and filters must have the same length")
    channels = config["input_filters"]
    for filters in config["pre_upsample_filters"] + config["resblock_upsample_filters"]:
        if channels % filters:
            raise ValueError(
                f"decoder channels {channels} must be divisible by stage filters {filters} "
                "(the causal ConvTranspose1d uses groups=out_channels)"
            )
        channels = filters


def convert_codec(
    codec_checkpoint: Path,
    output: Path,
    *,
    num_codebooks: int,
    num_levels_per_group: list[int],
    frame_stacking_factor: int,
) -> dict[str, Any]:
    nemo_config, state = read_nemo(codec_checkpoint)
    validate_decoder(nemo_config["audio_decoder"])
    config = codec_config(
        nemo_config,
        num_codebooks=num_codebooks,
        num_levels_per_group=num_levels_per_group,
        frame_stacking_factor=frame_stacking_factor,
    )
    restore_decoder(nemo_config["audio_decoder"], state)
    weights = decoder_weights(state)
    directory = output / CODEC_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    save_file(weights, directory / "model.safetensors")
    write_json(directory / "config.json", config)
    LOG.info(
        "Saved %d codec tensors (%d parameters)",
        len(weights),
        sum(w.numel() for w in weights.values()),
    )
    return config


__all__ = ["CODEC_SUBDIR", "convert_codec"]
