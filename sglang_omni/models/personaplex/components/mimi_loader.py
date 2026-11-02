# SPDX-License-Identifier: Apache-2.0
"""Selective loading and checkpoint name routing for the Mimi codec."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open

from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.mimi import MimiCodec

RENAMES = (
    (re.compile(r"\.conv\.conv\."), ".conv."),
    (re.compile(r"\.convtr\.convtr\."), ".convtr."),
    (re.compile(r"_transformer\.transformer\."), "_transformer."),
    (re.compile(r"\._codebook\."), ".codebook."),
)
ACOUSTIC_LAYER = re.compile(r"^quantizer\.rvq_rest\.vq\.layers\.(\d+)\.")


def rename_mimi_key(name: str) -> str | None:
    """Map a checkpoint tensor name onto this module tree; None drops it."""
    match = ACOUSTIC_LAYER.match(name)
    if (
        match
        and int(match.group(1)) >= MIMI.num_codebooks - MIMI.num_semantic_codebooks
    ):
        # Note (wilsonzheng0327): Trained with 32 codebooks; Moshi uses only 8.
        return None
    else:
        pass
    for pattern, replacement in RENAMES:
        # Note (wilsonzheng0327): The reference nests convolutions up to three deep
        # (downsample.conv.conv.conv); here each is one module.
        while True:
            renamed = pattern.sub(replacement, name, count=1)
            if renamed == name:
                break
            else:
                pass
            name = renamed
    return name


def resolve_mimi_weights(model_dir: str | Path, glob: str) -> Path:
    matches = sorted(Path(model_dir).glob(glob))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Mimi weight file matching {glob!r} in "
            f"{model_dir}, found {[m.name for m in matches]}"
        )
    else:
        pass
    return matches[0]


def load_mimi_codec(
    weights_path: str | Path,
    *,
    device: torch.device | str,
    mode: Literal["encode", "decode", "both"] = "both",
) -> MimiCodec:
    """Load only the requested Mimi path in float32 on the target device."""
    with torch.device("meta"):
        codec = MimiCodec(mode=mode)
    if mode == "encode":
        excluded_prefixes = ("decoder.", "decoder_transformer.", "upsample.") + tuple(
            f"quantizer.{quantizer_name}.output_proj."
            for quantizer_name in ("rvq_first", "rvq_rest")
        )
    elif mode == "decode":
        excluded_prefixes = ("encoder.", "encoder_transformer.", "downsample.") + tuple(
            f"quantizer.{quantizer_name}.input_proj."
            for quantizer_name in ("rvq_first", "rvq_rest")
        )
    else:
        excluded_prefixes = ()
    codec_weights: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_path), framework="pt") as checkpoint:
        for checkpoint_name in checkpoint.keys():
            renamed = rename_mimi_key(checkpoint_name)
            if renamed is None or renamed.startswith(excluded_prefixes):
                pass
            else:
                codec_weights[renamed] = checkpoint.get_tensor(checkpoint_name).to(
                    device=device, dtype=torch.float32, copy=True
                )
    codec.load_state_dict(codec_weights, strict=True, assign=True)
    return codec.eval()
