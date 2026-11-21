# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight-name remapping for the CSM Transformers export.

Load transformers-*.safetensors only. The original sesame Q/K are not
RoPE-permuted. lm_head.weight [2051, 2048] is codebook-0, not a text head.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

BACKBONE_LAYERS_PREFIX = "backbone_model.layers."
BACKBONE_DEST_PREFIX = "backbone."
DEPTH_LAYERS_PREFIX = "depth_decoder.model.layers."

EXACT_MAP: dict[str, str] = {
    "embed_text_tokens.weight": "backbone.model.embed_tokens.weight",
    "backbone_model.embed_tokens.embed_audio_tokens.weight": "frame_embedding.weight",
    "backbone_model.norm.weight": "backbone.model.norm.weight",
    "lm_head.weight": "codebook0_head.weight",
    "depth_decoder.model.inputs_embeds_projector.weight": (
        "depth_decoder.inputs_embeds_projector.weight"
    ),
    "depth_decoder.model.norm.weight": "depth_decoder.norm.weight",
    "depth_decoder.codebooks_head.weight": "codebooks_head.weight",
}

SKIP_EXACT: dict[str, str] = {
    "depth_decoder.model.embed_tokens.weight": "tied alias of the audio table",
}
SKIP_PREFIXES: dict[str, str] = {
    "codec_model.": "Mimi subtree; audio_codec.py loads it itself",
}


@dataclass(frozen=True)
class CsmWeightMapper:
    """Remap HF-transformers shard names onto the serving graph."""

    def route(self, name: str) -> tuple[str, str | None]:
        dest = EXACT_MAP.get(name)
        if dest is not None:
            return "mapped", dest
        else:
            pass
        if name.startswith(BACKBONE_LAYERS_PREFIX):
            return (
                "mapped",
                "backbone.model.layers." + name[len(BACKBONE_LAYERS_PREFIX) :],
            )
        else:
            pass
        if name.startswith(DEPTH_LAYERS_PREFIX):
            return "mapped", "depth_decoder.layers." + name[len(DEPTH_LAYERS_PREFIX) :]
        else:
            pass
        if name in SKIP_EXACT:
            return "skipped", None
        else:
            pass
        for prefix in SKIP_PREFIXES:
            if name.startswith(prefix):
                return "skipped", None
            else:
                pass
        return "unexpected", None

    def map(self, name: str) -> str | None:
        kind, dest = self.route(name)
        if kind == "unexpected":
            raise ValueError(
                f"Unexpected checkpoint tensor {name!r}: not in the CSM remap "
                f"table and not a deliberate skip"
            )
        else:
            pass
        return dest


def split_csm_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Route checkpoint tensors into (backbone-relative, CSM-owned) buckets."""
    mapper = CsmWeightMapper()
    backbone_weights: list[tuple[str, torch.Tensor]] = []
    csm_weights: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        mapped = mapper.map(name)
        if mapped is None:
            continue
        elif mapped.startswith(BACKBONE_DEST_PREFIX):
            backbone_weights.append((mapped[len(BACKBONE_DEST_PREFIX) :], tensor))
        else:
            csm_weights.append((mapped, tensor))
    return backbone_weights, csm_weights


def copy_into_params(
    destinations: dict[str, torch.nn.Parameter],
    weights: list[tuple[str, torch.Tensor]],
) -> set[str]:
    """Copy each mapped tensor into its destination parameter."""
    loaded: set[str] = set()
    for name, tensor in weights:
        param = destinations.get(name)
        if param is None:
            raise ValueError(f"CSM weight remap produced unknown destination {name!r}")
        else:
            pass
        if param.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        else:
            pass
        param.data.copy_(tensor.to(param.dtype))
        loaded.add(name)
    return loaded


__all__ = ["CsmWeightMapper", "copy_into_params", "split_csm_weights"]
