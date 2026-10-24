# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight-name remapping for the CSM Transformers export.

Load transformers-*.safetensors only. The original sesame Q/K are not
RoPE-permuted. lm_head.weight [2051, 2048] is codebook-0, not a text head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_BACKBONE_LAYERS_PREFIX = "backbone_model.layers."
_DEPTH_LAYERS_PREFIX = "depth_decoder.model.layers."

_EXACT_MAP: dict[str, str] = {
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

_SKIP_EXACT: dict[str, str] = {
    "depth_decoder.model.embed_tokens.weight": "tied alias of the audio table",
}
_SKIP_PREFIXES: dict[str, str] = {
    "codec_model.": "Mimi subtree; audio_codec.py loads it itself",
}


@dataclass(frozen=True)
class CsmWeightMapper:
    """Remap HF-transformers shard names onto the serving graph."""

    backbone_dtype: torch.dtype = torch.bfloat16

    def route(self, name: str) -> tuple[str, str | None]:
        dest = _EXACT_MAP.get(name)
        if dest is not None:
            return "mapped", dest
        if name.startswith(_BACKBONE_LAYERS_PREFIX):
            return (
                "mapped",
                "backbone.model.layers." + name[len(_BACKBONE_LAYERS_PREFIX) :],
            )
        if name.startswith(_DEPTH_LAYERS_PREFIX):
            return "mapped", "depth_decoder.layers." + name[len(_DEPTH_LAYERS_PREFIX) :]
        if name in _SKIP_EXACT:
            return "skipped", None
        for prefix in _SKIP_PREFIXES:
            if name.startswith(prefix):
                return "skipped", None
        return "unexpected", None

    def map(self, name: str) -> str | None:
        kind, dest = self.route(name)
        if kind == "unexpected":
            raise ValueError(
                f"Unexpected checkpoint tensor {name!r}: not in the CSM remap "
                f"table and not a deliberate skip"
            )
        return dest

    def census(self, names: list[str]) -> dict[str, int]:
        counts = {"mapped": 0, "skipped": 0, "unexpected": 0}
        for name in names:
            counts[self.route(name)[0]] += 1
        return counts


__all__ = ["CsmWeightMapper"]
