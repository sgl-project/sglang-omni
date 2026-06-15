# SPDX-License-Identifier: Apache-2.0
# Remaps the HuggingFace Transformers CSM checkpoint export (transformers/models/csm,
# Apache-2.0) onto the SGLang-integrated module names.
"""Checkpoint weight-name remapping for CSM-1B.

The ship artifact is the HF-transformers export: shards
``transformers-00001-of-00002.safetensors`` / ``transformers-00002-of-00002
.safetensors`` indexed by ``transformers.safetensors.index.json`` — 538
tensors, ALL fp32 (cast to backbone bf16 on copy). NEVER read the orig-format
``ckpt.pt`` / ``model.safetensors`` in the same repo: their Q/K are
un-permuted-RoPE.

Remap table:

==============================================================  =====================================
checkpoint name                                                 destination
==============================================================  =====================================
``embed_text_tokens.weight``                                    ``backbone.model.embed_tokens.weight``
``backbone_model.embed_tokens.embed_audio_tokens.weight``       ``frame_embedding.weight``
``backbone_model.layers.{i}.*``                                 ``backbone.model.layers.{i}.*``
``backbone_model.norm.weight``                                  ``backbone.model.norm.weight``
``lm_head.weight`` (shape ``[2051, 2048]``)                     ``codebook0_head.weight`` (NOT a text head!)
``depth_decoder.model.inputs_embeds_projector.weight``          ``depth_decoder.inputs_embeds_projector.weight``
``depth_decoder.model.layers.{0..3}.*``                         ``depth_decoder.layers.{i}.*`` (manual copies)
``depth_decoder.model.norm.weight``                             depth final norm
``depth_decoder.codebooks_head.weight``                         ``codebooks_head.weight``
``depth_decoder.model.embed_tokens.weight``                     SKIP — tied alias of the audio table
``codec_model.*``                                               SKIP — ``audio_codec.py`` loads that subtree
==============================================================  =====================================

The depth tied-embed duplicate IS present in the 538-tensor census (verified
vs ``tf_index.json``) so the skip path ALWAYS fires; the zero-unexpected
census counts it as deliberately-skipped, not missing. Llama
qkv/gate_up stacking on the backbone side is handled by
``LlamaForCausalLM.load_weights``; depth layers use their own dense modules
(no stacking).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_BACKBONE_LAYERS_PREFIX = "backbone_model.layers."
_DEPTH_LAYERS_PREFIX = "depth_decoder.model.layers."

# Exact-name routes (checkpoint name → destination).
_EXACT_MAP: dict[str, str] = {
    "embed_text_tokens.weight": "backbone.model.embed_tokens.weight",
    "backbone_model.embed_tokens.embed_audio_tokens.weight": "frame_embedding.weight",
    "backbone_model.norm.weight": "backbone.model.norm.weight",
    # The checkpoint's lm_head IS the codebook-0 head [2051, 2048]; routing it
    # to a text head would be catastrophic (the synthetic backbone config ties
    # word embeddings precisely so no text lm_head exists to mis-load).
    "lm_head.weight": "codebook0_head.weight",
    "depth_decoder.model.inputs_embeds_projector.weight": (
        "depth_decoder.inputs_embeds_projector.weight"
    ),
    "depth_decoder.model.norm.weight": "depth_decoder.norm.weight",
    "depth_decoder.codebooks_head.weight": "codebooks_head.weight",
}

# Deliberate skips (checkpoint name/prefix → reason).
_SKIP_EXACT: dict[str, str] = {
    # Tied alias of frame_embedding.weight; present in the 538-tensor census,
    # so this skip ALWAYS fires for the ship artifact.
    "depth_decoder.model.embed_tokens.weight": "tied alias of the audio table",
}
_SKIP_PREFIXES: dict[str, str] = {
    "codec_model.": "Mimi subtree; audio_codec.py loads it itself",
}


@dataclass(frozen=True)
class CsmWeightMapper:
    """Weight-name remapper for the CSM HF-transformers shards.

    ``map`` routes each checkpoint tensor name to (dest_name | None-to-skip);
    the model's ``load_weights`` splits the stream into backbone-vs-own params
    and casts everything to ``backbone_dtype`` on copy.
    """

    backbone_dtype: torch.dtype = torch.bfloat16

    def _route(self, name: str) -> tuple[str, str | None]:
        """``("mapped", dest) | ("skipped", None) | ("unexpected", None)``."""
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
        """Map a checkpoint tensor name to its destination name.

        Args:
            name: checkpoint key (e.g. ``"backbone_model.layers.3.self_attn.q_proj.weight"``).

        Returns:
            Destination parameter name, or ``None`` to deliberately skip
            (``depth_decoder.model.embed_tokens.weight`` tied alias and the
            whole ``codec_model.*`` subtree).

        Raises:
            ValueError: on any tensor name outside the known census — strict
                accounting; never silently drop or misroute.
        """
        kind, dest = self._route(name)
        if kind == "unexpected":
            raise ValueError(
                f"Unexpected checkpoint tensor {name!r}: not in the CSM remap "
                f"table and not a deliberate skip"
            )
        return dest

    def census(self, names: list[str]) -> dict[str, int]:
        """Tensor census over all checkpoint keys.

        Returns:
            ``{"mapped": int, "skipped": int, "unexpected": int}`` — must be a
            538/0-unexpected split for the ship artifact.
        """
        counts = {"mapped": 0, "skipped": 0, "unexpected": 0}
        for name in names:
            counts[self._route(name)[0]] += 1
        return counts


__all__ = ["CsmWeightMapper"]
