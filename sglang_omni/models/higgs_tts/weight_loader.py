# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight-name remapping for HiggsMultimodalQwen3 (discrete TTS path).

Higgs checkpoints use prefixes like ``tied.embedding.text_embedding.`` and
``body.layers.``; sglang expects its own parameter-tree layout. The mapping
function is parameterised by the destination prefix so different model
wrappers can supply their own layout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscreteWeightMapper:
    """Weight-name remapper for the discrete TTS path.

    ``tie_modality`` must match the ckpt's ``tie_word_embeddings`` flag;
    when True the modality_head shares the embedding weight and the ckpt's
    head copy is dropped.
    """

    text_prefix_map: dict[str, str]
    embedding_dest: str = "multimodal_embedding.modality_embedding_0."
    head_dest: str = "modality_head."
    tie_modality: bool = True

    def _instance_prefix_map(self) -> dict[str, str]:
        mapping = {
            "tied.embedding.modality_embeddings.0.embedding.": self.embedding_dest,
        }
        if not self.tie_modality:
            mapping["tied.head.modality_heads.0."] = self.head_dest
        return mapping

    def map(self, name: str) -> str | None:
        """Map ckpt name to downstream name; ``None`` to skip the weight."""
        for higgs_prefix, dest_prefix in self._instance_prefix_map().items():
            if name.startswith(higgs_prefix):
                return dest_prefix + name[len(higgs_prefix) :]

        # Audio tokenizer backbone — frozen, not in the serving graph.
        if name.startswith("tied.embedding.modality_embeddings.0.model."):
            return None

        for higgs_prefix, dest_prefix in self.text_prefix_map.items():
            if name.startswith(higgs_prefix):
                return dest_prefix + name[len(higgs_prefix) :]

        return name


__all__ = ["DiscreteWeightMapper"]
