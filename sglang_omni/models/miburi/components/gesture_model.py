# SPDX-License-Identifier: Apache-2.0
"""The gesture LM: per agent frame, 20 motion codes from the frame's text token and Mimi codes.

The agent frame becomes two memories, its audio codes' vector and its text token's vector, each projected once for the temporal and once for the depth stack. The temporal stack reads the previous frame's codes and predicts codebook 0; the depth stack then predicts codebooks 1 to 19, step s from the code of codebook s. loading.py maps the checkpoint's state-dict keys onto these module names.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass

import torch
from torch import nn

from sglang_omni.models.miburi.architecture import (
    INITIAL_CODE,
    NULL_CONDITION_ID,
    SPEECH_AUDIO_CODEBOOKS,
    GestureLMSpec,
)
from sglang_omni.models.miburi.components.gesture_transformer import (
    DepthTransformer,
    TemporalState,
    TemporalTransformer,
)
from sglang_omni.models.miburi.components.ring_attention import AttentionRing


class ConditionEmbedding(nn.Embedding):
    """One of the speech LM's input tables, in its bfloat16; the null id gives zero."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = super().forward(ids.clamp(min=0))
        return torch.where((ids == NULL_CONDITION_ID)[..., None], 0.0, rows)


class CodeEmbedding(nn.Embedding):
    """One codebook's code vectors, a learned vector for the pad code and zero for the initial code."""

    def __init__(self, codebook_size: int, dim: int) -> None:
        super().__init__(codebook_size, dim)
        self.pad_embedding = nn.Embedding(1, dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        rows = super().forward(codes.clamp(0, self.num_embeddings - 1))
        rows = torch.where(
            (codes == self.num_embeddings)[..., None],
            self.pad_embedding.weight[0],
            rows,
        )
        return torch.where((codes == INITIAL_CODE)[..., None], 0.0, rows)


@dataclass
class SpeechMemories:
    """An agent frame's audio and text vectors, projected for each stack [rows, 1, dim]."""

    temporal: tuple[torch.Tensor, torch.Tensor]
    depth: tuple[torch.Tensor, torch.Tensor]


class GestureModel(nn.Module):
    def __init__(self, spec: GestureLMSpec) -> None:
        super().__init__()
        self.spec = spec
        dim, depth_steps = spec.dim, spec.depth_steps

        def linears(count: int, in_features: int, *, bias: bool) -> nn.ModuleList:
            return nn.ModuleList(
                nn.Linear(in_features, dim, bias=bias) for _ in range(count)
            )

        self.text_embedding = ConditionEmbedding(
            spec.text_vocab, spec.condition_dim, dtype=torch.bfloat16
        )
        self.audio_embeddings = nn.ModuleList(
            ConditionEmbedding(
                spec.audio_vocab, spec.condition_dim, dtype=torch.bfloat16
            )
            for _ in range(SPEECH_AUDIO_CODEBOOKS)
        )
        # Index 0 projects the audio vector, index 1 the text vector.
        self.temporal_condition_projs = linears(2, spec.condition_dim, bias=True)
        self.depth_condition_projs = linears(2, spec.condition_dim, bias=True)
        self.temporal_speaker_embedding = nn.Embedding(spec.num_speakers, dim)
        self.depth_speaker_embeddings = nn.ModuleList(
            nn.Embedding(spec.num_speakers, dim) for _ in range(depth_steps)
        )
        self.temporal_code_embeddings = nn.ModuleList(
            CodeEmbedding(spec.codebook_size, spec.codebook_dim)
            for _ in range(spec.num_codebooks)
        )
        self.depth_code_embeddings = nn.ModuleList(
            CodeEmbedding(spec.codebook_size, spec.codebook_dim)
            for _ in range(depth_steps)
        )
        for codebook, embedding in enumerate(self.depth_code_embeddings):
            # Note (Junnan Li): The released model reads codebooks 0 to 18 on the temporal path through the depth path's code tables, one tensor for both; the checkpoint's own temporal tables for them are never read. The pad vectors stay separate.
            self.temporal_code_embeddings[codebook].weight = embedding.weight
        self.temporal_code_projs = linears(
            spec.num_codebooks, spec.codebook_dim, bias=False
        )
        self.depth_code_projs = linears(depth_steps, spec.codebook_dim, bias=False)
        self.depth_hidden_projs = linears(depth_steps, dim, bias=False)
        self.temporal_transformer = TemporalTransformer(spec)
        self.temporal_out_norm = nn.LayerNorm(dim)
        self.temporal_classifier = nn.Linear(dim, spec.codebook_size + 1, bias=False)
        self.depth_transformer = DepthTransformer(spec)
        self.depth_classifiers = nn.ModuleList(
            nn.Linear(dim, spec.codebook_size + 1, bias=False)
            for _ in range(depth_steps)
        )

    def speech_memories(self, condition: torch.Tensor) -> SpeechMemories:
        """condition [rows, 9, 1]: the frame's text token, then its Mimi codes."""
        text = self.text_embedding(condition[:, 0]).float()
        # Note (Junnan Li): The eight bfloat16 rows are summed in one reduction, which accumulates in float32 and rounds once; a running bfloat16 sum rounds differently.
        audio = (
            torch.stack(
                [
                    table(condition[:, 1 + index])
                    for index, table in enumerate(self.audio_embeddings)
                ]
            )
            .sum(dim=0)
            .float()
        )
        return SpeechMemories(
            temporal=(
                self.temporal_condition_projs[0](audio),
                self.temporal_condition_projs[1](text),
            ),
            depth=(
                self.depth_condition_projs[0](audio),
                self.depth_condition_projs[1](text),
            ),
        )

    def temporal_step(
        self,
        previous_codes: torch.Tensor,
        memories: SpeechMemories,
        speakers: torch.Tensor,
        state: TemporalState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """previous_codes [rows, codebooks, 1], speakers [rows, 1] → the frame's hidden state [rows, 1, dim] and codebook 0's logits [rows, 1, codebook_size + 1]."""
        projected_codes = [
            projection(embedding(previous_codes[:, codebook]))
            for codebook, (embedding, projection) in enumerate(
                zip(
                    self.temporal_code_embeddings,
                    self.temporal_code_projs,
                    strict=True,
                )
            )
        ]
        x = functools.reduce(
            operator.add, projected_codes
        ) + self.temporal_speaker_embedding(speakers)
        hidden = self.temporal_out_norm(
            self.temporal_transformer(x, memories.temporal, state)
        )
        return hidden, self.temporal_classifier(hidden)

    def depth_step(
        self,
        step: int,
        codes: torch.Tensor,
        hidden: torch.Tensor,
        memories: SpeechMemories,
        speakers: torch.Tensor,
        rings: list[AttentionRing],
    ) -> torch.Tensor:
        """codes [rows, 1] of codebook step → codebook step + 1's logits [rows, 1, codebook_size + 1]."""
        x = self.depth_hidden_projs[step](hidden) + self.depth_code_projs[step](
            self.depth_code_embeddings[step](codes)
        )
        x = self.depth_transformer(
            x + self.depth_speaker_embeddings[step](speakers),
            memories.depth,
            step,
            rings,
            step + 1 in self.spec.lower_body_codebooks,
        )
        return self.depth_classifiers[step](x)


__all__ = ["GestureModel", "SpeechMemories"]
