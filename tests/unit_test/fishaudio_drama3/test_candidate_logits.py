# SPDX-License-Identifier: Apache-2.0
"""Candidate-only lm_head against the full vocabulary projection."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.fishaudio_drama3.sglang_model import CandidateLogitsProcessor

VOCABULARY_SIZE = 64
HIDDEN_SIZE = 8
ROW_COUNT = 3


def candidate_processor(
    lm_head_weight: torch.Tensor, candidate_token_ids: torch.Tensor
) -> CandidateLogitsProcessor:
    processor = CandidateLogitsProcessor.__new__(CandidateLogitsProcessor)
    nn.Module.__init__(processor)
    processor.use_fp32_lm_head = False
    processor.candidate_token_ids = candidate_token_ids
    processor.candidate_lm_head_weight = lm_head_weight.index_select(
        0, candidate_token_ids
    ).contiguous()
    return processor


def test_candidate_columns_equal_the_full_projection() -> None:
    generator = torch.Generator().manual_seed(0)
    lm_head_weight = torch.randn(VOCABULARY_SIZE, HIDDEN_SIZE, generator=generator)
    hidden_states = torch.randn(ROW_COUNT, HIDDEN_SIZE, generator=generator)
    candidate_token_ids = torch.tensor([3, *range(10, 20), VOCABULARY_SIZE - 1])
    processor = candidate_processor(lm_head_weight, candidate_token_ids)

    logits = processor._compute_lm_head(
        hidden_states, SimpleNamespace(weight=lm_head_weight)
    )

    assert logits.shape == (ROW_COUNT, VOCABULARY_SIZE)
    torch.testing.assert_close(
        logits.index_select(1, candidate_token_ids),
        (hidden_states @ lm_head_weight.T).index_select(1, candidate_token_ids),
    )
    is_candidate = torch.zeros(VOCABULARY_SIZE, dtype=torch.bool)
    is_candidate[candidate_token_ids] = True
    assert torch.all(logits[:, ~is_candidate] == torch.finfo(logits.dtype).min)


def test_candidate_columns_win_every_softmax() -> None:
    generator = torch.Generator().manual_seed(1)
    lm_head_weight = torch.randn(VOCABULARY_SIZE, HIDDEN_SIZE, generator=generator)
    hidden_states = torch.randn(ROW_COUNT, HIDDEN_SIZE, generator=generator)
    candidate_token_ids = torch.tensor([0, 7, 8, 9])
    processor = candidate_processor(lm_head_weight, candidate_token_ids)

    logits = processor._compute_lm_head(
        hidden_states, SimpleNamespace(weight=lm_head_weight)
    )

    probabilities = torch.softmax(logits.float(), dim=-1)
    torch.testing.assert_close(
        probabilities.index_select(1, candidate_token_ids).sum(-1),
        torch.ones(ROW_COUNT),
    )
