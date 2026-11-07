# SPDX-License-Identifier: Apache-2.0
"""DSpark checkpoint vocabulary routing without accelerator execution."""

import torch
from sglang.srt.models.dspark import VanillaMarkov
from torch import nn

from sglang_omni.speculative.dspark_draft import ReducedVocabularyDSparkDraftModel


def test_reduced_heads_load_in_target_token_order_without_replacing_target_embedding() -> (
    None
):
    """Draft row ids must not become target ids in the head or Markov recurrence."""
    draft = ReducedVocabularyDSparkDraftModel.__new__(ReducedVocabularyDSparkDraftModel)
    nn.Module.__init__(draft)
    draft.projector_type = None
    draft.is_nemotron_35_draft = False
    draft.confidence_head = None
    draft.lm_head = nn.Linear(2, 3, bias=False)
    draft.markov_head = VanillaMarkov(vocab_size=10, markov_rank=2)
    draft.register_buffer("draft_to_target_token_ids", torch.empty(3, dtype=torch.long))
    output_weights = torch.arange(6, dtype=torch.float32).view(3, 2)
    markov_inputs = torch.arange(20, dtype=torch.float32).view(10, 2)
    markov_outputs = output_weights + 10
    draft.load_weights(
        [
            ("d2t", torch.tensor([2, 4, 7])),
            ("t2d", torch.zeros(10, dtype=torch.bool)),
            ("lm_head.weight", output_weights),
            ("markov_head.markov_w1.weight", markov_inputs),
            ("markov_head.markov_w2.weight", markov_outputs),
        ]
    )
    trained_head = draft.lm_head
    target_embedding = nn.Embedding(10, 2)
    draft.attach_shared_modules(embed_tokens=target_embedding, lm_head=nn.Linear(2, 10))

    assert draft.embed_tokens is target_embedding
    assert draft.lm_head is trained_head
    assert draft.draft_to_target_token_ids.tolist() == [2, 5, 9]
    torch.testing.assert_close(draft.lm_head.weight, output_weights)
    torch.testing.assert_close(draft.markov_head.markov_w1.weight, markov_inputs)
    torch.testing.assert_close(
        draft.markov_head.markov_w2.weight[[2, 5, 9]], markov_outputs
    )
    assert (
        torch.count_nonzero(draft.markov_head.markov_w2.weight[[0, 1, 3, 4, 6, 7, 8]])
        == 0
    )
