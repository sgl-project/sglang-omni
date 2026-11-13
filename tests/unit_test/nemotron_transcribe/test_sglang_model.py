# SPDX-License-Identifier: Apache-2.0
"""Nemotron Transcribe wrapper contracts with SGLang speculative decoding."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from torch import nn

from sglang_omni.models.nemotron_transcribe.sglang_model import (
    NemotronTranscribeForConditionalGeneration,
)


def test_nextn_shares_the_wrapper_head_module_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self.embed_tokens.weight, self.lm_head.weight

    class DraftModel:
        def __init__(self) -> None:
            self.hot_token_id: torch.Tensor | None = None
            self.embed_weight: torch.Tensor | None = None
            self.head_weight: torch.Tensor | None = None
            self.lm_head: nn.Module | None = None

        def set_embed_and_head(
            self, embed_weight: torch.Tensor, head_weight: torch.Tensor
        ) -> None:
            self.embed_weight = embed_weight
            self.head_weight = head_weight

        def set_lm_head_from_target(self, lm_head: nn.Module) -> None:
            self.lm_head = lm_head

    language_model = LanguageModel()
    target = NemotronTranscribeForConditionalGeneration.__new__(
        NemotronTranscribeForConditionalGeneration
    )
    nn.Module.__init__(target)
    target.language_model = language_model

    def resolve_shared_embed_and_head(
        worker: EagleDraftWorker,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del worker
        return target.get_embed_and_head()

    monkeypatch.setattr(
        EagleDraftWorker,
        "_resolve_shared_embed_and_head",
        resolve_shared_embed_and_head,
    )
    draft_model = DraftModel()
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=target))
    worker.draft_runner = SimpleNamespace(model=draft_model)
    worker.hot_token_id = None
    worker.speculative_algorithm = SpeculativeAlgorithm.from_string("NEXTN")

    worker.init_lm_head()

    assert target.lm_head is language_model.lm_head
    assert draft_model.embed_weight is language_model.embed_tokens.weight
    assert draft_model.head_weight is language_model.lm_head.weight
    assert draft_model.lm_head is language_model.lm_head
