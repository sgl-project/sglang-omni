# SPDX-License-Identifier: Apache-2.0
"""DSpark drafts with a trained reduced-vocabulary head and target-token Markov state."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.dspark import Qwen3DSparkModel
from torch import nn
from transformers import Qwen3Config


class ReducedVocabularyDSparkDraftModel(Qwen3DSparkModel):
    """Share target embeddings while preserving the draft's trained output distribution."""

    draft_to_target_token_ids: torch.Tensor

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        # note (Codex): the captured sampler sizes its proposal logits in target vocabulary space.
        self.lm_head.org_vocab_size: int = config.vocab_size
        self.register_buffer(
            "draft_to_target_token_ids",
            torch.empty(config.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        """The worker supplies both target modules; only its embedding is shared."""
        self.embed_tokens = embed_tokens

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draft_logits = self.lm_head(hidden.to(self.lm_head.weight.dtype))
        target_logits = draft_logits.new_full(
            (*draft_logits.shape[:-1], self.config.vocab_size), float("-inf")
        )
        target_logits.index_copy_(-1, self.draft_to_target_token_ids, draft_logits)
        return target_logits, hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        checkpoint_weights = dict(weights)
        token_id_offsets = checkpoint_weights.pop("d2t").long()
        checkpoint_weights.pop("t2d")
        output_weights = checkpoint_weights.pop("lm_head.weight")
        markov_output_weights = checkpoint_weights.pop("markov_head.markov_w2.weight")
        super().load_weights(checkpoint_weights.items())
        target_token_ids = token_id_offsets + torch.arange(
            token_id_offsets.numel(), device=token_id_offsets.device
        )
        with torch.no_grad():
            self.draft_to_target_token_ids.copy_(target_token_ids)
            self.lm_head.weight.copy_(output_weights)
            # note (Codex): verification and Markov recurrence consume target ids, including anchors outside the draft vocabulary.
            markov_output = self.markov_head.markov_w2.weight
            markov_output.zero_()
            markov_output.index_copy_(
                0,
                target_token_ids.to(markov_output.device),
                markov_output_weights.to(markov_output),
            )
