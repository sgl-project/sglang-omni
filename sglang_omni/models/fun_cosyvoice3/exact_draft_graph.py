# SPDX-License-Identifier: Apache-2.0
"""Capture the exact CosyVoice proposal in the draft graph tail."""

import torch
from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

from sglang_omni.models.fun_cosyvoice3.dspark_draft import (
    FunCosyVoice3DSparkDraftModel,
    fill_draft_block_noise,
)
from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    DraftSamplingState,
    sample_draft_block,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE


class ExactDraftGraphSampler(DsparkDraftSampler):
    """Draft graph tail that draws the mirrored block from buffers staged each step."""

    def __init__(
        self,
        *,
        draft_model: FunCosyVoice3DSparkDraftModel,
        gamma: int,
        max_batch_size: int,
        device: torch.device,
        draft_noise: DraftNoise,
        tp_sync: SpecTpSync,
    ) -> None:
        super().__init__(
            model=draft_model,
            gamma=gamma,
            max_bs=max_batch_size,
            device=device,
            noise=draft_noise,
            tp_sync=tp_sync,
            greedy_draft=False,
        )
        assert draft_model.confidence_head is None
        assert (
            self._megakernel_steps is None
        )  # noqa: leading-underscore  # a megakernel step would skip the exact markov block
        self.step_logits = torch.empty(
            (max_batch_size, gamma, TOTAL_VOCAB_SIZE),
            dtype=torch.float32,
            device=device,
        )
        self.exponential_noise = torch.ones_like(self.step_logits)
        self.presence_mask = torch.zeros(
            (max_batch_size, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=device
        )
        self.repetition_penalties = torch.ones(max_batch_size, device=device)
        self.output_lengths = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self.min_new_tokens = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self.temperatures = torch.ones(max_batch_size, device=device)
        self.greedy = torch.zeros(max_batch_size, dtype=torch.bool, device=device)
        self.compiled_sampler = torch.compile(sample_draft_block, dynamic=True)

    def stage_step_state(self, state: DraftSamplingState) -> None:
        """Copy one step's histories and noise into the buffers a replay reads."""
        request_count = state.output_lengths.shape[0]
        assert request_count <= self.max_bs
        self.presence_mask[:request_count].copy_(state.presence_mask)
        self.repetition_penalties[:request_count].copy_(state.repetition_penalties)
        self.output_lengths[:request_count].copy_(state.output_lengths)
        self.min_new_tokens[:request_count].copy_(state.min_new_tokens)
        self.temperatures[:request_count].copy_(state.temperatures)
        self.greedy[:request_count].copy_(state.greedy)
        fill_draft_block_noise(self.exponential_noise[:request_count], state)

    def _markov_block(  # noqa: leading-underscore  # inherited sampler hook
        self,
        model_hidden: torch.Tensor,
        sample_hidden: torch.Tensor,
        anchor: torch.Tensor,
        bs: int,
    ) -> tuple[torch.Tensor, None]:
        base_logits, confidence_features = self.model.compute_base_logits(model_hidden)
        assert confidence_features is None
        draft_tokens, shaped_logits = self.compiled_sampler(
            base_logits.view(bs, self.gamma, TOTAL_VOCAB_SIZE),
            anchor.long(),
            self.exponential_noise[:bs],
            self.markov_head.markov_w1.weight,
            self.markov_head.markov_w2.weight,
            self.presence_mask[:bs],
            self.repetition_penalties[:bs],
            self.output_lengths[:bs],
            self.min_new_tokens[:bs],
            self.temperatures[:bs],
            self.greedy[:bs],
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        self.step_logits[:bs].copy_(shaped_logits)
        return draft_tokens, None
