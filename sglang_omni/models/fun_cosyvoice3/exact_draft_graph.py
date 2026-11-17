# SPDX-License-Identifier: Apache-2.0
"""Capture the exact CosyVoice proposal in the draft graph tail."""

import torch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

from sglang_omni.models.fun_cosyvoice3.dspark_draft import FunCosyVoice3DSparkDraftModel
from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    DraftSamplingState,
    sample_draft_block,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE
from sglang_omni.sampling.speculative_noise import (
    DRAFT_NOISE_SEED_SALT,
    seeded_exponential_noise,
)


class ExactDraftGraphSampler(DsparkDraftSampler):
    requires_per_step_sampling_params: bool = True

    def __init__(
        self,
        *,
        model: FunCosyVoice3DSparkDraftModel,
        gamma: int,
        max_bs: int,
        device: torch.device,
        noise: DraftNoise,
        tp_sync: SpecTpSync,
        out: torch.Tensor | None,
    ) -> None:
        if not hasattr(DsparkDraftSampler, "requires_per_step_sampling_params"):
            raise RuntimeError(
                "Exact draft graphs require the SGLang per-step sampler staging API."
            )
        else:
            pass
        super().__init__(
            model=model,
            gamma=gamma,
            max_bs=max_bs,
            device=device,
            noise=noise,
            tp_sync=tp_sync,
            greedy_draft=False,
            out=out,
        )
        assert model.confidence_head is None
        self.step_logits = torch.empty(
            (max_bs, gamma, TOTAL_VOCAB_SIZE), dtype=torch.float32, device=device
        )
        self.exponential_noise = torch.ones_like(self.step_logits)
        self.presence_mask = torch.zeros(
            (max_bs, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=device
        )
        self.repetition_penalties = torch.ones(max_bs, device=device)
        self.output_lengths = torch.zeros(max_bs, dtype=torch.long, device=device)
        self.min_new_tokens = torch.zeros(max_bs, dtype=torch.long, device=device)
        self.temperatures = self.params.temperatures
        self.greedy = self.params.greedy_mask
        self.compiled_sampler = torch.compile(sample_draft_block, dynamic=True)

    def stage_sampling_params(
        self, *, bs: int, sampling_info: SamplingBatchInfo | None
    ) -> None:
        # note (yaourt): the stage state includes the exact temperatures and greedy mask.
        state = self.markov_head.sampling_state
        assert state is not None
        assert state.output_lengths.shape[0] == bs
        self.stage_exact_state(state)

    def stage_exact_state(self, state: DraftSamplingState) -> None:
        request_count = state.output_lengths.shape[0]
        self.presence_mask[:request_count].copy_(state.presence_mask)
        self.repetition_penalties[:request_count].copy_(state.repetition_penalties)
        self.output_lengths[:request_count].copy_(state.output_lengths)
        self.min_new_tokens[:request_count].copy_(state.min_new_tokens)
        self.temperatures[:request_count].copy_(state.temperatures)
        self.greedy[:request_count].copy_(state.greedy)
        if state.seeds is None:
            self.exponential_noise[:request_count].exponential_()
        else:
            self.exponential_noise[:request_count].copy_(
                seeded_exponential_noise(
                    state.seeds,
                    state.output_lengths,
                    salt=DRAFT_NOISE_SEED_SALT,
                    column_count=self.gamma * TOTAL_VOCAB_SIZE,
                ).view(request_count, self.gamma, TOTAL_VOCAB_SIZE)
            )

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
