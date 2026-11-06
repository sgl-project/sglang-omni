# SPDX-License-Identifier: Apache-2.0
"""Native Ling V3 thinker with audio-conditioned prefill."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.bailing_moe_nextn import BailingMoeForCausalLMNextN
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.types import SchedulerRequest


class Ming3AudioThinker(BailingMoeV3ForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput | PPProxyTensors:
        """Adapt prefill embeddings; omni_prefill_rids belongs to Omni's interface."""
        return super().forward(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if name.startswith(
                ("model.layers.", "model.word_embeddings.", "model.norm.", "lm_head.")
            )
        )


class Ming3AudioMTP(BailingMoeForCausalLMNextN):
    """Ling V3 MTP with the thinker's audio-conditioned prompt embeddings."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        input_embeddings = forward_batch.mm_input_embeds
        if (
            input_embeddings is not None
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # note (YzXiao): Draft prefill shifts prompt embeddings left, leaving each request's new tail token unfilled.
            last_token_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeddings[last_token_indices] = self.model.word_embeddings(
                input_ids[last_token_indices]
            )
        else:
            pass
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeddings
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )


class Ming3AudioModelRunner(ModelRunner):
    supports_speculative_async_decode: bool = True

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        embeddings = self.model.get_input_embeddings()(forward_batch.input_ids)
        offset = 0
        for request, length, prefix_length in zip(
            requests,
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_prefix_lens_cpu,
            strict=True,
        ):
            conditioning = request.data.model_inputs
            positions = conditioning["audio_positions"]
            selected = (positions >= prefix_length) & (
                positions < prefix_length + length
            )
            rows = positions[selected].to(embeddings.device) - prefix_length + offset
            audio_embeddings = conditioning["audio_embeddings"]
            embeddings[rows] = audio_embeddings[selected].to(embeddings)
            offset += length
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=embeddings)
        )
