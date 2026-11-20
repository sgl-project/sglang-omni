# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 draft for targets whose runner composes the multimodal prefill embeddings."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3


class OmniLlamaForCausalLMEagle3(LlamaForCausalLMEagle3):
    """Fills the draft-extend row the shifted prefill embeddings leave empty.

    The parent embeds each request's newly sampled last token only for requests with
    multimodal items, which an Omni runner composes before the scheduler sees them.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> LogitsProcessorOutput:
        prefill_embeds = forward_batch.mm_input_embeds
        if (
            input_embeds is None
            and prefill_embeds is not None
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.contains_mm_inputs()
        ):
            last_rows = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            prefill_embeds[last_rows] = self.model.embed_tokens(input_ids[last_rows])
        else:
            pass
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            get_embedding,
            pp_proxy_tensors,
        )
