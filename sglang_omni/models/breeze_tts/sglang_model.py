# SPDX-License-Identifier: Apache-2.0
"""Breeze backbone on SGLang Qwen3/RadixAttention, with its audio output head."""

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from .checkpoint import backbone_weights


class BreezeSGLangModel(Qwen3ForCausalLM):
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is None:
            raise RuntimeError(
                "Breeze backbone requires projected prompt/feedback embeddings"
            )
        hidden = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        if forward_batch.forward_mode.is_extend():
            indices = forward_batch.extend_seq_lens.cumsum(0).long() - 1
            hidden = hidden[indices]
        # The upstream fast/eager streaming path casts the audio LM head to
        # FP32. Use the same matmul precision before CFG and sampling.
        logits = torch.nn.functional.linear(hidden.float(), self.lm_head.weight.float())
        logits = logits[:, : self.config.vocab_size]
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)

    def load_weights(self, weights):
        # The SGLang embedding table is unused: all forwards receive continuous
        # embeddings. Codec, frontend and depth weights load strictly by stage.
        return super().load_weights(
            backbone_weights(weights, self.config.num_hidden_layers)
        )
