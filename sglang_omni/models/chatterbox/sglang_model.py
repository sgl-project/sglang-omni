# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Chatterbox T3 model: GPT-2 backbone + dual text/speech head."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gpt2 import GPT2Block

_CONV1D_WEIGHT_NAMES = ("c_attn", "c_proj", "c_fc")


class ChatterboxT3SGLangModel(nn.Module):
    """T3 AR model: 24 GPT-2 blocks plus dual-vocab embedding/head and a speaker
    conditioning prefix. Text and speech tokens use separate embedding tables;
    tfmr.wte is unused and skipped at load time."""

    def __init__(
        self,
        config: "ChatterboxT3Config",
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList(
            [
                GPT2Block(i, config, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.text_emb = VocabParallelEmbedding(
            config.text_tokens_dict_size, config.hidden_size
        )
        self.speech_emb = VocabParallelEmbedding(
            config.speech_tokens_dict_size, config.hidden_size
        )
        self.text_head = ParallelLMHead(
            config.text_tokens_dict_size, config.hidden_size
        )
        self.speech_head = ParallelLMHead(
            config.speech_tokens_dict_size, config.hidden_size, bias=True
        )

        self.cond_enc = nn.Module()
        self.cond_enc.spkr_enc = nn.Linear(
            config.speaker_embed_size, config.hidden_size, bias=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        if input_embeds is not None:
            token_embeds = input_embeds
        else:
            token_embeds = self.speech_emb(input_ids)
        hidden_states = token_embeds + self.wpe(positions)

        for block in self.h:
            hidden_states = block(hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)

        logits = self.speech_head(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("tfmr.wte."):
                continue
            if name.startswith("tfmr."):
                name = name[len("tfmr.") :]
            if name not in params_dict:
                continue

            if name.endswith(".weight") and any(
                suffix in name for suffix in _CONV1D_WEIGHT_NAMES
            ):
                # HF GPT-2 stores Conv1D weights transposed relative to Linear.
                loaded_weight = loaded_weight.t()

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = ChatterboxT3SGLangModel
