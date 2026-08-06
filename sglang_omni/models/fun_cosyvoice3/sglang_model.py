# SPDX-License-Identifier: Apache-2.0

"""Fun-CosyVoice3 LLM backbone for sglang-omni."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

VOCAB_SIZE = 6561
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 200

SOS_ID = VOCAB_SIZE + 0
EOS_ID = VOCAB_SIZE + 1
TASK_ID = VOCAB_SIZE + 2
FILL_ID = VOCAB_SIZE + 3


class FunCosyVoice3SGLangModel(Qwen2ForCausalLM):

    def __init__(
        self,
        config: Any,
        quant_config: Any = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)
        self.text_embed_tokens = self.model.embed_tokens
        self.speech_embedding = nn.Embedding(TOTAL_VOCAB_SIZE, config.hidden_size)
        self.llm_decoder = nn.Linear(config.hidden_size, TOTAL_VOCAB_SIZE, bias=False)
        self._cached_params_dict = dict(self.named_parameters())

    @property
    def vocab_size(self) -> int:
        return TOTAL_VOCAB_SIZE

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        pass

    def get_input_embeddings(self):
        return self.speech_embedding

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> Any:
        fwd_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = fwd_mode is not None and fwd_mode.is_decode()

        if is_decode:
            input_embeds = self.speech_embedding(input_ids)

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if fwd_mode is not None and fwd_mode.is_extend():
            extend_seq_lens = forward_batch.extend_seq_lens
            if extend_seq_lens is not None:
                last_indices = (
                    torch.cumsum(extend_seq_lens.to(device=hidden_states.device), dim=0)
                    - 1
                )
            else:
                last_indices = torch.tensor(
                    [hidden_states.shape[0] - 1], device=hidden_states.device
                )
            hidden_states = hidden_states[last_indices]

        logits = self.llm_decoder(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        backbone_weights = []
        custom_weights = {}

        for name, loaded_weight in weights:
            # note: Qwen2 backbone inside CosyVoice3LM uses llm.model.model.* prefix
            backbone_prefix = "llm.model.model."
            if name.startswith(backbone_prefix):
                backbone_weights.append(
                    ("model." + name[len(backbone_prefix) :], loaded_weight)
                )
                continue

            # note: skip Qwen2 text lm_head — CosyVoice3 uses llm_decoder instead
            if name == "llm.model.lm_head.weight":
                continue

            if name in (
                "speech_embedding.weight",
                "llm_decoder.weight",
                "llm_decoder.bias",
            ):
                param = self._cached_params_dict.get(name)
                if param is not None:
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                continue

            # note: pass through HF safetensors keys (model.*) to Qwen2 backbone
            backbone_weights.append((name, loaded_weight))

        if backbone_weights:
            super().load_weights(backbone_weights)



EntryClass = FunCosyVoice3SGLangModel
