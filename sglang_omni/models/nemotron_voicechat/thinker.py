from __future__ import annotations

import torch
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.nemotron_voicechat.fusion import AddFusion

FUNCTION_HEAD_KEY = "stt_model.function_head.weight"
BACKBONE_RENAMES_MAP = (
    ("stt_model.llm.", "backbone."),
    ("stt_model.embed_tokens.", "backbone.embeddings."),
    ("stt_model.lm_head.", "lm_head."),
)


class NemotronVoiceChatForCausalLM(nn.Module):
    def __init__(self, *, config, quant_config=None, prefix=""):
        super().__init__()
        self.llm = NemotronHForCausalLM(
            config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix)
        )
        self.function_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("function_head", prefix),
        )
        self.fusion = AddFusion(config.duplex)
        embedding = self.llm.get_input_embeddings().weight
        max_batch = get_global_server_args().max_running_requests
        self._fusion_buffer = torch.zeros(
            max_batch,
            config.hidden_size,
            device=embedding.device,
            dtype=embedding.dtype,
        )
        self._fusion_mask = torch.zeros(
            max_batch, dtype=torch.bool, device=embedding.device
        )
        self._function_ids = torch.zeros(
            max_batch, dtype=torch.long, device=embedding.device
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def forward(
        self, input_ids, positions, forward_batch, input_embeds=None, **omni_kwargs
    ):
        del omni_kwargs
        if input_embeds is None:
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            batch = input_embeds.shape[0]
            mask = self._fusion_mask[:batch]
            input_embeds = torch.where(
                mask.unsqueeze(-1),
                self._fusion_buffer[:batch].to(input_embeds.dtype),
                input_embeds,
            )
            self._fusion_mask[:batch] = False
        hidden = self.llm.model.forward(
            input_ids, positions, forward_batch, None, input_embeds
        )
        self._sample_function_ids(hidden, forward_batch)
        return self.llm.logits_processor(
            input_ids, hidden, self.llm.lm_head, forward_batch
        )

    def _sample_function_ids(self, hidden, forward_batch):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            last = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden = hidden[last]
        logits = self.function_head.quant_method.apply(self.function_head, hidden)
        batch = logits.shape[0]
        # The function id is only sampled at greedy sampling
        self._function_ids[:batch] = logits.argmax(dim=-1)

    def _backbone_weights_stream(self, parameters, weights):
        # Drop RNN weights from the stream.
        for name, weight in weights:
            if name == FUNCTION_HEAD_KEY:
                parameter = parameters["function_head.weight"]
                default_weight_loader(parameter, weight)
                continue
            for source, target in BACKBONE_RENAMES_MAP:
                if name.startswith(source):
                    yield target + name[len(source) :], weight
                    break

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        self.llm.load_weights(self._backbone_weights_stream(parameters, weights))


EntryClass = NemotronVoiceChatForCausalLM
