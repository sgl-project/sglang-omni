# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex language model as an SGLang model.

The temporal transformer is SGLang's Llama with paged KV; what is Moshi's
alone lives around it: the 17 summed input embeddings, the depformer that
spells out each frame's agent codes, and the buffers through which the model
runner hands a decode step its fused input row and takes the hidden state
back.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    DEPFORMER,
    NUM_AUDIO_STREAMS,
    NUM_STREAMS,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.components.depformer import Depformer

BACKBONE_LAYER_PREFIX = "transformer.layers."


def _backbone_weight(name: str, tensor: torch.Tensor):
    """Yield the Llama names for one temporal-transformer tensor."""
    index, _, tail = name[len(BACKBONE_LAYER_PREFIX) :].partition(".")
    base = f"model.layers.{index}."
    if tail == "self_attn.in_proj_weight":
        for proj, shard in zip(
            ("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0), strict=True
        ):
            yield base + f"self_attn.{proj}.weight", shard
    elif tail == "self_attn.out_proj.weight":
        yield base + "self_attn.o_proj.weight", tensor
    elif tail == "norm1.alpha":
        yield base + "input_layernorm.weight", tensor.reshape(-1)
    elif tail == "norm2.alpha":
        yield base + "post_attention_layernorm.weight", tensor.reshape(-1)
    elif tail == "gating.linear_in.weight":
        # Note (wilsonzheng0327): Gate first: the reference activates the first half.
        gate, up = tensor.chunk(2, dim=0)
        yield base + "mlp.gate_proj.weight", gate
        yield base + "mlp.up_proj.weight", up
    elif tail == "gating.linear_out.weight":
        yield base + "mlp.down_proj.weight", tensor
    else:
        raise KeyError(f"unexpected temporal transformer tensor {name!r}")


class PersonaPlexForCausalLM(nn.Module):
    def __init__(self, *, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.temporal = TEMPORAL_TRANSFORMER
        self.llm = LlamaForCausalLM(
            config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix)
        )
        for layer in self.llm.model.layers:
            layer.self_attn.attn.sliding_window_size = (
                self.get_attention_sliding_window_size()
            )

        dim = self.temporal.dim
        self.audio_emb = nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, dim) for _ in range(NUM_AUDIO_STREAMS)
        )
        self.text_emb = nn.Embedding(TEXT_CARD + 1, dim)
        self.depformer = Depformer(DEPFORMER)

        max_batch = get_global_server_args().max_running_requests
        dtype = torch.get_default_dtype()
        device = self.text_emb.weight.device
        self._fusion_buffer = torch.zeros(max_batch, dim, dtype=dtype, device=device)
        self._fusion_mask = torch.zeros(max_batch, dtype=torch.bool, device=device)
        self._hidden_out = torch.zeros(max_batch, dim, dtype=dtype, device=device)

    def get_attention_sliding_window_size(self) -> int:
        # Note (wilsonzheng0327): The reference attends where ``delta < context``: the
        # current step plus ``context - 1`` earlier ones.
        return self.temporal.context - 1

    def embed_rows(self, rows_NK: torch.Tensor) -> torch.Tensor:
        """Sum the 17 stream embeddings of each row, in the reference's order
        (agent codebooks, user codebooks, then text) so bf16 rounding matches."""
        assert rows_NK.shape[-1] == NUM_STREAMS, rows_NK.shape
        summed = self.audio_emb[0](rows_NK[:, 1])
        for k in range(1, NUM_AUDIO_STREAMS):
            summed = summed + self.audio_emb[k](rows_NK[:, 1 + k])
        return summed + self.text_emb(rows_NK[:, 0])

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **_):
        if input_embeds is None:
            batch = input_ids.shape[0]
            assert bool(
                self._fusion_mask[:batch].all()
            ), "PersonaPlex decode step reached the model without fused inputs"
            input_embeds = self._fusion_buffer[:batch]
            self._fusion_mask[:batch] = False
        hidden = self.llm.model(input_ids, positions, forward_batch, input_embeds)
        if forward_batch.forward_mode.is_decode():
            self._hidden_out[: hidden.shape[0]] = hidden
        else:
            last_rows = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            self._hidden_out[: last_rows.shape[0]] = hidden[last_rows]
        return self.llm.logits_processor(
            input_ids, hidden, self.llm.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        backbone: list[tuple[str, torch.Tensor]] = []
        depformer: dict[str, torch.Tensor] = {}
        embeddings: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith(BACKBONE_LAYER_PREFIX):
                backbone.extend(_backbone_weight(name, tensor))
            elif name == "out_norm.alpha":
                backbone.append(("model.norm.weight", tensor.reshape(-1)))
            elif name == "text_linear.weight":
                backbone.append(("lm_head.weight", tensor))
            elif name == "text_emb.weight":
                embeddings["text_emb.weight"] = tensor
                # Note (wilsonzheng0327): Never read (rows arrive pre-embedded); filled
                # so no parameter is left uninitialised.
                backbone.append(("model.embed_tokens.weight", tensor[:TEXT_CARD]))
            elif name.startswith("emb."):
                embeddings["audio_" + name] = tensor
            elif name.startswith(("depformer", "linears.")):
                depformer[name] = tensor
            else:
                raise KeyError(f"unexpected PersonaPlex tensor {name!r}")
        self.llm.load_weights(backbone)
        missing, unexpected = self.audio_emb.load_state_dict(
            {
                k[len("audio_emb.") :]: v
                for k, v in embeddings.items()
                if k != "text_emb.weight"
            },
            strict=False,
        )
        assert not missing and not unexpected, (missing, unexpected)
        self.text_emb.load_state_dict({"weight": embeddings["text_emb.weight"]})
        self.depformer.load_reference_weights(depformer)


EntryClass = PersonaPlexForCausalLM
