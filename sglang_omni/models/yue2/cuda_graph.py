# SPDX-License-Identifier: Apache-2.0
"""Reusable single-branch AR decode graph with device-resident cache length."""

import torch

from sglang_omni.models.yue2.layers import TokenCache
from sglang_omni.models.yue2.model import Yue2Model


class DecodeGraph:
    def __init__(self, model: Yue2Model) -> None:
        self.model = model
        weight = model.model.embed_tokens.weight
        self.capacity = model.config.max_position_embeddings
        shape = (
            1,
            self.capacity,
            model.config.num_key_value_heads,
            model.config.head_dim,
        )
        self.cache = TokenCache(capacity=self.capacity)
        for layer in model.model.layers:
            self.cache.keys.append(weight.new_zeros(shape).transpose(1, 2))
            self.cache.values.append(weight.new_zeros(shape).transpose(1, 2))
        self.position = torch.zeros(1, dtype=torch.long, device=weight.device)
        self.tokens = torch.zeros((1, 1), dtype=torch.long, device=weight.device)
        self.query_offsets = torch.tensor(
            [0, 1], dtype=torch.int32, device=weight.device
        )
        self.key_offsets = self.query_offsets * self.capacity
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None

    def forward(self) -> torch.Tensor:
        model = self.model
        cosine, sine = model.model.rotary_emb(self.position[None])
        hidden_states = model.model.embed_tokens(self.tokens)
        used_lengths = (self.position + 1).to(torch.int32)
        for layer, keys, values in zip(
            model.model.layers, self.cache.keys, self.cache.values
        ):
            query, key, value = layer.self_attn.project_qkv(
                layer.input_layernorm(hidden_states), cosine, sine
            )
            keys.index_copy_(2, self.position, key)
            values.index_copy_(2, self.position, value)
            # note (ruiling): The packed FlashAttention entrypoint honors used lengths; the fixed-batch entrypoint does not.
            attended = torch.ops.aten._flash_attention_forward(  # noqa: leading-underscore -- PyTorch operator name
                query.transpose(1, 2)[0],
                keys.transpose(1, 2)[0],
                values.transpose(1, 2)[0],
                self.query_offsets,
                self.key_offsets,
                1,
                self.capacity,
                0.0,
                False,
                False,
                seqused_k=used_lengths,
            )[
                0
            ]
            hidden_states = hidden_states + layer.self_attn.o_proj(
                attended.flatten(1)[None]
            )
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
        logits = model.lm_head(model.model.norm(hidden_states))[:, 0]
        self.position.add_(1)
        return logits

    def prefill(self, prefix: list[int]) -> torch.Tensor:
        self.cache.length = 0
        logits = self.model(
            torch.tensor([prefix], device=self.tokens.device), self.cache
        )
        self.position.fill_(len(prefix))
        if self.graph is None:
            stream = torch.cuda.Stream(device=self.tokens.device)
            stream.wait_stream(torch.cuda.current_stream(self.tokens.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.position.fill_(len(prefix))
                    self.forward()
                self.position.fill_(len(prefix))
            torch.cuda.current_stream(self.tokens.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward()
            self.graph, self.output = graph, output
            self.position.fill_(len(prefix))
        else:
            pass
        return logits

    def step(self, token: torch.Tensor) -> torch.Tensor:
        assert self.cache.length < self.capacity
        assert self.graph is not None and self.output is not None
        self.tokens.copy_(token)
        self.graph.replay()
        self.cache.length += 1
        return self.output
