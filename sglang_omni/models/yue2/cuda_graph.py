# SPDX-License-Identifier: Apache-2.0
"""Reusable AR decode graphs with independent request positions."""

import ctypes
import weakref

import torch

from sglang_omni.models.yue2.layers import TokenCache
from sglang_omni.models.yue2.model import Yue2Model


def release_capture(
    graph: torch.cuda.CUDAGraph, stream: torch.cuda.ExternalStream
) -> None:
    with torch.cuda.device(stream.device):
        graph.reset()
        torch.cuda.check_error(
            torch.cuda.cudart().cudaStreamDestroy(stream.cuda_stream)
        )


class CapturedGraph:
    def __init__(self, device: torch.device) -> None:
        # note (ruiling): PyTorch #193402 lets pooled-stream reuse invalidate another live graph's cuBLAS workspace.
        with torch.cuda.device(device):
            handle = ctypes.c_void_p()
            torch.cuda.check_error(
                torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(handle))
            )
            assert handle.value is not None
            try:
                self.stream = torch.cuda.ExternalStream(handle.value, device=device)
                self.graph = torch.cuda.CUDAGraph()
            except Exception:
                torch.cuda.check_error(
                    torch.cuda.cudart().cudaStreamDestroy(handle.value)
                )
                raise
        weakref.finalize(self, release_capture, self.graph, self.stream)

    def replay(self) -> None:
        self.graph.replay()


class DecodeGraph:
    def __init__(
        self, model: Yue2Model, request_count: int = 1, capacity: int | None = None
    ) -> None:
        capacity = (
            model.config.max_position_embeddings if capacity is None else capacity
        )
        self.model = model
        self.request_count = request_count
        self.capacity = capacity
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        shape = (
            request_count * capacity,
            model.config.num_key_value_heads,
            model.config.head_dim,
        )
        self.cache = [
            (weight.new_zeros(shape), weight.new_zeros(shape))
            for _ in model.model.layers
        ]
        self.positions = torch.zeros(
            request_count, dtype=torch.long, device=self.device
        )
        self.active = torch.ones_like(self.positions)
        self.tokens = torch.zeros(
            (request_count, 1), dtype=torch.long, device=self.device
        )
        self.row_offsets = torch.arange(request_count, device=self.device) * capacity
        self.query_offsets = torch.arange(
            request_count + 1, dtype=torch.int32, device=self.device
        )
        self.key_offsets = self.query_offsets * capacity
        self.graph: CapturedGraph | None = None
        self.output: torch.Tensor | None = None

    def forward(self) -> torch.Tensor:
        model = self.model
        rotary_positions = self.positions[:, None].to(torch.int32)
        cosine, sine = model.model.rotary_emb(rotary_positions)
        residual = model.model.embed_tokens(self.tokens)
        hidden_states = model.model.layers[0].input_layernorm(residual)
        write_indices = self.row_offsets + self.positions
        used_lengths = (self.positions + 1).to(torch.int32)
        for layer_index, (layer, (keys, values)) in enumerate(
            zip(model.model.layers, self.cache)
        ):
            query, key, value = layer.self_attn.project_qkv(
                hidden_states, cosine, sine, rotary_positions
            )
            keys.index_copy_(0, write_indices, key[:, :, 0])
            values.index_copy_(0, write_indices, value[:, :, 0])
            attended = torch.ops.aten._flash_attention_forward(  # noqa: leading-underscore -- PyTorch operator name
                query[:, :, 0],
                keys,
                values,
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
            attended = layer.self_attn.o_proj(attended.flatten(1)[:, None])
            hidden_states, residual = layer.post_attention_layernorm.add(
                attended, residual
            )
            next_norm = (
                model.model.layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(model.model.layers)
                else model.model.norm
            )
            hidden_states, residual = next_norm.add(layer.mlp(hidden_states), residual)
        logits = model.lm_head(hidden_states)[:, 0]
        self.positions.add_(self.active)
        return logits

    def prefill(self, prefixes: list[list[int]]) -> torch.Tensor:
        assert len(prefixes) == self.request_count
        logits: list[torch.Tensor] = []
        for row, prefix in enumerate(prefixes):
            assert 0 < len(prefix) < self.capacity
            start = row * self.capacity
            end = start + self.capacity
            cache = TokenCache(
                capacity=self.capacity,
                keys=[keys[start:end].transpose(0, 1)[None] for keys, _ in self.cache],
                values=[
                    values[start:end].transpose(0, 1)[None] for _, values in self.cache
                ],
            )
            logits.append(self.model(torch.tensor([prefix], device=self.device), cache))
        positions = torch.tensor(
            [len(prefix) for prefix in prefixes], device=self.device
        )
        self.positions.copy_(positions)
        self.active.fill_(1)
        if self.graph is None:
            graph = CapturedGraph(self.device)
            stream = graph.stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.positions.copy_(positions)
                    self.forward()
                self.positions.copy_(positions)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(graph.graph, stream=stream):
                output = self.forward()
            self.graph, self.output = graph, output
            self.positions.copy_(positions)
        else:
            pass
        return torch.cat(logits)

    def step(
        self, tokens: torch.Tensor, active: list[bool] | None = None
    ) -> torch.Tensor:
        assert self.graph is not None and self.output is not None
        self.tokens.copy_(tokens)
        if active is not None:
            self.active.copy_(torch.tensor(active, device=self.device))
        else:
            pass
        self.graph.replay()
        return self.output
