# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape CUDA graphs of the Qwen3-Omni D2 decoders over static KV caches.

Every mask spans the whole cache: written slots and earlier queries attend,
unwritten slots never do, so one capture serves every cursor.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache, StaticLayer


def static_cache(decoder: nn.Module, capacity_tokens: int) -> Cache:
    return Cache(layers=[StaticLayer(capacity_tokens) for _ in decoder.layers])


def set_cache_cursor(cache: Cache, tokens: int) -> None:
    for layer in cache.layers:
        layer.cumulative_length.fill_(tokens)


def reset_cache(cache: Cache) -> None:
    for layer in cache.layers:
        layer.reset()
    set_cache_cursor(cache, 0)


def causal_cache_mask(
    cursor: int, query_tokens: int, capacity_tokens: int, device: torch.device
) -> torch.Tensor:
    """[1, 1, query, capacity] boolean mask; query i attends to every slot up to cursor + i."""
    columns = torch.arange(capacity_tokens, device=device)
    rows = torch.arange(cursor, cursor + query_tokens, device=device)
    return (columns[None, :] <= rows[:, None])[None, None]


class DecoderGraph:
    """One decoder forward for a fixed number of query tokens, replayed at any cache cursor."""

    def __init__(
        self,
        decoder: nn.Module,
        cache: Cache,
        position_shape: tuple[int, ...],
        capture_cursor: int,
        hooked_layer: nn.Module | None = None,
    ) -> None:
        self.decoder = decoder
        self.cache = cache
        parameter = next(decoder.parameters())
        query_tokens = position_shape[-1]
        capacity_tokens = cache.layers[0].max_cache_len
        self.inputs_embeds = torch.zeros(
            (1, query_tokens, decoder.config.hidden_size),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.position_ids = torch.zeros(
            position_shape, device=parameter.device, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (1, 1, query_tokens, capacity_tokens),
            device=parameter.device,
            dtype=torch.bool,
        )
        self.columns = torch.arange(capacity_tokens, device=parameter.device)[None, :]
        self.query_offsets = torch.arange(query_tokens, device=parameter.device)[
            :, None
        ]
        self.hooked_states: torch.Tensor | None = None
        self.graph = torch.cuda.CUDAGraph()
        set_cache_cursor(cache, capture_cursor)
        self.write_mask(capture_cursor)
        self.position_ids.copy_(
            (capture_cursor + self.query_offsets[:, 0]).expand(position_shape)
        )
        # The eager warmup also allocates the static cache before capture.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.forward()
        set_cache_cursor(cache, capture_cursor)
        torch.cuda.synchronize(parameter.device)
        if hooked_layer is not None:
            handle = hooked_layer.register_forward_hook(self.keep_layer_output)
        else:
            handle = None
        try:
            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
            ):
                self.last_hidden_state = self.forward()
        finally:
            if handle is not None:
                handle.remove()
            else:
                pass

    def keep_layer_output(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.hooked_states = output

    def forward(self) -> torch.Tensor:
        return self.decoder(
            inputs_embeds=self.inputs_embeds,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            use_cache=True,
        ).last_hidden_state

    def write_mask(self, cursor: int) -> None:
        torch.le(
            self.columns, self.query_offsets + cursor, out=self.attention_mask[0, 0]
        )

    def replay(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, cursor: int
    ) -> torch.Tensor:
        """Append the queries at the cache's cursor and return the graph's last hidden states.

        The cache advances its own cursor inside the graph; cursor only shapes the mask.
        """
        self.inputs_embeds.copy_(inputs_embeds)
        self.position_ids.copy_(position_ids)
        self.write_mask(cursor)
        self.graph.replay()
        return self.last_hidden_state
