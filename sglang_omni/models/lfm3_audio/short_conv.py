# SPDX-License-Identifier: Apache-2.0
"""LFM3 gated short convolution on SGLang's MambaPool conv state.

The width-7 convolution runs in torch: the SGLang causal-conv1d kernels serve widths 2 to 5 only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import set_weight_attrs
from torch import nn

from sglang_omni.models.lfm3_audio.hf_config import Lfm3SGLangConfig


class Lfm3ShortConv(nn.Module):
    """Gated short convolution: in_proj -> (B, C, x) -> conv(B * x) -> C * conv -> out_proj.

    A request's MambaPool conv state holds the last kernel - 1 values of B * x per channel, shaped (hidden, kernel - 1).
    """

    def __init__(
        self, config: Lfm3SGLangConfig, layer_idx: int, prefix: str = ""
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.hidden_size = config.hidden_size
        tp_size = get_parallel().tp_size
        self.hidden_size_per_partition = self.hidden_size // tp_size
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,
            bias=bool(config.conv_bias),
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=bool(config.conv_bias),
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )
        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(
            self.conv_weight,
            {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
        )
        if config.conv_bias:
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias,
                {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
            )
        else:
            self.register_parameter("conv_bias", None)

    def conv_decode(
        self,
        gated_inputs: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """One token per request: dot the rolling window, then advance the state in place."""
        # note (ruilin): a graph's padding rows carry slot -1, which indexing would wrap to the last live slot; slot 0 is the pool's dummy.
        cache_indices = cache_indices.clamp(min=0)
        state = conv_state[cache_indices].to(gated_inputs.dtype)
        window = torch.cat([state, gated_inputs.unsqueeze(-1)], dim=-1)
        conv_outputs = (
            window * self.conv_weight.to(gated_inputs.dtype).unsqueeze(0)
        ).sum(dim=-1)
        if self.conv_bias is not None:
            conv_outputs = conv_outputs + self.conv_bias.to(conv_outputs.dtype)
        else:
            pass
        conv_state[cache_indices] = window[:, :, 1:].to(conv_state.dtype)
        return conv_outputs

    def conv_prefill(
        self,
        gated_inputs: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Causal depthwise conv of every request at once, without a host sync.

        Each request's segment is its cached state (zeros without a prefix) followed by its B * x, all segments laid end to end; one conv over the whole layout gives every token's output, and a segment's last kernel - 1 values are the request's new state.
        """
        state_width = self.conv_kernel - 1
        device = gated_inputs.device
        dtype = gated_inputs.dtype
        num_requests = cache_indices.shape[0]
        num_tokens = gated_inputs.shape[0]
        token_starts = query_start_loc.long()
        rows = torch.arange(num_requests, device=device)
        lanes = torch.arange(state_width, device=device)
        segment_starts = token_starts[:-1] + rows * state_width
        token_rows = torch.searchsorted(
            token_starts[1:], torch.arange(num_tokens, device=device), right=True
        )
        token_positions = (
            torch.arange(num_tokens, device=device) + (token_rows + 1) * state_width
        )
        states = conv_state[cache_indices].to(dtype)
        states = torch.where(has_initial_state[:, None, None], states, 0)
        layout = gated_inputs.new_empty(
            num_tokens + num_requests * state_width, gated_inputs.shape[1]
        )
        layout[(segment_starts[:, None] + lanes).flatten()] = states.transpose(
            1, 2
        ).flatten(0, 1)
        layout[token_positions] = gated_inputs
        conv_outputs = F.conv1d(
            layout.transpose(0, 1).unsqueeze(0),
            self.conv_weight.to(dtype).unsqueeze(1),
            bias=None if self.conv_bias is None else self.conv_bias.to(dtype),
            groups=gated_inputs.shape[1],
        )[0]
        segment_ends = token_starts[1:] + (rows + 1) * state_width
        conv_state[cache_indices] = (
            layout[segment_ends[:, None] - state_width + lanes]
            .transpose(1, 2)
            .to(conv_state.dtype)
        )
        return conv_outputs[:, token_positions - state_width].transpose(0, 1)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        conv_metadata = get_attn_backend().conv_state_metadata(
            self.layer_idx, forward_batch
        )
        conv_state = conv_metadata.layer_cache.conv[0]
        projected, _ = self.in_proj(hidden_states)
        b_gate, c_gate, x = projected.chunk(3, dim=-1)
        gated_inputs = b_gate * x
        if forward_batch.forward_mode.is_decode():
            conv_outputs = self.conv_decode(
                gated_inputs, conv_state, conv_metadata.cache_indices
            )
        else:
            conv_outputs = self.conv_prefill(
                gated_inputs,
                conv_state,
                conv_metadata.cache_indices,
                conv_metadata.query_start_loc,
                conv_metadata.has_initial_state,
            )
        output, _ = self.out_proj(c_gate * conv_outputs)
        return output
