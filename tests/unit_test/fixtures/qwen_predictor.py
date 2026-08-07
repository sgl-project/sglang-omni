# SPDX-License-Identifier: Apache-2.0
"""Toy Qwen3-Omni talker/predictor builders shared by predictor unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker


class TupleLinear(nn.Module):
    """Linear that returns (out, None) like SGLang parallel linear layers."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.proj(hidden_states), None


class IdentityRotary(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ):
        del positions, fused_set_kv_buffer_arg
        return q, k


def build_real_step_predictor_graph_talker(
    device: torch.device,
    num_heads: int = 2,
    num_kv_heads: int = 1,
) -> Qwen3OmniTalker:
    """Minimal talker whose predictor step runs the real attention/KV-cache code."""
    torch.manual_seed(1)
    hidden_size = 8
    head_dim = 4
    num_code_groups = 4
    vocab_size = 16
    max_batch_size = 4
    predictor_len = num_code_groups + 1

    talker = object.__new__(Qwen3OmniTalker)
    talker.training = False
    talker.config = SimpleNamespace(num_code_groups=num_code_groups)
    talker._predictor_input_buffer = torch.zeros(
        max_batch_size,
        predictor_len,
        hidden_size,
        device=device,
    )
    talker._output_codes = torch.zeros(
        max_batch_size,
        num_code_groups,
        dtype=torch.long,
        device=device,
    )
    talker._output_embeds = torch.zeros(max_batch_size, hidden_size, device=device)
    talker._predictor_positions = torch.arange(
        predictor_len,
        device=device,
        dtype=torch.long,
    )
    talker._predictor_k_cache = torch.zeros(
        1,
        max_batch_size,
        num_kv_heads,
        predictor_len,
        head_dim,
        device=device,
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    talker._predictor_decode_graph_batch_sizes = (1, 2, 4)
    talker._predictor_decode_graphs = {}
    talker._predictor_decode_graph_disabled = set()

    layer = SimpleNamespace(
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        mlp=nn.Linear(hidden_size, hidden_size, bias=False).to(device),
    )
    layer.self_attn = SimpleNamespace(
        q_size=num_heads * head_dim,
        kv_size=num_kv_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
        alt_stream=None,
        qkv_proj=TupleLinear(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim).to(
            device
        ),
        o_proj=TupleLinear(num_heads * head_dim, hidden_size).to(device),
        rotary_emb=IdentityRotary(),
    )
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[layer],
            norm=nn.Identity(),
            codec_embedding=nn.ModuleList(
                [nn.Embedding(vocab_size, hidden_size).to(device) for _ in range(3)]
            ),
        ),
        lm_head=nn.ModuleList(
            [TupleLinear(hidden_size, vocab_size).to(device) for _ in range(3)]
        ),
    )
    layer0_embedding = nn.Embedding(vocab_size, hidden_size).to(device)
    talker.get_input_embeddings = lambda: layer0_embedding
    return talker
