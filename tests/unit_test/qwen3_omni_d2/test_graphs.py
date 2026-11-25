# SPDX-License-Identifier: Apache-2.0
"""Captured D2 decoder replays equal eager forwards at every cache cursor."""

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
)

from sglang_omni.models.qwen3_omni_d2.graphs import (
    CapturedDecoder,
    causal_cache_mask,
    reset_cache,
    static_cache,
)

QUERY_TOKENS = 3
CHUNKS = 4


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@torch.no_grad()
def test_replays_at_advancing_cursors_match_eager_chunks() -> None:
    """The cache advances its own cursor inside the graph, so a stale cursor would shift every later chunk."""
    torch.manual_seed(0)
    config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e6,
            "mrope_section": [2, 3, 3],
            "interleaved": True,
        },
        attn_implementation="sdpa",
        experts_implementation="grouped_mm",
    )
    decoder = Qwen3OmniMoeThinkerTextModel(config).cuda().bfloat16().eval()
    capacity_tokens = QUERY_TOKENS * CHUNKS + 2
    captured = CapturedDecoder(
        decoder,
        static_cache(decoder, capacity_tokens),
        (4, 1, QUERY_TOKENS),
        capture_cursor=0,
        hooked_layer=decoder.layers[0],
    )
    reset_cache(captured.cache)
    eager_cache = static_cache(decoder, capacity_tokens)
    hooked: list[torch.Tensor] = []
    decoder.layers[0].register_forward_hook(
        lambda module, inputs, output: hooked.append(output.clone())
    )
    for chunk in range(CHUNKS):
        cursor = chunk * QUERY_TOKENS
        inputs = torch.randn(1, QUERY_TOKENS, 64, device="cuda", dtype=torch.bfloat16)
        token_order = torch.arange(cursor, cursor + QUERY_TOKENS, device="cuda")
        positions = torch.stack(
            (token_order, token_order // 3, token_order // 3, token_order // 3)
        )
        graph_output = captured.replay(inputs, positions[:, None], cursor).clone()
        graph_layer = captured.hooked_states.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager_output = decoder(
                inputs_embeds=inputs,
                attention_mask=causal_cache_mask(
                    cursor, QUERY_TOKENS, capacity_tokens, inputs.device
                ),
                position_ids=positions[:, None],
                past_key_values=eager_cache,
                use_cache=True,
            ).last_hidden_state
        torch.testing.assert_close(graph_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(graph_layer, hooked[-1], rtol=1e-2, atol=1e-2)
