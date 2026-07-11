# SPDX-License-Identifier: Apache-2.0
"""S0 regression gate: MOSS-TTS Local v1.5 frame-decode determinism.

Reproducer for this PR's bit-identity claim (the S0 gate cited in the
Verification section), runnable by anyone via pytest; the upcoming #734 and #736
changes reuse the same check. The frame-decode CUDA graph, replayed twice with
identical fixed-seed inputs, must produce bit-identical output. Marked ``gpu``
and auto-skipped without a CUDA device. Do not modify after initial commit.
"""
from __future__ import annotations

import pytest
import torch

_N_VQ = 12


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_s0_graph_replay_is_deterministic():
    """Two CUDA-graph replays with identical inputs must be bit-identical.

    Exercises the same decode-frame kernel as the production v1.5 pipeline
    (MossTTSLocalTransformer + sample_seeded_branchless loop).
    """
    from sglang_omni.models.moss_tts_local.local_transformer import (
        MossTTSLocalTransformer,
        sample_seeded_branchless,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=1,
        max_positions=_N_VQ + 1,
        rope_base=1_000_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    tables = [
        torch.randn(64, 64, device=device, dtype=torch.bfloat16) for _ in range(_N_VQ)
    ]

    def decode_frame(
        hidden: torch.Tensor,
        seeds: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        current = module.step(hidden, 0)
        codes = []
        for channel in range(_N_VQ):
            logits = (current.float() @ tables[channel].float().T)[:, :32]
            code = sample_seeded_branchless(
                logits,
                temperature=torch.full((hidden.shape[0],), 1.0, device=device),
                top_p=torch.full((hidden.shape[0],), 1.0, device=device),
                top_k=torch.full(
                    (hidden.shape[0],), 32, device=device, dtype=torch.long
                ),
                seeds=seeds,
                positions=base + channel + 1,
            )
            codes.append(code)
            if channel + 1 < _N_VQ:
                embed = torch.nn.functional.embedding(code, tables[channel][:32])
                current = module.step(embed.to(torch.bfloat16), channel + 1)
        return torch.stack(codes, dim=-1)

    batch = 4
    static_hidden = torch.zeros(batch, 64, device=device, dtype=torch.bfloat16)
    static_seeds = torch.zeros(batch, device=device, dtype=torch.long)
    static_base = torch.zeros(batch, device=device, dtype=torch.long)

    # Warmup two passes before capture (required for CUDA graph stability).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            decode_frame(static_hidden, static_seeds, static_base)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_codes = decode_frame(static_hidden, static_seeds, static_base)

    hidden = torch.randn(batch, 64, device=device, dtype=torch.bfloat16)
    seeds = torch.arange(batch, device=device, dtype=torch.long) * 1_234_567
    base = torch.full((batch,), 26, device=device, dtype=torch.long)

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    run1 = graphed_codes.clone()

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    run2 = graphed_codes.clone()

    assert torch.equal(run1, run2), (
        "CUDA graph replay with identical inputs must be bit-identical; "
        "any divergence indicates non-determinism in the decode kernel"
    )
