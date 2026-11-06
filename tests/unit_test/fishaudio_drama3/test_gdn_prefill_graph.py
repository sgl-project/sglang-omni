# SPDX-License-Identifier: Apache-2.0
"""The GDN extend captured on static prefill-graph tables replays the eager
chain's bits for every batch layout a bucket serves: at the kernel level, and
through the engine for mixed batches and the decodes that follow them."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

from sglang_omni.models.fishaudio_drama3.gdn_prefill_graph import (
    GDNPrefillGraphMetadata,
    gdn_extend_in_graph,
)
from tests.unit_test.fishaudio_drama3.graph_checks import (
    EnginePair,
    bit_problems,
    finite_problems,
    prefill_path_problems,
)
from tests.unit_test.fishaudio_drama3.graph_fixtures import (
    GraphEngineSpec,
    GraphRequestSpec,
    GraphScenario,
    bits,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    PROMPTS,
    prompt_rows,
    write_synthetic_checkpoint,
)
from tests.unit_test.fishaudio_drama3.test_graph_engine import run_engine

accelerator = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the GDN chain runs on CUDA only"
)
KEY_HEADS = 2
VALUE_HEADS = 4
HEAD_DIM = 128
CONV_WIDTH = 4
SLOT_COUNT = 12
SENTINEL = 7.0
# (bucket, sequence lengths, state slots): a two-prompt batch under the bucket,
# three prompts of which one ends mid-chunk, and a lone 32-token prompt whose
# eager chunk_fwd_o tiles at 32 rows while the 64-token bucket tiles at 64.
LAYOUTS = [
    (256, [97, 126], [3, 7]),
    (256, [60, 100, 63], [9, 1, 4]),
    (64, [32], [5]),
]


@dataclasses.dataclass
class Batch:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    lens: list[int]
    slots: list[int]


def make_layer(device: torch.device) -> SimpleNamespace:
    generator = torch.Generator(device="cpu").manual_seed(3)
    qkv_dim = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM
    return SimpleNamespace(
        layer_id=0,
        num_q_heads=KEY_HEADS,
        num_k_heads=KEY_HEADS,
        num_v_heads=VALUE_HEADS,
        head_q_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        q_dim=KEY_HEADS * HEAD_DIM,
        k_dim=KEY_HEADS * HEAD_DIM,
        v_dim=VALUE_HEADS * HEAD_DIM,
        conv_weights=(0.2 * torch.randn(qkv_dim, CONV_WIDTH, generator=generator)).to(
            device=device, dtype=torch.bfloat16
        ),
        bias=None,
        activation="silu",
        A_log=torch.randn(VALUE_HEADS, generator=generator).to(device),
        dt_bias=torch.randn(VALUE_HEADS, generator=generator).to(device),
    )


def make_batch(
    seed: int, lens: list[int], slots: list[int], device: torch.device
) -> Batch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_count = sum(lens)
    qkv_dim = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM
    return Batch(
        mixed_qkv=torch.randn(token_count, qkv_dim, generator=generator).to(
            device=device, dtype=torch.bfloat16
        ),
        a=torch.randn(token_count, VALUE_HEADS, generator=generator).to(
            device=device, dtype=torch.bfloat16
        ),
        b=torch.randn(token_count, VALUE_HEADS, generator=generator).to(
            device=device, dtype=torch.bfloat16
        ),
        lens=lens,
        slots=slots,
    )


def fresh_pools(
    slots: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sentinel everywhere but the batch's slots, which a fresh prompt reads as zero."""
    qkv_dim = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM
    conv_states = torch.full(
        (SLOT_COUNT, qkv_dim, CONV_WIDTH - 1),
        SENTINEL,
        dtype=torch.bfloat16,
        device=device,
    )
    ssm_states = torch.full(
        (SLOT_COUNT, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
        SENTINEL,
        dtype=torch.float32,
        device=device,
    )
    conv_states[slots] = 0
    ssm_states[slots] = 0
    return conv_states, ssm_states


def eager_extend(
    layer: SimpleNamespace,
    batch: Batch,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
) -> torch.Tensor:
    """GDNAttnBackend.forward_extend's chain for a batch of fresh prompts."""
    device = batch.mixed_qkv.device
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(batch.lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.tensor(batch.slots, dtype=torch.int32, device=device)
    convolved = causal_conv1d_fn(
        batch.mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        conv_states=conv_states,
        has_initial_state=torch.zeros(len(batch.lens), dtype=torch.bool, device=device),
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
        seq_lens_cpu=batch.lens,
    ).transpose(0, 1)[: batch.mixed_qkv.shape[0]]
    query, key, value = fused_qkv_split_gdn_prefill(
        convolved,
        layer.num_q_heads,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_q_dim,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    g, beta = fused_gdn_gating(layer.A_log, batch.a, batch.b, layer.dt_bias)
    output, _, _ = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        initial_state_indices=cache_indices,
        inplace_update=True,
    )
    return output


@accelerator
@pytest.mark.accelerator
def test_each_layout_replayed_through_one_bucket_matches_the_eager_bits() -> None:
    """One graph per bucket, replayed for every layout after a refill: the real
    rows and their state slots carry the eager bits, and no other slot changes."""
    device = torch.device("cuda")
    layer = make_layer(device)
    graphs: dict[
        int,
        tuple[
            torch.cuda.CUDAGraph,
            Batch,
            torch.Tensor,
            GDNPrefillGraphMetadata,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}
    with torch.no_grad():
        for seed, (bucket, lens, slots) in enumerate(LAYOUTS, start=100):
            batch = make_batch(seed, lens, slots, device)
            eager_conv, eager_ssm = fresh_pools(slots, device)
            eager = eager_extend(layer, batch, eager_conv, eager_ssm)
            if bucket not in graphs:
                qkv_dim = batch.mixed_qkv.shape[1]
                static = Batch(
                    mixed_qkv=torch.zeros(
                        bucket, qkv_dim, dtype=torch.bfloat16, device=device
                    ),
                    a=torch.zeros(
                        bucket, VALUE_HEADS, dtype=torch.bfloat16, device=device
                    ),
                    b=torch.zeros(
                        bucket, VALUE_HEADS, dtype=torch.bfloat16, device=device
                    ),
                    lens=[],
                    slots=[],
                )
                meta = GDNPrefillGraphMetadata.allocate(
                    num_tokens=bucket,
                    max_seqs_cap=8,
                    cache_index_dtype=torch.int32,
                    device=device,
                )
                conv_states, ssm_states = fresh_pools([], device)
                # The capture batch fills the bucket with one prompt, as the runner's does.
                meta.fill(
                    extend_seq_lens=[bucket],
                    cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
                    extend_prefix_lens=torch.zeros(1, dtype=torch.int64, device=device),
                )
                run = lambda: gdn_extend_in_graph(  # noqa: E731
                    layer=layer,
                    meta=meta,
                    conv_states=conv_states,
                    ssm_states=ssm_states,
                    mixed_qkv=static.mixed_qkv,
                    a=static.a,
                    b=static.b,
                )
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                graphs[bucket] = (
                    graph,
                    static,
                    captured,
                    meta,
                    conv_states,
                    ssm_states,
                )
            graph, static, captured, meta, conv_states, ssm_states = graphs[bucket]
            token_count = batch.mixed_qkv.shape[0]
            static.mixed_qkv.zero_()
            static.mixed_qkv[:token_count].copy_(batch.mixed_qkv)
            static.a[:token_count].copy_(batch.a)
            static.b[:token_count].copy_(batch.b)
            conv_states.fill_(SENTINEL)
            ssm_states.fill_(SENTINEL)
            conv_states[slots] = 0
            ssm_states[slots] = 0
            meta.fill(
                extend_seq_lens=lens,
                cache_indices=torch.tensor(slots, dtype=torch.int32, device=device),
                extend_prefix_lens=torch.zeros(
                    len(lens), dtype=torch.int64, device=device
                ),
            )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(
                bits(captured[:, :token_count]), bits(eager[:, :token_count])
            ), f"bucket {bucket} layout {lens}: output rows differ from eager"
            assert torch.all(captured[:, token_count:] == 0)
            assert torch.equal(bits(conv_states[slots]), bits(eager_conv[slots]))
            assert torch.equal(bits(ssm_states[slots]), bits(eager_ssm[slots]))
            untouched = torch.ones(SLOT_COUNT, dtype=torch.bool, device=device)
            untouched[slots] = False
            assert torch.all(conv_states[untouched] == SENTINEL)
            assert torch.all(ssm_states[untouched] == SENTINEL)


# Prompt lengths the stock set lacks: 64 fills a bucket on its own, 96 and 100
# cross the 64-token chunk boundary, 28 completes the 128 bucket after 100.
EXACT_PROMPT_LENGTHS = {"q28": 28, "q64": 64, "q96": 96, "q100": 100}
# Waves of one to five prompts whose token totals fill the 64, 128 and 192
# buckets exactly (64; 100 + 28; 2 + 23 + 31 + 40 + 96) or land mid-bucket
# (5 + 9 + 13 = 27 in the 32 bucket, 60 + 9 = 69 in the 128 bucket).
EXACT_WAVES = [
    ["q64"],
    ["q100", "q28"],
    ["p5", "p9", "p13"],
    ["p2", "p23", "p31", "p40", "q96"],
    ["p60", "p9"],
]
EXACT_BUCKETS = {32, 64, 128, 192}
DECODE_FRAMES_AFTER_PREFILL = 4


@accelerator
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_mixed_batches_through_the_captured_extend_keep_the_eager_bits(
    tmp_path: Path,
) -> None:
    """Waves of one to five prompts prefilled on the captured GDN extend and
    then decoded: every linear-attention state digest, final conv and recurrent
    state, written KV cache, logit and frame equals the eager engine's."""
    checkpoint_dir = tmp_path / "checkpoint"
    write_synthetic_checkpoint(checkpoint_dir)
    prompts = {
        **PROMPTS,
        **{
            name: prompt_rows(length, seed=1000 + index)
            for index, (name, length) in enumerate(EXACT_PROMPT_LENGTHS.items())
        },
    }
    scenario = GraphScenario(
        name="exact_buckets",
        sampling_mode="greedy",
        waves=[
            [
                GraphRequestSpec(
                    prompt_name=name, max_new_tokens=DECODE_FRAMES_AFTER_PREFILL
                )
                for name in wave
            ]
            for wave in EXACT_WAVES
        ],
    )
    runs = {
        kind: run_engine(
            f"exact-{kind}",
            GraphEngineSpec(
                model_dir=str(checkpoint_dir),
                kind=kind,
                prompts=prompts,
                scenarios=[scenario],
            ),
            tmp_path,
            {},
        )
        for kind in ("eager", "graph", "prefill_graph")
    }
    prefill_graph = runs["prefill_graph"]
    assert prefill_graph.facts.gdn_prefill_graph_status.startswith(
        "on"
    ), prefill_graph.facts.gdn_prefill_graph_status
    problems = prefill_path_problems(prefill_graph)
    assert not problems, "\n".join(problems)
    replayed = {
        bucket
        for record in prefill_graph.forwards
        if record.is_extend
        for bucket in record.prefill_replay_buckets
    }
    assert replayed == EXACT_BUCKETS
    problems = finite_problems(prefill_graph)
    assert not problems, "\n".join(problems)
    # The decode-graph engine differs from the prefill-graph engine only in the
    # prefill path, so every recorded value must agree.
    problems = bit_problems(
        EnginePair(eager=runs["graph"], graph=prefill_graph, scenarios=[scenario])
    )
    assert not problems, "\n".join(problems)
    # Against the eager engine, the staged RAS window is the one recorded value
    # eager and graph engines already disagree on without prefill graphs (see
    # test_graph_engine_matches_the_eager_engine_bit_for_bit on the eager/graph
    # pair); states, caches, logits and frames must still match bit for bit.
    problems = [
        problem
        for problem in bit_problems(
            EnginePair(eager=runs["eager"], graph=prefill_graph, scenarios=[scenario])
        )
        if "staged RAS window" not in problem
    ]
    assert not problems, "\n".join(problems)
