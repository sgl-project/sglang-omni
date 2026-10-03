# SPDX-License-Identifier: Apache-2.0
"""Hops over a cached prefix reproduce the whole-history causal solve."""

from __future__ import annotations

import copy

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    gather_rows,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    compile_forward_prefix,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    PrefixCudaGraphCache,
    PrefixCudaGraphCaptureInputs,
    PrefixCudaGraphEnvelope,
)

pytestmark = pytest.mark.accelerator

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

CHUNK = 50
CHANNELS = 80
HEADS, HEAD_DIM, LAYERS = 4, 32, 3
COMPILED_OVER_EAGER_ERROR = 1.1


def make_estimator() -> PackedDiT:
    torch.manual_seed(0)
    dit = (
        cosyvoice_dit.DiT(
            dim=HEADS * HEAD_DIM,
            depth=LAYERS,
            heads=HEADS,
            dim_head=HEAD_DIM,
            ff_mult=2,
            mel_dim=CHANNELS,
            mu_dim=CHANNELS,
            spk_dim=CHANNELS,
            out_channels=CHANNELS,
            static_chunk_size=CHUNK,
            num_decoding_left_chunks=-1,
        )
        .cuda()
        .eval()
    )
    # Note (Jiaxin Deng): a visible conv bias makes zero context differ from
    # zero padding after the first conv, which the cached path must reproduce.
    with torch.no_grad():
        for conv in (
            dit.input_embed.conv_pos_embed.conv1[0],
            dit.input_embed.conv_pos_embed.conv2[0],
        ):
            conv.bias.fill_(0.5)
        for block in dit.transformer_blocks:
            block.attn_norm.linear.bias.fill_(0.5)
        dit.norm_out.linear.bias.fill_(0.5)
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    estimator = PackedDiT(dit, device="cuda")
    if not estimator.is_ragged:
        pytest.skip("requires FA3")
    else:
        pass
    return estimator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("totals", [(100, 200, 400), (70, 130, 260)])
def test_prefix_hops_are_bit_identical_to_whole_history_hops(
    totals: tuple[int, ...],
) -> None:
    """Every hop equals the whole-history hop, also hops that end inside a chunk."""
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(1)
    rows = 2
    noise = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    mu = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    cond = torch.zeros_like(mu)
    cond[:, :, :50] = torch.randn(rows, CHANNELS, 50, device=device, dtype=dtype)
    spks = torch.randn(rows, CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    caches = [(PrefixCacheRow(), PrefixCacheRow()) for _ in range(rows)]
    previous = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for total in totals:
            packed = pack_rows([total] * rows, device)
            reference = solve_flow_euler_packed(
                estimator,
                gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                time_span,
                gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                spks,
                gather_rows(cond[:, :, :total].transpose(1, 2), packed),
                packed,
                cfg_rate=0.7,
                streaming=True,
            )
            for pair in caches:
                assert grow_rows(pool, list(pair), [total, total])
            start = caches[0][0].committed_frames
            new = [total - start] * rows
            take = lambda x: torch.cat(
                [x[row, :, start:total].transpose(0, 1) for row in range(rows)]
            ).unsqueeze(0)
            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                take(noise),
                time_span,
                take(mu),
                spks,
                take(cond),
                new,
                caches,
                cfg_rate=0.7,
            )
            for row in range(rows):
                expected = reference[0, row * total + previous : (row + 1) * total]
                actual = cached[
                    0,
                    row * (total - start)
                    + previous
                    - start : (row + 1) * (total - start),
                ]
                assert torch.equal(actual, expected), (total, row)
            previous = total
    used = 32 - len(pool.free_blocks)
    assert used == rows * 2 * ((totals[-1] + BLOCK_FRAMES - 1) // BLOCK_FRAMES)
    for pair in caches:
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_prefix_hops_follow_each_row_across_batches() -> None:
    """Compiled hops over rows with their own prompts, prefixes and lengths,
    one joining fresh beside a cached one and the order flipped on a later hop,
    stay as close to each row's float32 whole-history solve as its bf16 eager
    solve is."""
    estimator = make_estimator()
    reference_estimator = PackedDiT(copy.deepcopy(estimator.dit).float(), device="cuda")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    pool.forward = compile_forward_prefix()
    torch.manual_seed(2)
    streams = {}
    for name, prompt_frames in (("a", 50), ("b", 80)):
        mel_conditioning = torch.zeros(CHANNELS, 400, device=device, dtype=dtype)
        mel_conditioning[:, :prompt_frames] = torch.randn(
            CHANNELS, prompt_frames, device=device, dtype=dtype
        )
        streams[name] = {
            "noise": torch.randn(CHANNELS, 400, device=device, dtype=dtype),
            "mu": torch.randn(CHANNELS, 400, device=device, dtype=dtype),
            "mel_conditioning": mel_conditioning,
            "speaker_embeddings": torch.randn(CHANNELS, device=device, dtype=dtype),
        }
    caches = {name: (PrefixCacheRow(), PrefixCacheRow()) for name in streams}
    emitted = {name: 0 for name in streams}
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    hops = [[("a", 70)], [("b", 130), ("a", 160)], [("a", 260), ("b", 210)]]

    def error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(actual.float() - expected)
            / torch.linalg.vector_norm(expected)
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for hop in hops:
            names = [name for name, _ in hop]
            totals = [total for _, total in hop]
            for name, total in hop:
                assert grow_rows(pool, list(caches[name]), [total, total])
            starts = [caches[name][0].committed_frames for name in names]

            def take(key: str) -> torch.Tensor:
                return torch.cat(
                    [
                        streams[name][key][:, start:total].transpose(0, 1)
                        for name, start, total in zip(names, starts, totals)
                    ]
                ).unsqueeze(0)

            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                take("noise"),
                time_span,
                take("mu"),
                torch.stack([streams[name]["speaker_embeddings"] for name in names]),
                take("mel_conditioning"),
                [total - start for start, total in zip(starts, totals)],
                [caches[name] for name in names],
                cfg_rate=0.7,
            )
            offset = 0
            for name, start, total in zip(names, starts, totals):
                stream = streams[name]
                packed = pack_rows([total], device)

                def whole_history(
                    model: PackedDiT, value_dtype: torch.dtype
                ) -> torch.Tensor:
                    def rows_of(key: str) -> torch.Tensor:
                        return gather_rows(
                            stream[key][None, :, :total].transpose(1, 2), packed
                        ).to(value_dtype)

                    return solve_flow_euler_packed(
                        model,
                        rows_of("noise"),
                        time_span.to(value_dtype),
                        rows_of("mu"),
                        stream["speaker_embeddings"][None].to(value_dtype),
                        rows_of("mel_conditioning"),
                        packed,
                        cfg_rate=0.7,
                        streaming=True,
                    )[0, emitted[name] : total]

                eager = whole_history(estimator, dtype)
                with torch.autocast("cuda", enabled=False):
                    float32 = whole_history(reference_estimator, torch.float32)
                actual = cached[
                    0, offset + emitted[name] - start : offset + total - start
                ]
                assert error(actual, float32) <= COMPILED_OVER_EAGER_ERROR * error(
                    eager, float32
                ), (name, total)
                emitted[name] = total
                offset += total - start
    for pair in caches.values():
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


def test_grow_rows_takes_nothing_on_a_shortfall() -> None:
    pool = PrefixKVPool.__new__(PrefixKVPool)
    pool.free_blocks = [0, 1, 2]
    rows = [PrefixCacheRow(), PrefixCacheRow()]
    assert not grow_rows(pool, rows, [BLOCK_FRAMES * 2, BLOCK_FRAMES * 2])
    assert pool.free_blocks == [0, 1, 2] and rows[0].blocks == []
    assert grow_rows(pool, rows, [BLOCK_FRAMES, BLOCK_FRAMES * 2])
    assert rows[0].allocated_frames == BLOCK_FRAMES
    assert rows[1].allocated_frames == BLOCK_FRAMES * 2
    assert pool.free_blocks == []
    release_rows(pool, rows)
    assert sorted(pool.free_blocks) == [0, 1, 2] and rows[0].committed_frames == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_cuda_graph_matches_compiled_prefix_solver() -> None:
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    assert estimator.compile(dtype)
    graph_pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    reference_pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    graph_pool.forward = compile_forward_prefix()
    reference_pool.forward = compile_forward_prefix()
    envelope = PrefixCudaGraphEnvelope("B1-N100-M100-E128", 1, 100, 100, 128, (100,))
    noise_template = torch.randn(1, CHANNELS, 100, device=device, dtype=dtype)
    mu = torch.randn(1, 100, CHANNELS, device=device, dtype=dtype)
    mel_conditioning = torch.randn_like(mu)
    speaker_embeddings = torch.randn(1, CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    graph_cache = PrefixCudaGraphCache(
        estimator,
        graph_pool,
        device=device,
        autocast_dtype=dtype,
        cfg_rate=0.7,
        envelopes=(envelope,),
        max_slack_frames=100,
        capture_warmup_iterations=1,
    )

    def capture_inputs(
        new_frame_counts: tuple[int, ...],
    ) -> PrefixCudaGraphCaptureInputs:
        total_new_frame_count = sum(new_frame_counts)
        return (
            noise_template[:, :, :total_new_frame_count].transpose(1, 2).contiguous(),
            time_span.clone(),
            mu[:, :total_new_frame_count],
            speaker_embeddings,
            mel_conditioning[:, :total_new_frame_count],
        )

    graph_pair = (PrefixCacheRow(), PrefixCacheRow())
    reference_pair = (PrefixCacheRow(), PrefixCacheRow())

    def assert_cached_rows_match(
        graph_rows: tuple[PrefixCacheRow, PrefixCacheRow],
        reference_rows: tuple[PrefixCacheRow, PrefixCacheRow],
        committed_frames: int,
    ) -> None:
        for graph_row, reference_row in zip(graph_rows, reference_rows, strict=True):
            assert graph_row.committed_frames == reference_row.committed_frames
            assert graph_row.conv_context is not None
            assert reference_row.conv_context is not None
            assert torch.equal(graph_row.conv_context, reference_row.conv_context)
            graph_pages = graph_row.pages(device)[:committed_frames]
            reference_pages = reference_row.pages(device)[:committed_frames]
            for euler_step in range(10):
                for layer in range(LAYERS):
                    assert torch.equal(
                        graph_pool.keys[euler_step][layer][graph_pages],
                        reference_pool.keys[euler_step][layer][reference_pages],
                    )
                    assert torch.equal(
                        graph_pool.values[euler_step][layer][graph_pages],
                        reference_pool.values[euler_step][layer][reference_pages],
                    )

    try:
        graph_cache.capture(capture_inputs)
        assert graph_cache.capture_stats().graph_count == 1
        assert graph_cache.capture_stats().scratch_reserved_frames == 256
        assert grow_rows(graph_pool, list(graph_pair), [50, 50])
        assert grow_rows(reference_pool, list(reference_pair), [50, 50])
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            first_graph = graph_cache.run(
                noise=noise_template[:, :, :50].transpose(1, 2).contiguous(),
                time_span=time_span,
                mu=mu[:, :50],
                speaker_embeddings=speaker_embeddings,
                mel_conditioning=mel_conditioning[:, :50],
                new_frames=[50],
                total_frames=[50],
                caches=[graph_pair],
            )
            first_reference = solve_flow_euler_prefix(
                estimator,
                reference_pool,
                noise_template[:, :, :50].transpose(1, 2).contiguous(),
                time_span,
                mu[:, :50],
                speaker_embeddings,
                mel_conditioning[:, :50],
                [50],
                [reference_pair],
                cfg_rate=0.7,
            )
            assert first_graph is not None
            assert torch.equal(first_graph, first_reference)
            assert_cached_rows_match(graph_pair, reference_pair, 50)

            assert grow_rows(graph_pool, list(graph_pair), [100, 100])
            assert grow_rows(reference_pool, list(reference_pair), [100, 100])
            second_graph = graph_cache.run(
                noise=noise_template[:, :, 50:100].transpose(1, 2).contiguous(),
                time_span=time_span,
                mu=mu[:, 50:100],
                speaker_embeddings=speaker_embeddings,
                mel_conditioning=mel_conditioning[:, 50:100],
                new_frames=[50],
                total_frames=[100],
                caches=[graph_pair],
            )
            second_reference = solve_flow_euler_prefix(
                estimator,
                reference_pool,
                noise_template[:, :, 50:100].transpose(1, 2).contiguous(),
                time_span,
                mu[:, 50:100],
                speaker_embeddings,
                mel_conditioning[:, 50:100],
                [50],
                [reference_pair],
                cfg_rate=0.7,
            )
            assert second_graph is not None
            assert torch.equal(second_graph, second_reference)
            assert_cached_rows_match(graph_pair, reference_pair, 100)

            # Reusing fresh rows makes the N=100 call exercise zero slack.
            release_rows(graph_pool, list(graph_pair))
            release_rows(reference_pool, list(reference_pair))
            assert grow_rows(graph_pool, list(graph_pair), [100, 100])
            assert grow_rows(reference_pool, list(reference_pair), [100, 100])
            third_graph = graph_cache.run(
                noise=noise_template.transpose(1, 2).contiguous(),
                time_span=time_span,
                mu=mu,
                speaker_embeddings=speaker_embeddings,
                mel_conditioning=mel_conditioning,
                new_frames=[100],
                total_frames=[100],
                caches=[graph_pair],
            )
            third_reference = solve_flow_euler_prefix(
                estimator,
                reference_pool,
                noise_template.transpose(1, 2).contiguous(),
                time_span,
                mu,
                speaker_embeddings,
                mel_conditioning,
                [100],
                [reference_pair],
                cfg_rate=0.7,
            )
            assert third_graph is not None
            assert torch.equal(third_graph, third_reference)
            assert_cached_rows_match(graph_pair, reference_pair, 100)

        assert graph_cache.scratch_pair is not None
        assert graph_cache.scratch_pair[0].committed_frames == 0
        assert graph_cache.scratch_pair[1].committed_frames == 0
        assert graph_cache.scratch_pair[0].conv_context is None
        assert graph_cache.scratch_pair[1].conv_context is None
    finally:
        release_rows(graph_pool, list(graph_pair))
        release_rows(reference_pool, list(reference_pair))
        graph_cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_cuda_graph_matches_compiled_solver_for_b2_physical_reuse() -> None:
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    assert estimator.compile(dtype)
    graph_pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    reference_pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    graph_pool.forward = compile_forward_prefix()
    reference_pool.forward = compile_forward_prefix()
    envelope = PrefixCudaGraphEnvelope(
        "B2-N200-M100-E128", 2, 200, 100, 128, (100, 100)
    )
    noise_template = torch.randn(1, CHANNELS, 200, device=device, dtype=dtype)
    mu = torch.randn(1, 200, CHANNELS, device=device, dtype=dtype)
    mel_conditioning = torch.randn_like(mu)
    speaker_embeddings = torch.randn(2, CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    graph_cache = PrefixCudaGraphCache(
        estimator,
        graph_pool,
        device=device,
        autocast_dtype=dtype,
        cfg_rate=0.7,
        envelopes=(envelope,),
        max_slack_frames=100,
        capture_warmup_iterations=1,
    )

    def capture_inputs(
        new_frame_counts: tuple[int, ...],
    ) -> PrefixCudaGraphCaptureInputs:
        total_new_frame_count = sum(new_frame_counts)
        return (
            noise_template[:, :, :total_new_frame_count].transpose(1, 2).contiguous(),
            time_span.clone(),
            mu[:, :total_new_frame_count],
            speaker_embeddings,
            mel_conditioning[:, :total_new_frame_count],
        )

    graph_pairs = [
        (PrefixCacheRow(), PrefixCacheRow()),
        (PrefixCacheRow(), PrefixCacheRow()),
    ]
    reference_pairs = [
        (PrefixCacheRow(), PrefixCacheRow()),
        (PrefixCacheRow(), PrefixCacheRow()),
    ]

    def assert_cached_rows_match(
        graph_rows: list[tuple[PrefixCacheRow, PrefixCacheRow]],
        reference_rows: list[tuple[PrefixCacheRow, PrefixCacheRow]],
        committed_frame_counts: tuple[int, ...],
    ) -> None:
        for (
            graph_pair,
            reference_pair,
            committed_frame_count,
        ) in zip(graph_rows, reference_rows, committed_frame_counts, strict=True):
            for graph_row, reference_row in zip(
                graph_pair, reference_pair, strict=True
            ):
                assert graph_row.committed_frames == reference_row.committed_frames
                assert graph_row.conv_context is not None
                assert reference_row.conv_context is not None
                assert torch.equal(graph_row.conv_context, reference_row.conv_context)
                graph_pages = graph_row.pages(device)[:committed_frame_count]
                reference_pages = reference_row.pages(device)[:committed_frame_count]
                for euler_step in range(10):
                    for layer in range(LAYERS):
                        assert torch.equal(
                            graph_pool.keys[euler_step][layer][graph_pages],
                            reference_pool.keys[euler_step][layer][reference_pages],
                        )
                        assert torch.equal(
                            graph_pool.values[euler_step][layer][graph_pages],
                            reference_pool.values[euler_step][layer][reference_pages],
                        )

    try:
        graph_cache.capture(capture_inputs)
        assert graph_cache.capture_stats().graph_count == 1
        for new_frame_counts_tuple in ((50, 50), (100, 50), (100, 100)):
            new_frame_counts = list(new_frame_counts_tuple)
            total_new_frame_count = sum(new_frame_counts)
            graph_rows = [row for pair in graph_pairs for row in pair]
            reference_rows = [row for pair in reference_pairs for row in pair]
            allocated_frames = [
                frame_count for frame_count in new_frame_counts for _ in range(2)
            ]
            assert grow_rows(graph_pool, graph_rows, allocated_frames)
            assert grow_rows(reference_pool, reference_rows, allocated_frames)
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                graph_output = graph_cache.run(
                    noise=noise_template[:, :, :total_new_frame_count]
                    .transpose(1, 2)
                    .contiguous(),
                    time_span=time_span,
                    mu=mu[:, :total_new_frame_count],
                    speaker_embeddings=speaker_embeddings,
                    mel_conditioning=mel_conditioning[:, :total_new_frame_count],
                    new_frames=new_frame_counts,
                    total_frames=new_frame_counts,
                    caches=graph_pairs,
                )
                reference_output = solve_flow_euler_prefix(
                    estimator,
                    reference_pool,
                    noise_template[:, :, :total_new_frame_count]
                    .transpose(1, 2)
                    .contiguous(),
                    time_span,
                    mu[:, :total_new_frame_count],
                    speaker_embeddings,
                    mel_conditioning[:, :total_new_frame_count],
                    new_frame_counts,
                    reference_pairs,
                    cfg_rate=0.7,
                )
            assert graph_output is not None
            assert torch.equal(graph_output, reference_output)
            assert_cached_rows_match(
                graph_pairs,
                reference_pairs,
                tuple(new_frame_counts),
            )
            assert graph_cache.scratch_pair is not None
            assert graph_cache.scratch_pair[0].committed_frames == 0
            assert graph_cache.scratch_pair[1].committed_frames == 0
            assert graph_cache.scratch_pair[0].conv_context is None
            assert graph_cache.scratch_pair[1].conv_context is None
            release_rows(graph_pool, graph_rows)
            release_rows(reference_pool, reference_rows)
    finally:
        release_rows(
            graph_pool,
            [row for pair in graph_pairs for row in pair],
        )
        release_rows(
            reference_pool,
            [row for pair in reference_pairs for row in pair],
        )
        graph_cache.close()
