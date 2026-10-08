# SPDX-License-Identifier: Apache-2.0
"""Hops over a cached prefix reproduce the whole-history causal solve, and
graph replayed hops and finals their eager solves."""

from __future__ import annotations

import copy

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.causal_conv import FusedConvPositionEmbedding
from sglang_omni.models.fun_cosyvoice3.final_cuda_graph import FinalCudaGraphRunner
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
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.platforms import current_platform

pytestmark = pytest.mark.accelerator

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

CHUNK = 50
CHANNELS = 80
HEADS, HEAD_DIM, LAYERS = 4, 64, 3
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
    dit.input_embed.conv_pos_embed = FusedConvPositionEmbedding(
        dit.input_embed.conv_pos_embed
    )
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
            twin_rows = pack_rows([total] * 2 * rows, device)
            reference = solve_flow_euler_packed(
                estimator,
                gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                time_span,
                gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                spks,
                gather_rows(cond[:, :, :total].transpose(1, 2), packed),
                twin_rows,
                estimator.row_attention(twin_rows, streaming=True, dtype=dtype),
                estimator.rope_angles(total),
                cfg_rate=0.7,
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
                twin_rows = pack_rows([total, total], device)

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
                        twin_rows,
                        model.row_attention(
                            twin_rows, streaming=True, dtype=value_dtype
                        ),
                        model.rope_angles(total),
                        cfg_rate=0.7,
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
@pytest.mark.parametrize("compile_prefix", [False, True], ids=["eager", "compiled"])
def test_prefix_graph_replays_equal_the_eager_solve(compile_prefix: bool) -> None:
    """Replays with padding frames and unused row slots equal the eager solve,
    also when they resume from a step above the largest tier that ran eagerly."""
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    pools = [
        PrefixKVPool(
            layer_num=LAYERS,
            euler_steps=10,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            capacity_frames=64 * BLOCK_FRAMES,
            device=device,
            dtype=dtype,
        )
        for _ in range(2)
    ]
    graph_pool, eager_pool = pools
    if compile_prefix:
        assert estimator.compile(dtype)
        for pool in pools:
            pool.forward = compile_forward_prefix()
    else:
        pass
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    runner = PrefixCudaGraphRunner(
        estimator,
        graph_pool,
        backend=backend,
        autocast_dtype=dtype,
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=4,
        hop_frames=2 * CHUNK,
        min_hop_frames=CHUNK,
        max_frames=1024,
    )
    runner.capture()
    torch.manual_seed(3)
    streams = {}
    for name, prompt_frames in (("a", 100), ("b", 50), ("c", 50)):
        mel_conditioning = torch.zeros(CHANNELS, 650, device=device, dtype=dtype)
        mel_conditioning[:, :prompt_frames] = torch.randn(
            CHANNELS, prompt_frames, device=device, dtype=dtype
        )
        streams[name] = {
            "noise": torch.randn(CHANNELS, 650, device=device, dtype=dtype),
            "mu": torch.randn(CHANNELS, 650, device=device, dtype=dtype),
            "mel_conditioning": mel_conditioning,
            "speaker_embeddings": torch.randn(CHANNELS, device=device, dtype=dtype),
        }
    caches = {
        name: [(PrefixCacheRow(), PrefixCacheRow()) for _ in pools] for name in streams
    }
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    steps = [
        [("a", 150)],
        [("b", 100), ("a", 250)],
        [("a", 450), ("c", 70), ("b", 200)],
        [("c", 150), ("b", 400)],
        [("a", 600), ("b", 600), ("c", 300)],
        [("a", 650)],
    ]
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for step in steps:
            names = [name for name, _ in step]
            totals = [total for _, total in step]
            starts = [caches[name][0][0].committed_frames for name in names]
            new_frames = [total - start for start, total in zip(starts, totals)]

            def take(key: str) -> torch.Tensor:
                return (
                    torch.cat(
                        [
                            streams[name][key][:, start:total].transpose(0, 1)
                            for name, start, total in zip(names, starts, totals)
                        ]
                    )
                    .unsqueeze(0)
                    .float()
                )

            outputs = []
            for pool_index, pool in enumerate(pools):
                pairs = [caches[name][pool_index] for name in names]
                for pair, total in zip(pairs, totals):
                    assert grow_rows(pool, list(pair), [total, total])
                inputs = dict(
                    noise=take("noise"),
                    time_span=time_span.float(),
                    mu=take("mu"),
                    speaker_embeddings=torch.stack(
                        [streams[name]["speaker_embeddings"] for name in names]
                    ),
                    mel_conditioning=take("mel_conditioning"),
                )
                if pool is graph_pool and sum(new_frames) <= runner.tier_frames[-1]:
                    generated = runner.run(
                        **inputs, new_frames=new_frames, caches=pairs
                    )
                elif pool is graph_pool:
                    assert (
                        runner.run(**inputs, new_frames=new_frames, caches=pairs)
                        is None
                    )
                    generated = solve_flow_euler_prefix(
                        estimator,
                        pool,
                        **inputs,
                        new_frames=new_frames,
                        caches=pairs,
                        cfg_rate=0.7,
                    )
                else:
                    generated = solve_flow_euler_prefix(
                        estimator,
                        pool,
                        **inputs,
                        new_frames=new_frames,
                        caches=pairs,
                        cfg_rate=0.7,
                    )
                outputs.append(generated)
            assert torch.equal(outputs[0], outputs[1]), step
            for name in names:
                for graph_row, eager_row in zip(*caches[name]):
                    assert graph_row.committed_frames == eager_row.committed_frames
                    assert torch.equal(graph_row.conv_context, eager_row.conv_context)
                    graph_pages = graph_row.pages(device)[: graph_row.committed_frames]
                    eager_pages = eager_row.pages(device)[: eager_row.committed_frames]
                    for euler_step in range(10):
                        for layer in range(LAYERS):
                            assert torch.equal(
                                graph_pool.keys[euler_step][layer][graph_pages],
                                eager_pool.keys[euler_step][layer][eager_pages],
                            )
                            assert torch.equal(
                                graph_pool.values[euler_step][layer][graph_pages],
                                eager_pool.values[euler_step][layer][eager_pages],
                            )
    for pair_per_pool in caches.values():
        for pool, pair in zip(pools, pair_per_pool):
            release_rows(pool, list(pair))
    assert all(len(pool.free_blocks) == 64 for pool in pools)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_packed", [False, True], ids=["eager", "compiled"])
def test_final_graph_replays_equal_the_eager_solve(compile_packed: bool) -> None:
    """Replays with empty row slots and a padding row, after other steps left
    their frames in the tier's buffers, equal the eager solve of the rows alone.
    Steps above the largest tier or past the row slots are refused."""
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()

    def step_inputs(lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(sum(lengths))
        frames = (1, sum(lengths), CHANNELS)
        return dict(
            noise=torch.randn(frames, device=device, generator=generator),
            time_span=time_span,
            mu=torch.randn(frames, device=device, generator=generator),
            speaker_embeddings=torch.randn(
                len(lengths), CHANNELS, device=device, generator=generator
            ).to(dtype),
            mel_conditioning=torch.randn(frames, device=device, generator=generator),
        )

    def eager_solve(lengths: tuple[int, ...]) -> torch.Tensor:
        inputs = step_inputs(lengths)
        twin_rows = pack_rows(lengths * 2, device)
        return solve_flow_euler_packed(
            estimator,
            inputs["noise"],
            inputs["time_span"],
            inputs["mu"],
            inputs["speaker_embeddings"],
            inputs["mel_conditioning"],
            twin_rows,
            estimator.row_attention(twin_rows, streaming=False, dtype=dtype),
            estimator.rope_angles(twin_rows.width),
            cfg_rate=0.7,
        )

    if compile_packed:
        assert estimator.compile(dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for lengths in ((53,), (53, 78)):
                eager_solve(lengths)
    else:
        pass
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    runner = FinalCudaGraphRunner(
        estimator,
        backend=backend,
        device=device,
        autocast_dtype=dtype,
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=3,
        tier_frames=(128, 256),
        max_frames=1024,
    )
    runner.capture()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for lengths in ((40, 70), (100, 90, 50), (60,), (128,), (200, 56), (40, 70)):
            replayed = runner.run(**step_inputs(lengths), lengths=lengths)
            assert torch.equal(replayed, eager_solve(lengths)), lengths
        for lengths in ((150, 150), (20, 20, 20, 20)):
            assert runner.run(**step_inputs(lengths), lengths=lengths) is None
