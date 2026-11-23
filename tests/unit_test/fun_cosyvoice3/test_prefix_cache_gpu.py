# SPDX-License-Identifier: Apache-2.0
"""Hops over a cached prefix reproduce the whole-history causal solve, and
graph replayed hops and whole-history solves their eager solves."""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.causal_conv import FusedConvPositionEmbedding
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    CONV_CONTEXT_FRAMES,
    PackedDiT,
    gather_rows,
    pack_rows,
    solve_flow_euler_packed,
    uses_fa4,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    PrefixRowAttention,
    PrefixStepLayout,
    compile_forward_prefix,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import build_prefix_pool
from sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph import (
    WholeHistoryCudaGraphRunner,
)
from sglang_omni.platforms import current_platform

pytestmark = pytest.mark.accelerator

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

CHUNK = 50
CHANNELS = 80
HEADS, HEAD_DIM, LAYERS = 4, 64, 3
COMPILED_OVER_EAGER_ERROR = 1.1
# note (emre): error against the float32 reference over the BF16 baseline's error.
# Measured on SM120 at most 1.58 for solves, 1.51 for convs, 1.44 for K and V.
SOLVE_ERROR_RATIO = 2.0
CONV_ERROR_RATIO = 2.0
KV_ERROR_RATIO = 2.0


def make_estimator(dtype: torch.dtype = torch.bfloat16) -> PackedDiT:
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
            module.to(dtype)
        else:
            pass
    dit.input_embed.conv_pos_embed = FusedConvPositionEmbedding(
        dit.input_embed.conv_pos_embed
    )
    estimator = PackedDiT(dit, device="cuda")
    if not estimator.is_ragged:
        pytest.skip("requires ragged attention")
    else:
        pass
    return estimator


def max_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (actual.float() - reference.float()).abs().max().item()


def assert_within_baseline(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor | None,
    baseline: torch.Tensor | None,
    ratio: float,
) -> None:
    """Bit equal to expected on FA3. On SM120 BF16 rounding varies with the row count,
    so actual has to stay about as near the float32 reference as the BF16 baseline."""
    if uses_fa4(actual.device):
        baseline_error = max_error(baseline, reference)
        error = max_error(actual, reference)
        assert error <= ratio * baseline_error, (label, error, baseline_error)
    else:
        assert torch.equal(actual, expected), label


class RecordingAttention:
    """Keeps each layer's keys and values, in call order: Euler step, then layer."""

    def __init__(self, attention) -> None:
        self.attention = attention
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        self.keys.append(key.clone())
        self.values.append(value.clone())
        return self.attention(query, key, value)


@dataclass
class WholeHistoryStates:
    solve: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    conv_inputs: torch.Tensor


class WholeHistoryReference:
    """Eager whole-history solves of one seed in float32 and in BF16. Their difference
    is the rounding error a correct BF16 path has, for each kind of state."""

    def __init__(self, time_span: torch.Tensor) -> None:
        self.time_span = time_span.float()
        self.estimators = {
            torch.float32: make_estimator(torch.float32),
            torch.bfloat16: make_estimator(torch.bfloat16),
        }
        for dtype, estimator in self.estimators.items():
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                estimator.build_modulations(self.time_span.to(dtype), dtype)

    def run(
        self,
        dtype: torch.dtype,
        noise: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        lengths: list[int],
        *,
        streaming: bool = True,
    ) -> WholeHistoryStates:
        estimator = self.estimators[dtype]
        device = noise.device
        twin_rows = pack_rows(lengths * 2, device)
        attention = RecordingAttention(
            estimator.row_attention(twin_rows, streaming=streaming, dtype=dtype)
        )
        conv_inputs: list[torch.Tensor] = []
        positional_conv = type(estimator).positional_conv

        def recording_conv(x: torch.Tensor, conv_index: int) -> torch.Tensor:
            conv_inputs.append(x.clone())
            return positional_conv(estimator, x, conv_index)

        estimator.positional_conv = recording_conv
        try:
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                solve = solve_flow_euler_packed(
                    estimator,
                    noise.to(dtype),
                    self.time_span.to(dtype),
                    mu.to(dtype),
                    spks.to(dtype),
                    cond.to(dtype),
                    twin_rows,
                    attention,
                    estimator.rope_angles(twin_rows.width),
                    cfg_rate=0.7,
                )
        finally:
            del estimator.positional_conv
        return WholeHistoryStates(
            solve, attention.keys, attention.values, torch.stack(conv_inputs)
        )


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
        estimator.build_modulations(time_span, dtype)
        references = WholeHistoryReference(time_span)
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
            float32_solve = references.run(
                torch.float32,
                gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                spks,
                gather_rows(cond[:, :, :total].transpose(1, 2), packed),
                [total] * rows,
            ).solve
            for row in range(rows):
                expected = reference[0, row * total + previous : (row + 1) * total]
                float32_row = float32_solve[
                    0, row * total + previous : (row + 1) * total
                ]
                actual = cached[
                    0,
                    row * (total - start)
                    + previous
                    - start : (row + 1) * (total - start),
                ]
                assert_within_baseline(
                    f"hop_solve {total} {row}",
                    actual,
                    expected,
                    float32_row,
                    expected,
                    SOLVE_ERROR_RATIO,
                )
            previous = total
    used = 32 - len(pool.free_blocks)
    assert used == rows * 2 * ((totals[-1] + BLOCK_FRAMES - 1) // BLOCK_FRAMES)
    for pair in caches:
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_and_whole_history_hops_launch_without_a_host_wait() -> None:
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=16 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(1)
    lengths = [120, 70]
    noise, mu, cond = (
        torch.randn(1, sum(lengths), CHANNELS, device=device, dtype=dtype)
        for _ in range(3)
    )
    spks = torch.randn(len(lengths), CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    caches = [(PrefixCacheRow(), PrefixCacheRow()) for _ in lengths]
    for pair, length in zip(caches, lengths, strict=True):
        assert grow_rows(pool, list(pair), [length, length])
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        float32_solve = (
            WholeHistoryReference(time_span)
            .run(torch.float32, noise, mu, spks, cond, lengths)
            .solve
        )
        angles = estimator.rope_angles(max(lengths))
        torch.cuda.synchronize()
        sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            twin_rows = pack_rows(lengths * 2, device)
            whole = solve_flow_euler_packed(
                estimator,
                noise,
                time_span,
                mu,
                spks,
                cond,
                twin_rows,
                estimator.row_attention(twin_rows, streaming=True, dtype=dtype),
                angles,
                cfg_rate=0.7,
            )
            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                noise,
                time_span,
                mu,
                spks,
                cond,
                lengths,
                caches,
                cfg_rate=0.7,
            )
        finally:
            torch.cuda.set_sync_debug_mode(sync_debug_mode)
    assert_within_baseline(
        "no_wait_solve", cached, whole, float32_solve, whole, SOLVE_ERROR_RATIO
    )


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
        estimator.build_modulations(time_span, dtype)
        with torch.autocast("cuda", enabled=False):
            reference_estimator.build_modulations(time_span.float(), torch.float32)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_layouts_upload_from_pinned_memory_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    pool = PrefixKVPool(
        layer_num=1,
        euler_steps=1,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=128 * BLOCK_FRAMES,
        device=device,
        dtype=torch.bfloat16,
    )
    rows = [PrefixCacheRow(), PrefixCacheRow()]
    assert grow_rows(pool, rows, [2048, 2048])
    layout = PrefixStepLayout(
        half_frames=4096,
        row_slots=1,
        segment_count=2 * (-(-4096 // CHUNK) + 2),
        page_table_width=4096,
    )

    uploads: list[tuple[bool, bool]] = []
    to = torch.Tensor.to

    def recording_to(
        tensor: torch.Tensor,
        target: torch.device | torch.dtype,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        moved = to(tensor, target, non_blocking=non_blocking)
        if tensor.device.type == "cpu" and moved.device.type == "cuda":
            uploads.append((tensor.is_pinned(), non_blocking))
        else:
            pass
        return moved

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    PrefixRowAttention(
        rows=rows,
        new_frames=[2048, 2048],
        layout=layout,
        padding_block=pool.padding_block,
        chunk_size=CHUNK,
        device=device,
    )
    assert uploads
    assert all(pinned and non_blocking for pinned, non_blocking in uploads)


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
def test_prefix_pool_leaves_its_reserve_under_the_allocator_cap() -> None:
    """A pool sized under an allocator cap took its padding block and the allocator's
    rounding out of its reserve, so the step it was sized beside failed to allocate."""
    device = torch.device("cuda", torch.cuda.current_device())
    attention = SimpleNamespace(heads=16, inner_dim=1024)
    flow = SimpleNamespace(
        decoder=SimpleNamespace(
            estimator=SimpleNamespace(
                transformer_blocks=[SimpleNamespace(attn=attention)] * 22,
                input_embed=SimpleNamespace(proj=SimpleNamespace(out_features=1024)),
            )
        )
    )
    reserve_bytes = 6 * 2**30
    gc.collect()
    torch.cuda.empty_cache()
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    entry_fraction = torch.cuda.get_per_process_memory_fraction(device)
    torch.cuda.set_per_process_memory_fraction(
        (torch.cuda.memory_reserved(device) + 8 * 2**30) / total_bytes, device
    )
    try:
        pool = build_prefix_pool(
            flow,
            device,
            torch.bfloat16,
            size_gb=None,
            max_blocks=None,
            stage_reserve_bytes=reserve_bytes,
            headroom_bytes=0,
            share_ceiling_bytes=None,
        )
        assert pool is not None
        step = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        del step, pool
    finally:
        torch.cuda.set_per_process_memory_fraction(entry_fraction, device)
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_prefix", [False, True], ids=["eager", "compiled"])
def test_prefix_graph_replays_equal_the_eager_solve(compile_prefix: bool) -> None:
    """Replays with padding frames and unused row slots equal the eager solve,
    also when they resume from a step above the largest tier that ran eagerly and
    when a whole-history graph replays from the same pool between hops."""
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
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        references = WholeHistoryReference(time_span)
    if compile_prefix:
        assert estimator.compile(dtype)
        for pool in pools:
            pool.forward = compile_forward_prefix()
    else:
        pass
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    graphs = SolveGraphCapture(backend, device=device, autocast_dtype=dtype)
    runner = PrefixCudaGraphRunner(
        estimator,
        graph_pool,
        graphs=graphs,
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
    whole_history = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=graphs,
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=2,
        tier_frames=(512,),
    )
    runner.capture()
    whole_history.capture()
    torch.manual_seed(3)
    whole_history_inputs = dict(
        noise=torch.randn(1, 500, CHANNELS, device=device),
        time_span=time_span,
        mu=torch.randn(1, 500, CHANNELS, device=device),
        speaker_embeddings=torch.randn(2, CHANNELS, device=device, dtype=dtype),
        mel_conditioning=torch.randn(1, 500, CHANNELS, device=device),
        lengths=(300, 200),
    )
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

            assert whole_history.run(**whole_history_inputs) is not None
            outputs = []
            for pool_index, pool in enumerate(pools):
                pairs = [caches[name][pool_index] for name in names]
                for pair, total in zip(pairs, totals):
                    assert grow_rows(pool, list(pair), [total, total])
                inputs = dict(
                    noise=take("noise"),
                    time_span=time_span,
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
            states = {}
            for name, total in step:
                frames = [
                    streams[name][key][:, :total].transpose(0, 1).unsqueeze(0)
                    for key in ("noise", "mu", "mel_conditioning")
                ]
                speaker = streams[name]["speaker_embeddings"].unsqueeze(0)
                states[name] = (
                    references.run(
                        torch.float32,
                        frames[0],
                        frames[1],
                        speaker,
                        frames[2],
                        [total],
                    ),
                    references.run(
                        dtype, frames[0], frames[1], speaker, frames[2], [total]
                    ),
                )
            float32_parts = []
            bf16_parts = []
            for (name, total), start in zip(step, starts):
                float32_parts.append(states[name][0].solve[0, start:total])
                bf16_parts.append(states[name][1].solve[0, start:total])
            float32_output = torch.cat(float32_parts).unsqueeze(0)
            bf16_output = torch.cat(bf16_parts).unsqueeze(0)
            for pool_index, output in enumerate(outputs):
                assert_within_baseline(
                    f"replay_solve {pool_index}",
                    output,
                    outputs[1],
                    float32_output,
                    bf16_output,
                    SOLVE_ERROR_RATIO,
                )
            for name, total in step:
                float32_states, bf16_states = states[name]
                for twin, rows in enumerate(zip(*caches[name])):
                    graph_row, eager_row = rows
                    committed = graph_row.committed_frames
                    assert committed == eager_row.committed_frames
                    row_offset = twin * (CONV_CONTEXT_FRAMES + total) + committed
                    window = range(row_offset, row_offset + CONV_CONTEXT_FRAMES)
                    float32_conv = float32_states.conv_inputs[:, window]
                    bf16_conv = bf16_states.conv_inputs[:, window]
                    float32_conv = float32_conv.view(10, 2, CONV_CONTEXT_FRAMES, -1)
                    bf16_conv = bf16_conv.view(10, 2, CONV_CONTEXT_FRAMES, -1)
                    for pool_index, row in enumerate((graph_row, eager_row)):
                        assert_within_baseline(
                            f"replay_conv {pool_index}",
                            row.conv_context,
                            eager_row.conv_context,
                            float32_conv,
                            bf16_conv,
                            CONV_ERROR_RATIO,
                        )
                    frames = range(twin * total, twin * total + committed)
                    if committed == 0:
                        continue
                    else:
                        pass
                    pages = [
                        graph_row.pages(device)[:committed],
                        eager_row.pages(device)[:committed],
                    ]
                    for euler_step in range(10):
                        for layer in range(LAYERS):
                            call = euler_step * LAYERS + layer
                            for pool_index, pool in enumerate(pools):
                                for label, cached, float32_part, bf16_part in (
                                    (
                                        "key",
                                        pool.keys[euler_step][layer],
                                        float32_states.keys[call],
                                        bf16_states.keys[call],
                                    ),
                                    (
                                        "value",
                                        pool.values[euler_step][layer],
                                        float32_states.values[call],
                                        bf16_states.values[call],
                                    ),
                                ):
                                    eager_cached = getattr(eager_pool, f"{label}s")[
                                        euler_step
                                    ][layer][pages[1]]
                                    assert_within_baseline(
                                        f"replay_{label} {pool_index}",
                                        cached[pages[pool_index]].flatten(1),
                                        eager_cached.flatten(1),
                                        float32_part[0, frames],
                                        bf16_part[0, frames],
                                        KV_ERROR_RATIO,
                                    )
    for pair_per_pool in caches.values():
        for pool, pair in zip(pools, pair_per_pool):
            release_rows(pool, list(pair))
    assert all(len(pool.free_blocks) == 64 for pool in pools)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_graph_tiers_capture_over_one_set_of_input_buffers() -> None:
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=8 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    runner = PrefixCudaGraphRunner(
        estimator,
        pool,
        graphs=SolveGraphCapture(backend, device=device, autocast_dtype=dtype),
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
    assert len({captured.layout.row_slots for captured in runner.captured}) > 1
    for field in (
        "noise",
        "mu",
        "mel_conditioning",
        "speaker_embeddings",
        "time_span",
        "context",
        "next_context",
    ):
        storages = {
            getattr(captured, field).untyped_storage().data_ptr()
            for captured in runner.captured
        }
        assert len(storages) == 1, field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_packed", [False, True], ids=["eager", "compiled"])
def test_whole_history_graph_replays_equal_the_eager_solve(
    compile_packed: bool,
) -> None:
    """Replays with empty row slots and a padding row, after other steps left their
    frames in the tier's buffers, equal the eager solve of the rows alone, also
    through a tier whose padding row is longer than any row. Steps above the
    largest tier or past the row slots are refused."""
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)

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

    with torch.inference_mode():
        references = WholeHistoryReference(time_span)

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
    runner = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=SolveGraphCapture(backend, device=device, autocast_dtype=dtype),
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=3,
        tier_frames=(128, 256, 2048),
    )
    runner.capture()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for lengths in (
            (40, 70),
            (100, 90, 50),
            (60,),
            (128,),
            (200, 56),
            (900, 900),
            (40, 70),
        ):
            replayed = runner.run(**step_inputs(lengths), lengths=lengths)
            inputs = step_inputs(lengths)
            frames = (
                inputs["noise"],
                inputs["mu"],
                inputs["speaker_embeddings"],
                inputs["mel_conditioning"],
            )
            assert_within_baseline(
                f"whole_replay {lengths}",
                replayed,
                eager_solve(lengths),
                references.run(
                    torch.float32, *frames, list(lengths), streaming=False
                ).solve,
                references.run(dtype, *frames, list(lengths), streaming=False).solve,
                SOLVE_ERROR_RATIO,
            )
        for lengths in ((900, 900, 400), (20, 20, 20, 20)):
            assert runner.run(**step_inputs(lengths), lengths=lengths) is None
