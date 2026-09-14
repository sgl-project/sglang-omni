# SPDX-License-Identifier: Apache-2.0
"""Chronological streaming cache fusion, mutation, and fallback coverage."""

from __future__ import annotations

import copy

import pytest
import torch

from sglang_omni.models.moss_tts import attention as attention_impl
from sglang_omni.models.moss_tts import vocoder_kernels as kernels

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def _inputs(batch, heads, context, length, dim, dtype, *, device="cuda"):
    capacity = batch + 4
    # note (Zhang Yiyang): Exercise sliced cache rows and packed-QKV layout.
    cached_k = torch.randn(
        capacity, heads, context, dim * 2, device=device, dtype=dtype
    )[..., ::2]
    cached_v = torch.randn(
        capacity, context, heads, dim, device=device, dtype=dtype
    ).permute(0, 2, 1, 3)
    offsets = torch.arange(capacity, device=device) + context + 3
    positions = (
        offsets[:, None] - context + torch.arange(context, device=device)[None, :]
    )
    slots = torch.randperm(capacity, device=device)[:batch]
    valid = torch.arange(batch, device=device) % 3 != 1
    packed = torch.randn(batch, length, 3, heads, dim, device=device, dtype=dtype)
    current_k, current_v = packed[:, :, 1].transpose(1, 2), packed[:, :, 2].transpose(
        1, 2
    )
    query_positions = (
        offsets[slots, None] + torch.arange(length, device=device)[None, :]
    )
    return (
        cached_k,
        cached_v,
        positions,
        offsets,
        slots,
        valid,
        current_k,
        current_v,
        query_positions,
    )


def _reference(inputs):
    ck, cv, cp, offsets, slots, valid, k, v, qp = inputs
    context = ck.shape[2]
    all_k = torch.cat((ck.index_select(0, slots), k), dim=2)
    all_v = torch.cat((cv.index_select(0, slots), v), dim=2)
    positions = torch.cat((cp.index_select(0, slots), qp), dim=1)
    next_state = [x.clone() for x in (ck, cv, cp, offsets)]
    # note (Zhang Yiyang): Invalid rows can refer to inactive real slots.
    live_slots = slots[valid]
    next_state[0].index_copy_(0, live_slots, all_k[valid, :, -context:, :])
    next_state[1].index_copy_(0, live_slots, all_v[valid, :, -context:, :])
    next_state[2].index_copy_(0, live_slots, positions[valid, -context:])
    next_state[3][live_slots] += k.shape[2]
    return (all_k, all_v, positions), next_state


def _fused(inputs):
    ck, cv, cp, offsets, slots, valid, k, v, qp = inputs
    all_k, all_v, positions = kernels.gather_streaming_kv(ck, cv, cp, k, v, qp, slots)
    kernels.commit_streaming_kv_(
        ck, cv, cp, offsets, all_k, all_v, positions, slots, valid
    )
    return all_k, all_v, positions


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 1, 1, 8),
        (3, 3, 17, 1, 7),
        (3, 3, 17, 17, 7),
        (3, 3, 17, 21, 7),
        (2, 20, 125, 5, 64),
        (16, 12, 400, 160, 64),
        (4, 12, 400, 800, 64),
    ],
)
@torch.no_grad()
def test_streaming_kv_matches_chronological_reference(shape, dtype):
    inputs = _inputs(*shape, dtype)
    assert kernels.can_fuse_streaming_kv(*inputs[:6])
    expected, expected_state = _reference(inputs)
    actual = _fused(inputs)
    for got, want in zip(actual, expected):
        _assert_bytes_equal(got, want)
    for got, want in zip(inputs[:4], expected_state):
        _assert_bytes_equal(got, want)


@pytest.mark.accelerator
@requires_cuda
@torch.no_grad()
def test_streaming_kv_graph_replays_dynamic_slots_and_valid_rows():
    inputs = _inputs(4, 3, 17, 5, 8, torch.bfloat16)
    initial = [t.clone() for t in inputs[:4]]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _fused(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _fused(inputs)
    for tensor, value in zip(inputs[:4], initial):
        tensor.copy_(value)
    for step in range(6):
        inputs[4].copy_(inputs[4].roll(1))
        inputs[5].copy_(
            torch.tensor([step % 2 == 0, False, True, step % 3 == 0], device="cuda")
        )
        inputs[8].copy_(inputs[3][inputs[4], None] + torch.arange(5, device="cuda"))
        expected, expected_state = _reference(inputs)
        graph.replay()
        for got, want in zip(actual, expected):
            _assert_bytes_equal(got, want)
        for got, want in zip(inputs[:4], expected_state):
            _assert_bytes_equal(got, want)


@pytest.mark.accelerator
@requires_cuda
@torch.no_grad()
def test_streaming_kv_preserves_inactive_nan_and_signed_zero_bytes():
    inputs = _inputs(3, 2, 4, 5, 8, torch.bfloat16)
    # note (Zhang Yiyang): Invalid real slots retain stale/nonfinite bytes.
    # Cached values need no arithmetic, even when their positions are invalid.
    inputs[0].fill_(float("nan"))
    inputs[1].fill_(-0.0)
    inputs[2].fill_(-1)
    inputs[5].zero_()
    expected, expected_state = _reference(inputs)
    actual = _fused(inputs)
    for got, want in zip(actual, expected):
        _assert_bytes_equal(got, want)
    for got, want in zip(inputs[:4], expected_state):
        _assert_bytes_equal(got, want)


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize(
    "layout", ["slots", "valid", "offsets", "positions", "broadcast_cache"]
)
@torch.no_grad()
def test_streaming_kv_rejects_unsupported_metadata_or_writable_layout(layout):
    inputs = list(_inputs(3, 2, 4, 5, 8, torch.bfloat16))
    assert kernels.can_fuse_streaming_kv(*inputs[:6])
    index = {"slots": 4, "valid": 5, "offsets": 3, "positions": 2}.get(layout)
    if layout == "broadcast_cache":
        inputs[0] = inputs[0][:1].expand_as(inputs[0])
    else:
        original = inputs[index]
        backing = original.new_empty((*original.shape[:-1], original.shape[-1] * 2))
        backing[..., ::2].copy_(original)
        inputs[index] = backing[..., ::2]
    assert not kernels.can_fuse_streaming_kv(*inputs[:6])


@torch.no_grad()
def test_streaming_kv_cpu_uses_torch_fallback():
    inputs = _inputs(3, 2, 4, 5, 8, torch.float32, device="cpu")
    assert not kernels.can_fuse_streaming_kv(*inputs[:6])


@pytest.mark.accelerator
@requires_cuda
def test_streaming_kv_grad_and_missing_triton_use_fallback(monkeypatch):
    inputs = _inputs(3, 2, 4, 5, 8, torch.bfloat16)
    assert not kernels.can_fuse_streaming_kv(*inputs[:6])
    with torch.no_grad():
        assert kernels.can_fuse_streaming_kv(*inputs[:6])
        monkeypatch.setattr(kernels, "_streaming_kv_gather_kernel", None)
        assert not kernels.can_fuse_streaming_kv(*inputs[:6])


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize(
    "context,strided_metadata", [(17, False), (17, True), (None, False)]
)
@torch.no_grad()
def test_indexed_attention_fusion_and_fallback_match_torch(
    monkeypatch, context, strided_metadata
):
    model = attention_impl.MossAudioTokenizerAttention(
        in_proj=torch.nn.Linear(
            48, 144, bias=False, device="cuda", dtype=torch.bfloat16
        ),
        out_proj=torch.nn.Linear(
            48, 48, bias=False, device="cuda", dtype=torch.bfloat16
        ),
        embed_dim=48,
        num_heads=3,
        causal=True,
        context=context,
        rope=None,
    ).eval()
    reference = copy.deepcopy(model)
    can_fuse = attention_impl.can_fuse_streaming_kv
    used = []
    gather = attention_impl.gather_streaming_kv

    def record_gather(*args):
        used.append(True)
        return gather(*args)

    monkeypatch.setattr(attention_impl, "gather_streaming_kv", record_gather)
    with model.streaming(8), reference.streaming(8):
        for step, length in enumerate([1, 17, 21, 2, 5]):
            slots = torch.tensor([6, 1, 3], device="cuda").roll(step)
            valid = torch.tensor([True, False, step % 2 == 0], device="cuda")
            if strided_metadata:
                backing = torch.zeros(6, device="cuda", dtype=torch.long)
                backing[::2].copy_(slots)
                slots = backing[::2]
            if step == 3:
                for m in (model, reference):
                    m._streaming_state.reset_slots(torch.tensor([3], device="cuda"))
            chunk = torch.randn(3, length, 48, device="cuda", dtype=torch.bfloat16)
            execution = attention_impl.StreamingExecutionContext(slots, valid)
            with monkeypatch.context() as patch:
                patch.setattr(
                    attention_impl, "can_fuse_streaming_kv", lambda *args: False
                )
                expected = reference(chunk, execution_context=execution)
            assert attention_impl.can_fuse_streaming_kv is can_fuse
            actual = model(chunk, execution_context=execution)
            _assert_bytes_equal(actual, expected)
            for name in [
                "offset",
                "cached_keys",
                "cached_values",
                "cached_positions",
                "exec_mask",
            ]:
                _assert_bytes_equal(
                    getattr(model._streaming_state, name),
                    getattr(reference._streaming_state, name),
                )
        # note (Zhang Yiyang): Empty chunks retain output and state behavior.
        empty = chunk[:, :0]
        _assert_bytes_equal(
            model(empty, execution_context=execution),
            reference(empty, execution_context=execution),
        )
    assert bool(used) == (context is not None and not strided_metadata)


@pytest.mark.accelerator
@requires_cuda
@torch.no_grad()
def test_indexed_attention_commits_only_after_output_projection(monkeypatch):
    model = attention_impl.MossAudioTokenizerAttention(
        in_proj=torch.nn.Linear(
            48, 144, bias=False, device="cuda", dtype=torch.bfloat16
        ),
        out_proj=torch.nn.Linear(
            48, 48, bias=False, device="cuda", dtype=torch.bfloat16
        ),
        embed_dim=48,
        num_heads=3,
        causal=True,
        context=17,
        rope=None,
    )

    def fail_projection(x):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(model.out_proj, "forward", fail_projection)
    with model.streaming(4):
        state = model._streaming_state
        model._ensure_streaming_cache(state, 4, torch.device("cuda"), torch.bfloat16)
        names = ["offset", "cached_keys", "cached_values", "cached_positions"]
        before = {name: getattr(state, name).clone() for name in names}
        with pytest.raises(RuntimeError, match="projection failed"):
            model(
                torch.randn(2, 5, 48, device="cuda", dtype=torch.bfloat16),
                execution_context=attention_impl.StreamingExecutionContext(
                    torch.tensor([2, 0], device="cuda"),
                    torch.tensor([True, False], device="cuda"),
                ),
            )
        for name in names:
            _assert_bytes_equal(getattr(state, name), before[name])


@pytest.mark.accelerator
@requires_cuda
@torch.no_grad()
def test_streaming_kv_fullgraph_tracks_cache_mutations():
    inputs = _inputs(3, 2, 17, 5, 8, torch.bfloat16)
    compiled = torch.compile(_fused, fullgraph=True)
    for _ in range(3):
        expected, expected_state = _reference(inputs)
        actual = compiled(inputs)
        for got, want in zip(actual, expected):
            _assert_bytes_equal(got, want)
        for got, want in zip(inputs[:4], expected_state):
            _assert_bytes_equal(got, want)
