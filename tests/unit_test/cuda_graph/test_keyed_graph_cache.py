# SPDX-License-Identifier: Apache-2.0
"""Bookkeeping contract for the shared keyed CUDA-graph cache.

The cache owns what every per-step graph implementation currently reimplements:
bucket selection, signature keying, the key ceiling, capture-failure fusing, and
the shared memory pool. Capture itself stays with the caller.
"""

from __future__ import annotations

import pytest

from sglang_omni.cuda_graph import KeyedGraphCache


def _cache(**kwargs) -> KeyedGraphCache:
    kwargs.setdefault("name", "test-chain")
    kwargs.setdefault("batch_sizes", [1, 2, 4, 8])
    return KeyedGraphCache(**kwargs)


def test_bucket_selection_rounds_up_and_rejects_oversize():
    cache = _cache()
    assert cache.bucket_for(1) == 1
    assert cache.bucket_for(3) == 4
    assert cache.bucket_for(8) == 8
    assert cache.bucket_for(9) is None
    assert cache.bucket_for(0) is None


def test_batch_sizes_are_normalized_sorted_and_deduped():
    cache = _cache(batch_sizes=[8, 2, 2, 4, 0, -1, 1])
    assert cache.batch_sizes == (1, 2, 4, 8)


def test_captures_once_per_key_and_reuses():
    cache = _cache()
    calls = []

    def factory():
        calls.append(1)
        return object()

    first = cache.get_or_capture((4, "sampled"), factory)
    second = cache.get_or_capture((4, "sampled"), factory)
    assert first is second
    assert len(calls) == 1
    other = cache.get_or_capture((4, "argmax"), factory)
    assert other is not first
    assert len(calls) == 2


def test_key_ceiling_declines_instead_of_evicting_hot_entries():
    cache = _cache(max_keys=2)
    a = cache.get_or_capture(("a",), object)
    b = cache.get_or_capture(("b",), object)
    assert cache.get_or_capture(("c",), object) is None
    # the ceiling must not evict: existing keys still replay from cache
    assert cache.get_or_capture(("a",), object) is a
    assert cache.get_or_capture(("b",), object) is b


def test_failed_key_is_disabled_and_not_retried():
    cache = _cache()
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("capture failed")

    assert cache.get_or_capture(("k",), boom) is None
    assert cache.get_or_capture(("k",), boom) is None
    assert len(calls) == 1, "a disabled key must not attempt capture again"


def test_repeated_failures_fuse_the_whole_cache_off():
    cache = _cache(max_failures=2)

    def boom():
        raise RuntimeError("capture failed")

    assert cache.get_or_capture(("k1",), boom) is None
    assert cache.enabled is True
    assert cache.get_or_capture(("k2",), boom) is None
    assert cache.enabled is False
    # fused off: a fresh, capturable key is declined without calling the factory
    calls = []
    assert cache.get_or_capture(("k3",), lambda: calls.append(1) or object()) is None
    assert calls == []


def test_disabled_keys_do_not_consume_the_key_ceiling():
    """A burst of unusable signatures must not lock out later good ones."""
    cache = _cache(max_keys=2, max_failures=100)

    def boom():
        raise RuntimeError("capture failed")

    for i in range(5):
        assert cache.get_or_capture((f"bad{i}",), boom) is None
    good = cache.get_or_capture(("good",), object)
    assert good is not None


def test_env_gate_disables_before_any_capture(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_TEST_CHAIN_GRAPH", "0")
    cache = _cache(env_var="SGLANG_OMNI_TEST_CHAIN_GRAPH")
    calls = []
    assert cache.get_or_capture(("k",), lambda: calls.append(1)) is None
    assert cache.enabled is False
    assert calls == []


@pytest.mark.parametrize("value", ["1", "true", "", "yes"])
def test_env_gate_defaults_to_on(monkeypatch, value):
    monkeypatch.setenv("SGLANG_OMNI_TEST_CHAIN_GRAPH", value)
    cache = _cache(env_var="SGLANG_OMNI_TEST_CHAIN_GRAPH")
    assert cache.enabled is True


def test_disable_is_sticky_across_calls():
    cache = _cache()
    cache.disable("manual override")
    assert cache.enabled is False
    assert cache.get_or_capture(("k",), object) is None


def test_read_only_views_expose_state_without_letting_callers_mutate_it():
    cache = _cache()
    graph = cache.get_or_capture((4, "sampled"), object)
    assert dict(cache.graphs) == {(4, "sampled"): graph}
    with pytest.raises(TypeError):
        cache.graphs[(1,)] = object()

    def boom():
        raise RuntimeError("capture failed")

    cache.get_or_capture(("bad",), boom)
    assert cache.disabled_keys == frozenset({("bad",)})


def test_normalize_batch_sizes_defaults_and_bounds():
    from sglang_omni.cuda_graph import normalize_batch_sizes

    # explicit list: deduped, sorted, clamped, and always reaching the max
    assert normalize_batch_sizes([4, 1, 2, 99], max_batch_size=16) == (1, 2, 4, 16)
    assert normalize_batch_sizes([1, 2, 4, 8, 12, 16], max_batch_size=16) == (
        1,
        2,
        4,
        8,
        12,
        16,
    )
    # None falls back to the backbone's default capture ladder
    assert normalize_batch_sizes(None, max_batch_size=16) == (1, 2, 4, 8, 12, 16)
    assert normalize_batch_sizes([], max_batch_size=8) == (8,)


def test_warmup_captures_every_key_largest_first():
    """Sealed warmup exists so a stage can capture at startup instead of inside
    a live request. Order is descending because per-key graphs share one pool."""
    cache = _cache(batch_sizes=[1, 2, 4, 8])
    order = []
    warmed = cache.warmup([2, 8, 1, 4], lambda key: order.append(key) or object())
    assert order == [8, 4, 2, 1]
    assert warmed == 4
    assert sorted(cache.graphs) == [1, 2, 4, 8]


def test_warmup_skips_duplicates_and_respects_the_ceiling():
    cache = _cache(max_keys=2)
    captured = []
    warmed = cache.warmup([4, 4, 2, 1], lambda key: captured.append(key) or object())
    assert captured == [4, 2], "the ceiling stops capture, deduped, largest first"
    assert warmed == 2


def test_warmup_tolerates_a_failed_key_and_keeps_going():
    cache = _cache()
    seen = []

    def factory(key):
        seen.append(key)
        if key == 4:
            raise RuntimeError("capture failed")
        return object()

    warmed = cache.warmup([1, 2, 4], factory)
    assert seen == [4, 2, 1], "a failed key must not abort the remaining warmup"
    assert warmed == 2
    assert cache.disabled_keys == frozenset({4})


def test_pool_factory_lets_a_caller_keep_its_own_cuda_seam():
    """Code2Wav routes every CUDA call through an injectable API for testing;
    the shared pool must go through that seam too."""
    handles = []
    cache = _cache(pool_factory=lambda: handles.append(object()) or handles[-1])
    assert cache.memory_pool() is cache.memory_pool()
    assert len(handles) == 1


def test_clear_drops_graphs_and_pool_but_still_refuses_failed_keys():
    cache = _cache(pool_factory=object)
    cache.get_or_capture(("a",), object)

    def boom():
        raise RuntimeError("capture failed")

    assert cache.get_or_capture(("bad",), boom) is None
    pool = cache.memory_pool()

    cache.clear()

    assert dict(cache.graphs) == {}
    assert cache.memory_pool() is not pool
    retried = []
    assert cache.get_or_capture(("bad",), lambda: retried.append(1)) is None
    assert retried == []


def test_warmup_stops_when_a_precheck_declines():
    """A VRAM headroom check must be able to stop before capture, not after."""
    cache = _cache()
    captured = []
    warmed = cache.warmup(
        [1, 2, 4],
        lambda key: captured.append(key) or object(),
        precheck=lambda: len(captured) < 2,
    )
    assert captured == [4, 2]
    assert warmed == 2


def test_graphs_view_is_stable_and_tracks_clear():
    """Adopters read `graphs` on the replay path, so the proxy is built once;
    it must still reflect a later clear()."""
    cache = _cache()
    first_view = cache.graphs
    graph = cache.get_or_capture((4,), object)
    assert cache.graphs is first_view, "the view must not be rebuilt per access"
    assert dict(first_view) == {(4,): graph}
    cache.clear()
    assert dict(cache.graphs) == {}, "the held view must track a clear()"
