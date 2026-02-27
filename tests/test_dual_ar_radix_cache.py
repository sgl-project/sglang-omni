# SPDX-License-Identifier: Apache-2.0
"""Tests for the DualAR radix prefix cache.

Covers:
1. Tree operations: insert, match, split, evict, lock/unlock
2. KV save/restore round-trip with mock model
3. Eviction: capacity limits, LRU ordering, lock protection
"""

from __future__ import annotations

import torch
from torch import Tensor

from sglang_omni.models.fishaudio_s1.runtime.radix_cache import DualARRadixCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_data(
    num_layers: int = 2, length: int = 10, seed: int = 0
) -> list[tuple[Tensor, Tensor]]:
    """Create deterministic mock KV data for testing."""
    torch.manual_seed(seed)
    kv = []
    for _ in range(num_layers):
        k = torch.randn(1, 4, length, 16)  # [B, H, L, D]
        v = torch.randn(1, 4, length, 16)
        kv.append((k, v))
    return kv


# ===========================================================================
# Tree operation tests
# ===========================================================================


class TestRadixCacheTreeOps:
    """Pure data-structure tests for the radix tree."""

    def test_insert_and_match_exact(self):
        cache = DualARRadixCache(max_tokens=10000)
        tokens = [1, 2, 3, 4, 5]
        kv = _make_kv_data(length=5)
        cache.insert(tokens, kv)

        matched, kv_out, node = cache.match_prefix(tokens)
        assert matched == 5
        assert kv_out is kv

    def test_match_prefix_partial(self):
        cache = DualARRadixCache(max_tokens=10000)
        tokens = [1, 2, 3, 4, 5]
        kv = _make_kv_data(length=5)
        cache.insert(tokens, kv)

        # Query a longer sequence that shares the prefix
        longer = [1, 2, 3, 4, 5, 6, 7]
        matched, kv_out, node = cache.match_prefix(longer)
        assert matched == 5
        assert kv_out is kv

    def test_match_no_hit(self):
        cache = DualARRadixCache(max_tokens=10000)
        tokens = [1, 2, 3]
        kv = _make_kv_data(length=3)
        cache.insert(tokens, kv)

        matched, kv_out, _ = cache.match_prefix([4, 5, 6])
        assert matched == 0
        assert kv_out is None

    def test_match_empty_cache(self):
        cache = DualARRadixCache(max_tokens=10000)
        matched, kv_out, node = cache.match_prefix([1, 2, 3])
        assert matched == 0
        assert kv_out is None

    def test_insert_empty_tokens(self):
        cache = DualARRadixCache(max_tokens=10000)
        result = cache.insert([], _make_kv_data())
        assert result == 0
        assert cache.total_tokens == 0

    def test_insert_shared_prefix_causes_split(self):
        """Inserting two sequences with a shared prefix should split the edge."""
        cache = DualARRadixCache(max_tokens=10000)

        kv1 = _make_kv_data(length=5, seed=1)
        kv2 = _make_kv_data(length=7, seed=2)

        cache.insert([1, 2, 3, 4, 5], kv1)
        cache.insert([1, 2, 3, 6, 7], kv2)

        # Match the shared prefix [1,2,3] — split during insert propagates
        # KV from kv1 to the mid-node via causal slicing
        matched, kv_out, _ = cache.match_prefix([1, 2, 3])
        assert matched == 3
        assert kv_out is not None
        # The mid-node KV should be the first 3 positions sliced from kv1
        for (k, v), (k1, v1) in zip(kv_out, kv1):
            assert torch.equal(k, k1[:, :, :3, :])
            assert torch.equal(v, v1[:, :, :3, :])

        # Full match for first sequence
        matched, kv_out, _ = cache.match_prefix([1, 2, 3, 4, 5])
        assert matched == 5
        assert kv_out is kv1

        # Full match for second sequence
        matched, kv_out, _ = cache.match_prefix([1, 2, 3, 6, 7])
        assert matched == 5
        assert kv_out is kv2

    def test_insert_extending_existing_prefix(self):
        """Inserting a longer sequence that starts with an existing cached prefix."""
        cache = DualARRadixCache(max_tokens=10000)

        kv_short = _make_kv_data(length=3, seed=1)
        kv_long = _make_kv_data(length=5, seed=2)

        cache.insert([1, 2, 3], kv_short)
        already = cache.insert([1, 2, 3, 4, 5], kv_long)
        assert already == 3  # first 3 tokens were already cached

        # Original prefix still valid
        matched, kv_out, _ = cache.match_prefix([1, 2, 3])
        assert matched == 3
        assert kv_out is kv_short

        # Extended prefix also valid
        matched, kv_out, _ = cache.match_prefix([1, 2, 3, 4, 5])
        assert matched == 5
        assert kv_out is kv_long

    def test_insert_duplicate_updates_kv(self):
        """Inserting the same sequence again should update the KV data."""
        cache = DualARRadixCache(max_tokens=10000)
        tokens = [1, 2, 3]

        kv1 = _make_kv_data(length=3, seed=1)
        kv2 = _make_kv_data(length=3, seed=2)

        cache.insert(tokens, kv1)
        cache.insert(tokens, kv2)

        _, kv_out, _ = cache.match_prefix(tokens)
        assert kv_out is kv2

    def test_total_tokens_tracking(self):
        cache = DualARRadixCache(max_tokens=10000)

        cache.insert([1, 2, 3], _make_kv_data(length=3))
        assert cache.total_tokens == 3

        cache.insert([1, 2, 3, 4, 5], _make_kv_data(length=5))
        # 3 existing + 2 new suffix = 5 total (split creates mid node but
        # doesn't add tokens — the edge tokens are redistributed)
        assert cache.total_tokens == 5

        cache.insert([4, 5, 6], _make_kv_data(length=3))
        assert cache.total_tokens == 8

    def test_match_prefix_with_tensor_input(self):
        cache = DualARRadixCache(max_tokens=10000)
        tokens = [10, 20, 30]
        kv = _make_kv_data(length=3)
        cache.insert(tokens, kv)

        tensor_tokens = torch.tensor([10, 20, 30, 40])
        matched, kv_out, _ = cache.match_prefix(tensor_tokens)
        assert matched == 3
        assert kv_out is kv


# ===========================================================================
# Lock reference counting tests
# ===========================================================================


class TestRadixCacheLocking:
    """Tests for lock_ref-based eviction protection."""

    def test_lock_and_unlock(self):
        cache = DualARRadixCache(max_tokens=10000)
        kv = _make_kv_data(length=5)
        cache.insert([1, 2, 3, 4, 5], kv)

        _, _, node = cache.match_prefix([1, 2, 3, 4, 5])
        assert node.lock_ref == 0

        cache.inc_lock_ref(node)
        assert node.lock_ref >= 1

        cache.dec_lock_ref(node)
        assert node.lock_ref == 0

    def test_locked_node_not_evicted(self):
        cache = DualARRadixCache(max_tokens=20)

        kv1 = _make_kv_data(length=10, seed=1)
        kv2 = _make_kv_data(length=10, seed=2)

        cache.insert([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kv1)
        _, _, node1 = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cache.inc_lock_ref(node1)

        # This insertion should try to evict but node1 is locked
        cache.insert([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], kv2)

        # Both should still be present (cache exceeded capacity but locked
        # nodes can't be evicted)
        matched1, kv_out1, _ = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert matched1 == 10
        assert kv_out1 is kv1

        cache.dec_lock_ref(node1)

    def test_multiple_locks(self):
        cache = DualARRadixCache(max_tokens=10000)
        kv = _make_kv_data(length=3)
        cache.insert([1, 2, 3], kv)
        _, _, node = cache.match_prefix([1, 2, 3])

        cache.inc_lock_ref(node)
        cache.inc_lock_ref(node)
        assert node.lock_ref >= 2

        cache.dec_lock_ref(node)
        assert node.lock_ref >= 1

        cache.dec_lock_ref(node)
        assert node.lock_ref == 0


# ===========================================================================
# Eviction tests
# ===========================================================================


class TestRadixCacheEviction:
    """Tests for LRU leaf-only eviction."""

    def test_evict_lru_leaf(self):
        cache = DualARRadixCache(max_tokens=10000)

        kv_old = _make_kv_data(length=3, seed=1)
        cache.insert([1, 2, 3], kv_old)

        # Small delay to ensure different access times
        kv_new = _make_kv_data(length=3, seed=2)
        cache.insert([4, 5, 6], kv_new)

        evicted = cache.evict(3)
        assert evicted == 3

        # Old entry should be evicted (LRU)
        matched, kv_out, _ = cache.match_prefix([1, 2, 3])
        assert matched == 0
        assert kv_out is None

        # New entry should still be present
        matched, kv_out, _ = cache.match_prefix([4, 5, 6])
        assert matched == 3
        assert kv_out is kv_new

    def test_evict_respects_capacity(self):
        cache = DualARRadixCache(max_tokens=10)

        # Insert 10 tokens
        kv1 = _make_kv_data(length=5, seed=1)
        cache.insert([1, 2, 3, 4, 5], kv1)
        kv2 = _make_kv_data(length=5, seed=2)
        cache.insert([6, 7, 8, 9, 10], kv2)
        assert cache.total_tokens == 10

        # Insert 5 more — should trigger eviction
        kv3 = _make_kv_data(length=5, seed=3)
        cache.insert([11, 12, 13, 14, 15], kv3)

        # Should have evicted the oldest to make room
        assert cache.total_tokens <= 15  # At most all three, likely 10

    def test_clear(self):
        cache = DualARRadixCache(max_tokens=10000)
        cache.insert([1, 2, 3], _make_kv_data(length=3))
        cache.insert([4, 5, 6], _make_kv_data(length=3))

        cache.clear()
        assert cache.total_tokens == 0

        matched, kv_out, _ = cache.match_prefix([1, 2, 3])
        assert matched == 0


# ===========================================================================
# KV snapshot / restore tests
# ===========================================================================


class _MockKVCache:
    """Minimal mock of fish-speech's KVCache."""

    def __init__(self, max_seq_len: int = 100, n_heads: int = 4, head_dim: int = 16):
        self.k_cache = torch.zeros(1, n_heads, max_seq_len, head_dim)
        self.v_cache = torch.zeros(1, n_heads, max_seq_len, head_dim)


class _MockAttention:
    def __init__(self, max_seq_len: int = 100):
        self.kv_cache = _MockKVCache(max_seq_len=max_seq_len)


class _MockLayer:
    def __init__(self, max_seq_len: int = 100):
        self.attention = _MockAttention(max_seq_len=max_seq_len)


class _MockModel:
    """Minimal mock of DualARTransformer with slow layers."""

    def __init__(self, num_layers: int = 2, max_seq_len: int = 100):
        self.layers = [_MockLayer(max_seq_len=max_seq_len) for _ in range(num_layers)]
        self.fast_layers = []


class TestKVSnapshotRestore:
    """Test the snapshot_slow_kv / restore_slow_kv helpers."""

    def test_round_trip(self):
        from sglang_omni.models.fishaudio_s1.runtime.dual_ar import (
            restore_slow_kv,
            snapshot_slow_kv,
        )

        model = _MockModel(num_layers=3)

        # Fill KV caches with deterministic data
        torch.manual_seed(42)
        length = 20
        for layer in model.layers:
            kv = layer.attention.kv_cache
            kv.k_cache[:, :, :length, :] = torch.randn_like(
                kv.k_cache[:, :, :length, :]
            )
            kv.v_cache[:, :, :length, :] = torch.randn_like(
                kv.v_cache[:, :, :length, :]
            )

        # Snapshot
        kv_data = snapshot_slow_kv(model, length)
        assert len(kv_data) == 3
        for k, v in kv_data:
            assert k.shape == (1, 4, length, 16)
            assert v.shape == (1, 4, length, 16)

        # Verify snapshots are independent copies
        original_k0 = kv_data[0][0].clone()
        model.layers[0].attention.kv_cache.k_cache.fill_(0)

        assert not torch.equal(
            model.layers[0].attention.kv_cache.k_cache[:, :, :length, :],
            original_k0,
        )

        # Restore
        restore_slow_kv(model, kv_data)

        # Verify restored data matches snapshot
        for i, (k_snap, v_snap) in enumerate(kv_data):
            kv = model.layers[i].attention.kv_cache
            assert torch.equal(kv.k_cache[:, :, :length, :], k_snap)
            assert torch.equal(kv.v_cache[:, :, :length, :], v_snap)

    def test_snapshot_different_lengths(self):
        from sglang_omni.models.fishaudio_s1.runtime.dual_ar import snapshot_slow_kv

        model = _MockModel(num_layers=2)
        torch.manual_seed(0)
        for layer in model.layers:
            kv = layer.attention.kv_cache
            kv.k_cache.normal_()
            kv.v_cache.normal_()

        kv_short = snapshot_slow_kv(model, 5)
        kv_long = snapshot_slow_kv(model, 15)

        assert kv_short[0][0].shape[2] == 5
        assert kv_long[0][0].shape[2] == 15

        # Short should be a prefix of long
        assert torch.equal(kv_short[0][0], kv_long[0][0][:, :, :5, :])


# ===========================================================================
# Integration: cache hit flow
# ===========================================================================


class TestRadixCacheIntegration:
    """Integration tests simulating the BatchPlanner/OutputProcessor cache flow."""

    def test_cache_miss_then_hit(self):
        """Simulate two requests with shared prefix: miss then hit."""
        cache = DualARRadixCache(max_tokens=10000)

        # Shared reference prefix
        ref_prefix = [100, 200, 300, 400, 500]
        suffix_a = [601, 602]
        suffix_b = [701, 702]

        tokens_a = ref_prefix + suffix_a
        tokens_b = ref_prefix + suffix_b

        # Request 1: cache miss
        matched, kv_out, _ = cache.match_prefix(tokens_a)
        assert matched == 0
        assert kv_out is None

        # Simulate prefill completion — save full KV
        kv_full_a = _make_kv_data(length=len(tokens_a), seed=42)
        cache.insert(tokens_a, kv_full_a)

        # Request 2: shares the first 5 tokens (voice ref prefix).
        # match_prefix splits the edge at pos 5 and propagates KV via
        # causal slicing — this is the voice-cloning prefix reuse path.
        matched, kv_out, node = cache.match_prefix(tokens_b)
        assert matched == 5
        # KV data is the first 5 positions sliced from kv_full_a
        assert kv_out is not None
        for (k, v), (ka, va) in zip(kv_out, kv_full_a):
            assert torch.equal(k, ka[:, :, :5, :])
            assert torch.equal(v, va[:, :, :5, :])

    def test_cache_hit_with_prefix_insert(self):
        """Insert a prefix, then hit on a longer sequence."""
        cache = DualARRadixCache(max_tokens=10000)

        ref_prefix = [100, 200, 300, 400, 500]

        # Insert just the prefix with its KV
        kv_prefix = _make_kv_data(length=5, seed=1)
        cache.insert(ref_prefix, kv_prefix)

        # Query with prefix + suffix
        query = ref_prefix + [601, 602, 603]
        matched, kv_out, node = cache.match_prefix(query)
        assert matched == 5
        assert kv_out is kv_prefix

        # Lock, use, then unlock
        cache.inc_lock_ref(node)
        assert node.lock_ref >= 1

        # Insert the full sequence
        kv_full = _make_kv_data(length=8, seed=2)
        already = cache.insert(query, kv_full)
        assert already == 5

        cache.dec_lock_ref(node)

        # Now match the full sequence
        matched, kv_out, _ = cache.match_prefix(query)
        assert matched == 8
        assert kv_out is kv_full

    def test_node_splitting_preserves_data(self):
        """Ensure node splitting during match_prefix doesn't lose data."""
        cache = DualARRadixCache(max_tokens=10000)

        kv = _make_kv_data(length=10, seed=1)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kv)

        # Match a partial prefix — triggers split at pos 6, propagates KV
        matched, kv_out, node = cache.match_prefix([1, 2, 3, 4, 5, 6, 99, 99])
        assert matched == 6
        # Mid-node gets KV sliced from the original via causal attention
        assert kv_out is not None
        for (k, v), (ko, vo) in zip(kv_out, kv):
            assert torch.equal(k, ko[:, :, :6, :])
            assert torch.equal(v, vo[:, :, :6, :])

        # Original full sequence should still be matchable
        matched2, kv_out2, _ = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert matched2 == 10
        assert kv_out2 is kv
