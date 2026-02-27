# SPDX-License-Identifier: Apache-2.0
"""Radix-tree prefix cache for DualAR models in sglang-omni.

Usage:
    cache = DualARRadixCache(max_tokens=50000)
    # During prefill, after computing KV cache:
    cache.insert(prefix_tokens_row0, kv_data)
    # For a new request with the same voice reference:
    matched_len, kv_data, node = cache.match_prefix(new_request_tokens_row0)
    if matched_len > 0:
        cache.inc_lock_ref(node)  # protect from eviction while in use
        # restore KV, trim input, run suffix-only prefill
        ...
        cache.dec_lock_ref(node)  # release after request completes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the radix tree.

    Each node represents a position in the token sequence. The ``key``
    field holds the token IDs on the edge from parent to this node.
    ``kv_data`` stores cloned per-layer (k, v) tensor pairs when this
    node is an insertion point.
    """

    children: dict[int, TreeNode] = field(default_factory=dict)
    parent: TreeNode | None = None
    key: tuple[int, ...] = ()
    kv_data: list[tuple[Tensor, Tensor]] | None = None
    depth: int = 0
    lock_ref: int = 0
    last_access_time: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def num_tokens(self) -> int:
        """Number of tokens stored on the edge to this node."""
        return len(self.key)


# ---------------------------------------------------------------------------
# DualAR Radix Cache
# ---------------------------------------------------------------------------


class DualARRadixCache:
    """Radix-tree prefix cache for DualAR KV cache reuse.

    Keys are 1-D sequences of semantic token IDs (row 0 of the multi-row
    input). Values are per-layer ``(k_clone, v_clone)`` tensor pairs.

    Follows upstream SGLang patterns:
    - Edge splitting on insert when a new key diverges mid-edge
    - Lock reference counting (inc/dec walk root-to-leaf path)
    - Leaf-only LRU eviction (only evict leaves with lock_ref == 0)
    """

    def __init__(self, max_tokens: int = 50000) -> None:
        self._root = TreeNode()
        self._max_tokens = max_tokens
        self._total_tokens = 0

        # Observability counters
        self._num_matches = 0  # match_prefix calls that returned kv_data
        self._num_misses = 0  # match_prefix calls that returned None
        self._total_matched_tokens = 0  # sum of matched prefix lengths
        self._total_query_tokens = 0  # sum of query lengths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_prefix(
        self, tokens: list[int] | torch.Tensor
    ) -> tuple[int, list[tuple[Tensor, Tensor]] | None, TreeNode]:
        """Find the longest cached prefix match.

        Args:
            tokens: 1-D token sequence (row 0 of DualAR input).

        Returns:
            ``(matched_length, kv_data_or_None, last_matched_node)``
            where ``kv_data`` is the stored KV data at the deepest
            matching node that has cached data, or ``None`` if no match.
            ``last_matched_node`` is the deepest node reached (for lock
            management).
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        node = self._root
        best_matched = 0
        best_kv: list[tuple[Tensor, Tensor]] | None = None
        best_node = self._root
        pos = 0

        while pos < len(tokens):
            token = tokens[pos]
            child = node.children.get(token)
            if child is None:
                break

            edge_tokens = child.key
            match_len = 0
            for i, et in enumerate(edge_tokens):
                if pos + i >= len(tokens) or tokens[pos + i] != et:
                    break
                match_len += 1

            if match_len == 0:
                break

            if match_len == len(edge_tokens):
                # Full edge match — descend
                node = child
                pos += match_len
                node.last_access_time = time.monotonic()
                if node.kv_data is not None:
                    best_kv = node.kv_data
                    best_node = node
                    best_matched = pos
            else:
                # Partial edge match — split to create a node boundary
                # so KV data (propagated by _split_node) can be returned.
                mid = self._split_node(child, match_len)
                pos += match_len
                mid.last_access_time = time.monotonic()
                if mid.kv_data is not None:
                    best_kv = mid.kv_data
                    best_node = mid
                    best_matched = pos
                break

        # Update stats
        self._total_query_tokens += len(tokens)
        if best_kv is not None:
            self._num_matches += 1
            self._total_matched_tokens += best_matched
            logger.info(
                "Cache HIT: matched %d/%d tokens (%.1f%%)",
                best_matched,
                len(tokens),
                100.0 * best_matched / len(tokens) if tokens else 0,
            )
        else:
            self._num_misses += 1
            logger.debug("Cache MISS: no prefix match for %d tokens", len(tokens))

        return best_matched, best_kv, best_node

    def insert(
        self,
        tokens: list[int] | torch.Tensor,
        kv_data: list[tuple[Tensor, Tensor]],
    ) -> int:
        """Insert a prefix and its associated KV cache data.

        If the prefix (or a prefix of it) is already cached, only the
        new suffix is added. Returns the length of the already-cached
        prefix (for dedup awareness).

        Args:
            tokens: 1-D token sequence (row 0).
            kv_data: Per-layer ``(k_clone, v_clone)`` tensor pairs.

        Returns:
            Length of the prefix that was already cached.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if not tokens:
            return 0

        node = self._root
        pos = 0
        already_cached = 0

        while pos < len(tokens):
            token = tokens[pos]
            child = node.children.get(token)

            if child is None:
                remaining = tuple(tokens[pos:])
                new_tokens = len(remaining)
                if not self._evict_to_fit(new_tokens):
                    return already_cached  # cache full, refuse insert

                new_node = TreeNode(
                    parent=node,
                    key=remaining,
                    kv_data=kv_data,
                    depth=node.depth + new_tokens,
                    last_access_time=time.monotonic(),
                )
                node.children[token] = new_node
                self._total_tokens += new_tokens
                return already_cached

            edge_tokens = child.key
            match_len = 0
            for i, et in enumerate(edge_tokens):
                if pos + i >= len(tokens) or tokens[pos + i] != et:
                    break
                match_len += 1

            if match_len == len(edge_tokens):
                # Full edge match — descend
                node = child
                pos += match_len
                if node.kv_data is not None:
                    already_cached = pos
            else:
                # Partial match — split and insert remainder
                mid_node = self._split_node(child, match_len)
                pos += match_len

                rest_new = tuple(tokens[pos:])
                if rest_new:
                    new_tokens = len(rest_new)
                    if not self._evict_to_fit(new_tokens):
                        return already_cached  # cache full, refuse insert

                    new_node = TreeNode(
                        parent=mid_node,
                        key=rest_new,
                        kv_data=kv_data,
                        depth=mid_node.depth + new_tokens,
                        last_access_time=time.monotonic(),
                    )
                    mid_node.children[rest_new[0]] = new_node
                    self._total_tokens += new_tokens
                else:
                    # Exact match at split point — store data here
                    mid_node.kv_data = kv_data
                    mid_node.last_access_time = time.monotonic()
                return already_cached

        # Exact match of entire sequence — update existing node
        node.kv_data = kv_data
        node.last_access_time = time.monotonic()
        already_cached = pos
        return already_cached

    def inc_lock_ref(self, node: TreeNode) -> None:
        """Lock path from node to root (protects from eviction).

        Following upstream SGLang: walk from the given node up to root,
        incrementing ``lock_ref`` on each node.
        """
        current: TreeNode | None = node
        while current is not None and current is not self._root:
            current.lock_ref += 1
            current = current.parent

    def dec_lock_ref(self, node: TreeNode) -> None:
        """Unlock path from node to root.

        Decrements ``lock_ref`` on each node from the given node to root.
        Nodes that reach ``lock_ref == 0`` become eligible for eviction.
        """
        current: TreeNode | None = node
        while current is not None and current is not self._root:
            current.lock_ref = max(0, current.lock_ref - 1)
            current = current.parent

    def evict(self, num_tokens: int) -> int:
        """Evict LRU leaves to free at least ``num_tokens`` of capacity.

        Only evicts leaf nodes with ``lock_ref == 0``. Following upstream
        SGLang: after evicting a leaf, its parent may become a new leaf
        eligible for eviction.

        Returns:
            Number of tokens actually evicted.
        """
        evicted = 0
        while evicted < num_tokens:
            leaf = self._find_lru_leaf()
            if leaf is None:
                break  # No more evictable leaves
            evicted += self._remove_leaf(leaf)
        return evicted

    def clear(self) -> None:
        """Remove all cached entries."""
        self._root = TreeNode()
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        """Total number of tokens stored across all edges."""
        return self._total_tokens

    def stats(self) -> dict:
        """Return cache statistics following SGLang conventions.

        Returns a dict with:
        - ``num_matches``: number of match_prefix calls that returned KV data
        - ``num_misses``: number of match_prefix calls that returned None
        - ``hit_rate``: matches / (matches + misses), or 0.0 if no queries
        - ``token_hit_rate``: matched_tokens / query_tokens, or 0.0
        - ``total_matched_tokens``: cumulative matched prefix tokens
        - ``total_query_tokens``: cumulative queried tokens
        - ``total_cached_tokens``: tokens currently stored in the tree
        - ``num_entries``: number of leaf nodes (cached sequences)
        """
        total = self._num_matches + self._num_misses
        return {
            "num_matches": self._num_matches,
            "num_misses": self._num_misses,
            "hit_rate": self._num_matches / total if total > 0 else 0.0,
            "token_hit_rate": (
                self._total_matched_tokens / self._total_query_tokens
                if self._total_query_tokens > 0
                else 0.0
            ),
            "total_matched_tokens": self._total_matched_tokens,
            "total_query_tokens": self._total_query_tokens,
            "total_cached_tokens": self._total_tokens,
            "num_entries": self._count_leaves(),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters (does not clear cached data)."""
        self._num_matches = 0
        self._num_misses = 0
        self._total_matched_tokens = 0
        self._total_query_tokens = 0

    def _count_leaves(self) -> int:
        """Count leaf nodes with KV data."""
        count = 0

        def _walk(node: TreeNode) -> None:
            nonlocal count
            if node.kv_data is not None:
                count += 1
            for child in node.children.values():
                _walk(child)

        _walk(self._root)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_node(self, node: TreeNode, split_pos: int) -> TreeNode:
        """Split a node's edge at ``split_pos``.

        Given a node with edge key ``[a, b, c, d, e]`` and split_pos=2,
        creates a new mid-node with key ``[a, b]`` and pushes the
        original node down with key ``[c, d, e]``.

        Returns the new mid-node.
        """
        assert (
            0 < split_pos < len(node.key)
        ), f"split_pos={split_pos} out of range for key length {len(node.key)}"

        parent = node.parent
        assert parent is not None

        prefix_key = node.key[:split_pos]
        suffix_key = node.key[split_pos:]

        mid_node = TreeNode(
            parent=parent,
            key=prefix_key,
            depth=parent.depth + len(prefix_key),
            lock_ref=node.lock_ref,
            last_access_time=time.monotonic(),
        )

        # Derive prefix KV from the original node's full KV.
        # Causal attention guarantees kv[..., :prefix_len, :] is
        # identical to a fresh prefill of just the prefix tokens.
        if node.kv_data is not None:
            d = mid_node.depth
            mid_node.kv_data = [
                (k[:, :, :d, :].clone(), v[:, :, :d, :].clone())
                for k, v in node.kv_data
            ]

        node.key = suffix_key
        node.parent = mid_node
        mid_node.children[suffix_key[0]] = node

        # Re-link parent
        parent.children[prefix_key[0]] = mid_node

        return mid_node

    def _find_lru_leaf(self) -> TreeNode | None:
        """Find the least-recently-used evictable leaf node."""
        best: TreeNode | None = None
        best_time = float("inf")

        def _walk(node: TreeNode) -> None:
            nonlocal best, best_time
            if node.is_leaf and node is not self._root and node.lock_ref == 0:
                if node.last_access_time < best_time:
                    best_time = node.last_access_time
                    best = node
            for child in node.children.values():
                _walk(child)

        _walk(self._root)
        return best

    def _remove_leaf(self, leaf: TreeNode) -> int:
        """Remove a leaf node and return the number of tokens freed."""
        assert leaf.is_leaf
        assert leaf.parent is not None

        tokens_freed = leaf.num_tokens
        parent = leaf.parent

        first_token = leaf.key[0]
        if first_token in parent.children and parent.children[first_token] is leaf:
            del parent.children[first_token]

        leaf.kv_data = None
        leaf.parent = None

        self._total_tokens -= tokens_freed
        return tokens_freed

    def _evict_to_fit(self, needed_tokens: int) -> bool:
        """Evict leaves until we have room for ``needed_tokens``.

        Returns True if enough space was freed, False if the cache is full
        and all leaves are locked (insert should be refused).
        """
        while self._total_tokens + needed_tokens > self._max_tokens:
            leaf = self._find_lru_leaf()
            if leaf is None:
                logger.warning(
                    "Cannot evict: need %d tokens but all leaves are locked "
                    "(total=%d, max=%d)",
                    needed_tokens, self._total_tokens, self._max_tokens,
                )
                return False
            self._remove_leaf(leaf)
        return True
