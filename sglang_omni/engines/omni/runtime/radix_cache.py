# SPDX-License-Identifier: Apache-2.0
"""Radix-tree prefix cache for DualAR models in sglang-omni.

Standard LLM radix caches key on a 1-D token sequence. DualAR models
use a multi-row token representation ``[num_codebooks+1, seq_len]`` where
row 0 contains text/semantic tokens and rows 1..N contain codebook values.

This module patches the cache to handle multi-row keys while preserving
the core radix-tree structure. The primary use case is **voice-cloning
prefix reuse**: when multiple TTS requests share the same reference voice,
the KV cache computed during the reference-audio portion of the prompt can
be shared.

Design decisions:
- Cache key is derived from row 0 only (semantic tokens). Codebook values
  at the same position are deterministic given row 0, so matching row 0
  is sufficient for correctness.
- The cache stores slow-transformer KV cache entries only. The fast
  transformer's KV cache is reset every decode step and never cached.
- This is a **patch** on top of the existing ``SimpleCacheManager``.
  When sglang-omni integrates SGLang's full radix cache, this module
  provides the adapter layer.

Usage:
    cache = DualARRadixCache(max_entries=256)
    # During prefill, after computing KV cache:
    cache.insert(prefix_tokens_row0, kv_cache_state)
    # For a new request with the same voice reference:
    matched_len, kv = cache.match(new_request_tokens_row0)
    if matched_len > 0:
        # Skip recomputing the first matched_len tokens
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from ..types import RequestOutput, SchedulerRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Radix tree node
# ---------------------------------------------------------------------------


@dataclass
class _RadixNode:
    """A node in the radix tree.

    Each edge is labelled with a tuple of token IDs (a prefix segment).
    """

    children: dict[int, _RadixEdge] = field(default_factory=dict)
    kv_cache: Any = None
    depth: int = 0
    ref_count: int = 0


@dataclass
class _RadixEdge:
    """An edge connecting two radix nodes, labelled with a token span."""

    tokens: tuple[int, ...]
    child: _RadixNode


# ---------------------------------------------------------------------------
# DualAR Radix Cache
# ---------------------------------------------------------------------------


class DualARRadixCache:
    """Radix-tree prefix cache for DualAR KV cache reuse.

    Keys are 1-D sequences of semantic token IDs (row 0 of the multi-row
    input). Values are opaque KV-cache state objects.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._root = _RadixNode()
        self._max_entries = max_entries
        self._num_entries = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, tokens: torch.Tensor | list[int]) -> tuple[int, Any]:
        """Find the longest prefix match.

        Args:
            tokens: 1-D token sequence (row 0 of DualAR input).

        Returns:
            ``(matched_length, kv_cache_state)`` where ``kv_cache_state``
            is the stored state at the deepest matching node, or ``None``
            if no match.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        node = self._root
        matched = 0
        best_kv = None
        pos = 0

        while pos < len(tokens):
            token = tokens[pos]
            edge = node.children.get(token)
            if edge is None:
                break

            edge_tokens = edge.tokens
            match_len = 0
            for i, et in enumerate(edge_tokens):
                if pos + i >= len(tokens) or tokens[pos + i] != et:
                    break
                match_len += 1

            pos += match_len
            matched = pos

            if match_len == len(edge_tokens):
                node = edge.child
                if node.kv_cache is not None:
                    best_kv = node.kv_cache
                    node.ref_count += 1
            else:
                break

        return matched, best_kv

    def insert(
        self,
        tokens: torch.Tensor | list[int],
        kv_cache: Any,
    ) -> None:
        """Insert a prefix and its associated KV cache state.

        If the prefix already exists, the KV cache is updated in place.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if not tokens:
            return

        self._evict_if_needed()

        node = self._root
        pos = 0

        while pos < len(tokens):
            token = tokens[pos]
            edge = node.children.get(token)

            if edge is None:
                remaining = tuple(tokens[pos:])
                new_node = _RadixNode(
                    depth=node.depth + len(remaining),
                    kv_cache=kv_cache,
                )
                node.children[token] = _RadixEdge(
                    tokens=remaining, child=new_node
                )
                self._num_entries += 1
                return

            edge_tokens = edge.tokens
            match_len = 0
            for i, et in enumerate(edge_tokens):
                if pos + i >= len(tokens) or tokens[pos + i] != et:
                    break
                match_len += 1

            if match_len == len(edge_tokens):
                node = edge.child
                pos += match_len
            else:
                # Split the edge
                shared = edge_tokens[:match_len]
                rest_existing = edge_tokens[match_len:]
                rest_new = tuple(tokens[pos + match_len :])

                mid_node = _RadixNode(depth=node.depth + len(shared))

                mid_node.children[rest_existing[0]] = _RadixEdge(
                    tokens=rest_existing, child=edge.child
                )

                if rest_new:
                    new_node = _RadixNode(
                        depth=mid_node.depth + len(rest_new),
                        kv_cache=kv_cache,
                    )
                    mid_node.children[rest_new[0]] = _RadixEdge(
                        tokens=rest_new, child=new_node
                    )
                    self._num_entries += 1
                else:
                    mid_node.kv_cache = kv_cache

                node.children[token] = _RadixEdge(
                    tokens=shared, child=mid_node
                )
                self._num_entries += 1
                return

        # Exact match — update existing node
        node.kv_cache = kv_cache

    def invalidate(self, tokens: torch.Tensor | list[int]) -> bool:
        """Remove a prefix entry. Returns True if found and removed."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        node = self._root
        pos = 0

        while pos < len(tokens):
            token = tokens[pos]
            edge = node.children.get(token)
            if edge is None:
                return False

            edge_tokens = edge.tokens
            if pos + len(edge_tokens) > len(tokens):
                return False

            for i, et in enumerate(edge_tokens):
                if tokens[pos + i] != et:
                    return False

            pos += len(edge_tokens)
            node = edge.child

        if node.kv_cache is not None:
            node.kv_cache = None
            self._num_entries = max(0, self._num_entries - 1)
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        self._root = _RadixNode()
        self._num_entries = 0

    @property
    def size(self) -> int:
        return self._num_entries

    # ------------------------------------------------------------------
    # CacheManager protocol adapter
    # ------------------------------------------------------------------

    def get(self, request: SchedulerRequest) -> RequestOutput | None:
        """``CacheManager`` protocol: try to hit cache for a request."""
        data = getattr(request, "data", None)
        if data is None:
            return None

        input_values = getattr(data, "input_values", None)
        if input_values is None:
            return None

        row0 = input_values[0] if input_values.dim() == 2 else input_values
        matched_len, kv = self.match(row0)
        if matched_len == 0 or kv is None:
            return None

        return RequestOutput(
            request_id=request.request_id,
            data={"matched_prefix_len": matched_len, "kv_cache": kv},
            finished=False,
        )

    def put(self, request: SchedulerRequest, output: RequestOutput) -> None:
        """``CacheManager`` protocol: store KV cache after prefill."""
        data = getattr(request, "data", None)
        if data is None:
            return

        input_values = getattr(data, "input_values", None)
        kv = getattr(data, "_slow_kv_cache", None)
        if input_values is None or kv is None:
            return

        row0 = input_values[0] if input_values.dim() == 2 else input_values
        self.insert(row0, kv)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Simple LRU-style eviction: remove the least-referenced leaf."""
        if self._num_entries < self._max_entries:
            return

        best_node: _RadixNode | None = None
        best_parent: _RadixNode | None = None
        best_key: int | None = None
        min_ref = float("inf")

        def _walk(node: _RadixNode, parent: _RadixNode | None, key: int | None) -> None:
            nonlocal best_node, best_parent, best_key, min_ref
            if node.kv_cache is not None and not node.children:
                if node.ref_count < min_ref:
                    min_ref = node.ref_count
                    best_node = node
                    best_parent = parent
                    best_key = key
            for k, edge in node.children.items():
                _walk(edge.child, node, k)

        _walk(self._root, None, None)

        if best_node is not None and best_parent is not None and best_key is not None:
            del best_parent.children[best_key]
            self._num_entries = max(0, self._num_entries - 1)
