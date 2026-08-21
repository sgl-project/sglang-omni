# SPDX-License-Identifier: Apache-2.0
"""Bucketed, signature-keyed CUDA-graph cache shared by per-step decode chains.

Every model that graphs a per-token or per-frame chain needs the same
bookkeeping around the capture itself: pick a batch bucket, key the graph by the
host branches the capture pins, keep the key count bounded, never retry a
signature whose capture failed, and give up entirely when captures keep failing.
This holds that logic so each model only supplies its buffers and its chain.

Capture stays with the caller: the buffers, the warmup and the captured region
are model-shaped, and hiding them behind another abstraction would obscure what
a replay actually reproduces.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger(__name__)

_FALSY = frozenset({"0", "false", "off", "no"})

DEFAULT_MAX_KEYS = 32
DEFAULT_MAX_FAILURES = 8


def env_graph_enabled(env_var: str) -> bool:
    """Graphs are on unless the env var is explicitly falsy."""
    return os.environ.get(env_var, "1").strip().lower() not in _FALSY


def normalize_batch_sizes(
    raw_batch_sizes: Iterable[int] | None,
    *,
    max_batch_size: int,
) -> tuple[int, ...]:
    """Sorted, deduped capture buckets bounded by ``max_batch_size``.

    Falls back to the backbone's default capture list and always ends at
    ``max_batch_size`` so a full batch never misses a bucket.
    """
    max_batch_size = int(max_batch_size)
    if raw_batch_sizes is None:
        raw_batch_sizes = (1, 2, 4, *range(8, max_batch_size + 1, 4))
    normalized = sorted(
        {
            int(batch_size)
            for batch_size in raw_batch_sizes
            if 1 <= int(batch_size) <= max_batch_size
        }
    )
    if not normalized or normalized[-1] < max_batch_size:
        normalized.append(max_batch_size)
    return tuple(normalized)


class KeyedGraphCache:
    """Holds captured graphs for one chain, keyed by (bucket, *signature)."""

    def __init__(
        self,
        *,
        name: str,
        batch_sizes: Iterable[int],
        env_var: str | None = None,
        max_keys: int = DEFAULT_MAX_KEYS,
        max_failures: int = DEFAULT_MAX_FAILURES,
        pool_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.name = name
        self.batch_sizes = tuple(sorted({int(b) for b in batch_sizes if int(b) >= 1}))
        self.env_var = env_var
        self.max_keys = int(max_keys)
        self.max_failures = int(max_failures)
        self._graphs: dict[tuple, Any] = {}
        self._graphs_view = MappingProxyType(self._graphs)
        self._disabled_keys: set[tuple] = set()
        self._failure_count = 0
        self._enabled: bool | None = None
        self._pool: Any | None = None
        self._pool_factory = pool_factory

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = self.env_var is None or env_graph_enabled(self.env_var)
        return self._enabled

    @property
    def graphs(self) -> Mapping[tuple, Any]:
        """Captured graphs by key, read-only.

        The proxy is built once: adopters read this on the replay path, and a
        fresh proxy per access shows up there (measured ~100 ns).
        """
        return self._graphs_view

    @property
    def disabled_keys(self) -> frozenset:
        return frozenset(self._disabled_keys)

    def disable(self, reason: str) -> None:
        if self._enabled is not False:
            logger.warning("Disabling %s CUDA graphs: %s", self.name, reason)
        self._enabled = False

    def bucket_for(self, batch_size: int) -> int | None:
        """Smallest capture bucket that fits ``batch_size``."""
        if batch_size < 1:
            return None
        for bucket_size in self.batch_sizes:
            if bucket_size >= batch_size:
                return bucket_size
        return None

    def memory_pool(self):
        """Shared graph pool so per-key captures do not each reserve one.

        Note: (Jiaxin Deng) ``pool_factory`` exists for a caller that routes
        every CUDA call through an injectable seam; the pool must not bypass it.
        """
        if self._pool is None:
            factory = self._pool_factory or torch.cuda.graph_pool_handle
            self._pool = factory()
        return self._pool

    def clear(self) -> None:
        """Drop every captured graph and release the shared pool.

        For a caller that tears its graphs down on failure. Disabled keys and
        the failure count survive, so clearing never revives a failed key.
        """
        self._graphs.clear()
        self._pool = None

    def warmup(
        self,
        keys: Iterable[Any],
        factory: Callable[[Any], Any],
        *,
        precheck: Callable[[], bool] | None = None,
    ) -> int:
        """Capture ``keys`` up front, largest first, and return how many landed.

        Startup capture keeps captures out of live requests, which matters when
        the captured region has side effects (a sampler advancing a shared RNG,
        say). Descending order is for peak reservation, not correctness: the
        smaller captures then reuse the largest graph's freed blocks instead of
        forcing new pool segments (measured ~7% less reserved on a four-graph
        set). Replay stays correct in any capture order. ``precheck`` is polled
        before each capture so a caller can stop on a VRAM headroom rule. A key
        that fails to capture is disabled and the rest still proceed.
        """
        captured = 0
        for key in sorted(dict.fromkeys(keys), reverse=True):
            if key in self._graphs or key in self._disabled_keys:
                continue
            if len(self._graphs) >= self.max_keys:
                logger.warning(
                    "%s: graph key ceiling %d reached during warmup",
                    self.name,
                    self.max_keys,
                )
                break
            if precheck is not None and not precheck():
                break
            if self.get_or_capture(key, lambda k=key: factory(k)) is not None:
                captured += 1
        return captured

    def get_or_capture(self, key: tuple, factory: Callable[[], Any]) -> Any | None:
        """Return the graph for ``key``, capturing on first use.

        Returns ``None`` whenever the caller must run eagerly: cache off, key
        previously failed, key ceiling reached, or this capture just failed.
        """
        if not self.enabled:
            return None
        graph = self._graphs.get(key)
        if graph is not None:
            return graph
        if key in self._disabled_keys:
            return None
        if len(self._graphs) >= self.max_keys:
            return None
        try:
            graph = factory()
        except Exception:
            self._disabled_keys.add(key)
            self._failure_count += 1
            logger.warning(
                "Disabling %s CUDA graph for key=%s", self.name, key, exc_info=True
            )
            if self._failure_count >= self.max_failures:
                self.disable(f"{self._failure_count} capture failures")
            return None
        self._graphs[key] = graph
        logger.info("Captured %s CUDA graph for key=%s", self.name, key)
        return graph
