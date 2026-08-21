# SPDX-License-Identifier: Apache-2.0
"""Shared CUDA-graph plumbing for per-step decode chains."""

from sglang_omni.cuda_graph.keyed_graph_cache import (
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_KEYS,
    KeyedGraphCache,
    env_graph_enabled,
    normalize_batch_sizes,
)
from sglang_omni.cuda_graph.persistent_state import (
    PersistentStateError,
    PersistentStateRegistry,
)

__all__ = [
    "PersistentStateError",
    "PersistentStateRegistry",
    "DEFAULT_MAX_FAILURES",
    "DEFAULT_MAX_KEYS",
    "KeyedGraphCache",
    "env_graph_enabled",
    "normalize_batch_sizes",
]
