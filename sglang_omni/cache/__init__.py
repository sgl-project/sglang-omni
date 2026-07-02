# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.cache.local import LocalCachePlane
from sglang_omni.cache.schema import (
    ArtifactHandle,
    CacheAffinity,
    CacheEntry,
    CacheKey,
    CacheLease,
    CacheMeta,
    CacheOwner,
    CacheSelector,
    CacheStats,
)

_GLOBAL_CACHE_PLANE = LocalCachePlane()


def get_global_cache_plane() -> LocalCachePlane:
    return _GLOBAL_CACHE_PLANE


def reset_global_cache_plane() -> None:
    _GLOBAL_CACHE_PLANE.clear()


__all__ = [
    "ArtifactHandle",
    "CacheAffinity",
    "CacheEntry",
    "CacheKey",
    "CacheLease",
    "CacheMeta",
    "CacheOwner",
    "CacheSelector",
    "CacheStats",
    "LocalCachePlane",
    "get_global_cache_plane",
    "reset_global_cache_plane",
]
