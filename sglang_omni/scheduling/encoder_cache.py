# SPDX-License-Identifier: Apache-2.0
"""Optional encoder output storage; the default remains the local stage LRU."""

from __future__ import annotations

import json
from typing import Any

from sglang_omni.scheduling.stage_cache import StageOutputCache


def create_encoder_output_cache(
    *,
    model_path: str,
    stage: str,
    dtype: str | None,
    max_entries: int,
    max_bytes: int,
    lmcache_config_file: str | None = None,
    lmcache_namespace: str | None = None,
) -> Any:
    """Build the optional LMCache EC adapter for a complete encoder result.

    The namespace must identify immutable weights and preprocessing settings.
    Stage, model path and requested dtype are additionally isolated here.
    LMCache imports only when explicitly enabled. The caller owns ``close()``
    on the optional adapter and connects it to its scheduler shutdown callback.
    """
    if lmcache_config_file is None:
        if lmcache_namespace is not None:
            raise ValueError("lmcache_namespace requires lmcache_config_file")
        return StageOutputCache(
            max_size=max_entries, max_bytes=max_bytes, cache_device="cpu"
        )
    if not lmcache_namespace or not lmcache_namespace.strip():
        raise ValueError(
            "LMCache encoder caching requires lmcache_namespace identifying "
            "the immutable model revision, dtype and preprocessing configuration"
        )
    from lmcache.integration.sglang_omni.encoder_cache import create_encoder_cache

    namespace = json.dumps(
        ["qwen3-omni", model_path, stage, dtype, lmcache_namespace],
        separators=(",", ":"),
    )
    return create_encoder_cache(lmcache_config_file, namespace)
