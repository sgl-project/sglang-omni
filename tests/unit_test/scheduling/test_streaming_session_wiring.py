# SPDX-License-Identifier: Apache-2.0
"""Session-aware cache and controller identity contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import StreamingSession

from sglang_omni.scheduling import bootstrap
from sglang_omni.scheduling import sglang_backend as backend_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import cache as cache_module


def _server_args(
    *,
    enable_streaming_session: bool,
    disable_radix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        attention_backend="flashinfer",
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
        enable_streaming_session=enable_streaming_session,
        disable_radix_cache=disable_radix_cache,
        disable_overlap_schedule=True,
        page_size=1,
        chunked_prefill_size=0,
        max_prefill_tokens=32,
    )


def _compat_server_args(*, enable_streaming_session: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enable_streaming_session=enable_streaming_session,
        enable_hisparse=False,
        enable_priority_scheduling=False,
        disable_priority_preemption=False,
    )


def _compat_scheduler(tree_cache: Any) -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler.tree_cache = tree_cache
    scheduler.device = torch.device("cpu")
    return scheduler


def test_tree_cache_disabled_keeps_raw_radix_cache() -> None:
    tree_cache = cache_module.create_tree_cache(
        _server_args(enable_streaming_session=False),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
    )

    assert isinstance(tree_cache, RadixCache)
    assert not isinstance(tree_cache, StreamingSession)


@pytest.mark.parametrize(
    ("disable_radix_cache", "inner_type"),
    ((False, RadixCache), (True, ChunkCache)),
)
def test_tree_cache_enabled_wraps_radix_and_chunk_once(
    disable_radix_cache: bool,
    inner_type: type,
) -> None:
    tree_cache = cache_module.create_tree_cache(
        _server_args(
            enable_streaming_session=True,
            disable_radix_cache=disable_radix_cache,
        ),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
    )

    assert isinstance(tree_cache, StreamingSession)
    assert isinstance(tree_cache.inner, inner_type)
    assert not isinstance(tree_cache.inner, StreamingSession)
    assert tree_cache.supports_streaming_session() is True


def test_tree_cache_does_not_wrap_native_streaming_cache(monkeypatch) -> None:
    class NativeStreamingCache:
        def __init__(self, params: Any) -> None:
            self.params = params

        @staticmethod
        def supports_streaming_session() -> bool:
            return True

    # The lru default selects EvictHeapRadixCache; a cache that natively
    # supports streaming sessions must not be wrapped in StreamingSession.
    monkeypatch.setattr(cache_module, "EvictHeapRadixCache", NativeStreamingCache)

    tree_cache = cache_module.create_tree_cache(
        _server_args(enable_streaming_session=True),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        page_size=1,
    )

    assert isinstance(tree_cache, NativeStreamingCache)


def test_infrastructure_returns_session_wrapped_cache(monkeypatch) -> None:
    from sglang_omni.model_runner import model_worker as model_worker_module

    class FakeWorker:
        def __init__(
            self,
            *,
            config: Any,
            server_args: Any,
            gpu_id: int,
            tp_rank: int,
        ) -> None:
            del config, server_args
            self.gpu_id = gpu_id
            self.tp_rank = tp_rank
            self.model_runner = SimpleNamespace(
                model=object(),
                alloc_memory_pool=lambda: None,
                init_attention_backends=lambda: None,
                init_cuda_graphs=lambda: None,
            )
            self.model_config = object()
            self.enable_prefill_input_embeds = False

        @staticmethod
        def get_memory_pool() -> tuple[None, None]:
            return None, None

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(
        model_worker_module,
        "ModelWorkerConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    infrastructure = bootstrap.create_sglang_infrastructure(
        _server_args(enable_streaming_session=True),
        gpu_id=0,
    )

    # The prefill/decode managers the 7-tuple used to expose are gone: the
    # scheduler keeps the wrapped cache through its own session controller
    # (covered by test_enabled_scheduler_builds_real_controller_on_same_cache).
    model_worker, tree_cache, req_pool, kv_pool, model_config = infrastructure
    assert isinstance(tree_cache, StreamingSession)
    assert req_pool is None
    assert kv_pool is None
    assert model_config is model_worker.model_config


def test_enabled_scheduler_builds_real_controller_on_same_cache() -> None:
    tree_cache = StreamingSession(RadixCache.create_simulated())
    scheduler = _compat_scheduler(tree_cache)

    scheduler._init_upstream_compat_flags(
        _compat_server_args(enable_streaming_session=True)
    )

    assert isinstance(scheduler.session_controller, SessionController)
    assert scheduler.session_controller.tree_cache is tree_cache


def test_disabled_scheduler_builds_controller_on_raw_cache() -> None:
    tree_cache = RadixCache.create_simulated()
    scheduler = _compat_scheduler(tree_cache)

    scheduler._init_upstream_compat_flags(
        _compat_server_args(enable_streaming_session=False)
    )

    assert isinstance(scheduler.session_controller, SessionController)
    assert scheduler.session_controller.tree_cache is tree_cache


def test_backend_export_still_points_to_session_aware_factory() -> None:
    assert backend_module.create_tree_cache is cache_module.create_tree_cache
