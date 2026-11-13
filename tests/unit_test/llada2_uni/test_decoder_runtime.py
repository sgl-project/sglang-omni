# SPDX-License-Identifier: Apache-2.0
"""Lifecycle checks for the process-local SGLang decoder runtime."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.llada2_uni.components import decoder_runtime as runtime


@pytest.fixture
def owned_runtime():
    events: list[str] = []

    class PrecisionState(threading.local):
        state = None

    tls = PrecisionState()

    def set_policy(**kwargs):
        tls.state = SimpleNamespace(**kwargs)

    parallel_state = SimpleNamespace(
        destroy_model_parallel=lambda: events.append("model"),
        destroy_distributed_environment=lambda: events.append("world"),
    )
    server_args = SimpleNamespace(
        set_global_server_args=lambda value: events.append("args")
    )
    context = SimpleNamespace(reset_context=lambda: events.append("context"))
    precision = SimpleNamespace(
        _mixed_precision_state=tls,
        set_mixed_precision_policy=set_policy,
    )
    handle = runtime.DecoderRuntimeHandle(
        torch.device("cpu"),
        torch.bfloat16,
        "torch_sdpa",
        parallel_state,
        server_args,
        context,
        precision,
    )
    handle.world_started = True
    handle.model_started = True
    handle.published = True
    return handle, parallel_state, tls, events


def test_compute_context_is_thread_local_and_restored(owned_runtime):
    handle, _, tls, events = owned_runtime
    main_state = object()
    tls.state = main_state

    def scheduler_work():
        assert tls.state is None
        with handle.compute_context():
            assert tls.state.param_dtype == torch.bfloat16
        assert tls.state is None
        handle.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(scheduler_work).result()

    assert tls.state is main_state
    handle.close()
    assert events == ["model", "world", "args", "context"]


def test_compute_context_restores_state_after_failure(owned_runtime):
    handle, _, tls, _ = owned_runtime
    previous = object()
    tls.state = previous
    with pytest.raises(ValueError, match="compute failure"), handle.compute_context():
        raise ValueError("compute failure")
    assert tls.state is previous


def test_close_attempts_all_owned_cleanup(owned_runtime):
    handle, parallel_state, _, events = owned_runtime

    def fail():
        events.append("model")
        raise RuntimeError("model cleanup error")

    parallel_state.destroy_model_parallel = fail
    with pytest.raises(RuntimeError, match="model cleanup error"):
        handle.close()
    assert events == ["model", "world", "args", "context"]


def test_initialization_rejects_existing_distributed_runtime(monkeypatch):
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="dedicated process"):
        runtime.initialize_decoder_runtime("unused")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_id": -1},
        {"dist_timeout": 0},
        {"dtype": torch.int32},
        {"attention_backend": "auto"},
    ],
)
def test_invalid_initialization_settings(kwargs):
    with pytest.raises(ValueError):
        runtime.initialize_decoder_runtime("unused", **kwargs)


@pytest.mark.parametrize(
    "failure_point", ["_sync_srt_world_group", "_sync_srt_tp_group"]
)
def test_partial_initialization_releases_groups(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    parallel_state = pytest.importorskip(
        "sglang.multimodal_gen.runtime.distributed.parallel_state"
    )
    from sglang.multimodal_gen.runtime.server_args import server_args
    from sglang.srt import runtime_context

    error = RuntimeError("initialization failed after group creation")
    monkeypatch.setattr(parallel_state, failure_point, Mock(side_effect=error))
    try:
        with pytest.raises(RuntimeError) as raised:
            runtime.initialize_decoder_runtime("unused", gpu_id=None)
        assert raised.value is error
        assert not runtime.dist.is_initialized()
        assert not parallel_state.world_group_is_initialized()
        assert not parallel_state.model_parallel_is_initialized()
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        server_args.set_global_server_args(None)
        runtime_context.reset_context()
