# SPDX-License-Identifier: Apache-2.0
"""Lifecycle checks for the process-local SGLang decoder runtime."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang_omni.models.llada2_uni.components import decoder_runtime as runtime


def collective_worker(rank, rendezvous):
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    group = dist.group.WORLD
    parallel_state = SimpleNamespace(
        model_parallel_is_initialized=lambda: True,
        get_world_size=dist.get_world_size,
        get_tp_world_size=lambda: 1,
        get_sp_world_size=dist.get_world_size,
        get_sp_parallel_rank=dist.get_rank,
        get_ulysses_parallel_world_size=lambda: 2,
        get_ring_parallel_world_size=lambda: 1,
        get_sp_group=lambda: SimpleNamespace(device_group=group, cpu_group=group),
    )
    handle = runtime.DecoderRuntimeHandle(
        torch.device("cpu"),
        torch.float32,
        "torch_sdpa",
        parallel_state,
        None,
        None,
        SimpleNamespace(get_compute_dtype=lambda: torch.float32),
        sp_rank=rank,
        sp_size=2,
        ulysses_degree=2,
    )
    try:
        metadata = (4, 8, "normal", 3, 2)
        seed = handle.request_seed(metadata, None)
        noise = torch.randn(8, generator=torch.Generator().manual_seed(seed))
        peers = [torch.empty_like(noise) for _ in range(2)]
        dist.all_gather(peers, noise)
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        assert handle.request_seed(metadata, 17) == 17
        features = torch.full((4, 16), 7.0 if rank == 0 else -1.0)
        torch.testing.assert_close(
            handle.broadcast_features(features), torch.full_like(features, 7.0)
        )
        with pytest.raises(ValueError, match="inconsistent request settings"):
            handle.request_seed(metadata, rank)
        with pytest.raises(RuntimeError, match="conditioning failed across ranks"):
            with handle.preparation("conditioning"):
                if rank == 1:
                    raise ValueError("bad conditioning")
        assert handle.request_seed(metadata, 23) == 23
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_sp_collective_request_contract(tmp_path):
    mp.spawn(
        collective_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.fixture
def owned_runtime():
    events: list[str] = []
    tls = threading.local()

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
    handle._world_started = True
    handle._model_started = True
    handle._published = True
    return handle, parallel_state, tls, events


def test_compute_context_is_thread_local_and_restored(owned_runtime):
    handle, _, tls, events = owned_runtime
    main_state = object()
    tls.state = main_state

    def scheduler_work():
        assert not hasattr(tls, "state")
        with handle.compute_context():
            assert tls.state.param_dtype == torch.bfloat16
        assert not hasattr(tls, "state")
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
        {"sp_size": 2, "ulysses_degree": 2, "stage_role": "leader"},
        {"sp_size": 2, "stage_role": "leader", "nccl_port": 23456},
        {"sp_rank": 1},
        {"stage_role": "follower"},
        {"nccl_port": 0},
        {
            "sp_size": 2,
            "ring_degree": 2,
            "stage_role": "leader",
            "nccl_port": 23456,
            "attention_backend": "torch_sdpa",
        },
    ],
)
def test_invalid_initialization_settings(kwargs):
    with pytest.raises(ValueError):
        runtime.initialize_decoder_runtime("unused", **kwargs)


def test_sp_topology_validation(owned_runtime, monkeypatch):
    handle, ps, _, _ = owned_runtime
    handle.sp_size = handle.ulysses_degree = 2
    device_group, cpu_group = object(), object()
    ps.model_parallel_is_initialized = lambda: True
    ps.get_world_size = ps.get_sp_world_size = ps.get_ulysses_parallel_world_size = (
        lambda: 2
    )
    ps.get_tp_world_size = ps.get_ring_parallel_world_size = lambda: 1
    ps.get_sp_parallel_rank = lambda: 0
    ps.get_sp_group = lambda: SimpleNamespace(
        device_group=device_group, cpu_group=cpu_group
    )
    handle._precision.get_compute_dtype = lambda: torch.bfloat16
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    handle.validate()
    assert handle.group is device_group and handle.cpu_group is cpu_group
    ps.get_sp_parallel_rank = lambda: 1
    with pytest.raises(RuntimeError, match="topology"):
        handle.validate()


@pytest.mark.parametrize("rank", [0, 1])
def test_sp_requests_share_seed_and_reject_mismatched_settings(
    owned_runtime, monkeypatch, rank
):
    handle, _, _, _ = owned_runtime
    handle.sp_size, handle.sp_rank = 2, rank
    handle.cpu_group = object()
    monkeypatch.setattr(handle, "validate", lambda: None)
    mismatch = False

    def gather(output, value, group):
        assert group is handle.cpu_group
        output[:] = [value, ((9, 9, "normal", 50, 2), 7) if mismatch else value]

    def broadcast(values, src, group):
        assert group is handle.cpu_group and src == 0
        values[0] = 123

    monkeypatch.setattr(runtime.dist, "all_gather_object", gather)
    monkeypatch.setattr(runtime.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(runtime.dist, "get_global_rank", lambda group, rank: rank)
    assert handle.request_seed((4, 4, "normal", 50, 2), None) == 123
    mismatch = True
    with pytest.raises(ValueError, match="inconsistent"):
        handle.request_seed((4, 4, "normal", 50, 2), 7)


def test_sp_preparation_reports_peer_failure(owned_runtime, monkeypatch):
    handle, _, _, _ = owned_runtime
    handle.sp_size = 2

    def gather(output, value, group):
        output[:] = [value, "ValueError: peer load failed"]

    monkeypatch.setattr(runtime.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="peer load failed"):
        with handle.preparation("weight loading"):
            pass


def test_sp1_preparation_preserves_original_exception(owned_runtime):
    handle, _, _, _ = owned_runtime
    original = ValueError("load failed")
    with pytest.raises(ValueError) as error:
        with handle.preparation("weight loading"):
            raise original
    assert error.value is original
