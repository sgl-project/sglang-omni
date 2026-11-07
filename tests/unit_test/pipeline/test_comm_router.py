# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.comm import router as router_module
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter


@pytest.fixture(autouse=True)
def force_cuda_transport_policy(monkeypatch):
    """Transport selection is platform policy, so these tests pin CUDA to assert the
    cuda_ipc contract on any host. XPU has its own cases below.
    """
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
        raising=False,
    )
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )


def test_comm_router_uses_cuda_ipc_for_same_node_gpu_payload_edges() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets={"local"},
        gpu_stage_names={"decode"},
        comm_config={},
    )

    assert router.outbound("local") is TransportKind.LOCAL_OBJECT
    assert router.outbound("decode") is TransportKind.CUDA_IPC
    assert router.outbound_stream("decode", torch.empty(1)) is TransportKind.SHM


def test_comm_router_uses_mooncake_only_for_remote_edges() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"decode"},
        remote_stage_names={"remote_decode"},
        comm_config={},
    )

    assert router.outbound("decode") is TransportKind.CUDA_IPC
    assert router.outbound("cpu_decode") is TransportKind.SHM
    assert router.outbound("remote_decode") is TransportKind.MOONCAKE
    assert router.outbound_stream("remote_decode", torch.empty(1)) is (
        TransportKind.MOONCAKE
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_comm_router_uses_cuda_ipc_for_mixed_gpu_payloads() -> None:
    router = CommRouter(
        stage_name="mm_aggregate",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"thinker"},
        comm_config={},
    )
    payload = {
        "hidden": torch.empty(1, device="cuda:0"),
        "lengths": torch.empty(1),
    }

    assert router.outbound_payload("thinker", payload) is TransportKind.CUDA_IPC


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_comm_router_uses_shm_for_cuda_payloads_to_cpu_targets() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names=set(),
        comm_config={},
    )
    payload = {"token_ids": torch.empty(1, device="cuda:0")}

    assert router.outbound_payload("decode", payload) is TransportKind.SHM


ACCEL = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "meta"


def pin_non_cuda_platform(monkeypatch) -> None:
    """Pin every platform attribute the router reads, so these cases hold on a CUDA
    host too. Patching device_type alone left is_cuda_alike() reporting the host.
    """
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.SHM,
        raising=False,
    )
    monkeypatch.setattr(platforms.current_platform, "device_type", ACCEL, raising=False)
    monkeypatch.setattr(
        platforms.current_platform, "is_cuda_alike", lambda: False, raising=False
    )


def xpu_router(monkeypatch, **kwargs) -> CommRouter:
    pin_non_cuda_platform(monkeypatch)
    return CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"decode"},
        comm_config={},
        **kwargs,
    )


def test_comm_router_never_selects_cuda_ipc_without_platform_support(
    monkeypatch,
) -> None:
    """One policy: every decision point must agree. Previously outbound/inbound and
    the direct-IPC namespace still chose cuda_ipc on a platform without it.
    """
    router = xpu_router(monkeypatch, stage_gpu_ids={"decode": (0,)})

    assert router.outbound("decode") is TransportKind.SHM
    assert router.physical_outbound("decode") is TransportKind.SHM
    assert router.inbound("decode") is TransportKind.SHM
    assert not router.can_use_direct_cuda_ipc("decode")


def test_comm_router_uses_shm_for_accelerator_payloads_and_streams(
    monkeypatch,
) -> None:
    router = xpu_router(monkeypatch)
    payload = {"hidden": torch.empty(1, device=ACCEL), "lengths": torch.empty(1)}

    assert router.outbound_payload("decode", payload) is TransportKind.SHM
    assert (
        router.outbound_stream("decode", torch.empty(1, device=ACCEL))
        is TransportKind.SHM
    )


def test_comm_router_still_routes_remote_edges_over_mooncake(monkeypatch) -> None:
    """Losing cuda_ipc must not swallow the cross-node decision."""
    router = xpu_router(monkeypatch, remote_stage_names={"remote_decode"})

    assert router.outbound("remote_decode") is TransportKind.MOONCAKE
    assert router.inbound("remote_decode") is TransportKind.MOONCAKE


def test_a_mooncake_relay_is_refused_on_a_non_cuda_accelerator(monkeypatch) -> None:
    """Selecting the transport is a topology decision, but the relay itself is built
    on a literal cuda device, so an accelerator edge must fail loudly here instead of
    allocating on CUDA at the first transfer.
    """
    router = xpu_router(monkeypatch, remote_stage_names={"remote_decode"})

    with pytest.raises(NotImplementedError, match="not supported yet"):
        router.relay(TransportKind.MOONCAKE)


def test_a_host_only_stage_still_gets_a_mooncake_relay(monkeypatch) -> None:
    """A stage with no card uses the cpu path, which is unaffected."""
    pin_non_cuda_platform(monkeypatch)
    relay_kwargs: dict[str, object] = {}
    built_relay = object()

    def record_relay(kind: str, **kwargs: object) -> object:
        relay_kwargs.update(kwargs, kind=kind)
        return built_relay

    monkeypatch.setattr(router_module, "create_relay", record_relay)
    router = CommRouter(
        stage_name="preprocessing",
        gpu_id=None,
        same_process_targets=set(),
        gpu_stage_names=set(),
        remote_stage_names={"remote_decode"},
    )

    assert router.relay(TransportKind.MOONCAKE) is built_relay
    assert relay_kwargs["kind"] == "mooncake"
    assert relay_kwargs["device"] == "cpu"


def test_comm_router_admits_compatible_direct_cuda_ipc_namespace() -> None:
    router = CommRouter(
        stage_name="talker_ar",
        gpu_id=1,
        placement_gpu_id=1,
        same_process_targets={"local_code2wav"},
        gpu_stage_names={"same_code2wav", "cross_code2wav", "tp_decode"},
        stage_gpu_ids={
            "same_code2wav": (1,),
            "cross_code2wav": (0,),
            "tp_decode": (0, 1),
            "local_code2wav": (1,),
        },
        comm_config={},
    )

    assert router.can_use_direct_cuda_ipc("same_code2wav")
    assert not router.can_use_direct_cuda_ipc("cross_code2wav")
    assert not router.can_use_direct_cuda_ipc("tp_decode")
    assert not router.can_use_direct_cuda_ipc("local_code2wav")


def test_comm_router_rejects_narrowed_direct_cuda_ipc_namespace() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        placement_gpu_id=2,
        same_process_targets=set(),
        gpu_stage_names={"talker"},
        stage_gpu_ids={"talker": (2,)},
        comm_config={},
    )

    assert not router.can_use_direct_cuda_ipc("talker")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_comm_router_uses_cuda_ipc_for_cuda_stream_chunks_only() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"decode"},
        comm_config={},
    )

    assert (
        router.outbound_stream("decode", torch.empty(1, device="cuda:0"))
        is TransportKind.CUDA_IPC
    )


def cross_gpu_router() -> CommRouter:
    return CommRouter(
        stage_name="minimax_ttm_ar",
        gpu_id=0,
        placement_gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"dit_dav"},
        stage_gpu_ids={"dit_dav": (1,)},
        comm_config={},
    )


def test_comm_router_caches_cuda_peer_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def can_access(target: int, source: int) -> bool:
        calls.append((target, source))
        return True

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access)
    router = cross_gpu_router()

    assert router.outbound("dit_dav") is TransportKind.CUDA_IPC
    assert router.outbound("dit_dav") is TransportKind.CUDA_IPC
    assert calls == [(1, 0)]


def test_comm_router_falls_back_to_shm_when_cuda_peer_access_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def cannot_access(_target: int, source: int) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", cannot_access)
    router = cross_gpu_router()

    assert router.outbound("dit_dav") is TransportKind.SHM
    assert router.outbound("dit_dav") is TransportKind.SHM
    assert calls == 1


def test_a_non_cuda_platform_never_runs_the_cuda_peer_probe(monkeypatch) -> None:
    """The probe reads torch.cuda.device_count() and warns about a CUDA fallback, so
    a cross-device XPU edge must decide on platform policy before it is reached.
    """
    pin_non_cuda_platform(monkeypatch)
    router = CommRouter(
        stage_name="talker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"vocoder"},
        stage_gpu_ids={"vocoder": (1,)},
    )

    def fail(target, device_kind):
        raise AssertionError(f"{device_kind} peer probe ran for {target!r} off CUDA")

    monkeypatch.setattr(router, "device_ipc_peer_available", fail)

    assert router.outbound("vocoder") is TransportKind.SHM
    assert router.inbound("vocoder") is TransportKind.SHM


def pin_level_zero_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    pin_non_cuda_platform(monkeypatch)
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.LEVEL_ZERO_IPC,
        raising=False,
    )


def level_zero_router(**comm_config: int) -> CommRouter:
    return CommRouter(
        stage_name="talker_ar",
        gpu_id=0,
        placement_gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"code2wav"},
        stage_gpu_ids={"code2wav": (1,)},
        comm_config=comm_config,
    )


def accelerator_bytes(num_bytes: int) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device=ACCEL)


def test_level_zero_edges_ask_the_xpu_runtime_for_peer_access_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin_level_zero_platform(monkeypatch)
    calls: list[tuple[int, int]] = []

    def can_access(target: int, source: int) -> bool:
        calls.append((target, source))
        return True

    def cuda_probe(target: int, source: int) -> bool:
        raise AssertionError("an XPU edge queried the CUDA peer namespace")

    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "can_device_access_peer", can_access)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", cuda_probe)
    router = level_zero_router()

    payload = {"hidden": accelerator_bytes(2 << 20)}
    assert router.outbound_payload("code2wav", payload) is TransportKind.LEVEL_ZERO_IPC
    assert (
        router.outbound_stream("code2wav", accelerator_bytes(2 << 20))
        is TransportKind.LEVEL_ZERO_IPC
    )
    assert calls == [(1, 0)]


def test_level_zero_edges_fall_back_to_shm_when_xpu_peers_cannot_talk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin_level_zero_platform(monkeypatch)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.xpu, "can_device_access_peer", lambda target, source: False
    )
    router = level_zero_router()

    payload = {"hidden": accelerator_bytes(2 << 20)}
    assert router.outbound_payload("code2wav", payload) is TransportKind.SHM
    assert (
        router.outbound_stream("code2wav", accelerator_bytes(2 << 20))
        is TransportKind.SHM
    )


def test_level_zero_ipc_leaves_paged_kv_on_the_path_it_takes_without_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paged KV routes by edge alone and supports only cuda_ipc, so it must see
    shm, as on main, rather than a Level-Zero relay nothing has sized."""
    pin_level_zero_platform(monkeypatch)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.xpu, "can_device_access_peer", lambda target, source: True
    )
    router = level_zero_router()

    assert router.outbound("code2wav") is TransportKind.SHM
    assert router.inbound("code2wav") is TransportKind.SHM
    payload = {"hidden": accelerator_bytes(2 << 20)}
    assert router.outbound_payload("code2wav", payload) is TransportKind.LEVEL_ZERO_IPC


def test_level_zero_ipc_carries_only_transfers_between_the_minimum_and_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small and oversized transfers stay on shm; the pool limit counts the packed
    payload, which can overflow even when every tensor fits on its own."""
    pin_level_zero_platform(monkeypatch)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.xpu, "can_device_access_peer", lambda target, source: True
    )
    router = level_zero_router(
        level_zero_ipc_slot_size_kb=64,
        level_zero_ipc_pool_size_mb=4,
        level_zero_ipc_min_transfer_kb=256,
    )

    assert (
        router.outbound_stream("code2wav", accelerator_bytes(4096)) is TransportKind.SHM
    )
    assert (
        router.outbound_stream("code2wav", accelerator_bytes(256 << 10))
        is TransportKind.LEVEL_ZERO_IPC
    )
    assert (
        router.outbound_stream("code2wav", accelerator_bytes(4 << 20))
        is TransportKind.LEVEL_ZERO_IPC
    )
    assert (
        router.outbound_stream("code2wav", accelerator_bytes((4 << 20) + 1))
        is TransportKind.SHM
    )
    split_payload = {"a": accelerator_bytes(3 << 20), "b": accelerator_bytes(3 << 20)}
    assert router.outbound_payload("code2wav", split_payload) is TransportKind.SHM


def test_level_zero_ipc_carries_only_stream_chunks_without_metadata_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata tensors would take slots of their own, so such a chunk stays on shm."""
    pin_level_zero_platform(monkeypatch)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.xpu, "can_device_access_peer", lambda target, source: True
    )
    router = level_zero_router(
        level_zero_ipc_slot_size_kb=64,
        level_zero_ipc_pool_size_mb=4,
        level_zero_ipc_min_transfer_kb=256,
    )
    chunk = accelerator_bytes(1 << 20)

    plain = {"stream": True, "chunk_index": 3}
    assert (
        router.outbound_stream("code2wav", chunk, plain) is TransportKind.LEVEL_ZERO_IPC
    )
    with_tensor = {"lengths": torch.ones(4, dtype=torch.int64)}
    assert router.outbound_stream("code2wav", chunk, with_tensor) is TransportKind.SHM
