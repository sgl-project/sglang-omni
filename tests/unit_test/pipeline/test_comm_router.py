# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter


@pytest.fixture(autouse=True)
def _force_cuda_transport_policy(monkeypatch):
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


_ACCEL = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "meta"


def _pin_non_cuda_platform(monkeypatch) -> None:
    """Pin every platform attribute the router reads, so these cases hold on a CUDA
    host too. Patching device_type alone left is_cuda_alike() reporting the host.
    """
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.SHM,
        raising=False,
    )
    monkeypatch.setattr(
        platforms.current_platform, "device_type", _ACCEL, raising=False
    )
    monkeypatch.setattr(
        platforms.current_platform, "is_cuda_alike", lambda: False, raising=False
    )


def _xpu_router(monkeypatch, **kwargs) -> CommRouter:
    _pin_non_cuda_platform(monkeypatch)
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
    router = _xpu_router(monkeypatch, stage_gpu_ids={"decode": (0,)})

    assert router.outbound("decode") is TransportKind.SHM
    assert router._physical_outbound("decode") is TransportKind.SHM
    assert router.inbound("decode") is TransportKind.SHM
    assert not router.can_use_direct_cuda_ipc("decode")


def test_comm_router_uses_shm_for_accelerator_payloads_and_streams(
    monkeypatch,
) -> None:
    router = _xpu_router(monkeypatch)
    payload = {"hidden": torch.empty(1, device=_ACCEL), "lengths": torch.empty(1)}

    assert router.outbound_payload("decode", payload) is TransportKind.SHM
    assert (
        router.outbound_stream("decode", torch.empty(1, device=_ACCEL))
        is TransportKind.SHM
    )


def test_comm_router_still_routes_remote_edges_over_mooncake(monkeypatch) -> None:
    """Losing cuda_ipc must not swallow the cross-node decision."""
    router = _xpu_router(monkeypatch, remote_stage_names={"remote_decode"})

    assert router.outbound("remote_decode") is TransportKind.MOONCAKE
    assert router.inbound("remote_decode") is TransportKind.MOONCAKE


def test_a_mooncake_relay_is_refused_on_a_non_cuda_accelerator(monkeypatch) -> None:
    """Selecting the transport is a topology decision, but the relay itself is built
    on a literal cuda device, so an accelerator edge must fail loudly here instead of
    allocating on CUDA at the first transfer.
    """
    router = _xpu_router(monkeypatch, remote_stage_names={"remote_decode"})

    with pytest.raises(NotImplementedError, match="not supported yet"):
        router.relay(TransportKind.MOONCAKE)


def test_a_host_only_stage_still_gets_a_mooncake_relay(monkeypatch) -> None:
    """A stage with no card uses the cpu path, which is unaffected."""
    _pin_non_cuda_platform(monkeypatch)
    router = CommRouter(
        stage_name="preprocessing",
        gpu_id=None,
        same_process_targets=set(),
        gpu_stage_names=set(),
        remote_stage_names={"remote_decode"},
        injected_relay=object(),
    )

    assert router.relay(TransportKind.MOONCAKE) is router.injected_relay


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


def _cross_gpu_router() -> CommRouter:
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

    def _can_access(target: int, source: int) -> bool:
        calls.append((target, source))
        return True

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", _can_access)
    router = _cross_gpu_router()

    assert router.outbound("dit_dav") is TransportKind.CUDA_IPC
    assert router.outbound("dit_dav") is TransportKind.CUDA_IPC
    assert calls == [(1, 0)]


def test_comm_router_falls_back_to_shm_when_cuda_peer_access_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def _cannot_access(_target: int, _source: int) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", _cannot_access)
    router = _cross_gpu_router()

    assert router.outbound("dit_dav") is TransportKind.SHM
    assert router.outbound("dit_dav") is TransportKind.SHM
    assert calls == 1


def test_a_non_cuda_platform_never_runs_the_cuda_peer_probe(monkeypatch) -> None:
    """The probe reads torch.cuda.device_count() and warns about a CUDA fallback, so
    a cross-device XPU edge must decide on platform policy before it is reached.
    """
    _pin_non_cuda_platform(monkeypatch)
    router = CommRouter(
        stage_name="talker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"vocoder"},
        stage_gpu_ids={"vocoder": (1,)},
    )

    def _fail(target):
        raise AssertionError(f"peer probe ran for {target!r} off CUDA")

    monkeypatch.setattr(router, "_cuda_ipc_peer_available", _fail)

    assert router.outbound("vocoder") is TransportKind.SHM
    assert router.inbound("vocoder") is TransportKind.SHM


def test_a_cuda_platform_still_falls_back_when_peers_cannot_talk(monkeypatch) -> None:
    """CUDA keeps its peer fallback: the probe decides, not the platform alone."""
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
        raising=False,
    )
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )
    router = CommRouter(
        stage_name="talker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"vocoder"},
        stage_gpu_ids={"vocoder": (1,)},
    )

    monkeypatch.setattr(router, "_cuda_ipc_peer_available", lambda target: False)
    assert router.outbound("vocoder") is TransportKind.SHM

    monkeypatch.setattr(router, "_cuda_ipc_peer_available", lambda target: True)
    assert router.outbound("vocoder") is TransportKind.CUDA_IPC
