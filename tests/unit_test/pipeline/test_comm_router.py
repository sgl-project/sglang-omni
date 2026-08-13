# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter


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
