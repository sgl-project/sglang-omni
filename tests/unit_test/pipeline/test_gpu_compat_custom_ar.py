# SPDX-License-Identifier: Apache-2.0
"""Tests for topology-gated custom all-reduce detection in gpu_compat."""

from __future__ import annotations

from types import ModuleType

import pytest

import sglang_omni.utils.gpu_compat as gpu_compat


@pytest.fixture(autouse=True)
def use_cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_compat, "gpu_device_type", lambda: "cuda")


class FakeP2PNVML(ModuleType):
    """Minimal pynvml stub exposing only the P2P-status surface we query."""

    NVML_P2P_STATUS_OK = 0
    NVML_P2P_CAPS_INDEX_READ = 0

    def __init__(
        self,
        *,
        not_ok_pairs: set[tuple[int, int]] | None = None,
        init_error: Exception | None = None,
        query_error: Exception | None = None,
        drop_status_fn: bool = False,
    ) -> None:
        super().__init__("pynvml")
        self.not_ok_pairs = not_ok_pairs or set()
        self.init_error = init_error
        self.query_error = query_error
        self.shutdown_called = False
        if drop_status_fn:
            self.nvmlDeviceGetP2PStatus = None

    def nvmlInit(self) -> None:
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, device_id: int) -> int:
        return device_id

    def nvmlDeviceGetP2PStatus(self, handle_a: int, handle_b: int, _index: int) -> int:
        if self.query_error is not None:
            raise self.query_error
        if (handle_a, handle_b) in self.not_ok_pairs:
            return 1
        return self.NVML_P2P_STATUS_OK


def patch_pynvml(monkeypatch, fake: ModuleType | None) -> None:
    monkeypatch.setattr(gpu_compat, "try_import_pynvml", lambda: fake)


def test_should_disable_with_no_or_single_gpu(monkeypatch) -> None:
    patch_pynvml(monkeypatch, FakeP2PNVML(not_ok_pairs={(0, 1)}))
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus(None, env={}) is True
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([], env={}) is True
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0], env={}) is True


def test_disabled_when_pynvml_unavailable(monkeypatch) -> None:
    patch_pynvml(monkeypatch, None)
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1], env={}) is None
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1], env={}) is True


def test_enabled_on_full_p2p_mesh(monkeypatch) -> None:
    fake = FakeP2PNVML()
    patch_pynvml(monkeypatch, fake)
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1, 2, 3], env={}) is True
    assert (
        gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1, 2, 3], env={})
        is False
    )
    assert fake.shutdown_called is True


def test_disabled_when_any_pair_lacks_p2p(monkeypatch) -> None:
    patch_pynvml(monkeypatch, FakeP2PNVML(not_ok_pairs={(0, 3), (3, 0)}))
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1, 2, 3], env={}) is False
    assert (
        gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1, 2, 3], env={})
        is True
    )


def test_disabled_on_nvml_errors(monkeypatch) -> None:
    patch_pynvml(monkeypatch, FakeP2PNVML(init_error=RuntimeError("nvml init")))
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1], env={}) is True

    patch_pynvml(monkeypatch, FakeP2PNVML(query_error=RuntimeError("nvml query")))
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1], env={}) is True


def test_disabled_when_p2p_status_api_missing(monkeypatch) -> None:
    patch_pynvml(monkeypatch, FakeP2PNVML(drop_status_fn=True))
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1], env={}) is None
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1], env={}) is True


@pytest.mark.parametrize("missing_reverse_peer", [False, True])
def test_xpu_queries_logical_indices_in_both_directions(
    monkeypatch: pytest.MonkeyPatch, missing_reverse_peer: bool
) -> None:
    pairs: list[tuple[int, int]] = []

    def can_access_peer(source: int, destination: int) -> bool:
        pairs.append((source, destination))
        return not (missing_reverse_peer and (source, destination) == (1, 0))

    monkeypatch.setattr(gpu_compat, "gpu_device_type", lambda: "xpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,5")
    monkeypatch.setattr(gpu_compat.torch.xpu, "can_device_access_peer", can_access_peer)
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1, 0]) is not missing_reverse_peer
    assert pairs == [(0, 1), (1, 0)]
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1]) is True


def test_xpu_unknown_peer_access_keeps_custom_all_reduce_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(source: int, destination: int) -> bool:
        raise RuntimeError("peer access unavailable")

    monkeypatch.setattr(gpu_compat, "gpu_device_type", lambda: "xpu")
    monkeypatch.setattr(gpu_compat.torch.xpu, "can_device_access_peer", unavailable)
    assert gpu_compat.gpu_ids_support_p2p_mesh([0, 1]) is None
    assert gpu_compat.should_disable_custom_all_reduce_for_gpus([0, 1]) is True


def test_xpu_does_not_probe_a_different_visibility_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_probe(source: int, destination: int) -> bool:
        pytest.fail("current process cannot probe another XPU mask")

    monkeypatch.setattr(gpu_compat, "gpu_device_type", lambda: "xpu")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1")
    monkeypatch.setattr(gpu_compat.torch.xpu, "can_device_access_peer", refuse_probe)
    assert (
        gpu_compat.gpu_ids_support_p2p_mesh([0, 1], {"ZE_AFFINITY_MASK": "2,3"}) is None
    )
