# SPDX-License-Identifier: Apache-2.0
"""Serial offload coordinator behind --stage-offload-components ar,dit."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minimax_music3.serial_offload import (
    STALL_REPORT_SECONDS,
    SerialOffloadCoordinator,
    StageResidency,
    get_coordinator,
)


def _registered() -> SerialOffloadCoordinator:
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    return coordinator


def test_get_coordinator_returns_a_process_wide_singleton() -> None:
    assert get_coordinator() is get_coordinator()


def test_disabled_coordinator_never_blocks_admission_and_handoffs_are_noops() -> None:
    coordinator = SerialOffloadCoordinator()

    assert coordinator.enabled is False
    assert coordinator.ar_can_admit() is True
    coordinator.begin_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is True


def test_register_ar_enables_the_coordinator_and_starts_ar_active() -> None:
    coordinator = _registered()

    assert coordinator.enabled is True
    assert coordinator.ar_can_admit() is True


def test_begin_dit_handoff_moves_ar_off_the_gpu_and_blocks_admission() -> None:
    coordinator = _registered()

    coordinator.begin_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is False


def test_end_dit_handoff_restores_ar_and_reopens_admission() -> None:
    coordinator = _registered()
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is True


def test_handoff_calls_are_idempotent() -> None:
    coordinator = _registered()

    coordinator.begin_dit_handoff("req-1")
    coordinator.begin_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is False

    coordinator.end_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is True


def test_ar_stays_parked_until_every_outstanding_request_retires() -> None:
    """The wake is driven by the outstanding set, not by the last event."""
    coordinator = _registered()
    coordinator.begin_dit_handoff("req-1")
    coordinator.begin_dit_handoff("req-2")

    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is False

    coordinator.end_dit_handoff("req-2")
    assert coordinator.ar_can_admit() is True


def test_end_for_a_request_that_never_handed_off_does_not_wake_ar() -> None:
    coordinator = _registered()
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-unknown")

    assert coordinator.ar_can_admit() is False


def test_a_stalled_handoff_is_reported_once_and_never_force_woken(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = _registered()
    coordinator.begin_dit_handoff("req-1")
    # Backdate the pause past the reporting threshold.
    coordinator._paused_at -= STALL_REPORT_SECONDS + 1.0

    with caplog.at_level("ERROR"):
        assert coordinator.ar_can_admit() is False
        assert coordinator.ar_can_admit() is False

    stall_records = [r for r in caplog.records if "off the GPU for" in r.message]
    assert len(stall_records) == 1
    assert "req-1" in stall_records[0].message


def test_handoff_without_registration_raises_if_force_enabled() -> None:
    """Defensive guard for a caller that enables without registering."""
    coordinator = SerialOffloadCoordinator()
    coordinator._enabled = True

    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.begin_dit_handoff("req-1")
    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.end_dit_handoff("req-1")


def test_residency_sleep_and_wake_preserve_weights_and_state() -> None:
    module = torch.nn.Linear(2, 2)
    expected = module.weight.detach().clone()
    residency = StageResidency({"module": module}, torch.device("cpu"))

    residency.sleep()
    assert residency.resident is False
    residency.wake()

    assert residency.resident is True
    assert torch.equal(module.weight, expected)


def test_residency_reuses_one_host_copy_instead_of_recopying_each_sleep() -> None:
    """The weights are immutable, so only the first sleep may snapshot them."""
    module = torch.nn.Linear(2, 2)
    residency = StageResidency({"module": module}, torch.device("cpu"))

    residency.sleep()
    snapshot = residency._host[("module", "weight")]
    residency.wake()
    residency.sleep()

    assert residency._host[("module", "weight")] is snapshot


def test_a_host_built_module_is_asleep_and_never_snapshots_from_the_gpu() -> None:
    module = torch.nn.Linear(2, 2)
    residency = StageResidency({"module": module}, torch.device("cpu"), resident=False)

    assert residency.resident is False
    assert residency._host[("module", "weight")] is not module.weight
    assert residency._host[("module", "weight")].data_ptr() == module.weight.data_ptr()

    residency.wake()
    assert residency.resident is True


def test_residency_keeps_tied_weights_tied_across_a_round_trip() -> None:
    module = torch.nn.Linear(4, 4)
    tied = torch.nn.Linear(4, 4)
    tied.weight = module.weight
    parent = torch.nn.Sequential(module, tied)
    residency = StageResidency({"module": parent}, torch.device("cpu"))

    residency.sleep()
    residency.wake()

    assert module.weight is tied.weight
    assert module.weight.data_ptr() == tied.weight.data_ptr()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_serial_offload_round_trip_moves_weights_between_devices() -> None:
    coordinator = SerialOffloadCoordinator()
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4).to(device)
    coordinator.register_ar(model, device)

    coordinator.begin_dit_handoff("req-1")
    assert next(model.parameters()).device.type == "cpu"

    coordinator.end_dit_handoff("req-1")
    assert next(model.parameters()).device == device


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_host_built_residency_keeps_canonical_copy_on_cpu() -> None:
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4)
    expected = model.weight.detach().clone()
    residency = StageResidency({"module": model}, device, resident=False)
    host_weight = residency._host[("module", "weight")]

    residency.wake()
    assert model.weight.device == device
    assert host_weight.device.type == "cpu"

    residency.sleep()
    assert model.weight.device.type == "cpu"
    assert model.weight.data_ptr() == host_weight.data_ptr()
    assert torch.equal(model.weight, expected)

    residency.wake()
    assert model.weight.device == device
    assert host_weight.device.type == "cpu"
    assert torch.equal(model.weight.cpu(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stage_group_wakes_once_and_keeps_allocator_blocks_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    dit = torch.nn.Linear(4, 4)
    dav = torch.nn.Linear(4, 4)
    residency = StageResidency(
        {"dit": dit, "dav": dav}, device, resident=False, label="dit/dav"
    )
    synchronize_calls: list[torch.device] = []
    empty_cache_calls = 0
    synchronize = torch.cuda.synchronize
    empty_cache = torch.cuda.empty_cache

    def tracked_synchronize(target: torch.device) -> None:
        synchronize_calls.append(target)
        synchronize(target)

    def tracked_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1
        empty_cache()

    monkeypatch.setattr(torch.cuda, "synchronize", tracked_synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", tracked_empty_cache)

    residency.wake()
    assert next(dit.parameters()).device == device
    assert next(dav.parameters()).device == device
    assert synchronize_calls == [device]

    residency.sleep()
    assert next(dit.parameters()).device.type == "cpu"
    assert next(dav.parameters()).device.type == "cpu"
    assert empty_cache_calls == 0
