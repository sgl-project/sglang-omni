# SPDX-License-Identifier: Apache-2.0
"""Serial offload coordinator behind --layerwise-offload-components ar,dit."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minimax_music3.serial_offload import (
    SerialOffloadCoordinator,
    get_coordinator,
    move_module_to_device,
)


def test_get_coordinator_returns_a_process_wide_singleton() -> None:
    assert get_coordinator() is get_coordinator()


def test_disabled_coordinator_never_blocks_admission_and_handoffs_are_noops() -> None:
    coordinator = SerialOffloadCoordinator()

    assert coordinator.enabled is False
    assert coordinator.ar_can_admit() is True
    coordinator.begin_dit_handoff()
    coordinator.end_dit_handoff()
    assert coordinator.ar_can_admit() is True


def test_register_ar_enables_the_coordinator_and_starts_ar_active() -> None:
    coordinator = SerialOffloadCoordinator()
    model = torch.nn.Linear(2, 2)

    coordinator.register_ar(model, torch.device("cpu"))

    assert coordinator.enabled is True
    assert coordinator.ar_can_admit() is True


def test_begin_dit_handoff_moves_ar_off_the_gpu_and_blocks_admission() -> None:
    coordinator = SerialOffloadCoordinator()
    model = torch.nn.Linear(2, 2)
    coordinator.register_ar(model, torch.device("cpu"))

    coordinator.begin_dit_handoff()

    assert coordinator.ar_can_admit() is False


def test_end_dit_handoff_restores_ar_and_reopens_admission() -> None:
    coordinator = SerialOffloadCoordinator()
    model = torch.nn.Linear(2, 2)
    coordinator.register_ar(model, torch.device("cpu"))
    coordinator.begin_dit_handoff()

    coordinator.end_dit_handoff()

    assert coordinator.ar_can_admit() is True


def test_handoff_calls_are_idempotent() -> None:
    coordinator = SerialOffloadCoordinator()
    model = torch.nn.Linear(2, 2)
    coordinator.register_ar(model, torch.device("cpu"))

    coordinator.begin_dit_handoff()
    coordinator.begin_dit_handoff()
    assert coordinator.ar_can_admit() is False

    coordinator.end_dit_handoff()
    coordinator.end_dit_handoff()
    assert coordinator.ar_can_admit() is True


def test_handoff_without_registration_raises_if_force_enabled() -> None:
    """Defensive guard for a caller that enables without registering."""
    coordinator = SerialOffloadCoordinator()
    coordinator._enabled = True

    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.begin_dit_handoff()
    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.end_dit_handoff()


def test_move_module_to_device_relocates_every_parameter() -> None:
    module = torch.nn.Linear(2, 2)

    move_module_to_device(module, torch.device("cpu"))

    assert next(module.parameters()).device.type == "cpu"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_serial_offload_round_trip_moves_weights_between_devices() -> None:
    coordinator = SerialOffloadCoordinator()
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4).to(device)
    coordinator.register_ar(model, device)

    coordinator.begin_dit_handoff()
    assert next(model.parameters()).device.type == "cpu"

    coordinator.end_dit_handoff()
    assert next(model.parameters()).device == device
