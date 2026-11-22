# SPDX-License-Identifier: Apache-2.0
"""Deterministic request ownership, cancellation and failure boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from sglang_omni.models.auk.offload_scheduler import AuKOffloadScheduler
from sglang_omni.models.auk.serial_offload import AuKOffloadCoordinator, StageName
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.stage_residency import StageResidency
from sglang_omni.scheduling.weight_cache import StagingRing


def payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id, request=OmniRequest(inputs="hello"), data={}
    )


def test_next_request_waits_for_terminal_delivery() -> None:
    """Finishing conditioning alone must not admit a competing request."""
    coordinator = AuKOffloadCoordinator(torch.device("cpu"))
    waiting = Event()

    def check_cancellation(request_id: str) -> bool:
        assert request_id == "second"
        waiting.set()
        return False

    coordinator.begin("first", "conditioning", lambda request_id: False)
    coordinator.finish("first", "conditioning", succeeded=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        next_request = executor.submit(
            coordinator.begin, "second", "conditioning", check_cancellation
        )
        assert waiting.wait(timeout=5)
        try:
            assert not next_request.done()
            for stage in ("auk_engine", "decode"):
                coordinator.begin("first", stage, lambda request_id: False)
                coordinator.finish("first", stage, succeeded=True)
            assert coordinator.claim_result("first")
            assert not next_request.done()
        finally:
            coordinator.release_result("first", terminal=True, delivered=True)
        next_request.result(timeout=5)
    coordinator.finish("second", "conditioning", succeeded=True)
    for stage in ("auk_engine", "decode"):
        coordinator.begin("second", stage, lambda request_id: False)
        coordinator.finish("second", stage, succeeded=True)
    coordinator.release_result("second", terminal=True, delivered=True)
    assert coordinator.owner_request_id is None
    assert not coordinator.abort_event.is_set()


@pytest.mark.parametrize("stage", ["conditioning", "auk_engine", "decode"])
def test_dropped_handoff_releases_owner_and_late_abort_preserves_next_request(
    stage: StageName,
) -> None:
    coordinator = AuKOffloadCoordinator(torch.device("cpu"))
    for current_stage in ("conditioning", "auk_engine", "decode"):
        coordinator.begin("first", current_stage, lambda request_id: False)
        coordinator.finish("first", current_stage, succeeded=True)
        if current_stage == stage:
            break
    coordinator.release_result("first", terminal=stage == "decode", delivered=False)
    coordinator.begin("second", "conditioning", lambda request_id: False)
    coordinator.release_result("first", terminal=True, delivered=True)
    coordinator.cancel("first")
    assert coordinator.claim_result("second")
    coordinator.finish("second", "conditioning", succeeded=False)


def test_abort_does_not_release_owner_during_compute() -> None:
    coordinator = AuKOffloadCoordinator(torch.device("cpu"))
    entered = Event()
    finish_compute = Event()

    def compute(payloads: list[StagePayload]) -> list[StagePayload]:
        entered.set()
        assert finish_compute.wait(timeout=5)
        return payloads

    scheduler = AuKOffloadScheduler(compute, coordinator, "conditioning")
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(scheduler.fn, payload("first"))
        assert entered.wait(timeout=5)
        try:
            scheduler.abort("first")
            assert coordinator.owner_request_id == "first"
            assert coordinator.is_computing
            assert not coordinator.claim_result("first")
        finally:
            finish_compute.set()
        active.result(timeout=5)
    coordinator.begin("second", "conditioning", lambda request_id: False)
    coordinator.finish("second", "conditioning", succeeded=False)


def test_failed_restore_escapes_scheduler_request_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AuKOffloadCoordinator(torch.device("cpu"))
    residency = StageResidency(
        {"dit": torch.nn.Linear(2, 2)}, torch.device("cpu"), source="ram"
    )
    residency.sleep()
    coordinator.residencies["dit"] = residency

    def fail_restore(staging_ring: StagingRing | None) -> None:
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(residency.weights, "restore", fail_restore)
    scheduler = AuKOffloadScheduler(
        lambda payloads: payloads, coordinator, "conditioning"
    )
    with pytest.raises(OSError, match="checkpoint unavailable"):
        coordinator.switch("dit")
    with pytest.raises(RuntimeError, match="restart the worker"):
        scheduler.next_message()
    with pytest.raises(RuntimeError, match="restart the worker"):
        coordinator.begin("next", "conditioning", lambda request_id: False)


def test_stop_wakes_waiting_admission() -> None:
    coordinator = AuKOffloadCoordinator(torch.device("cpu"))
    waiting = Event()

    def is_aborted(request_id: str) -> bool:
        waiting.set()
        return False

    coordinator.begin("first", "conditioning", lambda request_id: False)
    coordinator.finish("first", "conditioning", succeeded=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(
            coordinator.begin, "second", "conditioning", is_aborted
        )
        assert waiting.wait(timeout=5)
        coordinator.stop()
        with pytest.raises(InterruptedError):
            blocked.result(timeout=5)
