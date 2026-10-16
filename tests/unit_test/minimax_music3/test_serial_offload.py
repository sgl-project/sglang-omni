# SPDX-License-Identifier: Apache-2.0
"""Serial offload coordinator behind --stage-offload-components ar,dit."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minimax_music3 import acoustic
from sglang_omni.models.minimax_music3.model_runner import MiniMaxMusic3ModelRunner
from sglang_omni.models.minimax_music3.serial_offload import (
    STALL_REPORT_SECONDS,
    SerialOffloadCoordinator,
    StageResidency,
    get_coordinator,
)


def registered_coordinator() -> SerialOffloadCoordinator:
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    return coordinator


def test_get_coordinator_returns_a_process_wide_singleton() -> None:
    assert get_coordinator() is get_coordinator()


def test_disabled_coordinator_never_blocks_admission_and_handoffs_are_noops() -> None:
    coordinator = SerialOffloadCoordinator()

    assert coordinator.enabled is False
    assert coordinator.try_acquire_ar("req-1") is True
    coordinator.begin_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2") is True


def test_register_ar_enables_the_coordinator_and_starts_ar_active() -> None:
    coordinator = registered_coordinator()

    assert coordinator.enabled is True
    assert coordinator.try_acquire_ar("req-1") is True


def test_begin_dit_handoff_moves_ar_off_the_gpu_and_blocks_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")

    assert coordinator.try_acquire_ar("req-2") is False


def test_end_dit_handoff_restores_ar_and_reopens_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-1")

    assert coordinator.try_acquire_ar("req-2") is True


def test_handoff_calls_are_idempotent() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")
    coordinator.begin_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2") is False

    coordinator.end_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2") is True


def test_second_request_cannot_enter_the_first_request_lifecycle() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    assert coordinator.try_acquire_ar("req-2") is False
    coordinator.begin_dit_handoff("req-1")

    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.begin_dit_handoff("req-2")


def test_end_for_a_request_that_never_handed_off_does_not_wake_ar() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-unknown")

    assert coordinator.try_acquire_ar("req-2") is False


def test_late_terminal_for_old_request_does_not_release_new_owner() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-old")
    coordinator.begin_dit_handoff("req-old")
    coordinator.end_dit_handoff("req-old")
    assert coordinator.try_acquire_ar("req-new")
    coordinator.begin_dit_handoff("req-new")
    released: list[str] = []

    coordinator.end_dit_handoff(
        "req-old", release_acoustic=lambda: released.append("old")
    )

    assert released == []
    assert coordinator.try_acquire_ar("req-other") is False


def test_acoustic_release_failure_fails_closed() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    def fail_release() -> None:
        raise OSError("release failed")

    with pytest.raises(OSError, match="release failed"):
        coordinator.end_dit_handoff("req-1", release_acoustic=fail_release)
    with pytest.raises(RuntimeError, match="transition failed"):
        coordinator.try_acquire_ar("req-2")


def test_cancel_before_handoff_releases_owner() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.cancel_ar("req-1")

    assert coordinator.try_acquire_ar("req-2") is True


def test_a_stalled_handoff_is_reported_once_and_never_force_woken(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")
    # Backdate the pause past the reporting threshold.
    coordinator.paused_at_seconds -= STALL_REPORT_SECONDS + 1.0

    with caplog.at_level("ERROR"):
        assert coordinator.try_acquire_ar("req-2") is False
        assert coordinator.try_acquire_ar("req-2") is False

    stall_records = [r for r in caplog.records if "off the GPU for" in r.message]
    assert len(stall_records) == 1
    assert "req-1" in stall_records[0].message


def test_ownership_without_cuda_residency_supports_mlx() -> None:
    coordinator = SerialOffloadCoordinator()
    coordinator.enable()

    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2")


def test_transition_failure_fails_closed() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    assert coordinator.ar_residency is not None

    def fail_sleep() -> None:
        raise OSError("copy failed")

    coordinator.ar_residency.sleep = fail_sleep
    with pytest.raises(OSError, match="copy failed"):
        coordinator.begin_dit_handoff("req-1")
    with pytest.raises(RuntimeError, match="transition failed"):
        coordinator.try_acquire_ar("req-2")


def test_builder_serial_mode_uses_one_cfg_pair_and_disables_graphs() -> None:
    from sglang_omni.models.minimax_music3.engine_builder import (
        MiniMaxMusic3EngineBuilder,
    )

    builder = MiniMaxMusic3EngineBuilder(
        max_running_requests=8,
        enable_serial_offload=True,
    )
    overrides = builder.generation_defaults(dtype="bfloat16")

    builder.adjust_overrides(overrides)

    assert builder.max_running_requests == 1
    assert overrides["max_running_requests"] == 2
    assert overrides["disable_cuda_graph"] is True
    assert overrides["enable_torch_compile"] is False


def test_builder_default_mode_preserves_requested_concurrency() -> None:
    from sglang_omni.models.minimax_music3.engine_builder import (
        MiniMaxMusic3EngineBuilder,
    )

    builder = MiniMaxMusic3EngineBuilder(max_running_requests=8)
    overrides = builder.generation_defaults(dtype="bfloat16")

    builder.adjust_overrides(overrides)

    assert builder.max_running_requests == 8
    assert overrides["max_running_requests"] == 16
    assert overrides["disable_cuda_graph"] is False


def test_builder_registers_complete_ar_model_with_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.minimax_music3 import serial_offload
    from sglang_omni.models.minimax_music3.engine_builder import (
        MiniMaxMusic3EngineBuilder,
    )

    coordinator = SerialOffloadCoordinator()
    monkeypatch.setattr(serial_offload, "COORDINATOR", coordinator)
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.audio_decoder = torch.nn.Linear(2, 2)
    builder = MiniMaxMusic3EngineBuilder(enable_serial_offload=True)
    builder.device = "cpu"

    builder.setup_runtime_resources(model, SimpleNamespace())

    assert coordinator.enabled
    assert coordinator.ar_residency is not None
    assert coordinator.ar_residency.modules == {"ar": model}
    assert coordinator.try_acquire_ar("request")


def test_cuda_factory_forwards_serial_offload_to_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.minimax_music3 import engine_builder, stages

    observed = {}

    class Builder:
        max_running_requests = 1
        device = "cuda:0"

        def __init__(self, **kwargs):
            observed.update(init=kwargs)

        def build(self, model_path, **kwargs):
            observed.update(model_path=model_path, build=kwargs)
            return "scheduler"

    monkeypatch.setattr(stages, "use_mlx_backend", lambda: False)
    monkeypatch.setattr(engine_builder, "MiniMaxMusic3EngineBuilder", Builder)

    result = stages.create_ar_executor(
        "checkpoint",
        device="cuda",
        gpu_id=0,
        enable_serial_offload=True,
    )

    assert result == "scheduler"
    assert observed["init"]["enable_serial_offload"] is True
    assert observed["build"]["device"] == "cuda"
    assert observed["build"]["gpu_id"] == 0


def test_scheduler_admission_claims_only_one_logical_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.minimax_music3 import scheduler as scheduler_module
    from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler

    coordinator = SerialOffloadCoordinator()
    coordinator.enable()
    monkeypatch.setattr(scheduler_module, "get_coordinator", lambda: coordinator)
    scheduler = SimpleNamespace(
        max_prefill_tokens=100,
        get_num_allocatable_reqs=lambda running: 4 - running,
    )
    queue = [
        SimpleNamespace(rid="req-1", origin_input_ids=[1]),
        SimpleNamespace(rid="req-1-cfg", origin_input_ids=[1]),
        SimpleNamespace(rid="req-2", origin_input_ids=[1]),
        SimpleNamespace(rid="req-2-cfg", origin_input_ids=[1]),
    ]

    limit = MiniMaxMusic3Scheduler.pair_admission_limit(
        scheduler, queue, SimpleNamespace(reqs=[])
    )

    assert limit == 2
    assert coordinator.try_acquire_ar("req-2") is False


def test_ar_reset_releases_owner_before_handoff() -> None:
    from sglang_omni.models.minimax_music3.model_runner import MiniMaxMusic3ModelRunner

    coordinator = SerialOffloadCoordinator()
    coordinator.enable()
    assert coordinator.try_acquire_ar("req-1")
    runner = object.__new__(MiniMaxMusic3ModelRunner)
    runner.request_data = {"req-1": SimpleNamespace(ar_state=object())}
    runner.serial_offload = coordinator

    runner.reset_request("req-1")

    assert coordinator.try_acquire_ar("req-2")


def test_acoustic_scheduler_releases_only_the_current_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.minimax_music3 import acoustic

    coordinator = SerialOffloadCoordinator()
    coordinator.enable()
    assert coordinator.try_acquire_ar("req-current")
    coordinator.begin_dit_handoff("req-current")
    monkeypatch.setattr(acoustic, "get_coordinator", lambda: coordinator)

    class Decoder:
        serial_offload = True

        def __init__(self) -> None:
            self.release_calls = 0

        def release_residency(self) -> None:
            self.release_calls += 1

    decoder = Decoder()
    scheduler = acoustic.MiniMaxMusic3AcousticScheduler(decoder)

    scheduler.clear_stream_state("req-old")
    assert decoder.release_calls == 0
    assert coordinator.try_acquire_ar("req-next") is False

    scheduler.abort("req-current")
    assert decoder.release_calls == 1
    assert coordinator.try_acquire_ar("req-next")


@pytest.mark.parametrize("abort_before_terminal", [True, False])
def test_acoustic_abort_during_ar_terminal_releases_after_unloading(
    monkeypatch: pytest.MonkeyPatch,
    abort_before_terminal: bool,
) -> None:
    coordinator = registered_coordinator()
    monkeypatch.setattr(acoustic, "get_coordinator", lambda: coordinator)
    releases: list[str] = []
    scheduler = acoustic.MiniMaxMusic3AcousticScheduler(
        SimpleNamespace(
            serial_offload=True,
            release_residency=lambda: releases.append("acoustic"),
        )
    )
    unloading_started = threading.Event()
    finish_unloading = threading.Event()
    residency = coordinator.ar_residency
    sleep = residency.sleep

    def blocked_sleep() -> None:
        unloading_started.set()
        assert finish_unloading.wait(timeout=5)
        sleep()

    monkeypatch.setattr(residency, "sleep", blocked_sleep)
    assert coordinator.try_acquire_ar("cancelled")
    runner = MiniMaxMusic3ModelRunner.__new__(MiniMaxMusic3ModelRunner)
    runner.serial_offload = coordinator
    runner.request_data = {}
    runner.dump_reference_ranks = lambda state: None
    request_data = SimpleNamespace(
        ar_state=SimpleNamespace(
            generated_frames=1,
            emitted_windows=100,
            finish_reason="stop",
            frames=SimpleNamespace(peak_frames=1),
            started_s=time.perf_counter(),
        ),
        minimax_state=SimpleNamespace(),
        prompt_tokens=1,
    )
    if abort_before_terminal:
        scheduler.abort("cancelled")
    errors: list[BaseException] = []

    def finish_request() -> None:
        try:
            runner.on_request_finished("cancelled", request_data)
            runner.reset_request("cancelled")
        except BaseException as error:
            errors.append(error)

    terminal_thread = threading.Thread(target=finish_request)
    terminal_thread.start()
    try:
        assert unloading_started.wait(timeout=5)
        scheduler.abort("cancelled")
        scheduler.abort("cancelled")
        assert not coordinator.try_acquire_ar("next")
        assert releases == []
    finally:
        finish_unloading.set()
        terminal_thread.join(timeout=5)
    assert not terminal_thread.is_alive()
    assert errors == []
    assert scheduler.is_aborted("cancelled")
    assert releases == ["acoustic"]
    assert residency.resident
    assert coordinator.try_acquire_ar("next")


def test_cancel_ar_clears_early_consumer_retirement() -> None:
    coordinator = registered_coordinator()
    released: list[str] = []
    assert coordinator.try_acquire_ar("cancelled")
    coordinator.end_dit_handoff(
        "cancelled", release_acoustic=lambda: released.append("cancelled")
    )
    coordinator.cancel_ar("cancelled")
    assert coordinator.try_acquire_ar("next")
    coordinator.begin_dit_handoff("next")
    assert not coordinator.try_acquire_ar("third")
    coordinator.end_dit_handoff("next")
    assert coordinator.try_acquire_ar("third")
    assert released == []


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
    snapshot = residency.host_weights[("module", "weight")]
    residency.wake()
    residency.sleep()

    assert residency.host_weights[("module", "weight")] is snapshot


def test_a_host_built_module_is_asleep_and_never_snapshots_from_the_gpu() -> None:
    module = torch.nn.Linear(2, 2)
    residency = StageResidency({"module": module}, torch.device("cpu"), resident=False)

    assert residency.resident is False
    assert residency.host_weights[("module", "weight")] is not module.weight
    assert (
        residency.host_weights[("module", "weight")].data_ptr()
        == module.weight.data_ptr()
    )

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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_serial_offload_round_trip_moves_weights_between_devices() -> None:
    coordinator = SerialOffloadCoordinator()
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4).to(device)
    coordinator.register_ar(model, device)
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")
    assert next(model.parameters()).device.type == "cpu"

    coordinator.end_dit_handoff("req-1")
    assert next(model.parameters()).device == device


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_host_built_residency_keeps_canonical_copy_on_cpu() -> None:
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4)
    expected = model.weight.detach().clone()
    residency = StageResidency({"module": model}, device, resident=False)
    host_weight = residency.host_weights[("module", "weight")]

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


@pytest.mark.accelerator
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
    assert synchronize_calls == [device, device]
    assert empty_cache_calls == 0
