# SPDX-License-Identifier: Apache-2.0
"""Weight-free checks for single-image stage residency and configuration."""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path
from queue import Queue
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.config.schema import ProcessConfig
from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    THINKER_STAGE,
    LLaDA2UniInterleavedPipelineConfig,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.models.llada2_uni.offload_scheduler import SerialOffloadDllmScheduler
from sglang_omni.models.llada2_uni.serial_offload import SerialOffloadCoordinator
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


@pytest.mark.parametrize(
    ("keep_decoder_process", "replica_devices", "target_replica_devices"),
    [(False, None, None), (False, [0], None), (False, [1], [1]), (True, None, None)],
)
def test_offload_handles_explicit_source_process_policy(
    keep_decoder_process: bool,
    replica_devices: list[int] | None,
    target_replica_devices: list[int] | None,
) -> None:
    """Colocation must remove orphan policies while preserving processes with members."""
    resident = LLaDA2UniOmniPipelineConfig(
        model_path="unused",
        processes={
            "pipeline": ProcessConfig(replica_devices=target_replica_devices),
            IMAGE_DECODE_STAGE: ProcessConfig(
                num_replicas=1, replica_devices=replica_devices
            ),
        },
    )
    if keep_decoder_process:
        resident.stage_named(DECODE_STAGE).process = IMAGE_DECODE_STAGE
    else:
        pass
    resident = LLaDA2UniOmniPipelineConfig(**resident.model_dump())
    original_configuration = resident.model_dump()

    offload = resident.with_serial_offload(frozenset({"thinker", "decoder", "vae"}))

    assert offload.stage_named(IMAGE_DECODE_STAGE).process == "pipeline"
    assert offload.processes["pipeline"] == resident.processes["pipeline"]
    if keep_decoder_process:
        assert (
            offload.processes[IMAGE_DECODE_STAGE]
            == resident.processes[IMAGE_DECODE_STAGE]
        )
        assert offload.stage_named(DECODE_STAGE).process == IMAGE_DECODE_STAGE
    else:
        assert IMAGE_DECODE_STAGE not in offload.processes
    assert resident.model_dump() == original_configuration
    assert LLaDA2UniOmniPipelineConfig(**offload.model_dump()) == offload


@pytest.mark.parametrize(
    ("source_policy", "error"),
    [
        (ProcessConfig(num_replicas=2, replica_devices=[0, 1]), "one process replica"),
        (ProcessConfig(replica_devices=[1]), "replica_devices.*incompatible"),
    ],
)
def test_offload_rejects_incompatible_source_process_policy(
    source_policy: ProcessConfig, error: str
) -> None:
    """Colocation must not silently discard replica or device constraints."""
    resident = LLaDA2UniOmniPipelineConfig(
        model_path="unused", processes={IMAGE_DECODE_STAGE: source_policy}
    )

    with pytest.raises(ValueError, match=error):
        resident.with_serial_offload(frozenset({"thinker", "decoder", "vae"}))


@pytest.mark.parametrize(
    ("configured_preload", "inherited_preload", "expected_preload"),
    [
        (None, "libprofile.so", "{allocator}:libprofile.so"),
        ("", "libprofile.so", "{allocator}:libprofile.so"),
        ("libtrace.so", "", "{allocator}:libtrace.so"),
        ("libtrace.so", "libprofile.so", "{allocator}:libtrace.so:libprofile.so"),
        ("libtrace.so", "{allocator}", "libtrace.so:{allocator}"),
        ("", "", "{allocator}"),
        (
            "{allocator} libtrace.so",
            "libprofile.so",
            "{allocator} libtrace.so:libprofile.so",
        ),
    ],
)
def test_offload_preserves_configured_and_inherited_preload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_preload: str | None,
    inherited_preload: str,
    expected_preload: str,
) -> None:
    """Offload must preserve configured and inherited preload libraries."""
    package_directory = tmp_path / "torch_memory_saver"
    package_directory.mkdir()
    allocator = tmp_path / "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
    allocator.touch()
    monkeypatch.setattr(
        "sglang_omni.scheduling.stage_residency.find_spec",
        Mock(
            return_value=ModuleSpec(
                "torch_memory_saver",
                loader=None,
                origin=str(package_directory / "__init__.py"),
            )
        ),
    )
    monkeypatch.setattr(
        "sglang_omni.models.llada2_uni.config.version", Mock(return_value="0.0.10")
    )
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setenv("LD_PRELOAD", inherited_preload.format(allocator=allocator))
    environment = {"OMP_NUM_THREADS": "2"}
    if configured_preload is not None:
        environment["LD_PRELOAD"] = configured_preload.format(allocator=allocator)
    else:
        pass
    original_environment = dict(environment)
    configuration = LLaDA2UniOmniPipelineConfig(
        model_path="unused", env_defaults=environment
    ).with_serial_offload(frozenset({"thinker", "decoder", "vae"}))
    resolved = configuration.resolved_env_defaults()
    assert configuration.env_defaults == original_environment

    assert resolved["LD_PRELOAD"] == expected_preload.format(allocator=allocator)
    assert resolved["OMP_NUM_THREADS"] == "2"
    assert environment == original_environment


@pytest.mark.parametrize("setting", ["process", "gpu", "backend", "partial"])
def test_invalid_offload_layout_is_rejected(setting: str) -> None:
    offload = LLaDA2UniOmniPipelineConfig(model_path="unused").with_serial_offload(
        frozenset({"thinker", "decoder", "vae"})
    )
    configuration = offload.model_dump()
    decoder = next(
        stage
        for stage in configuration["stages"]
        if stage["name"] == IMAGE_DECODE_STAGE
    )
    if setting == "process":
        decoder["process"] = "separate"
    elif setting == "gpu":
        decoder["gpu"] = 1
    elif setting == "backend":
        decoder["factory"]["backend"] = "sglang"
    else:
        decoder["factory"]["enable_serial_offload"] = False
    with pytest.raises(ValidationError, match="Serial offload"):
        LLaDA2UniOmniPipelineConfig(**configuration)


def test_interleaved_rejects_offload() -> None:
    configuration = LLaDA2UniInterleavedPipelineConfig(model_path="unused").model_dump()
    thinker = next(
        stage for stage in configuration["stages"] if stage["name"] == THINKER_STAGE
    )
    thinker["factory"]["enable_serial_offload"] = True
    with pytest.raises(ValidationError, match="single-image"):
        LLaDA2UniInterleavedPipelineConfig(**configuration)


def test_compute_failure_releases_residency_for_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transfers and compute must stay exclusive, including cleanup after an error."""
    device = torch.device("cpu")
    coordinator = SerialOffloadCoordinator(device)
    module = torch.nn.Linear(2, 2)
    expected = module(torch.ones(1, 2))
    original_transfer = module.to

    def transfer(target_device: str | torch.device) -> torch.nn.Module:
        assert coordinator.lock.locked()
        return original_transfer(target_device)

    tracked_transfer = Mock(side_effect=transfer)
    monkeypatch.setattr(module, "to", tracked_transfer)
    with pytest.raises(ValueError, match="compute failed"):
        with coordinator.use_module(module, "decoder"):
            assert coordinator.lock.locked()
            torch.testing.assert_close(module(torch.ones(1, 2)), expected)
            raise ValueError("compute failed")
    assert tracked_transfer.call_args_list == [call(device), call("cpu")]
    assert not coordinator.lock.locked()
    with coordinator.use_module(module, "next"):
        torch.testing.assert_close(module(torch.ones(1, 2)), expected)
    assert not coordinator.lock.locked()


@pytest.mark.parametrize("fail_on_restore", [True, False])
def test_failed_transfer_blocks_subsequent_compute(
    monkeypatch: pytest.MonkeyPatch, fail_on_restore: bool
) -> None:
    coordinator = SerialOffloadCoordinator(torch.device("cpu"))
    module = torch.nn.Linear(2, 2)
    failure = RuntimeError("transfer failed")
    transfer = Mock(side_effect=[failure] if fail_on_restore else [module, failure])
    monkeypatch.setattr(module, "to", transfer)
    with pytest.raises(RuntimeError, match="transfer failed"):
        with coordinator.use_module(module, "decoder"):
            pass
    with pytest.raises(RuntimeError, match="unavailable"):
        with coordinator.use_module(module, "next"):
            pytest.fail("compute must not run after an incomplete transfer")
    assert not coordinator.lock.locked()


@pytest.mark.accelerator
def test_cuda_module_release_and_restore_preserve_output() -> None:
    device = torch.device("cuda", 0)
    coordinator = SerialOffloadCoordinator(device)
    module = torch.nn.Embedding(1024, 256).eval()
    input_ids = torch.tensor([1, 3, 7], device=device)
    with torch.inference_mode():
        expected = module(input_ids.cpu())
        baseline_bytes = torch.cuda.memory_allocated(device)
        weight_bytes = module.weight.numel() * module.weight.element_size()
        for _ in range(2):
            with coordinator.use_module(module, "embedding"):
                assert module.weight.device == device
                assert (
                    torch.cuda.memory_allocated(device) >= baseline_bytes + weight_bytes
                )
                actual = module(input_ids).cpu()
            assert module.weight.device.type == "cpu"
            assert torch.cuda.memory_allocated(device) == baseline_bytes
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def offload_scheduler(monkeypatch: pytest.MonkeyPatch) -> SerialOffloadDllmScheduler:
    monkeypatch.setattr(
        "sglang_omni.models.llada2_uni.offload_scheduler.get_offload_coordinator",
        Mock(return_value=SerialOffloadCoordinator(torch.device("cpu"))),
    )
    scheduler = SerialOffloadDllmScheduler.__new__(SerialOffloadDllmScheduler)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(memory_saver_adapter=Mock(enabled=True))
    )
    scheduler.configure_offload(torch.device("cpu"))
    scheduler.waiting_queue = [Mock()]
    scheduler.staging_queue = []
    scheduler.tp_worker.model_runner.memory_saver_adapter.reset_mock()
    return scheduler


def test_thinker_retains_weights_while_requests_are_staged(
    monkeypatch: pytest.MonkeyPatch, offload_scheduler: SerialOffloadDllmScheduler
) -> None:
    scheduler = offload_scheduler
    batch = Mock()
    monkeypatch.setattr(DllmScheduler, "drain_and_purge", lambda self: True)
    monkeypatch.setattr(DllmScheduler, "schedule_next_batch", lambda self: batch)
    monkeypatch.setattr(DllmScheduler, "post_step", lambda self, batch: None)
    assert scheduler.schedule_next_batch() is batch
    scheduler.staging_queue = [Mock(), Mock(), Mock()]
    scheduler.post_step(batch)
    adapter = scheduler.tp_worker.model_runner.memory_saver_adapter
    adapter.pause.assert_not_called()
    assert scheduler.coordinator.lock.locked()
    assert scheduler.schedule_next_batch() is batch
    scheduler.staging_queue = []
    scheduler.post_step(batch)
    adapter.resume.assert_called_once_with("weights")
    adapter.pause.assert_called_once_with("weights")
    assert not scheduler.coordinator.lock.locked()
    scheduler.waiting_queue.clear()
    assert scheduler.schedule_next_batch() is None
    adapter.resume.assert_called_once_with("weights")


def test_thinker_releases_weights_after_base_scheduler_purges_requests(
    monkeypatch: pytest.MonkeyPatch, offload_scheduler: SerialOffloadDllmScheduler
) -> None:
    scheduler = offload_scheduler
    monkeypatch.setattr(DllmScheduler, "drain_and_purge", lambda self: True)
    monkeypatch.setattr(DllmScheduler, "schedule_next_batch", lambda self: Mock())
    scheduler.schedule_next_batch()
    scheduler.staging_queue = [Mock()]
    scheduler.drain_and_purge()
    assert scheduler.has_residency

    def retire_cancelled(self: DllmScheduler) -> bool:
        assert scheduler.has_residency
        self.staging_queue.clear()
        self.waiting_queue.clear()
        return True

    monkeypatch.setattr(DllmScheduler, "drain_and_purge", retire_cancelled)
    scheduler.drain_and_purge()
    assert not scheduler.has_residency
    scheduler.tp_worker.model_runner.memory_saver_adapter.pause.assert_called_once_with(
        "weights"
    )


def test_offload_event_loop_schedules_until_stopped(
    monkeypatch: pytest.MonkeyPatch, offload_scheduler: SerialOffloadDllmScheduler
) -> None:
    """The real base drain result must keep the serving loop alive until stop."""
    scheduler = offload_scheduler
    scheduler.inbox = Queue()
    scheduler.abort_lock = Lock()
    scheduler.aborted_request_ids = set()
    scheduler.waiting_queue.clear()
    scheduler.tp_size = 1
    scheduler.orphaned_uncond_rids = set()

    def stop_after_scheduling() -> None:
        scheduler.stop()

    schedule = Mock(side_effect=stop_after_scheduling)
    monkeypatch.setattr(scheduler, "schedule_next_batch", schedule)

    scheduler.start()

    schedule.assert_called_once_with()
    assert not scheduler.running
    assert not scheduler.has_residency
    assert not scheduler.coordinator.lock.locked()
