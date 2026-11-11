# SPDX-License-Identifier: Apache-2.0
"""Weight-free checks for single-image stage residency and admission."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
    LLaDA2UniInterleavedPipelineConfig,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.models.llada2_uni.offload_scheduler import SerialOffloadDllmScheduler
from sglang_omni.models.llada2_uni.serial_offload import SerialOffloadCoordinator
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


def test_offload_colocates_stages_without_mutating_resident_config() -> None:
    resident = LLaDA2UniOmniPipelineConfig(model_path="unused")
    offload = resident.with_serial_offload(frozenset({"thinker", "decoder", "vae"}))
    stages = [
        offload.stage_named(name)
        for name in (IMAGE_STAGE, THINKER_STAGE, IMAGE_DECODE_STAGE)
    ]
    assert len({stage.process for stage in stages}) == 1
    assert all(stage.factory.enable_serial_offload for stage in stages)
    assert not resident.stage_named(THINKER_STAGE).factory.enable_serial_offload
    assert LLaDA2UniOmniPipelineConfig(**offload.model_dump()) == offload


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


def test_concurrent_stages_wait_until_compute_and_release_finish() -> None:
    coordinator = SerialOffloadCoordinator(torch.device("cpu"))
    first_entered = threading.Event()
    second_waiting = threading.Event()
    finish_first = threading.Event()
    second_entered = threading.Event()
    module = torch.nn.Linear(2, 2)

    def first_stage() -> None:
        with coordinator.use_module(module, "first"):
            first_entered.set()
            assert finish_first.wait(5)

    def second_stage() -> None:
        second_waiting.set()
        with coordinator.use_module(module, "second"):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_stage)
        assert first_entered.wait(5)
        second = executor.submit(second_stage)
        assert second_waiting.wait(5)
        try:
            assert coordinator.lock.locked()
            assert not second_entered.is_set()
        finally:
            finish_first.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert second_entered.is_set()


def test_compute_failure_releases_module_and_allows_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = SerialOffloadCoordinator(torch.device("cpu"))
    module = torch.nn.Linear(2, 2)
    transfer = Mock(wraps=module.to)
    monkeypatch.setattr(module, "to", transfer)
    with pytest.raises(ValueError, match="compute failed"):
        with coordinator.use_module(module, "decoder"):
            raise ValueError("compute failed")
    assert transfer.call_args.args == ("cpu",)
    with coordinator.use_module(module, "next"):
        assert module(torch.ones(1, 2)).shape == (1, 2)


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
def offload_scheduler() -> SerialOffloadDllmScheduler:
    scheduler = SerialOffloadDllmScheduler.__new__(SerialOffloadDllmScheduler)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(memory_saver_adapter=Mock(enabled=True))
    )
    scheduler.configure_offload(torch.device("cpu"))
    scheduler.waiting_queue = [Mock()]
    scheduler.staging_queue = []
    scheduler.tp_worker.model_runner.memory_saver_adapter.reset_mock()
    return scheduler


def test_thinker_holds_residency_until_cfg_companions_retire(
    monkeypatch: pytest.MonkeyPatch, offload_scheduler: SerialOffloadDllmScheduler
) -> None:
    scheduler = offload_scheduler
    batch = Mock()
    monkeypatch.setattr(DllmScheduler, "drain_and_purge", lambda self: None)
    monkeypatch.setattr(DllmScheduler, "schedule_next_batch", lambda self: batch)
    monkeypatch.setattr(DllmScheduler, "post_step", lambda self, batch: None)
    assert scheduler.schedule_next_batch() is batch
    scheduler.staging_queue = [Mock(), Mock(), Mock()]
    scheduler.post_step(batch)
    adapter = scheduler.tp_worker.model_runner.memory_saver_adapter
    adapter.pause.assert_not_called()
    assert scheduler.coordinator.lock.locked()
    scheduler.staging_queue = []
    scheduler.post_step(batch)
    adapter.resume.assert_called_once_with("weights")
    adapter.pause.assert_called_once_with("weights")
    assert not scheduler.coordinator.lock.locked()


def test_cancellation_retires_before_weights_are_released(
    monkeypatch: pytest.MonkeyPatch, offload_scheduler: SerialOffloadDllmScheduler
) -> None:
    scheduler = offload_scheduler
    monkeypatch.setattr(DllmScheduler, "drain_and_purge", lambda self: None)
    monkeypatch.setattr(DllmScheduler, "schedule_next_batch", lambda self: Mock())
    scheduler.schedule_next_batch()
    scheduler.staging_queue = [Mock()]
    scheduler.drain_and_purge()
    assert scheduler.has_residency

    def retire_cancelled(self: DllmScheduler) -> None:
        assert scheduler.has_residency
        self.staging_queue.clear()
        self.waiting_queue.clear()

    monkeypatch.setattr(DllmScheduler, "drain_and_purge", retire_cancelled)
    scheduler.drain_and_purge()
    assert not scheduler.has_residency
    scheduler.tp_worker.model_runner.memory_saver_adapter.pause.assert_called_once_with(
        "weights"
    )
