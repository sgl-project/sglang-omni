# SPDX-License-Identifier: Apache-2.0
"""Serial offload coordinator behind --stage-offload-components ar,dit."""

from __future__ import annotations

import json
import mmap
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from torch_memory_saver import torch_memory_saver

from sglang_omni.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticDecoder,
    MiniMaxMusic3AcousticScheduler,
)
from sglang_omni.models.minimax_music3.engine_builder import MiniMaxMusic3EngineBuilder
from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.models.minimax_music3.serial_offload import (
    STALL_REPORT_SECONDS,
    SerialOffloadCoordinator,
    StageResidency,
    get_coordinator,
)
from sglang_omni.models.minimax_music3.weight_cache import (
    CHUNK_BYTES,
    SHARD_BYTES,
    RuntimeWeights,
    StagingRing,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


@pytest.mark.parametrize("enable_serial_offload", [False, True])
def test_offload_enforces_one_cfg_pair_and_preserves_graph_selection(
    enable_serial_offload: bool,
) -> None:
    builder = MiniMaxMusic3EngineBuilder(enable_serial_offload=enable_serial_offload)
    overrides = build_generation_batch_overrides(
        server_args_overrides={
            "max_running_requests": 16,
            "disable_cuda_graph": False,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "full",
            "cuda_graph_config": {
                "decode": {"backend": "full"},
                "prefill": {"backend": "full"},
            },
            "enable_torch_compile": True,
            "disable_overlap_schedule": False,
        },
        **builder.generation_defaults(dtype="bfloat16"),
    )

    builder.adjust_overrides(overrides)
    assert overrides["enable_deterministic_inference"] is True

    if enable_serial_offload:
        assert builder.max_running_requests == 1
        assert overrides["max_running_requests"] == 2
        assert overrides["disable_cuda_graph"] is False
        assert overrides["cuda_graph_backend_decode"] == "full"
        assert overrides["cuda_graph_backend_prefill"] == "full"
        assert "cuda_graph_config" in overrides
        assert overrides["enable_memory_saver"] is True
        assert overrides["enable_weights_cpu_backup"] is False
        assert overrides["enable_torch_compile"] is False
        assert overrides["disable_overlap_schedule"] is True
    else:
        assert builder.max_running_requests == 16
        assert overrides["max_running_requests"] == 32
        assert overrides["disable_cuda_graph"] is False
        assert overrides["cuda_graph_backend_decode"] == "full"
        assert overrides["cuda_graph_backend_prefill"] == "full"
        assert "cuda_graph_config" in overrides
        assert overrides["enable_torch_compile"] is True
        assert overrides["disable_overlap_schedule"] is False


def registered_coordinator() -> SerialOffloadCoordinator:
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
    coordinator = registered_coordinator()

    assert coordinator.enabled is True
    assert coordinator.ar_can_admit() is True


def test_begin_dit_handoff_moves_ar_off_the_gpu_and_blocks_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is False


def test_end_dit_handoff_restores_ar_and_reopens_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is True


def test_handoff_calls_are_idempotent() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")
    coordinator.begin_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is False

    coordinator.end_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is True


def test_one_owner_excludes_other_requests_until_acoustic_retires() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    assert not coordinator.try_acquire_ar("req-2")
    coordinator.begin_dit_handoff("req-1")
    assert not coordinator.try_acquire_ar("req-2")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.begin_dit_handoff("req-2")

    coordinator.end_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2")


def test_end_for_a_request_that_never_handed_off_does_not_wake_ar() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-unknown")

    assert coordinator.ar_can_admit() is False


def test_a_stalled_handoff_is_reported_once_and_never_force_woken(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")
    coordinator.paused_at_seconds -= STALL_REPORT_SECONDS + 1.0

    with caplog.at_level("ERROR"):
        assert coordinator.ar_can_admit() is False
        assert coordinator.ar_can_admit() is False

    stall_records = [r for r in caplog.records if "off the GPU for" in r.message]
    assert len(stall_records) == 1
    assert "req-1" in stall_records[0].message


def test_handoff_without_registration_raises_if_force_enabled() -> None:
    """Defensive guard for a caller that enables without registering."""
    coordinator = SerialOffloadCoordinator()
    coordinator.is_enabled = True

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
    snapshot = residency.weights.chunks[0].values
    residency.wake()
    residency.sleep()

    assert residency.weights.chunks[0].values is snapshot


def test_restore_keeps_storage_and_tensor_identity() -> None:
    module = torch.nn.Linear(2, 2)
    parameter = module.weight
    address = parameter.data_ptr()
    expected = parameter.detach().clone()
    residency = StageResidency({"module": module}, torch.device("cpu"))
    residency.sleep()
    with torch.no_grad():
        parameter.zero_()
    residency.wake()
    assert module.weight is parameter
    assert parameter.data_ptr() == address
    assert torch.equal(parameter, expected)


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
@pytest.mark.parametrize("source", ["ram", "mmap"])
def test_cuda_cycles_release_physical_memory_and_replay_graph_without_recapture(
    tmp_path: Path, source: Literal["ram", "mmap"]
) -> None:
    device = torch.device("cuda:0")
    tag = f"test_music3_{source}"
    with torch_memory_saver.region(tag=tag, enable_cpu_backup=False):
        model = torch.nn.Linear(4096, 4096, bias=False).to(device)
    expected = model.weight.detach().clone()
    address = model.weight.data_ptr()
    staging_ring = StagingRing(device)
    assert sum(buffer.numel() for buffer in staging_ring.buffers) <= 2 * SHARD_BYTES
    residency = StageResidency(
        {"model": model},
        device,
        tags=(tag,),
        source=source,
        cache_dir=str(tmp_path),
        staging_ring=staging_ring,
    )
    inputs = torch.ones((2, 4096), device=device)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.no_grad():
        model(inputs)
    capture_stream.synchronize()
    with torch.cuda.graph(graph), torch.no_grad():
        output = model(inputs)
    graph.replay()
    torch.cuda.synchronize()
    expected_output = output.clone()
    for _ in range(3):
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info(device)
        residency.sleep()
        free_after, _ = torch.cuda.mem_get_info(device)
        assert free_after - free_before >= model.weight.nbytes
        assert model.weight.device == device
        assert model.weight.data_ptr() == address
        residency.wake()
        assert torch.equal(model.weight, expected)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected_output)


def test_ar_abort_releases_only_its_owner_before_handoff() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.cancel_ar("unknown")
    assert not coordinator.try_acquire_ar("req-2")
    coordinator.cancel_ar("req-1")
    assert coordinator.try_acquire_ar("req-2")
    coordinator.begin_dit_handoff("req-2")
    coordinator.cancel_ar("req-2")
    assert not coordinator.try_acquire_ar("req-3")


class RecordingAcousticDecoder:
    serial_offload = True

    def __init__(self) -> None:
        self.release_count = 0

    def offload_to_cpu(self) -> None:
        self.release_count += 1


def test_late_acoustic_cleanup_cannot_release_a_new_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = registered_coordinator()
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.acoustic.get_coordinator",
        lambda: coordinator,
    )
    decoder = RecordingAcousticDecoder()
    scheduler = MiniMaxMusic3AcousticScheduler(decoder)
    assert coordinator.try_acquire_ar("old")
    coordinator.begin_dit_handoff("old")
    scheduler.clear_stream_state("old")
    assert decoder.release_count == 1

    assert coordinator.try_acquire_ar("new")
    coordinator.begin_dit_handoff("new")
    scheduler.clear_stream_state("old")
    scheduler.clear_stream_state("unknown")
    assert decoder.release_count == 1
    coordinator.require_acoustic("new")
    assert not coordinator.try_acquire_ar("next")
    scheduler.clear_stream_state("new")
    assert decoder.release_count == 2
    assert coordinator.try_acquire_ar("next")


@pytest.mark.parametrize("operation", ["sleep", "wake", "release"])
def test_failed_transition_prevents_further_admission(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    def fail() -> None:
        raise RuntimeError("transfer failed")

    if operation == "sleep":
        monkeypatch.setattr(coordinator.ar_residency, "sleep", fail)
        with pytest.raises(RuntimeError, match="transfer failed"):
            coordinator.begin_dit_handoff("req-1")
    else:
        coordinator.begin_dit_handoff("req-1")
        if operation == "wake":
            monkeypatch.setattr(coordinator.ar_residency, "wake", fail)
            release_acoustic = None
        else:
            release_acoustic = fail
        with pytest.raises(RuntimeError, match="transfer failed"):
            coordinator.end_dit_handoff("req-1", release_acoustic=release_acoustic)

    with pytest.raises(RuntimeError, match="transition failed"):
        coordinator.try_acquire_ar("req-2")


def test_acoustic_compute_requires_a_completed_handoff() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.require_acoustic("req-1")
    coordinator.begin_dit_handoff("req-1")
    coordinator.require_acoustic("req-1")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.require_acoustic("req-2")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sleep_waits_for_compute_on_another_cuda_stream() -> None:
    device = torch.device("cuda:0")
    with torch_memory_saver.region(tag="test_music3_stream", enable_cpu_backup=False):
        model = torch.nn.Linear(256, 256, bias=False).to(device)
    expected = model.weight.detach().cpu().clone()
    residency = StageResidency(
        {"module": model},
        device,
        tags=("test_music3_stream",),
        staging_ring=StagingRing(device),
    )
    residency.sleep()
    residency.wake()
    compute_stream = torch.cuda.Stream(device=device)
    completed = torch.cuda.Event()
    with torch.cuda.stream(compute_stream):
        torch.cuda._sleep(
            10_000_000
        )  # noqa: leading-underscore  # upstream CUDA test API
        output = model(torch.eye(256, device=device))
        completed.record()

    residency.sleep()

    assert completed.query()
    torch.testing.assert_close(output.cpu(), expected.T)


def test_scheduler_admits_only_the_owning_cfg_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = registered_coordinator()
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.scheduler.get_coordinator",
        lambda: coordinator,
    )
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.max_prefill_tokens = 100
    monkeypatch.setattr(scheduler, "get_num_allocatable_reqs", lambda count: 4)
    queue = [
        Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        for request_id in ("first", "first-uncond", "second", "second-uncond")
    ]
    running_batch = ScheduleBatch.__new__(ScheduleBatch)
    running_batch.reqs = []

    assert scheduler.pair_admission_limit(queue, running_batch) == 2
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 0
    coordinator.begin_dit_handoff("first")
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 0
    coordinator.end_dit_handoff("first")
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 2


@pytest.mark.parametrize("source", ["mmap", "ram"])
def test_runtime_cache_restores_shared_storage_and_strided_views(
    tmp_path: Path, source: Literal["mmap", "ram"]
) -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.arange(32.0).view(4, 8))
    )
    module.register_buffer("view", module.weight.detach().T[1:])
    addresses = (module.weight.data_ptr(), module.view.data_ptr())
    expected_view = module.view.clone()
    weights = RuntimeWeights(
        {"model": module},
        source=source,
        cache_dir=str(tmp_path),
        checkpoint_contents={"checkpoint": "content"},
        folding_backend="cpu",
    )
    assert len(weights.storages) == 1
    with torch.no_grad():
        module.weight.zero_()
    weights.restore(None)
    assert torch.equal(module.view, expected_view)
    assert addresses == (module.weight.data_ptr(), module.view.data_ptr())
    assert all(chunk.values.numel() <= CHUNK_BYTES for chunk in weights.chunks)
    if source == "mmap":
        assert all(isinstance(mapping, mmap.mmap) for mapping in weights.mappings)
        assert all(
            path.stat().st_size <= SHARD_BYTES
            for path in weights.cache_path.glob("*.safetensors")
        )
    weights.close()


@pytest.mark.parametrize("corruption", ["shard", "layout", "missing"])
def test_existing_cache_corruption_fails_visibly(
    tmp_path: Path, corruption: str
) -> None:
    module = torch.nn.Linear(4, 4)
    kwargs = dict(
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={"c": "1"},
        folding_backend="cpu",
    )
    weights = RuntimeWeights({"model": module}, **kwargs)
    cache_path = weights.cache_path
    weights.close()
    if corruption == "shard":
        shard = next(cache_path.glob("*.safetensors"))
        shard.write_bytes(b"broken")
    elif corruption == "layout":
        manifest_path = cache_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["layouts"][0]["offset"] = 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        next(cache_path.glob("*.safetensors")).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        RuntimeWeights({"model": module}, **kwargs)


def test_tmpfs_cache_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-tmpfs"):
        RuntimeWeights(
            {"model": torch.nn.Linear(2, 2)},
            source="mmap",
            cache_dir="/dev/shm/music3-test",
            checkpoint_contents={},
            folding_backend="cpu",
        )


def test_explicit_eager_selection_survives_offload_configuration() -> None:
    builder = MiniMaxMusic3EngineBuilder(enable_serial_offload=True)
    overrides = {"disable_cuda_graph": True}
    builder.adjust_overrides(overrides)
    assert overrides["disable_cuda_graph"] is True


def test_failed_acoustic_wake_blocks_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("owner")
    coordinator.begin_dit_handoff("owner")
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.acoustic.get_coordinator",
        lambda: coordinator,
    )
    decoder = MiniMaxMusic3AcousticDecoder.__new__(MiniMaxMusic3AcousticDecoder)
    decoder.residency = coordinator.ar_residency

    def fail() -> None:
        raise RuntimeError("acoustic wake failed")

    monkeypatch.setattr(decoder.residency, "wake", fail)
    with pytest.raises(RuntimeError, match="acoustic wake failed"):
        decoder.ensure_gpu_resident()
    with pytest.raises(RuntimeError, match="transition failed"):
        coordinator.try_acquire_ar("next")


def test_cache_reuse_and_checkpoint_invalidation(tmp_path: Path) -> None:
    module = torch.nn.Linear(2, 2)
    first = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={"c": "first"},
        folding_backend="cpu",
    )
    cache_path = first.cache_path
    modified_at = (cache_path / "manifest.json").stat().st_mtime_ns
    first.close()
    reused = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={"c": "first"},
        folding_backend="cpu",
    )
    assert reused.cache_path == cache_path
    assert (cache_path / "manifest.json").stat().st_mtime_ns == modified_at
    reused.close()
    changed = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={"c": "changed"},
        folding_backend="cpu",
    )
    assert changed.cache_path != cache_path
    changed.close()


def test_insufficient_cache_disk_space_has_no_ram_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disk_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.weight_cache.shutil.disk_usage",
        lambda directory: type(disk_usage)(disk_usage.total, disk_usage.used, 0),
    )
    with pytest.raises(OSError, match="Insufficient disk space"):
        RuntimeWeights(
            {"model": torch.nn.Linear(2, 2)},
            source="mmap",
            cache_dir=str(tmp_path),
            checkpoint_contents={},
            folding_backend="cpu",
        )
    assert not list(tmp_path.glob("*/manifest.json"))


def test_stale_cache_publication_is_cleaned_under_lock(tmp_path: Path) -> None:
    module = torch.nn.Linear(2, 2)
    weights = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={},
        folding_backend="cpu",
    )
    identity = weights.cache_path.name
    weights.close()
    cache_path = tmp_path / identity
    cache_path.rename(tmp_path / f".{identity}-interrupted")
    rebuilt = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={},
        folding_backend="cpu",
    )
    assert rebuilt.cache_path.is_dir()
    assert not (tmp_path / f".{identity}-interrupted").exists()
    rebuilt.close()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staging_ring_waits_before_reusing_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.weight_cache.SHARD_BYTES", 1024
    )
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.weight_cache.CHUNK_BYTES", 1024
    )
    device = torch.device("cuda:0")
    module = torch.nn.Linear(64, 64).to(device)
    expected = module.weight.detach().clone()
    weights = RuntimeWeights(
        {"model": module},
        source="ram",
        cache_dir=None,
        checkpoint_contents={},
        folding_backend="cpu",
    )
    staging_ring = StagingRing(device)
    buffer_addresses = [buffer.data_ptr() for buffer in staging_ring.buffers]
    for _ in range(3):
        with torch.no_grad():
            module.weight.zero_()
        weights.restore(staging_ring)
        assert torch.equal(module.weight, expected)
        assert buffer_addresses == [
            buffer.data_ptr() for buffer in staging_ring.buffers
        ]


def test_cache_publication_serializes_concurrent_creators(tmp_path: Path) -> None:
    module = torch.nn.Linear(2, 2)

    def snapshot() -> RuntimeWeights:
        return RuntimeWeights(
            {"model": module},
            source="mmap",
            cache_dir=str(tmp_path),
            checkpoint_contents={},
            folding_backend="cpu",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        weights = [
            future.result() for future in [executor.submit(snapshot) for _ in range(2)]
        ]
    assert weights[0].cache_path == weights[1].cache_path
    assert len(list(tmp_path.glob("*/manifest.json"))) == 1
    for snapshot_weights in weights:
        snapshot_weights.close()


def test_large_storage_uses_bounded_shards_and_restores_every_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.weight_cache.CHUNK_BYTES", 1024
    )
    module = torch.nn.Linear(64, 64)
    expected = module.weight.detach().clone()
    weights = RuntimeWeights(
        {"model": module},
        source="mmap",
        cache_dir=str(tmp_path),
        checkpoint_contents={},
        folding_backend="cpu",
    )
    assert len(weights.chunks) > 2
    assert all(chunk.values.numel() <= 1024 for chunk in weights.chunks)
    with torch.no_grad():
        module.weight.zero_()
    weights.restore(None)
    assert torch.equal(module.weight, expected)
    weights.close()
