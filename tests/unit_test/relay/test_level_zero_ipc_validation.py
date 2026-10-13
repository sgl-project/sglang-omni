# SPDX-License-Identifier: Apache-2.0
"""What the Level-Zero IPC relay refuses before it touches a device."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.platforms.level_zero as level_zero
from sglang_omni.comm.router import CommRouter
from sglang_omni.config.schema import CommConfig
from sglang_omni.pipeline.runtime_config import build_comm_config
from sglang_omni.platforms.level_zero import (
    ZE_MAX_IPC_HANDLE_SIZE,
    LevelZeroDriver,
    LevelZeroError,
    dma_buf_descriptor,
    is_level_zero_ipc_available,
)
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay
from sglang_omni.relay.level_zero_ipc_metadata import (
    PeerPoolReference,
    TransferPlacement,
    parse_metadata,
)

SLOT_SIZE_KB = 64
POOL_SIZE_MB = 8


def valid_metadata() -> dict[str, object]:
    return {
        "engine_id": "sender",
        "transfer_info": {
            "size": 4096,
            "offset": 65536,
            "slot_index": 1,
            "slot_size": 65536,
            "num_slots": 1,
            "allocation_size": 65536,
        },
        "level_zero_ipc": {
            "pool_id": "sender:1:abc",
            "socket_name": "sglang-omni-level-zero-1-abc",
            "segment_offset": 0,
            "pool_bytes": 262144,
            "source_device_index": 0,
        },
    }


@pytest.mark.parametrize("device", ["cpu", "cpu:0", "cuda:0"])
def test_level_zero_ipc_relay_refuses_a_device_that_is_not_xpu(device: str) -> None:
    with pytest.raises(ValueError, match="requires an XPU device"):
        LevelZeroIpcRelay(
            engine_id="sender",
            device=device,
            slot_size_kb=SLOT_SIZE_KB,
            pool_size_mb=POOL_SIZE_MB,
        )


@pytest.mark.parametrize(
    "field",
    [
        {"level_zero_ipc_slot_size_kb": 0},
        {"level_zero_ipc_pool_size_mb": 0},
        {"level_zero_ipc_pool_size_mb": -64},
        {"level_zero_ipc_min_transfer_kb": -1},
    ],
)
def test_level_zero_ipc_sizes_must_be_positive_in_the_config(
    field: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        CommConfig(**field)


def router_with(comm_config: dict[str, int] | None) -> CommRouter:
    return CommRouter(
        stage_name="talker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names=set(),
        comm_config=comm_config,
    )


@pytest.mark.parametrize(
    ("pool_size_mb", "slot_size_kb", "accepted"),
    [(1, 2048, False), (1, 1024, True)],
    ids=["slot-larger-than-pool", "one-slot-pool"],
)
def test_a_level_zero_ipc_pool_must_hold_a_slot(
    pool_size_mb: int, slot_size_kb: int, accepted: bool
) -> None:
    """A pool smaller than one slot would send every transfer to shm while the
    opt-in looks enabled, so it is refused in the config and in the router."""
    sizes = {
        "level_zero_ipc_pool_size_mb": pool_size_mb,
        "level_zero_ipc_slot_size_kb": slot_size_kb,
    }
    if accepted:
        CommConfig(**sizes)
        router_with(sizes)
    else:
        with pytest.raises(ValueError, match="must hold at least one"):
            CommConfig(**sizes)
        with pytest.raises(ValueError, match="must hold at least one"):
            router_with(sizes)


@pytest.mark.parametrize(
    "sizes",
    [
        {"level_zero_ipc_slot_size_kb": 0},
        {"level_zero_ipc_slot_size_kb": -64},
        {"level_zero_ipc_pool_size_mb": 0},
        {"level_zero_ipc_min_transfer_kb": -1},
    ],
    ids=["zero-slot", "negative-slot", "zero-pool", "negative-minimum"],
)
def test_a_router_refuses_sizes_it_could_not_route_by(sizes: dict[str, int]) -> None:
    """A direct caller skips CommConfig, and a zero slot size would otherwise
    divide by zero on the first transfer routed."""
    with pytest.raises(ValueError, match="must"):
        router_with(sizes)


def test_the_router_and_an_unconfigured_stage_use_the_config_defaults() -> None:
    defaults = CommConfig()
    router = router_with(None)
    unconfigured = build_comm_config(SimpleNamespace(comm=None))

    for name in (
        "level_zero_ipc_slot_size_kb",
        "level_zero_ipc_pool_size_mb",
        "level_zero_ipc_min_transfer_kb",
    ):
        assert getattr(router, name) == getattr(defaults, name)
        assert unconfigured[name] == getattr(defaults, name)


def test_parse_metadata_reads_a_valid_transfer() -> None:
    placement, reference = parse_metadata(valid_metadata())

    assert placement == TransferPlacement(
        size_bytes=4096,
        offset_bytes=65536,
        slot_index=1,
        slot_size_bytes=65536,
        num_slots=1,
    )
    assert reference == PeerPoolReference(
        pool_id="sender:1:abc",
        socket_name="sglang-omni-level-zero-1-abc",
        segment_offset_bytes=0,
        pool_bytes=262144,
        source_device_index=0,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("offset", -64, "must be non-negative"),
        ("offset", 1024, "must be slot aligned"),
        ("slot_size", 0, "must be positive"),
        ("num_slots", 0, "must be positive"),
        ("size", 1 << 20, "num_slots is too small"),
    ],
)
def test_parse_metadata_rejects_an_inconsistent_transfer(
    field: str, value: int, message: str
) -> None:
    metadata = valid_metadata()
    metadata["transfer_info"][field] = value

    with pytest.raises(ValueError, match=message):
        parse_metadata(metadata)


def test_parse_metadata_rejects_a_range_past_the_pool() -> None:
    metadata = valid_metadata()
    metadata["level_zero_ipc"]["pool_bytes"] = 65536

    with pytest.raises(ValueError, match="exceeds pool size"):
        parse_metadata(metadata)


@pytest.mark.parametrize(
    "driver", [None, LevelZeroDriver(handle=1, devices=())], ids=["none", "empty"]
)
def test_a_driver_without_devices_leaves_level_zero_ipc_unavailable(
    monkeypatch: pytest.MonkeyPatch, driver: LevelZeroDriver | None
) -> None:
    monkeypatch.setattr(torch.version, "xpu", "20260000")
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 0)
    monkeypatch.setattr(level_zero, "level_zero_driver", lambda: driver)
    is_level_zero_ipc_available.cache_clear()
    try:
        assert not is_level_zero_ipc_available()
    finally:
        is_level_zero_ipc_available.cache_clear()


def handle_carrying(descriptor: int) -> bytes:
    """An IPC handle blob whose first four bytes hold this descriptor."""
    blob = bytearray(ZE_MAX_IPC_HANDLE_SIZE)
    struct.pack_into("<i", blob, 0, descriptor)
    return bytes(blob)


def test_a_dma_buf_descriptor_is_read_out_of_a_handle(tmp_path: Path) -> None:
    """The handle is opaque, so the descriptor is only taken once the kernel
    agrees it is a dma_buf big enough for what was exported."""
    plain_file = tmp_path / "not-a-dma-buf"
    plain_file.write_bytes(bytes(8192))
    descriptor = os.open(plain_file, os.O_RDONLY)
    try:
        # A regular file is not a dma_buf, the layout change this guards against.
        with pytest.raises(LevelZeroError, match="not a dma_buf"):
            dma_buf_descriptor(handle_carrying(descriptor), 4096)
    finally:
        os.close(descriptor)

    with pytest.raises(LevelZeroError, match="dma_buf descriptor was expected"):
        dma_buf_descriptor(handle_carrying(-1), 4096)

    closed = os.open(os.devnull, os.O_RDONLY)
    os.close(closed)
    with pytest.raises(LevelZeroError, match="cannot be inspected"):
        dma_buf_descriptor(handle_carrying(closed), 4096)


@pytest.mark.accelerator
def test_a_real_exported_handle_carries_a_dma_buf_of_the_exported_size() -> None:
    if not level_zero.is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass
    context = level_zero.LevelZeroContext()
    # Not a page multiple, so a driver that rounds up still has to cover it.
    requested_bytes = 4096 + 17
    pointer = context.allocate_exportable(0, requested_bytes)
    segment = None
    try:
        segment = context.export_segment(pointer)
        assert os.readlink(f"/proc/self/fd/{segment.file_descriptor}").startswith(
            level_zero.DMA_BUF_LINK_TARGET_PREFIX
        )
        assert os.fstat(segment.file_descriptor).st_size >= requested_bytes
        # A short claim must be refused even though the descriptor is a dma_buf.
        with pytest.raises(LevelZeroError, match="short of"):
            dma_buf_descriptor(segment.handle_blob, 1 << 30)
    finally:
        if segment is not None:
            context.release_segment(segment)
        else:
            pass
        context.free(pointer)
        context.destroy()


@pytest.mark.accelerator
def test_a_driver_that_hides_torch_allocations_leaves_the_relay_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch copies through the pool on SYCL's context, so a driver that keeps
    allocations per context must not be used at all."""
    if not level_zero.is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass
    monkeypatch.setattr(
        level_zero.LevelZeroContext,
        "allocation_identifier",
        lambda self, device_pointer: None,
    )
    assert not level_zero.torch_reads_private_context_memory()


@pytest.mark.accelerator
def test_memory_torch_cannot_read_back_leaves_the_relay_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not level_zero.is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass
    # Wrapping a pointer the probe never wrote stands in for a driver whose
    # cross-context mapping does not carry the bytes.
    monkeypatch.setattr(
        level_zero,
        "CONTEXT_PROBE_PATTERN",
        level_zero.CONTEXT_PROBE_PATTERN ^ 0xFF,
        raising=True,
    )
    original_fill = torch.Tensor.fill_
    monkeypatch.setattr(
        torch.Tensor,
        "fill_",
        lambda self, value: original_fill(self, 0),
    )
    assert not level_zero.torch_reads_private_context_memory()


@pytest.mark.parametrize("failing_probe", ["context", "handles"])
def test_either_driver_probe_failing_keeps_every_edge_on_shm(
    monkeypatch: pytest.MonkeyPatch, failing_probe: str
) -> None:
    monkeypatch.setattr(torch.version, "xpu", "20260000")
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(
        level_zero, "level_zero_driver", lambda: LevelZeroDriver(handle=1, devices=(7,))
    )
    monkeypatch.setattr(level_zero, "exports_dma_buf", lambda device: True)
    monkeypatch.setattr(
        level_zero,
        "torch_reads_private_context_memory",
        lambda: failing_probe != "context",
    )
    monkeypatch.setattr(
        level_zero, "exports_dma_buf_handles", lambda: failing_probe != "handles"
    )
    is_level_zero_ipc_available.cache_clear()
    try:
        assert not is_level_zero_ipc_available()
    finally:
        is_level_zero_ipc_available.cache_clear()


@pytest.mark.accelerator
def test_a_rejected_handle_is_put_back_before_the_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export_segment raising leaves the caller no segment to release, so the
    export it already took has to go back here or its descriptor is stranded."""
    if not level_zero.is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass
    context = level_zero.LevelZeroContext()
    pointer = context.allocate_exportable(0, 4096)
    put_back: list[str] = []
    real_put = context.library.zeMemPutIpcHandle

    def record_put(handle_context, handle):
        put_back.append("put")
        return real_put(handle_context, handle)

    monkeypatch.setattr(context.library, "zeMemPutIpcHandle", record_put)
    monkeypatch.setattr(
        level_zero,
        "dma_buf_descriptor",
        Mock(side_effect=LevelZeroError("not a dma_buf")),
    )
    try:
        before = len(os.listdir("/proc/self/fd"))
        with pytest.raises(LevelZeroError, match="not a dma_buf"):
            context.export_segment(pointer)
        assert put_back == ["put"]
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        context.free(pointer)
        context.destroy()


@pytest.mark.parametrize("failing_step", ["release", "free"])
def test_a_probe_destroys_its_context_even_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    """The probe owns its context, so a caller that falls back to shm has nothing
    left to retry its teardown with."""
    events: list[str] = []

    class FailingContext:
        def __init__(self) -> None:
            events.append("created")

        def allocate_exportable(self, device_index: int, num_bytes: int) -> int:
            return 4096

        def export_segment(self, device_pointer: int) -> SimpleNamespace:
            return SimpleNamespace(base_pointer=device_pointer)

        def release_segment(self, segment: SimpleNamespace) -> None:
            if failing_step == "release":
                raise LevelZeroError("release failed")
            else:
                events.append("released")

        def free(self, device_pointer: int) -> None:
            if failing_step == "free":
                raise LevelZeroError("free failed")
            else:
                events.append("freed")

        def destroy(self) -> None:
            events.append("destroyed")

    monkeypatch.setattr(level_zero, "LevelZeroContext", FailingContext)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)

    # The cleanup failure surfaces instead of the probe's own False; either way
    # is_level_zero_ipc_available falls back to shm, so the context must be gone.
    with pytest.raises(LevelZeroError):
        level_zero.exports_dma_buf_handles()
    assert events[-1] == "destroyed"
