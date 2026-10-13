# SPDX-License-Identifier: Apache-2.0
"""What the Level-Zero IPC relay refuses before it touches a device."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms.level_zero as level_zero
from sglang_omni.comm.router import CommRouter
from sglang_omni.config.schema import CommConfig
from sglang_omni.pipeline.runtime_config import build_comm_config
from sglang_omni.platforms.level_zero import (
    LevelZeroDriver,
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
        segment_offset=0,
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
