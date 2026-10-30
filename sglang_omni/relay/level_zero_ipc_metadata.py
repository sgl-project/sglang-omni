# SPDX-License-Identifier: Apache-2.0
"""Control-plane metadata of a Level-Zero IPC transfer and its validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from sglang_omni.comm.data_ref import required
from sglang_omni.relay.slot_pool import slots_for_size


class LevelZeroPoolInfo(TypedDict):
    pool_id: str
    socket_name: str
    segment_offset: int
    pool_bytes: int
    source_device_index: int


LevelZeroIpcMetadata = dict[str, str | dict[str, int] | LevelZeroPoolInfo]


@dataclass(frozen=True, kw_only=True)
class TransferPlacement:
    """Where one payload sits inside the sender's pool."""

    size_bytes: int
    offset_bytes: int
    slot_index: int
    slot_size_bytes: int
    num_slots: int


@dataclass(frozen=True, kw_only=True)
class PeerPoolReference:
    """How a receiver reaches the pool that holds a payload."""

    pool_id: str
    socket_name: str
    segment_offset: int
    pool_bytes: int
    source_device_index: int


def parse_metadata(
    metadata: Mapping[str, object],
) -> tuple[TransferPlacement, PeerPoolReference]:
    transfer_info = required(metadata, "transfer_info", dict)
    pool_info = required(metadata, "level_zero_ipc", dict)
    placement = TransferPlacement(
        size_bytes=required(transfer_info, "size", int),
        offset_bytes=required(transfer_info, "offset", int),
        slot_index=required(transfer_info, "slot_index", int),
        slot_size_bytes=required(transfer_info, "slot_size", int),
        num_slots=required(transfer_info, "num_slots", int),
    )
    reference = PeerPoolReference(
        pool_id=required(pool_info, "pool_id", str),
        socket_name=required(pool_info, "socket_name", str),
        segment_offset=required(pool_info, "segment_offset", int),
        pool_bytes=required(pool_info, "pool_bytes", int),
        source_device_index=required(pool_info, "source_device_index", int),
    )

    if placement.offset_bytes < 0:
        raise ValueError("level_zero_ipc transfer offset must be non-negative")
    elif placement.slot_size_bytes <= 0 or placement.num_slots <= 0:
        raise ValueError("level_zero_ipc slot_size and num_slots must be positive")
    elif placement.offset_bytes % placement.slot_size_bytes != 0:
        raise ValueError("level_zero_ipc transfer offset must be slot aligned")
    elif placement.num_slots < slots_for_size(
        placement.size_bytes, placement.slot_size_bytes
    ):
        raise ValueError("level_zero_ipc num_slots is too small for transfer size")
    elif (
        placement.offset_bytes + placement.num_slots * placement.slot_size_bytes
        > reference.pool_bytes
    ):
        raise ValueError("level_zero_ipc allocation range exceeds pool size")
    else:
        pass
    return placement, reference
