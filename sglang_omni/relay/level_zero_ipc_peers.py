# SPDX-License-Identifier: Apache-2.0
"""Peer pools a receiving Level-Zero IPC relay maps once per sender pool."""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from sglang_omni.platforms.level_zero import LevelZeroContext
from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes
from sglang_omni.relay.level_zero_ipc_metadata import PeerPoolReference
from sglang_omni.relay.level_zero_rendezvous import fetch_peer_segment

logger = logging.getLogger(__name__)

PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()


def ensure_peer_access(source_device_index: int, destination_device_index: int) -> None:
    """Reject a peer copy the importing device cannot perform."""
    if source_device_index == destination_device_index:
        return
    else:
        pass
    device_count = torch.xpu.device_count()
    if source_device_index >= device_count:
        warn_key = (destination_device_index, source_device_index, device_count)
        if warn_key not in PEER_VISIBILITY_WARNED:
            PEER_VISIBILITY_WARNED.add(warn_key)
            logger.warning(
                f"level_zero_ipc source device {source_device_index} is outside this "
                f"receiver's visible range [0, {device_count}); peer-access "
                f"validation skipped. This is expected only when sender and receiver "
                f"use different ZE_AFFINITY_MASK namespaces."
            )
        else:
            pass
        return
    elif not torch.xpu.can_device_access_peer(
        destination_device_index, source_device_index
    ):
        raise RuntimeError(
            f"level_zero_ipc cross-GPU transfer requires peer access, but XPU "
            f"{destination_device_index} cannot access XPU {source_device_index}; "
            f"use the shm relay for host-staged transfer"
        )
    else:
        pass


@dataclass(kw_only=True)
class PeerPool:
    """A peer pool this process has mapped, kept for the relay's lifetime."""

    peer_pointer: int
    pool_tensor: torch.Tensor


class PeerPoolMappings:
    """The peer pools one receiving relay has mapped, keyed by pool id."""

    def __init__(
        self,
        *,
        level_zero: LevelZeroContext,
        wait_executor: ThreadPoolExecutor,
        engine_id: str,
    ) -> None:
        self.level_zero = level_zero
        self.wait_executor = wait_executor
        self.engine_id = engine_id
        self.mappings: dict[str, Future[PeerPool]] = {}

    def map_peer_pool(
        self,
        reference: PeerPoolReference,
        destination_device_index: int,
    ) -> PeerPool:
        """Collect a sender's handle and map its pool. Blocks on the sender's socket."""
        handle_blob, file_descriptor = fetch_peer_segment(
            reference.socket_name, reference.pool_id
        )
        try:
            peer_pointer = self.level_zero.open_peer_segment(
                handle_blob, file_descriptor, destination_device_index
            )
        finally:
            # The driver holds its own reference to the mapping once it is open.
            os.close(file_descriptor)
        try:
            # The window comes off the wire, so it has to fit what was mapped.
            mapping = self.level_zero.address_range(peer_pointer)
            mapped_bytes = mapping.base_pointer + mapping.segment_bytes - peer_pointer
            window_end = reference.segment_offset + reference.pool_bytes
            if (
                reference.segment_offset < 0
                or reference.pool_bytes <= 0
                or window_end > mapped_bytes
            ):
                raise ValueError(
                    f"level_zero_ipc pool {reference.pool_id} claims bytes "
                    f"[{reference.segment_offset}, {window_end}) of a peer mapping "
                    f"that holds {mapped_bytes}"
                )
            else:
                pass
            pool_tensor = wrap_device_bytes(
                peer_pointer + reference.segment_offset,
                reference.pool_bytes,
                destination_device_index,
            )
        except BaseException:
            # The cache drops a failed mapping, so nothing else would unmap it.
            self.level_zero.close_peer_segment(peer_pointer)
            raise
        logger.info(
            f"[{self.engine_id}] Mapped Level-Zero IPC pool {reference.pool_id} "
            f"from XPU {reference.source_device_index} onto XPU "
            f"{destination_device_index}"
        )
        return PeerPool(peer_pointer=peer_pointer, pool_tensor=pool_tensor)

    async def pool_tensor_for(
        self,
        reference: PeerPoolReference,
        destination_device_index: int,
    ) -> torch.Tensor:
        """Map a sender's pool once, off the event loop, and reuse the mapping."""
        mapping = self.mappings.get(reference.pool_id)
        if mapping is None:
            mapping = self.wait_executor.submit(
                self.map_peer_pool, reference, destination_device_index
            )
            self.mappings[reference.pool_id] = mapping
        else:
            pass
        try:
            # A cancelled read must not abandon a mapping the worker is still opening.
            peer_pool = await asyncio.shield(asyncio.wrap_future(mapping))
        except Exception:
            if self.mappings.get(reference.pool_id) is mapping:
                self.mappings.pop(reference.pool_id)
            else:
                pass
            raise
        return peer_pool.pool_tensor

    def close(self) -> None:
        """Unmap every mapped pool, attempting each and raising the first failure."""
        first_error: Exception | None = None
        for pool_id, mapping in self.mappings.items():
            if mapping.cancelled() or mapping.exception() is not None:
                continue
            else:
                pass
            try:
                self.level_zero.close_peer_segment(mapping.result().peer_pointer)
                logger.debug(f"[{self.engine_id}] Released peer pool {pool_id}")
            except Exception as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.warning(
                        f"[{self.engine_id}] Could not release peer pool {pool_id}: "
                        f"{error}"
                    )
        self.mappings.clear()
        if first_error is not None:
            raise first_error
        else:
            pass
