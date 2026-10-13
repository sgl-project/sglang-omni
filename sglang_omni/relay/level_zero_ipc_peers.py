# SPDX-License-Identifier: Apache-2.0
"""Peer pools a receiving Level-Zero IPC relay maps once per sender pool."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from sglang_omni.platforms.level_zero import LevelZeroContext
from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes
from sglang_omni.relay.level_zero_ipc_metadata import PeerPoolReference
from sglang_omni.relay.level_zero_rendezvous import fetch_peer_segment

logger = logging.getLogger(__name__)

PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()
# A mapping holds its sender's whole pool alive on the sender's card, so a
# receiver that outlives several senders must not keep every pool it ever read.
MAX_MAPPED_PEER_POOLS = 8
PROCESS_UNIX_SOCKETS = "/proc/net/unix"


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
    """A peer pool this process has mapped, and what keeps it mapped."""

    peer_pointer: int
    pool_tensor: torch.Tensor
    socket_name: str
    # Reads that still copy out of this mapping; it may not be unmapped under them.
    readers: int = 0
    last_used_ns: int = 0


def bound_rendezvous_names() -> set[str] | None:
    """Abstract socket names bound in this network namespace, or None if unreadable.

    A sender closes its rendezvous before it frees the pool, so a name that is
    gone means the only thing keeping that memory alive is our mapping. This
    reads the table instead of connecting: a connect would hand the sender's
    serve thread an empty request to log and turn away.
    """
    try:
        with open(PROCESS_UNIX_SOCKETS) as table:
            # The last column is the path; the kernel prefixes abstract names.
            return {
                path[1:]
                for line in table
                if (path := line.rsplit(maxsplit=1)[-1]).startswith("@")
            }
    except OSError as error:
        # Without the table nothing is provably gone, so nothing is unmapped.
        logger.warning(f"Could not read {PROCESS_UNIX_SOCKETS}: {error}")
        return None


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
        # Reads that asked for a pool and are waiting on it. Counted before the
        # wait yields, so a read opening another pool cannot evict the mapping
        # this one is about to be handed.
        self.waiting_readers: dict[str, int] = {}
        # Peer pointers whose unmap failed with no cache entry left to hold them,
        # so close() is the only thing that can try again.
        self.pointers_still_mapped: list[int] = []

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
            window_end = reference.segment_offset_bytes + reference.pool_bytes
            if (
                reference.segment_offset_bytes < 0
                or reference.pool_bytes <= 0
                or window_end > mapped_bytes
            ):
                raise ValueError(
                    f"level_zero_ipc pool {reference.pool_id} claims bytes "
                    f"[{reference.segment_offset_bytes}, {window_end}) of a peer mapping "
                    f"that holds {mapped_bytes}"
                )
            else:
                pass
            pool_tensor = wrap_device_bytes(
                peer_pointer + reference.segment_offset_bytes,
                reference.pool_bytes,
                destination_device_index,
            )
        except BaseException:
            # The cache drops a failed mapping, so this is the only chance to
            # unmap it; a pointer that will not close is kept for close(), and
            # its error must not replace the one that brought us here.
            try:
                self.level_zero.close_peer_segment(peer_pointer)
            except Exception as error:
                self.pointers_still_mapped.append(peer_pointer)
                logger.warning(
                    f"[{self.engine_id}] Could not release the peer mapping at "
                    f"{peer_pointer} that failed to open, keeping it for "
                    f"close(): {error}"
                )
            raise
        logger.info(
            f"[{self.engine_id}] Mapped Level-Zero IPC pool {reference.pool_id} "
            f"from XPU {reference.source_device_index} onto XPU "
            f"{destination_device_index}"
        )
        return PeerPool(
            peer_pointer=peer_pointer,
            pool_tensor=pool_tensor,
            socket_name=reference.socket_name,
        )

    async def pool_tensor_for(
        self,
        reference: PeerPoolReference,
        destination_device_index: int,
    ) -> torch.Tensor:
        """Map a sender's pool once, off the event loop, and reuse the mapping."""
        mapping = self.mappings.get(reference.pool_id)
        if mapping is None:
            self.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)
            mapping = self.wait_executor.submit(
                self.map_peer_pool, reference, destination_device_index
            )
            self.mappings[reference.pool_id] = mapping
            self.forget_when_it_fails(reference.pool_id, mapping)
        else:
            pass
        self.waiting_readers[reference.pool_id] = (
            self.waiting_readers.get(reference.pool_id, 0) + 1
        )
        try:
            # A cancelled read must not abandon a mapping the worker is still opening.
            peer_pool = await asyncio.shield(asyncio.wrap_future(mapping))
        except Exception:
            if self.mappings.get(reference.pool_id) is mapping:
                self.mappings.pop(reference.pool_id)
            else:
                pass
            raise
        finally:
            self.drop_waiting_reader(reference.pool_id)
        # Held until the read's copy drains, so nothing unmaps it underneath.
        peer_pool.readers += 1
        peer_pool.last_used_ns = time.monotonic_ns()
        return peer_pool.pool_tensor

    def forget_when_it_fails(self, pool_id: str, mapping: Future[PeerPool]) -> None:
        """Drop this mapping from the cache if its worker ends up failing.

        The read that asked for it drops a failed mapping itself, but a cancelled
        read never reaches that handler: CancelledError is a BaseException. Left
        cached, the failure would be replayed to the next read instead of
        importing again, and would sit in the cache counting against the pool cap
        with nothing able to sweep it.
        """
        loop = asyncio.get_running_loop()

        def forget_on_the_loop() -> None:
            if self.mappings.get(pool_id) is not mapping:
                # A later read already replaced or dropped it.
                return
            elif mapping.cancelled() or mapping.exception() is not None:
                self.mappings.pop(pool_id, None)
            else:
                pass

        def settled(finished: Future[PeerPool]) -> None:
            # Runs on the worker thread, and the cache belongs to the loop.
            try:
                loop.call_soon_threadsafe(forget_on_the_loop)
            except RuntimeError:
                # The loop is gone, so close() is what releases this mapping.
                pass

        mapping.add_done_callback(settled)

    def drop_waiting_reader(self, pool_id: str) -> None:
        waiting = self.waiting_readers.get(pool_id, 0)
        if waiting > 1:
            self.waiting_readers[pool_id] = waiting - 1
        else:
            self.waiting_readers.pop(pool_id, None)

    def pool_is_busy(self, pool_id: str, peer_pool: PeerPool) -> bool:
        """Whether a read holds this pool or is waiting to be handed it."""
        return peer_pool.readers > 0 or self.waiting_readers.get(pool_id, 0) > 0

    def release_reader(self, pool_id: str) -> None:
        """Report that one read finished copying out of a pool."""
        peer_pool = self.mapped_pool(pool_id)
        if peer_pool is None:
            return
        elif peer_pool.readers > 0:
            peer_pool.readers -= 1
            peer_pool.last_used_ns = time.monotonic_ns()
            # Pools admitted while every mapping was busy can only be reclaimed
            # once their reads drain, and a run of transfers to pools already
            # mapped never reaches the sweep that admitting a new one does. The
            # length test keeps the usual case off /proc/net/unix.
            if len(self.mappings) > MAX_MAPPED_PEER_POOLS:
                self.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS)
            else:
                pass
        else:
            pass

    def mapped_pool(self, pool_id: str) -> PeerPool | None:
        """The pool mapped under this id, or None while it is opening or failed."""
        mapping = self.mappings.get(pool_id)
        if mapping is None or not mapping.done():
            return None
        elif mapping.cancelled() or mapping.exception() is not None:
            return None
        else:
            return mapping.result()

    def release_spent_pools(self, keep_at_most: int) -> None:
        """Unmap the pools no read holds: dead senders' first, then over the cap.

        A sender that restarts publishes a new pool id, so without this every
        pool a long-lived receiver ever read would stay mapped, holding that much
        memory on each sender's card.

        ``keep_at_most`` is the ceiling to leave behind: one below the cap when
        a new mapping is about to be added, the cap itself when trimming pools
        that were admitted over it.

        MAX_MAPPED_PEER_POOLS bounds the pools no read holds, not every mapping.
        A pool being read, or held by a read whose wait timed out, cannot be
        unmapped without pulling memory a copy may still be sourcing, and
        refusing the next transfer instead would fail a request over a condition
        it has no part in. Those pools are reported here and released by close(),
        which synchronizes the device first.
        """
        serving = bound_rendezvous_names()
        for pool_id in list(self.mappings) if serving is not None else ():
            peer_pool = self.mapped_pool(pool_id)
            if peer_pool is None or self.pool_is_busy(pool_id, peer_pool):
                continue
            elif peer_pool.socket_name not in serving:
                if not self.unmap_pool(pool_id, "its sender stopped serving"):
                    return
                else:
                    pass
            else:
                pass

        while len(self.mappings) > keep_at_most:
            idle = [
                (peer_pool.last_used_ns, pool_id)
                for pool_id in self.mappings
                if (peer_pool := self.mapped_pool(pool_id)) is not None
                and not self.pool_is_busy(pool_id, peer_pool)
            ]
            if not idle:
                logger.warning(
                    f"[{self.engine_id}] {len(self.mappings)} Level-Zero IPC peer "
                    f"pools are mapped and every one is being read, so the next "
                    f"sender's pool is mapped over the {MAX_MAPPED_PEER_POOLS} "
                    f"cap; close() releases them"
                )
                return
            else:
                pass
            # A pool that will not unmap would be picked again every round.
            if not self.unmap_pool(min(idle)[1], "the pool cap was reached"):
                return
            else:
                pass

    def unmap_pool(self, pool_id: str, reason: str) -> bool:
        """Unmap one pool; report whether the cache no longer holds it.

        A failed unmap keeps the entry, so close() retries it rather than the
        sender's pool staying mapped with nothing tracking it.
        """
        peer_pool = self.mapped_pool(pool_id)
        if peer_pool is None:
            # Pending or failed: there is no mapping to close.
            self.mappings.pop(pool_id, None)
            return True
        else:
            pass
        try:
            self.level_zero.close_peer_segment(peer_pool.peer_pointer)
        except Exception as error:
            logger.warning(
                f"[{self.engine_id}] Could not release peer pool {pool_id}, "
                f"keeping it for close(): {error}"
            )
            return False
        self.mappings.pop(pool_id, None)
        logger.info(
            f"[{self.engine_id}] Released Level-Zero IPC peer pool {pool_id} "
            f"because {reason}"
        )
        return True

    def close(self) -> None:
        """Unmap every mapped pool, attempting each and raising the first failure."""
        first_error: Exception | None = None
        for peer_pointer in self.pointers_still_mapped:
            try:
                self.level_zero.close_peer_segment(peer_pointer)
                logger.debug(
                    f"[{self.engine_id}] Released the peer mapping at {peer_pointer}"
                )
            except Exception as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.warning(
                        f"[{self.engine_id}] Could not release the peer mapping at "
                        f"{peer_pointer}: {error}"
                    )
        self.pointers_still_mapped.clear()
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
