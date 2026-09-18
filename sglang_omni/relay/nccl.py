# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any, Generic, TypedDict, TypeVar

import torch
import torch.distributed as dist

from .base import CreditAllocator, Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

NIXL_AVAILABLE = dist.is_available()

NcclMetadataT = TypeVar("NcclMetadataT")


class NcclAgentMetadata(TypedDict):
    rank: int
    engine_id: str


class NcclTransferInfo(TypedDict):
    size: int
    device_id: int
    shape: list[int]
    dtype: str


class NcclPutMetadata(TypedDict):
    engine_id: str
    agent_meta: NcclAgentMetadata
    transfer_info: NcclTransferInfo


class Connection:
    """
    Manages NCCL Process Group connection with explicit send/recv topology.
    """

    def __init__(
        self,
        engine_id: str,
        rank: int,
        world_size: int,
        send_ranks: list[int],
        recv_ranks: list[int],
    ) -> None:
        self.name = engine_id
        self.rank = rank
        self.world_size = world_size
        self.send_ranks = send_ranks
        self.recv_ranks = recv_ranks

        if any(r >= world_size or r < 0 for r in send_ranks + recv_ranks):
            raise ValueError(
                f"Invalid rank in topology: send={send_ranks}, recv={recv_ranks}, world_size={world_size}"
            )

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")

            if torch.cuda.is_available():
                self.device_id = rank % torch.cuda.device_count()
                torch.cuda.set_device(self.device_id)
            else:
                self.device_id = 0

            dist.init_process_group(
                "nccl",
                rank=rank,
                world_size=world_size,
                device_id=torch.device(f"cuda:{self.device_id}"),
            )
        else:
            self.device_id = (
                torch.cuda.current_device() if torch.cuda.is_available() else 0
            )

        self.group = dist.new_group(list(range(world_size)))

        logger.info(
            f"[{engine_id}] Connection initialized. Rank: {rank}, Send->{send_ranks}, Recv<-{recv_ranks}"
        )

    def get_agent_metadata(self) -> NcclAgentMetadata:
        return {"rank": self.rank, "engine_id": self.name}

    def ensure_remote_agent(
        self,
        remote_engine_id: str,
        remote_meta_bytes: NcclAgentMetadata,
    ) -> int:
        target_rank = remote_meta_bytes.get("rank", 0)
        if target_rank not in self.recv_ranks:
            logger.warning(
                f"[{self.name}] Receiving data from rank {target_rank} which is NOT in recv_ranks {self.recv_ranks}!"
            )
        return target_rank


class NcclOperation(RelayOperation, Generic[NcclMetadataT]):
    """
    Base class for NCCL async operations.
    """

    def __init__(
        self,
        connection: Connection,
        work_handle: dist.Work | None,
        tensor_ref: object,
        metadata: NcclMetadataT | None = None,
    ) -> None:
        self._conn = connection
        self._work = work_handle
        self._tensor_ref = tensor_ref
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> NcclMetadataT | None:
        return self._metadata


class PutOperation(NcclOperation[NcclMetadataT]):
    """Handle for a Put operation (NCCL isend)."""

    def __init__(
        self,
        connection: Connection,
        work_handle: dist.Work | None,
        tensor_ref: torch.Tensor,
        metadata: NcclMetadataT,
        on_completion_cb: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(connection, work_handle, tensor_ref, metadata)
        self._on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        start = time.time()
        try:
            while not self._work.is_completed():
                if time.time() - start > timeout:
                    raise TimeoutError(f"PutOperation timed out")
                await asyncio.sleep(0.0001)

            self._work.wait()

        finally:
            self._completed = True
            if self._on_completion_cb:
                self._on_completion_cb()


class GetOperation(NcclOperation[None]):
    """
    Handle for a Get operation (NCCL irecv).
    """

    def __init__(
        self,
        connection: Connection,
        work_handle: dist.Work | None,
        dest_tensor: torch.Tensor,
    ) -> None:
        super().__init__(connection, work_handle, dest_tensor, metadata=None)

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        start = time.time()
        try:
            while not self._work.is_completed():
                if time.time() - start > timeout:
                    raise TimeoutError(f"GetOperation timed out")
                await asyncio.sleep(0.0001)

            self._work.wait()
        finally:
            self._completed = True


@register_relay("nccl")
class NcclRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        send_to_ranks: list[int],
        recv_from_ranks: list[int],
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cuda",
        rank: int | None = None,
        world_size: int = 2,
    ) -> None:
        self.engine_id = engine_id
        self.device = device

        self.device_id = 0
        if "cuda" in device and ":" in device:
            try:
                self.device_id = int(device.split(":")[1])
            except ValueError:
                self.device_id = 0

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)

        if rank is None:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 2))

        self.connection = Connection(
            engine_id,
            rank,
            world_size,
            send_ranks=send_to_ranks,
            recv_ranks=recv_from_ranks,
        )
        self.allocator = CreditAllocator(credits=credits)

        try:
            dist.barrier(group=self.connection.group)
        except Exception as e:
            logger.error(f"Barrier failed: {e}")
            raise e

        logger.info(
            f"[{engine_id}] Initialized NCCL Relay on {device} (Rank {rank}). Starting Warmup..."
        )

        dummy_tensor = torch.tensor([1.0], device=f"cuda:{self.device_id}")
        warmup_reqs = []

        try:
            for dst in self.connection.send_ranks:
                req = dist.isend(dummy_tensor, dst=dst, group=self.connection.group)
                warmup_reqs.append(req)

            for src in self.connection.recv_ranks:
                req = dist.irecv(dummy_tensor, src=src, group=self.connection.group)
                warmup_reqs.append(req)

            if warmup_reqs:
                for req in warmup_reqs:
                    req.wait()

            dist.barrier(group=self.connection.group)

        except Exception as e:
            logger.error(
                f"[{engine_id}] NCCL Warmup failed! Check topology consistency."
            )
            raise e

        logger.info(f"[{engine_id}] Rank {rank}: Warmup complete. Ready.")

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> PutOperation[NcclPutMetadata]:
        if dst_rank is None:
            if len(self.connection.send_ranks) == 1:
                dst_rank = self.connection.send_ranks[0]
            else:
                raise ValueError(
                    f"Ambiguous destination! send_ranks={self.connection.send_ranks}, but dst_rank is None."
                )

        if dst_rank not in self.connection.send_ranks:
            logger.warning(
                f"Sending to rank {dst_rank} which is NOT in send_ranks whitelist!"
            )

        credit_id = await self.allocator.acquire_async()

        work_handle = dist.isend(
            tensor=tensor, dst=dst_rank, group=self.connection.group
        )

        payload: NcclPutMetadata = {
            "engine_id": self.engine_id,
            "agent_meta": self.connection.get_agent_metadata(),
            "transfer_info": {
                "size": tensor.numel() * tensor.element_size(),
                "device_id": self.device_id,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
        }

        return PutOperation(
            connection=self.connection,
            work_handle=work_handle,
            tensor_ref=tensor,
            metadata=payload,
            on_completion_cb=lambda: self.allocator.release(credit_id),
        )

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
        src_rank: int | None = None,
    ) -> GetOperation:
        """Asynchronously get tensor via NCCL Zero-Copy."""
        remote_engine_id = metadata["engine_id"]
        remote_agent_meta = metadata["agent_meta"]

        if src_rank is None:
            src_rank = self.connection.ensure_remote_agent(
                remote_engine_id, remote_agent_meta
            )

        work = dist.irecv(tensor=dest_tensor, src=src_rank, group=self.connection.group)

        return GetOperation(
            connection=self.connection,
            work_handle=work,
            dest_tensor=dest_tensor,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
