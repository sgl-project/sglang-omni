# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict

import numpy as np
import torch

from .base import CreditAllocator, Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

# ==========================================
# Dependency Check
# ==========================================
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config

    NIXL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import nixl: {e}. NixlRelay will not work.")
    NIXL_AVAILABLE = False


class Connection:
    def __init__(self, engine_id: str, num_threads: int = 2):
        self.name = engine_id
        config = nixl_agent_config(num_threads=num_threads)
        self._nixl = NixlAgent(str(uuid.uuid4()), config)
        self._remote_agents: Dict[str, str] = {}

    def get_agent_metadata(self) -> bytes:
        return self._nixl.get_agent_metadata()

    def ensure_remote_agent(
        self, remote_engine_id: str, remote_meta_bytes: bytes
    ) -> str:
        if remote_engine_id not in self._remote_agents:
            agent_name = self._nixl.add_remote_agent(remote_meta_bytes)
            self._remote_agents[remote_engine_id] = agent_name
        return self._remote_agents[remote_engine_id]


class NixlOperation(RelayOperation):
    """Base class for async operations."""

    def __init__(self, connection: Connection, metadata: Any = None):
        self._conn = connection
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> Any:
        return self._metadata


class PutOperation(NixlOperation):
    """
    Handle for a Put operation.
    Waits for a notification from the receiver indicating they have finished reading.
    Releases the sender's credit upon completion.
    """

    def __init__(
        self,
        connection: Connection,
        metadata: Any,
        expected_notification: bytes,
        on_completion_cb: Callable[[], None],
    ):
        super().__init__(connection, metadata)
        self._expected_notification = expected_notification
        self._on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        start = time.time()
        try:
            while True:
                notifs = self._conn._nixl.get_new_notifs()
                found = False
                for msgs in notifs.values():
                    if self._expected_notification in msgs:
                        found = True
                        break

                if found:
                    break

                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"PutOperation timed out waiting for {self._expected_notification}"
                    )

                # Non-blocking wait
                await asyncio.sleep(0.0001)
        finally:
            # Regardless of success or timeout, we mark complete.
            # In a real system, you might want distinct handling for timeout vs success.
            self._completed = True
            # Release the credit so new puts can happen
            self._on_completion_cb()


class GetOperation(NixlOperation):
    """
    Handle for a Get operation.
    Waits for the RDMA transfer handle to complete.
    Upon completion, copies data from Pool to Dest Tensor, then releases local credit.
    """

    def __init__(
        self,
        connection: Connection,
        handle: int,
        src_pool_tensor: torch.Tensor,
        dest_tensor: torch.Tensor,
        copy_size: int,
        on_completion_cb: Callable[[], None],
    ):
        super().__init__(
            connection, metadata=None
        )  # Get usually doesn't return metadata
        self._handle = handle
        self._src_pool_tensor = src_pool_tensor
        self._dest_tensor = dest_tensor
        self._copy_size = copy_size
        self._on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        try:
            # 1. Wait for RDMA Transfer
            while True:
                state = self._conn._nixl.check_xfer_state(self._handle)
                if state == "DONE":
                    break
                elif state != "PROC":
                    raise RuntimeError(f"Transfer failed with state: {state}")

                await asyncio.sleep(0.00001)

            # 2. Cleanup Handle
            self._conn._nixl.release_xfer_handle(self._handle)
            self._handle = None

            # 3. Perform Copy (Pool -> Dest)
            # This ensures data is valid before the user gets control back
            # Note: This is a GPU-GPU copy (fast), but conceptually blocking the stream.
            src_view = self._src_pool_tensor[: self._copy_size]
            dest_view = self._dest_tensor.view(torch.uint8).reshape(-1)
            dest_view.copy_(src_view)

        finally:
            self._completed = True
            # 4. Release Local Credit (Buffer is now free)
            self._on_completion_cb()


# ==========================================
# NixlRelay (Async Only)
# ==========================================


@register_relay("nixl")
class NixlRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cuda",
    ):
        self.engine_id = engine_id
        self.device = device
        self.connection = Connection(engine_id)

        # 1. Parse Device ID
        self.device_id = 0
        if "cuda" in device and ":" in device:
            try:
                self.device_id = int(device.split(":")[1])
            except ValueError:
                self.device_id = 0

        # 2. Initialize memory pool
        slot_bytes = slot_size_mb * 1024 * 1024
        total_pool_bytes = slot_bytes * credits

        self.pool_tensor = torch.zeros(
            total_pool_bytes, dtype=torch.uint8, device=device
        )
        self.pool_ptr = self.pool_tensor.data_ptr()

        self.allocator = CreditAllocator(
            credits=credits, slot_size=slot_bytes, base_ptr=self.pool_ptr
        )

        # 3. Register memory pool
        logger.info(
            f"[{engine_id}] Registering Pool ({total_pool_bytes / 1024**2:.2f} MB) on {device}..."
        )
        if NIXL_AVAILABLE:
            mem_type = "VRAM" if "cuda" in device else "DRAM"
            reg_list = [(self.pool_ptr, total_pool_bytes, self.device_id, mem_type)]
            self.pool_handle = self.connection._nixl.register_memory(reg_list, mem_type)
        else:
            self.pool_handle = 1

    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> PutOperation:
        """
        Asynchronously put tensor. Returns a PutOperation.
        """
        size_bytes = tensor.numel() * tensor.element_size()
        if size_bytes > self.allocator.slot_size:
            raise ValueError(f"Tensor size {size_bytes} exceeds slot size")

        # 1. Async Wait for Credit
        offset = await self.allocator.acquire_async()

        try:
            # 2. Copy Data to Pool
            pool_slice = self.pool_tensor[offset : offset + size_bytes]
            tensor_view = tensor.view(torch.uint8).reshape(-1)
            pool_slice.copy_(tensor_view)

            # 3. Prepare Metadata
            mem_type = "VRAM" if "cuda" in self.device else "DRAM"
            payload = {
                "engine_id": self.engine_id,
                "agent_meta": self.connection.get_agent_metadata(),
                "mem_type": mem_type,
                "transfer_info": {
                    "offset": offset,
                    "size": size_bytes,
                    "ptr": self.pool_ptr + offset,
                    "device_id": self.device_id,
                },
            }

            # 4. Create Operation
            # Credit will be released when PutOperation.wait() completes
            return PutOperation(
                connection=self.connection,
                metadata=payload,
                expected_notification=b"done",
                on_completion_cb=lambda: self.allocator.release(offset),
            )

        except Exception as e:
            # If creating the op fails, release immediately
            self.allocator.release(offset)
            raise e

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> GetOperation:
        """
        Asynchronously get tensor. Returns a GetOperation.
        """
        # Parse Metadata
        remote_engine_id = metadata["engine_id"]
        remote_agent_meta = metadata["agent_meta"]

        if "transfer_info" in metadata:
            xfer_info = metadata["transfer_info"]
            remote_ptr = xfer_info["ptr"]
            data_size = xfer_info["size"]
            remote_device_id = xfer_info.get("device_id", 0)
        else:
            raise ValueError("Invalid metadata format")

        if data_size > self.allocator.slot_size:
            raise ValueError("Data size exceeds local slot size")

        # 1. Async Wait for Local Credit (Buffer)
        local_offset = await self.allocator.acquire_async()

        try:
            remote_agent_name = self.connection.ensure_remote_agent(
                remote_engine_id, remote_agent_meta
            )
            mem_type = "VRAM" if "cuda" in self.device else "DRAM"
            remote_mem_type = metadata.get("mem_type", mem_type)

            # 2. Prepare RDMA
            local_phys_addr = self.pool_ptr + local_offset
            local_descs = self.connection._nixl.get_xfer_descs(
                [(local_phys_addr, data_size, self.device_id)], mem_type
            )
            local_handle = self.connection._nixl.prep_xfer_dlist(
                "NIXL_INIT_AGENT", local_descs
            )

            remote_descs = self.connection._nixl.get_xfer_descs(
                [(remote_ptr, data_size, remote_device_id)], remote_mem_type
            )
            remote_handle = self.connection._nixl.prep_xfer_dlist(
                remote_agent_name, remote_descs
            )

            # 3. Trigger Transfer
            indices = np.arange(1, dtype=np.int64)
            xfer_handle = self.connection._nixl.make_prepped_xfer(
                "READ",
                local_handle,
                indices,
                remote_handle,
                indices,
                notif_msg=f"done".encode(),
            )
            self.connection._nixl.transfer(xfer_handle)

            # 4. Create Operation
            # Pass the pool slice (buffer) so the Operation can copy it out later
            pool_slice = self.pool_tensor[local_offset : local_offset + data_size]

            return GetOperation(
                connection=self.connection,
                handle=xfer_handle,
                src_pool_tensor=pool_slice,
                dest_tensor=dest_tensor,
                copy_size=data_size,
                on_completion_cb=lambda: self.allocator.release(local_offset),
            )

        except Exception as e:
            self.allocator.release(local_offset)
            raise e

    def cleanup(self, request_id: str):
        pass

    def close(self):
        if NIXL_AVAILABLE and hasattr(self, "pool_handle"):
            try:
                self.connection._nixl.deregister_memory(self.pool_handle)
            except:
                pass
