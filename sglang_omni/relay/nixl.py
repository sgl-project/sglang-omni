# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ==========================================
# Dependency Check
# ==========================================

from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config

NIXL_AVAILABLE = True


# ==========================================
# Helper Classes
# ==========================================
class LinearAllocator:
    def __init__(self, total_size: int, base_ptr: int):
        self.total_size = total_size
        self.base_ptr = base_ptr
        self.current_offset = 0

    def allocate(self, size: int) -> int:
        aligned_size = (size + 255) & ~255
        if self.current_offset + aligned_size > self.total_size:
            raise MemoryError("Memory Pool exhausted!")
        offset = self.current_offset
        self.current_offset += aligned_size
        return offset

    def reset(self):
        self.current_offset = 0


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


class NixlOperation:
    def __init__(
        self,
        connection: Connection,
        handle: Optional[int] = None,
        metadata: Any = None,
        expected_notification: Optional[bytes] = None,
    ):
        self._conn = connection
        self._handle = handle
        self._metadata = metadata
        self._expected_notification = expected_notification
        # If there's a handle, wait for transfer; if there's expected_notification, wait for notification
        self._completed = False if (handle or expected_notification) else True

    def metadata(self) -> Any:
        return self._metadata

    def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        # If there's a handle, wait for transfer completion
        if self._handle:
            while True:
                state = self._conn._nixl.check_xfer_state(self._handle)
                if state == "DONE":
                    break
                elif state != "PROC":
                    raise RuntimeError(f"Transfer failed: {state}")
                # Busy wait
            if self._handle:
                self._conn._nixl.release_xfer_handle(self._handle)
            self._completed = True

        # If there's expected_notification, wait for notification
        elif self._expected_notification:
            start = time.time()
            while True:
                notifs = self._conn._nixl.get_new_notifs()
                for msgs in notifs.values():
                    if self._expected_notification in msgs:
                        self._completed = True
                        return
                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"Wait for {self._expected_notification} timed out"
                    )
                time.sleep(0.0001)

    async def wait_for_completion_async(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        # If there's a handle, wait for transfer completion
        if self._handle:
            while True:
                state = self._conn._nixl.check_xfer_state(self._handle)
                if state == "DONE":
                    break
                elif state != "PROC":
                    raise RuntimeError(f"Transfer failed: {state}")
                await asyncio.sleep(0.001)
            if self._handle:
                self._conn._nixl.release_xfer_handle(self._handle)
            self._completed = True

        # If there's expected_notification, wait for notification
        elif self._expected_notification:
            start = time.time()
            while True:
                notifs = self._conn._nixl.get_new_notifs()
                for msgs in notifs.values():
                    if self._expected_notification in msgs:
                        self._completed = True
                        return
                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"Wait for {self._expected_notification} timed out"
                    )
                await asyncio.sleep(0.001)


# ==========================================
# Core Class NixlRelay (Pool Implementation)
# ==========================================
class NixlRelay:
    def __init__(self, engine_id: str, pool_size_mb: int = 128, device: str = "cuda"):
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
        pool_bytes = pool_size_mb * 1024 * 1024
        self.pool_tensor = torch.zeros(pool_bytes, dtype=torch.uint8, device=device)
        self.pool_ptr = self.pool_tensor.data_ptr()
        self.pool_size = pool_bytes

        self.allocator = LinearAllocator(self.pool_size, self.pool_ptr)

        # 3. Register memory pool
        logger.info(
            f"[{engine_id}] Registering Pool ({pool_size_mb} MB) on {device} (ID: {self.device_id})..."
        )
        if NIXL_AVAILABLE:
            mem_type = "VRAM" if "cuda" in device else "DRAM"
            reg_list = [(self.pool_ptr, self.pool_size, self.device_id, mem_type)]
            self.pool_handle = self.connection._nixl.register_memory(reg_list, mem_type)
        else:
            self.pool_handle = 1

    def put(self, tensor: torch.Tensor, request_id: str = None) -> NixlOperation:
        size_bytes = tensor.numel() * tensor.element_size()
        offset = self.allocator.allocate(size_bytes)

        # D2D Copy
        pool_slice = self.pool_tensor[offset : offset + size_bytes]
        tensor_view = tensor.view(torch.uint8).reshape(-1)
        pool_slice.copy_(tensor_view)

        payload = {
            "engine_id": self.engine_id,
            "agent_meta": self.connection.get_agent_metadata(),
            "transfer_info": {
                "offset": offset,
                "size": size_bytes,
                "ptr": self.pool_ptr + offset,
                "device_id": self.device_id,  # Sender informs its own device_id
            },
        }
        # Set expected notification so wait_for_completion() can wait for receiver to finish reading
        expected_notification = b"done"
        return NixlOperation(
            self.connection,
            metadata=payload,
            expected_notification=expected_notification,
        )

    # Add async interface for compatibility with Stage code
    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None
    ) -> NixlOperation:
        return self.put(tensor, request_id)

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> NixlOperation:
        """Async version of get method for compatibility with stage.py"""
        # Run synchronous get method in event loop executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get, metadata, dest_tensor, request_id
        )

    def get(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> NixlOperation:
        remote_engine_id = metadata["engine_id"]
        remote_agent_meta = metadata["agent_meta"]

        # Compatible with different metadata formats
        if "transfer_info" in metadata:
            xfer_info = metadata["transfer_info"]
            remote_ptr = xfer_info["ptr"]
            data_size = xfer_info["size"]
            # [Key] Get remote device_id, default to 0
            remote_device_id = xfer_info.get("device_id", 0)
        elif "descriptors" in metadata:
            # Compatible with legacy format
            desc = metadata["descriptors"][0]
            remote_ptr = desc["ptr"]
            data_size = desc["size"]
            # Try to parse ID from device string
            remote_device_str = desc.get("device", "cuda:0")
            remote_device_id = 0
            if ":" in remote_device_str:
                try:
                    remote_device_id = int(remote_device_str.split(":")[1])
                except:
                    pass
        else:
            raise ValueError("Invalid metadata format")

        local_offset = self.allocator.allocate(data_size)
        remote_agent_name = self.connection.ensure_remote_agent(
            remote_engine_id, remote_agent_meta
        )

        mem_type = "VRAM" if "cuda" in self.device else "DRAM"

        # Local DList
        local_phys_addr = self.pool_ptr + local_offset
        local_descs = self.connection._nixl.get_xfer_descs(
            [(local_phys_addr, data_size, self.device_id)], mem_type
        )
        local_handle = self.connection._nixl.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_descs
        )

        # Remote DList
        # [Key fix] Use remote_device_id obtained from metadata
        remote_descs = self.connection._nixl.get_xfer_descs(
            [(remote_ptr, data_size, remote_device_id)], mem_type
        )
        remote_handle = self.connection._nixl.prep_xfer_dlist(
            remote_agent_name, remote_descs
        )

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

        op = NixlOperation(self.connection, handle=xfer_handle)
        op.wait_for_completion()

        # D2D Copy
        pool_slice = self.pool_tensor[local_offset : local_offset + data_size]
        dest_view = dest_tensor.view(torch.uint8).reshape(-1)
        dest_view.copy_(pool_slice)

        return op

    def wait_for_notification(self, expected_msg: bytes, timeout: float = 30.0):
        start = time.time()
        while True:
            notifs = self.connection._nixl.get_new_notifs()
            for msgs in notifs.values():
                if expected_msg in msgs:
                    return
            if time.time() - start > timeout:
                raise TimeoutError(f"Wait for {expected_msg} timed out")
            time.sleep(0.0001)

    def reset_pool(self):
        if "cuda" in self.device:
            torch.cuda.synchronize()
        self.allocator.reset()

    def cleanup(self, request_id: str):
        pass

    def close(self):
        if NIXL_AVAILABLE and hasattr(self, "pool_handle"):
            try:
                self.connection._nixl.deregister_memory(self.pool_handle)
            except:
                pass
