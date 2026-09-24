# SPDX-License-Identifier: Apache-2.0
"""Host shared-memory relay.

This is the host-memory transport: it carries CPU tensors between same-node
processes. GPU-to-GPU edges go through the cuda_ipc relay (NVLink), so the
transport router only selects shm for host-resident data and always builds it
with ``device="cpu"``. Buffers therefore arrive here already on the host.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from multiprocessing import shared_memory as _shm
from typing import Any, Callable

import numpy as np
import torch

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)


def shm_create_from_tensor(tensor: torch.Tensor) -> _shm.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    size = t_np.nbytes

    shm = _shm.SharedMemory(create=True, size=size)
    shm_view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    shm_view[:] = t_np[:]

    return shm


class ShmOperation(RelayOperation):
    """Base class implementation for SHM operations."""

    def __init__(self, metadata: Any):
        self._metadata = metadata  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.completed = False

    @property
    def metadata(self) -> Any:
        return (
            self._metadata
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken


class ShmPutOperation(ShmOperation):
    """Sender-side handle; completion means the receiver consumed the block."""

    def __init__(
        self,
        metadata: Any,
        shm_obj: _shm.SharedMemory,
        *,
        shm_name: str,
        release_cb: Callable[[], None],
    ):
        super().__init__(metadata)
        self.shm_name = shm_name
        self.release_cb = release_cb
        self.receiver_done = asyncio.get_running_loop().create_future()
        shm_obj.close()

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        try:
            await asyncio.wait_for(self.receiver_done, timeout=timeout)
        except TimeoutError as exc:
            self.unlink_if_present()
            raise TimeoutError(
                f"SHM block {self.shm_name} was not consumed in time"
            ) from exc
        except Exception:
            self.unlink_if_present()
            raise
        finally:
            self.completed = True
            self.release_cb()

    def mark_receiver_done(self) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_result(None)
        else:
            pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_exception(exc)
        else:
            pass

    def unlink_if_present(self) -> None:
        try:
            shm = _shm.SharedMemory(name=self.shm_name)
        except FileNotFoundError:
            return
        try:
            shm.unlink()
        finally:
            shm.close()


class ShmGetOperation(ShmOperation):
    """Receiver-side copy from SHM to destination tensor."""

    def __init__(self, metadata: Any, dest_tensor: torch.Tensor):
        super().__init__(metadata)
        self.transfer_info = metadata["transfer_info"]
        self.dest_tensor = dest_tensor

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass

        shm_name = self.transfer_info["shm_name"]
        size = self.transfer_info["size"]

        try:
            try:
                existing_shm = _shm.SharedMemory(name=shm_name)
            except FileNotFoundError:
                raise RuntimeError(f"SHM block {shm_name} not found.")

            try:
                shm_array = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
                src_tensor = torch.from_numpy(shm_array)

                dest_view = self.dest_tensor.view(torch.uint8).reshape(-1)
                if dest_view.numel() < size:
                    raise ValueError(
                        f"SHM destination has {dest_view.numel()} bytes, "
                        f"but transfer requires {size} bytes"
                    )
                else:
                    pass
                dest_view[:size].copy_(src_tensor[:size])

            finally:
                existing_shm.close()
                try:
                    existing_shm.unlink()
                except FileNotFoundError:
                    logger.debug("SHM block %s was already unlinked", shm_name)

        finally:
            self.completed = True


@register_relay("shm")
class ShmRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cpu",
    ):
        self.engine_id = engine_id
        self.device = device
        self.sem = asyncio.Semaphore(credits)
        self.slot_size_bytes = slot_size_mb * 1024 * 1024

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> RelayOperation:
        if request_id is None:
            request_id = str(uuid.uuid4())
        else:
            pass

        await self.sem.acquire()

        try:
            shm = shm_create_from_tensor(tensor)
            size_bytes = shm.size
            metadata = {
                "engine_id": self.engine_id,
                "transfer_info": {
                    "shm_name": shm.name,
                    "size": size_bytes,
                    "req_id": request_id,
                },
            }
            return ShmPutOperation(
                metadata,
                shm,
                shm_name=shm.name,
                release_cb=self.sem.release,
            )

        except Exception:
            self.sem.release()
            raise

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> RelayOperation:
        # Note: metadata validation is implicit here based on usage in test
        return ShmGetOperation(metadata=metadata, dest_tensor=dest_tensor)

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        pass

    # Optional hook for tests
    def reset_pool(self):
        pass
