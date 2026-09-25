from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from collections.abc import Mapping
from typing import Callable, Dict, Generic, TypeVar

import torch

from .base import CreditAllocator, Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)
try:
    from mooncake.engine import TransferEngine, TransferNotify, TransferOpcode

    MOONCAKE_AVAILABLE = True
except ImportError as e:
    logger.error(
        f"Failed to import mooncake: {e}. MooncakeRelay will not work. Install with: pip install mooncake-transfer-engine"
    )
    MOONCAKE_AVAILABLE = False

    class TransferEngine:

        def __init__(self):
            pass

        def initialize(self, *args):
            return -1

        def get_rpc_port(self):
            return 0

        def register_memory(self, *args):
            return -1

        def unregister_memory(self, *args):
            pass

        def transfer_sync_write(self, *args):
            return -1

        def transfer_sync_read(self, *args):
            return -1

        def transfer_sync(self, *args, **kwargs):
            return -1

        def get_notifies(self):
            return []

    class TransferNotify:

        def __init__(self, name, message):
            self.name = name
            self.message = message

    class TransferOpcode:
        Read = 0
        Write = 1


class MooncakeConnection:
    """
    Wrapper for Mooncake TransferEngine connection.
    Uses P2P handshake mode - no etcd required!
    """

    def __init__(
        self,
        engine_id: str,
        hostname: str,
        protocol: str = "tcp",
        device_name: str = "",
    ):
        """
        Initialize Mooncake connection using P2P handshake mode.

        Args:
            engine_id: Unique identifier for this engine instance
            hostname: Local hostname or IP address
            protocol: Transfer protocol ("tcp", "rdma", "nvlink")
            device_name: RDMA device name (e.g., "mlx5_0") for RDMA protocol
        """
        self.engine_id = engine_id
        self.hostname = hostname
        self.protocol = protocol
        self.device_name = device_name
        self.engine = TransferEngine()
        logger.debug(
            f"[{engine_id}] Initializing TransferEngine with protocol='{protocol}'"
        )
        ret = self.engine.initialize(
            hostname, "P2PHANDSHAKE", protocol, device_name if device_name else ""
        )
        logger.debug(f"[{engine_id}] TransferEngine initialization returned: {ret}")
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake TransferEngine (error code: {ret})"
            )
        else:
            pass
        self.session_id = f"{hostname}:{self.engine.get_rpc_port()}"
        self.memory_handles: Dict[int, int] = {}
        logger.info(
            f"[{engine_id}] Mooncake connection initialized: protocol={protocol}, session_id={self.session_id}"
        )

    def register_memory(self, ptr: int, size: int) -> int:
        """
        Register GPU/CPU memory with Mooncake.

        Args:
            ptr: Memory pointer
            size: Memory size in bytes

        Returns:
            Memory handle (ptr itself)
        """
        if ptr in self.memory_handles:
            logger.warning(f"Memory at {hex(ptr)} already registered")
            return self.memory_handles[ptr]
        else:
            pass
        ret = self.engine.register_memory(ptr, size)
        if ret != 0:
            raise RuntimeError(f"Failed to register memory (error code: {ret})")
        else:
            pass
        self.memory_handles[ptr] = ptr
        logger.debug(f"Registered memory: ptr={hex(ptr)}, size={size} bytes")
        return ptr

    def deregister_memory(self, handle: int) -> None:
        """Deregister memory from Mooncake."""
        if handle not in self.memory_handles:
            logger.warning(f"Memory handle {hex(handle)} not found")
            return
        else:
            pass
        try:
            self.engine.unregister_memory(handle)
            del self.memory_handles[handle]
            logger.debug(f"Deregistered memory: handle={hex(handle)}")
        except Exception as e:
            logger.error(f"Failed to deregister memory: {e}")

    def transfer_sync_write(
        self, session_id: str, src_ptr: int, dst_ptr: int, size: int
    ) -> int:
        """
        Synchronously transfer data to remote peer.

        Args:
            session_id: Remote peer session ID (hostname:port)
            src_ptr: Source buffer pointer
            dst_ptr: Destination buffer pointer
            size: Transfer size in bytes

        Returns:
            0 on success, negative on error
        """
        ret = self.engine.transfer_sync_write(session_id, src_ptr, dst_ptr, size)
        if ret < 0:
            raise RuntimeError(f"Transfer failed with error code: {ret}")
        else:
            pass
        return ret

    def transfer_sync_read(
        self, session_id: str, local_ptr: int, remote_ptr: int, size: int
    ) -> int:
        """
        Synchronously read data from remote peer.

        Args:
            session_id: Remote peer session ID (hostname:port)
            local_ptr: Local buffer pointer (destination)
            remote_ptr: Remote buffer pointer (source)
            size: Transfer size in bytes

        Returns:
            0 on success, negative on error
        """
        ret = self.engine.transfer_sync_read(session_id, local_ptr, remote_ptr, size)
        if ret < 0:
            raise RuntimeError(f"Transfer failed with error code: {ret}")
        else:
            pass
        return ret

    def transfer_sync(
        self,
        session_id: str,
        local_ptr: int,
        remote_ptr: int,
        size: int,
        opcode,
        notify=None,
    ) -> int:
        """
        Synchronously transfer data with optional notification.

        Args:
            session_id: Remote peer session ID (hostname:port)
            local_ptr: Local buffer pointer
            remote_ptr: Remote buffer pointer
            size: Transfer size in bytes
            opcode: Transfer operation (TransferOpcode.READ or WRITE)
            notify: Optional TransferNotify object

        Returns:
            0 on success, negative on error
        """
        ret = self.engine.transfer_sync(
            session_id, local_ptr, remote_ptr, size, opcode, notify
        )
        if ret < 0:
            raise RuntimeError(f"Transfer failed with error code: {ret}")
        else:
            pass
        return ret

    def get_notifies(self):
        """
        Get pending transfer notifications from remote peers.

        Returns:
            List of TransferNotify objects
        """
        return self.engine.get_notifies()

    def close(self) -> None:
        """Close the Mooncake connection and cleanup resources."""
        for handle in list(self.memory_handles.keys()):
            self.deregister_memory(handle)
        logger.info(f"[{self.engine_id}] Mooncake connection closed")


MooncakeMetadataT = TypeVar("MooncakeMetadataT")


class MooncakeOperation(RelayOperation, Generic[MooncakeMetadataT]):
    """Base class for Mooncake async operations."""

    def __init__(
        self,
        connection: MooncakeConnection,
        metadata: MooncakeMetadataT | None = None,
    ) -> None:
        self.conn = connection
        self._metadata = metadata  # noqa: leading-underscore
        self.completed = False

    @property
    def metadata(self) -> MooncakeMetadataT | None:
        return self._metadata  # noqa: leading-underscore


class PutOperation(MooncakeOperation[MooncakeMetadataT]):
    """
    Handle for a Put operation.
    In P2P mode, data is prepared in buffer and receiver pulls it.
    Waits for receiver notification before releasing credit.
    """

    def __init__(
        self,
        connection: MooncakeConnection,
        metadata: MooncakeMetadataT,
        transfer_id: str,
        tensor_ref: torch.Tensor,
        on_completion_cb: Callable[[], None],
    ) -> None:
        super().__init__(connection, metadata)
        self.transfer_id = transfer_id
        self.tensor_ref = tensor_ref
        self.on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """
        Wait for receiver notification before completing.
        This ensures the receiver has finished reading before we reuse the buffer.
        Uses Mooncake's built-in notification mechanism via get_notifies().
        """
        if self.completed:
            return
        else:
            pass
        try:
            await MooncakeRelay.wait_for_mooncake_notification(
                self.transfer_id, timeout
            )
        finally:
            self.completed = True
            if self.on_completion_cb:
                self.on_completion_cb()
            else:
                pass


class GetOperation(MooncakeOperation[None]):
    """
    Handle for a Get operation using memory pool.
    Transfers data from remote peer to memory pool, then copies to dest_tensor.
    Sends completion notification to sender after transfer completes.
    """

    def __init__(
        self,
        connection: MooncakeConnection,
        remote_session_id: str,
        local_ptr: int,
        remote_ptr: int,
        size: int,
        transfer_id: str,
        on_completion_cb: Callable[[], None],
    ) -> None:
        super().__init__(connection, metadata=None)
        self.remote_session_id = remote_session_id
        self.local_ptr = local_ptr
        self.remote_ptr = remote_ptr
        self.size = size
        self.transfer_id = transfer_id
        self.on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        try:
            notify = TransferNotify(self.transfer_id, "transfer_complete")
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.conn.transfer_sync,
                self.remote_session_id,
                self.local_ptr,
                self.remote_ptr,
                self.size,
                TransferOpcode.Read,
                notify,
            )
        finally:
            self.completed = True
            if self.on_completion_cb:
                self.on_completion_cb()
            else:
                pass


@register_relay("mooncake")
class MooncakeRelay(Relay):
    """
    Mooncake Transfer Engine based Relay for high-performance data transfer.

    Features:
    - P2P handshake mode (no etcd required!)
    - Multi-protocol support (RDMA, TCP, NVLink)
    - Credit-based flow control
    - Pre-allocated memory pool for stable RDMA registration
    - Completion notification mechanism for safe buffer reuse
    """

    notification_registry: Dict[str, asyncio.Event] = {}
    registry_lock = asyncio.Lock()

    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cuda",
        hostname: str = None,
        protocol: str = "tcp",
        device_name: str = "",
        **kwargs,
    ):
        """
        Initialize Mooncake Relay.

        Args:
            engine_id: Unique identifier for this relay instance
            slot_size_mb: Size of each buffer slot in MB
            credits: Number of concurrent transfers allowed
            device: Device to use (e.g., "cuda:0")
            hostname: Local hostname or IP (auto-detected if None)
            protocol: Transfer protocol ("tcp", "rdma", "nvlink")
            device_name: RDMA device name (e.g., "mlx5_0")
        """
        self.engine_id = engine_id
        self.device = device
        self.torch_device = torch.device(device)
        self.device_id = 0
        if "cuda" in device and ":" in device:
            try:
                self.device_id = int(device.split(":")[1])
            except ValueError:
                self.device_id = 0
        else:
            pass
        if hostname is None:
            hostname = socket.gethostname()
        else:
            pass
        self.connection = MooncakeConnection(
            engine_id=engine_id,
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
        )
        self.slot_size = slot_size_mb * 1024 * 1024
        total_pool_bytes = self.slot_size * credits
        logger.info(
            f"[{engine_id}] Allocating memory pool: {total_pool_bytes / 1024 ** 2:.2f} MB on {device}"
        )
        self.pool_tensor = torch.zeros(
            total_pool_bytes, dtype=torch.uint8, device=self.torch_device
        )
        self.pool_ptr = self.pool_tensor.data_ptr()
        if MOONCAKE_AVAILABLE:
            self.pool_handle = self.connection.register_memory(
                self.pool_ptr, total_pool_bytes
            )
            logger.info(
                f"[{engine_id}] Memory pool registered: handle={hex(self.pool_handle)}"
            )
        else:
            self.pool_handle = self.pool_ptr
            logger.warning(f"[{engine_id}] Mooncake not available, using mock handle")
        self.allocator = CreditAllocator(
            credits=credits, slot_size=self.slot_size, base_ptr=self.pool_ptr
        )
        self.running = True
        self.listener_task = None
        logger.info(
            f"[{engine_id}] MooncakeRelay initialized: protocol={protocol}, device={device}, session_id={self.connection.session_id}"
        )

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> PutOperation[dict[str, object]]:
        """
        Asynchronously send tensor via Mooncake using memory pool.
        Copies tensor data to pre-registered memory pool to avoid RDMA registration overhead.

        Args:
            tensor: Tensor to send
            request_id: Optional request identifier
            dst_rank: Destination rank (not used in P2P mode)
            receiver_id: Stable destination process (unused in P2P mode)

        Returns:
            PutOperation handle with metadata
        """
        self.ensure_listener_started()
        size_bytes = tensor.numel() * tensor.element_size()
        if size_bytes > self.slot_size:
            raise ValueError(
                f"Tensor size {size_bytes} exceeds slot size {self.slot_size}"
            )
        else:
            pass
        transfer_id = f"{self.engine_id}_{uuid.uuid4().hex[:8]}"
        logger.debug(
            f"[{self.engine_id}] put_async: transfer_id={transfer_id}, size={size_bytes}"
        )
        await self.register_notification(transfer_id)
        credit_id = await self.allocator.acquire_async()
        try:
            pool_slice = self.pool_tensor[credit_id : credit_id + size_bytes]
            tensor_view = tensor.view(torch.uint8).reshape(-1)
            pool_slice.copy_(tensor_view)
            buffer_ptr = self.pool_ptr + credit_id
            metadata: dict[str, object] = {
                "engine_id": self.engine_id,
                "session_id": self.connection.session_id,
                "protocol": self.connection.protocol,
                "transfer_id": transfer_id,
                "mooncake": {"buffer_ptr": buffer_ptr, "device_id": self.device_id},
                "transfer_info": {
                    "size": size_bytes,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "credit_id": credit_id,
                },
            }

            def cleanup_callback():
                self.allocator.release(credit_id)

            return PutOperation(
                connection=self.connection,
                metadata=metadata,
                transfer_id=transfer_id,
                tensor_ref=self.pool_tensor,
                on_completion_cb=cleanup_callback,
            )
        except Exception as e:
            self.allocator.release(credit_id)
            raise e

    async def get_async(
        self,
        metadata: Mapping[str, object],
        dest_tensor: torch.Tensor,
        request_id: str = None,
    ) -> GetOperation:
        """
        Asynchronously receive tensor via Mooncake using memory pool.
        Transfers data to pre-registered memory pool, then copies to destination tensor.

        Args:
            metadata: Metadata from sender's put_async
            dest_tensor: Destination tensor to receive data
            request_id: Optional request identifier

        Returns:
            GetOperation handle
        """
        self.ensure_listener_started()
        remote_session_id = metadata["session_id"]
        transfer_info = metadata["transfer_info"]
        mooncake_info = metadata["mooncake"]
        transfer_id = metadata["transfer_id"]
        data_size = transfer_info["size"]
        remote_ptr = mooncake_info["buffer_ptr"]
        logger.debug(
            f"[{self.engine_id}] get_async: transfer_id={transfer_id}, size={data_size}, remote_session={remote_session_id}"
        )
        if data_size > self.slot_size:
            raise ValueError(
                f"Data size {data_size} exceeds slot size {self.slot_size}"
            )
        else:
            pass
        local_credit_id = await self.allocator.acquire_async()
        try:
            dest_size = dest_tensor.numel() * dest_tensor.element_size()
            if dest_size < data_size:
                raise ValueError(
                    f"Destination tensor size {dest_size} is smaller than data size {data_size}"
                )
            else:
                pass
            local_ptr = self.pool_ptr + local_credit_id

            def cleanup_callback():
                pool_slice = self.pool_tensor[
                    local_credit_id : local_credit_id + data_size
                ]
                dest_view = dest_tensor.view(torch.uint8).reshape(-1)[:data_size]
                dest_view.copy_(pool_slice)
                self.allocator.release(local_credit_id)

            return GetOperation(
                connection=self.connection,
                remote_session_id=remote_session_id,
                local_ptr=local_ptr,
                remote_ptr=remote_ptr,
                size=data_size,
                transfer_id=transfer_id,
                on_completion_cb=cleanup_callback,
            )
        except Exception as e:
            self.allocator.release(local_credit_id)
            raise e

    def ensure_listener_started(self) -> None:
        """Ensure the notification listener task is running."""
        if self.listener_task is None or self.listener_task.done():
            try:
                loop = asyncio.get_event_loop()
                self.listener_task = loop.create_task(self.notification_listener_loop())
                logger.debug(
                    f"[{self.engine_id}] Mooncake notification listener started"
                )
            except RuntimeError as e:
                logger.error(
                    f"[{self.engine_id}] Failed to start notification listener: {e}"
                )
        else:
            pass

    async def notification_listener_loop(self) -> None:
        """
        Background task that polls Mooncake for transfer completion notifications.
        Receives notifications sent by remote peers via get_notifies().
        """
        logger.debug(f"[{self.engine_id}] Notification listener loop started")
        while self.running:
            try:
                notifies = self.connection.get_notifies()
                for notify in notifies:
                    transfer_id = notify.name
                    logger.debug(
                        f"[{self.engine_id}] Received Mooncake notification: {transfer_id}"
                    )
                    async with self.registry_lock:
                        if transfer_id in self.notification_registry:
                            self.notification_registry[transfer_id].set()
                            logger.debug(
                                f"[{self.engine_id}] Triggered event for {transfer_id}"
                            )
                        else:
                            pass
                await asyncio.sleep(0.001)
            except Exception as e:
                if self.running:
                    logger.error(
                        f"[{self.engine_id}] Error in notification listener: {e}"
                    )
                    await asyncio.sleep(0.1)
                else:
                    pass
        logger.debug(f"[{self.engine_id}] Notification listener loop stopped")

    @classmethod
    async def register_notification(cls, transfer_id: str) -> asyncio.Event:
        """Register a notification event for a transfer."""
        async with cls.registry_lock:
            event = asyncio.Event()
            cls.notification_registry[transfer_id] = event
            return event

    @classmethod
    async def wait_for_mooncake_notification(
        cls, transfer_id: str, timeout: float
    ) -> None:
        """
        Wait for a Mooncake notification with timeout.
        The notification will be received via get_notifies() in the listener loop.
        """
        event = await cls.register_notification(transfer_id)
        logger.debug(f"Waiting for Mooncake notification: {transfer_id}")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            logger.debug(f"Received Mooncake notification: {transfer_id}")
        except asyncio.TimeoutError:
            logger.error(
                f"Mooncake notification timeout for {transfer_id} after {timeout}s"
            )
            raise asyncio.TimeoutError(f"Notification timeout for {transfer_id}")
        finally:
            async with cls.registry_lock:
                cls.notification_registry.pop(transfer_id, None)

    def cleanup(self, request_id: str) -> None:
        """
        Clean up resources for a specific request.

        Args:
            request_id: Request identifier
        """

    def health(self) -> dict:
        """Return health status of the relay."""
        return {
            "status": "healthy",
            "engine_id": self.engine_id,
            "protocol": self.connection.protocol,
            "session_id": self.connection.session_id,
            "device": self.device,
            "slot_size_mb": self.slot_size // (1024 * 1024),
            "credits": self.allocator.credits,
        }

    def close(self) -> None:
        """Shutdown relay and release all resources."""
        logger.info(f"[{self.engine_id}] Closing MooncakeRelay...")
        self.running = False
        if self.listener_task is not None and (not self.listener_task.done()):
            self.listener_task.cancel()
            logger.info(f"[{self.engine_id}] Notification listener task cancelled")
        else:
            pass
        self.connection.deregister_memory(self.pool_handle)
        logger.info(f"[{self.engine_id}] Memory pool deregistered")
        self.connection.close()
        logger.info(f"[{self.engine_id}] MooncakeRelay closed")
