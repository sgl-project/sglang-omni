# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from typing import Any, Callable, Dict

import torch

from .base import CreditAllocator, Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

# ==========================================
# Dependency Check
# ==========================================
try:
    from mooncake.engine import TransferEngine, TransferNotify, TransferOpcode

    MOONCAKE_AVAILABLE = True
except ImportError as e:
    logger.error(
        f"Failed to import mooncake: {e}. MooncakeRelay will not work. "
        "Install with: pip install mooncake-transfer-engine"
    )
    MOONCAKE_AVAILABLE = False

    # Mock classes
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


# ==========================================
# Helper Classes
# ==========================================


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

        # Initialize TransferEngine
        self.engine = TransferEngine()

        # Initialize with P2P handshake mode (no etcd needed!)
        logger.debug(
            f"[{engine_id}] Initializing TransferEngine with protocol='{protocol}'"
        )
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",  # P2P mode - no metadata server required
            protocol,
            device_name if device_name else "",
        )
        logger.debug(f"[{engine_id}] TransferEngine initialization returned: {ret}")

        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake TransferEngine (error code: {ret})"
            )

        # Get session ID for peer communication
        self.session_id = f"{hostname}:{self.engine.get_rpc_port()}"

        # Track registered memory
        self._memory_handles: Dict[int, int] = {}  # ptr -> handle

        logger.info(
            f"[{engine_id}] Mooncake connection initialized: "
            f"protocol={protocol}, session_id={self.session_id}"
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
        if ptr in self._memory_handles:
            logger.warning(f"Memory at {hex(ptr)} already registered")
            return self._memory_handles[ptr]

        # Register with Mooncake
        ret = self.engine.register_memory(ptr, size)

        if ret != 0:
            raise RuntimeError(f"Failed to register memory (error code: {ret})")

        # Use ptr as handle
        self._memory_handles[ptr] = ptr

        logger.debug(f"Registered memory: ptr={hex(ptr)}, size={size} bytes")
        return ptr

    def deregister_memory(self, handle: int) -> None:
        """Deregister memory from Mooncake."""
        if handle not in self._memory_handles:
            logger.warning(f"Memory handle {hex(handle)} not found")
            return

        try:
            self.engine.unregister_memory(handle)
            del self._memory_handles[handle]
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
        # Deregister all memory
        for handle in list(self._memory_handles.keys()):
            self.deregister_memory(handle)

        logger.info(f"[{self.engine_id}] Mooncake connection closed")


# ==========================================
# Operations Hierarchy
# ==========================================


class MooncakeOperation(RelayOperation):
    """Base class for Mooncake async operations."""

    def __init__(self, connection: MooncakeConnection, metadata: Any = None):
        self._conn = connection
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> Any:
        return self._metadata


class PutOperation(MooncakeOperation):
    """
    Handle for a Put operation.
    In P2P mode, data is prepared in buffer and receiver pulls it.
    Waits for receiver notification before releasing credit.
    """

    def __init__(
        self,
        connection: MooncakeConnection,
        metadata: Any,
        transfer_id: str,
        tensor_ref: torch.Tensor,
        on_completion_cb: Callable[[], None],
    ):
        super().__init__(connection, metadata)
        self._transfer_id = transfer_id
        self._tensor_ref = tensor_ref
        self._on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """
        Wait for receiver notification before completing.
        This ensures the receiver has finished reading before we reuse the buffer.
        Uses Mooncake's built-in notification mechanism via get_notifies().
        """
        if self._completed:
            return

        try:
            # Wait for receiver to send completion notification via Mooncake
            await MooncakeRelay._wait_for_mooncake_notification(
                self._transfer_id, timeout
            )
        finally:
            self._completed = True
            if self._on_completion_cb:
                self._on_completion_cb()


class GetOperation(MooncakeOperation):
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
    ):
        super().__init__(connection, metadata=None)
        self._remote_session_id = remote_session_id
        self._local_ptr = local_ptr
        self._remote_ptr = remote_ptr
        self._size = size
        self._transfer_id = transfer_id
        self._on_completion_cb = on_completion_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        try:
            # Create notification for sender (Mooncake will send this automatically)
            notify = TransferNotify(self._transfer_id, "transfer_complete")

            # Execute synchronous transfer with notification (to memory pool)
            # Data will be copied from pool to dest_tensor in cleanup callback
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._conn.transfer_sync,
                self._remote_session_id,
                self._local_ptr,
                self._remote_ptr,
                self._size,
                TransferOpcode.Read,
                notify,
            )

            # No copy needed - data transferred directly to dest_tensor!

        finally:
            self._completed = True
            if self._on_completion_cb:
                self._on_completion_cb()


# ==========================================
# MooncakeRelay (Async Only)
# ==========================================


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

    # Class-level notification registry for same-process communication
    # For cross-process scenarios, users should implement external notification mechanism
    _notification_registry: Dict[str, asyncio.Event] = {}
    _registry_lock = asyncio.Lock()

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
        self.device = torch.device(device)

        # Parse device ID
        self.device_id = 0
        if "cuda" in device and ":" in device:
            try:
                self.device_id = int(device.split(":")[1])
            except ValueError:
                self.device_id = 0

        # Auto-detect hostname if not provided
        if hostname is None:
            hostname = socket.gethostname()

        # Initialize Mooncake connection (P2P mode, no etcd!)
        self.connection = MooncakeConnection(
            engine_id=engine_id,
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
        )

        # Initialize memory pool
        self.slot_size = slot_size_mb * 1024 * 1024
        total_pool_bytes = self.slot_size * credits

        logger.info(
            f"[{engine_id}] Allocating memory pool: "
            f"{total_pool_bytes / 1024**2:.2f} MB on {device}"
        )

        self.pool_tensor = torch.zeros(
            total_pool_bytes, dtype=torch.uint8, device=self.device
        )
        self.pool_ptr = self.pool_tensor.data_ptr()

        # Register memory pool once with Mooncake
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

        # Initialize credit allocator with pool base pointer
        self.allocator = CreditAllocator(
            credits=credits,
            slot_size=self.slot_size,
            base_ptr=self.pool_ptr,
        )

        # Initialize notification listener for Mooncake notifications
        self._running = True
        self._listener_task = None

        logger.info(
            f"[{engine_id}] MooncakeRelay initialized: "
            f"protocol={protocol}, device={device}, "
            f"session_id={self.connection.session_id}"
        )

    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> PutOperation:
        """
        Asynchronously send tensor via Mooncake using memory pool.
        Copies tensor data to pre-registered memory pool to avoid RDMA registration overhead.

        Args:
            tensor: Tensor to send
            request_id: Optional request identifier
            dst_rank: Destination rank (not used in P2P mode)

        Returns:
            PutOperation handle with metadata
        """
        # Ensure notification listener is running
        self._ensure_listener_started()

        size_bytes = tensor.numel() * tensor.element_size()
        if size_bytes > self.slot_size:
            raise ValueError(
                f"Tensor size {size_bytes} exceeds slot size {self.slot_size}"
            )

        # Generate unique transfer ID
        transfer_id = f"{self.engine_id}_{uuid.uuid4().hex[:8]}"
        logger.debug(
            f"[{self.engine_id}] put_async: transfer_id={transfer_id}, size={size_bytes}"
        )

        # Register notification event for this transfer
        await self._register_notification(transfer_id)

        # Acquire credit for flow control and buffer slot allocation
        credit_id = await self.allocator.acquire_async()

        try:
            # Copy tensor data to memory pool
            pool_slice = self.pool_tensor[credit_id : credit_id + size_bytes]
            tensor_view = tensor.view(torch.uint8).reshape(-1)
            pool_slice.copy_(tensor_view)

            # Calculate buffer pointer in pool
            buffer_ptr = self.pool_ptr + credit_id

            # Prepare metadata for receiver (includes session_id for P2P)
            metadata = {
                "engine_id": self.engine_id,
                "session_id": self.connection.session_id,  # For P2P handshake
                "protocol": self.connection.protocol,
                "transfer_id": transfer_id,  # For completion notification
                "mooncake": {
                    "buffer_ptr": buffer_ptr,  # Pool buffer pointer
                    "device_id": self.device_id,
                },
                "transfer_info": {
                    "size": size_bytes,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "credit_id": credit_id,
                },
            }

            # Create operation handle with cleanup callback
            def cleanup_callback():
                # Release credit (no need to deregister, pool stays registered)
                self.allocator.release(credit_id)

            return PutOperation(
                connection=self.connection,
                metadata=metadata,
                transfer_id=transfer_id,
                tensor_ref=self.pool_tensor,  # Keep pool tensor alive
                on_completion_cb=cleanup_callback,
            )

        except Exception as e:
            # Release credit on error
            self.allocator.release(credit_id)
            raise e

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
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
        # Ensure notification listener is running
        self._ensure_listener_started()

        # Parse metadata
        remote_session_id = metadata["session_id"]  # For P2P handshake
        transfer_info = metadata["transfer_info"]
        mooncake_info = metadata["mooncake"]
        transfer_id = metadata["transfer_id"]  # For completion notification

        data_size = transfer_info["size"]
        remote_ptr = mooncake_info["buffer_ptr"]

        logger.debug(
            f"[{self.engine_id}] get_async: transfer_id={transfer_id}, size={data_size}, remote_session={remote_session_id}"
        )

        if data_size > self.slot_size:
            raise ValueError(
                f"Data size {data_size} exceeds slot size {self.slot_size}"
            )

        # Acquire local credit for flow control and buffer slot allocation
        local_credit_id = await self.allocator.acquire_async()

        try:
            # Validate dest_tensor size
            dest_size = dest_tensor.numel() * dest_tensor.element_size()

            if dest_size < data_size:
                raise ValueError(
                    f"Destination tensor size {dest_size} is smaller than data size {data_size}"
                )

            # Calculate local buffer pointer in pool
            local_ptr = self.pool_ptr + local_credit_id

            # Create operation handle with cleanup callback
            def cleanup_callback():
                # Copy data from pool to dest_tensor
                pool_slice = self.pool_tensor[
                    local_credit_id : local_credit_id + data_size
                ]
                dest_view = dest_tensor.view(torch.uint8).reshape(-1)[:data_size]
                dest_view.copy_(pool_slice)
                # Release credit (no need to deregister, pool stays registered)
                self.allocator.release(local_credit_id)

            # The actual transfer will happen in wait_for_completion()
            return GetOperation(
                connection=self.connection,
                remote_session_id=remote_session_id,
                local_ptr=local_ptr,  # Pool buffer pointer
                remote_ptr=remote_ptr,
                size=data_size,
                transfer_id=transfer_id,
                on_completion_cb=cleanup_callback,
            )

        except Exception as e:
            self.allocator.release(local_credit_id)
            raise e

    def _ensure_listener_started(self) -> None:
        """Ensure the notification listener task is running."""
        if self._listener_task is None or self._listener_task.done():
            try:
                loop = asyncio.get_event_loop()
                self._listener_task = loop.create_task(
                    self._notification_listener_loop()
                )
                logger.debug(
                    f"[{self.engine_id}] Mooncake notification listener started"
                )
            except RuntimeError as e:
                logger.error(
                    f"[{self.engine_id}] Failed to start notification listener: {e}"
                )

    async def _notification_listener_loop(self) -> None:
        """
        Background task that polls Mooncake for transfer completion notifications.
        Receives notifications sent by remote peers via get_notifies().
        """
        logger.debug(f"[{self.engine_id}] Notification listener loop started")

        while self._running:
            try:
                # Poll Mooncake for notifications from remote peers
                notifies = self.connection.get_notifies()

                # Process each notification
                for notify in notifies:
                    transfer_id = (
                        notify.name
                    )  # TransferNotify.name contains the transfer_id
                    logger.debug(
                        f"[{self.engine_id}] Received Mooncake notification: {transfer_id}"
                    )

                    # Set the event for this transfer_id
                    async with self._registry_lock:
                        if transfer_id in self._notification_registry:
                            self._notification_registry[transfer_id].set()
                            logger.debug(
                                f"[{self.engine_id}] Triggered event for {transfer_id}"
                            )

                # Sleep briefly to avoid busy-waiting
                await asyncio.sleep(0.001)  # 1ms polling interval

            except Exception as e:
                if self._running:
                    logger.error(
                        f"[{self.engine_id}] Error in notification listener: {e}"
                    )
                    await asyncio.sleep(0.1)  # Back off on error

        logger.debug(f"[{self.engine_id}] Notification listener loop stopped")

    @classmethod
    async def _register_notification(cls, transfer_id: str) -> asyncio.Event:
        """Register a notification event for a transfer."""
        async with cls._registry_lock:
            event = asyncio.Event()
            cls._notification_registry[transfer_id] = event
            return event

    @classmethod
    async def _wait_for_mooncake_notification(
        cls, transfer_id: str, timeout: float
    ) -> None:
        """
        Wait for a Mooncake notification with timeout.
        The notification will be received via get_notifies() in the listener loop.
        """
        # Register event for this transfer
        event = await cls._register_notification(transfer_id)

        logger.debug(f"Waiting for Mooncake notification: {transfer_id}")

        try:
            # Wait for the event to be set by the notification listener
            await asyncio.wait_for(event.wait(), timeout=timeout)
            logger.debug(f"Received Mooncake notification: {transfer_id}")
        except asyncio.TimeoutError:
            logger.error(
                f"Mooncake notification timeout for {transfer_id} after {timeout}s"
            )
            raise asyncio.TimeoutError(f"Notification timeout for {transfer_id}")
        finally:
            # Clean up the event from registry
            async with cls._registry_lock:
                cls._notification_registry.pop(transfer_id, None)

    def cleanup(self, request_id: str) -> None:
        """
        Clean up resources for a specific request.

        Args:
            request_id: Request identifier
        """
        # Currently no per-request cleanup needed
        # Buffers are reused via credit allocator

    def health(self) -> dict:
        """Return health status of the relay."""
        return {
            "status": "healthy",
            "engine_id": self.engine_id,
            "protocol": self.connection.protocol,
            "session_id": self.connection.session_id,
            "device": str(self.device),
            "slot_size_mb": self.slot_size // (1024 * 1024),
            "credits": self.allocator.credits,
        }

    def close(self) -> None:
        """Shutdown relay and release all resources."""
        logger.info(f"[{self.engine_id}] Closing MooncakeRelay...")

        # Stop notification listener task
        self._running = False
        if self._listener_task is not None and not self._listener_task.done():
            self._listener_task.cancel()
            logger.info(f"[{self.engine_id}] Notification listener task cancelled")

        # Deregister memory pool
        if hasattr(self, "pool_handle"):
            self.connection.deregister_memory(self.pool_handle)
            logger.info(f"[{self.engine_id}] Memory pool deregistered")

        # Close Mooncake connection
        self.connection.close()

        logger.info(f"[{self.engine_id}] MooncakeRelay closed")
