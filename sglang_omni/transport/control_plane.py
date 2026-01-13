# SPDX-License-Identifier: Apache-2.0
"""Control plane for inter-stage communication via ZMQ."""

import asyncio
import logging

import msgpack
import zmq
import zmq.asyncio

from sglang_omni.proto import (
    AbortMessage,
    CompleteMessage,
    DataReadyMessage,
    ShutdownMessage,
    SubmitMessage,
    parse_message,
)

logger = logging.getLogger(__name__)


def serialize_message(
    msg: DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage,
) -> bytes:
    """Serialize a message to bytes."""
    return msgpack.packb(msg.to_dict(), use_bin_type=True)


def deserialize_message(
    data: bytes,
) -> DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage:
    """Deserialize bytes to a message."""
    d = msgpack.unpackb(data, raw=False)
    return parse_message(d)


class ControlPlaneContext:
    """Shared ZMQ context for control plane."""

    _instance: "ControlPlaneContext | None" = None
    _context: zmq.asyncio.Context | None = None

    @classmethod
    def get(cls) -> zmq.asyncio.Context:
        """Get or create the shared ZMQ context."""
        if cls._context is None:
            cls._context = zmq.asyncio.Context()
        return cls._context

    @classmethod
    def close(cls) -> None:
        """Close the shared context."""
        if cls._context is not None:
            cls._context.term()
            cls._context = None


class PushSocket:
    """Async PUSH socket for sending messages to a single destination."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._socket: zmq.asyncio.Socket | None = None

    async def connect(self) -> None:
        """Connect to the endpoint."""
        ctx = ControlPlaneContext.get()
        self._socket = ctx.socket(zmq.PUSH)
        self._socket.connect(self.endpoint)
        logger.debug("PUSH socket connected to %s", self.endpoint)

    async def send(
        self, msg: DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage
    ) -> None:
        """Send a message."""
        if self._socket is None:
            raise RuntimeError("Socket not connected")
        data = serialize_message(msg)
        await self._socket.send(data)
        logger.debug("PUSH sent %s to %s", type(msg).__name__, self.endpoint)

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class PullSocket:
    """Async PULL socket for receiving messages."""

    def __init__(self, endpoint: str, bind: bool = True):
        self.endpoint = endpoint
        self.bind = bind
        self._socket: zmq.asyncio.Socket | None = None

    async def start(self) -> None:
        """Bind or connect the socket."""
        ctx = ControlPlaneContext.get()
        self._socket = ctx.socket(zmq.PULL)
        if self.bind:
            self._socket.bind(self.endpoint)
            logger.debug("PULL socket bound to %s", self.endpoint)
        else:
            self._socket.connect(self.endpoint)
            logger.debug("PULL socket connected to %s", self.endpoint)

    async def recv(
        self,
    ) -> DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage:
        """Receive a message (blocking)."""
        if self._socket is None:
            raise RuntimeError("Socket not started")
        data = await self._socket.recv()
        msg = deserialize_message(data)
        logger.debug("PULL received %s", type(msg).__name__)
        return msg

    async def recv_nowait(
        self,
    ) -> DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage | None:
        """Try to receive a message (non-blocking)."""
        if self._socket is None:
            raise RuntimeError("Socket not started")
        try:
            data = await asyncio.wait_for(self._socket.recv(), timeout=0)
            return deserialize_message(data)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class PubSocket:
    """Async PUB socket for broadcasting messages (e.g., abort)."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._socket: zmq.asyncio.Socket | None = None

    async def bind(self) -> None:
        """Bind the socket."""
        ctx = ControlPlaneContext.get()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.bind(self.endpoint)
        # Give subscribers time to connect
        await asyncio.sleep(0.1)
        logger.debug("PUB socket bound to %s", self.endpoint)

    async def publish(self, msg: AbortMessage) -> None:
        """Publish a message to all subscribers."""
        if self._socket is None:
            raise RuntimeError("Socket not bound")
        data = serialize_message(msg)
        await self._socket.send(data)
        logger.debug("PUB published %s", type(msg).__name__)

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class SubSocket:
    """Async SUB socket for receiving broadcast messages."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._socket: zmq.asyncio.Socket | None = None

    async def connect(self) -> None:
        """Connect to the publisher."""
        ctx = ControlPlaneContext.get()
        self._socket = ctx.socket(zmq.SUB)
        self._socket.connect(self.endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        logger.debug("SUB socket connected to %s", self.endpoint)

    async def recv(self) -> AbortMessage:
        """Receive a broadcast message (blocking)."""
        if self._socket is None:
            raise RuntimeError("Socket not connected")
        data = await self._socket.recv()
        msg = deserialize_message(data)
        if not isinstance(msg, AbortMessage):
            raise ValueError(f"Expected AbortMessage, got {type(msg)}")
        logger.debug("SUB received %s", type(msg).__name__)
        return msg

    def poll(self, timeout_ms: int = 0) -> bool:
        """Check if a message is available."""
        if self._socket is None:
            raise RuntimeError("Socket not connected")
        return self._socket.poll(timeout_ms, zmq.POLLIN) != 0

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class StageControlPlane:
    """Control plane interface for a Stage.

    Handles:
    - Receiving work (PULL from coordinator or previous stage)
    - Sending work to next stage (PUSH)
    - Receiving abort broadcasts (SUB)
    - Sending completion to coordinator (PUSH)
    """

    def __init__(
        self,
        stage_name: str,
        recv_endpoint: str,
        coordinator_endpoint: str,
        abort_endpoint: str,
    ):
        self.stage_name = stage_name
        self.recv_endpoint = recv_endpoint
        self.coordinator_endpoint = coordinator_endpoint
        self.abort_endpoint = abort_endpoint

        self._recv_socket: PullSocket | None = None
        self._coordinator_socket: PushSocket | None = None
        self._abort_socket: SubSocket | None = None
        self._next_stage_sockets: dict[str, PushSocket] = {}

    async def start(self) -> None:
        """Initialize all sockets."""
        # Socket to receive work
        self._recv_socket = PullSocket(self.recv_endpoint, bind=True)
        await self._recv_socket.start()

        # Socket to send completions to coordinator
        self._coordinator_socket = PushSocket(self.coordinator_endpoint)
        await self._coordinator_socket.connect()

        # Socket to receive abort broadcasts
        self._abort_socket = SubSocket(self.abort_endpoint)
        await self._abort_socket.connect()

        logger.info("Stage %s control plane started", self.stage_name)

    async def recv(self) -> DataReadyMessage | SubmitMessage | ShutdownMessage:
        """Receive work from previous stage or coordinator."""
        if self._recv_socket is None:
            raise RuntimeError("Control plane not started")
        msg = await self._recv_socket.recv()
        if isinstance(msg, (DataReadyMessage, SubmitMessage, ShutdownMessage)):
            return msg
        raise ValueError(f"Unexpected message type: {type(msg)}")

    async def send_to_stage(
        self, next_stage: str, next_stage_endpoint: str, msg: DataReadyMessage
    ) -> None:
        """Send data ready notification to next stage."""
        if next_stage not in self._next_stage_sockets:
            sock = PushSocket(next_stage_endpoint)
            await sock.connect()
            self._next_stage_sockets[next_stage] = sock

        await self._next_stage_sockets[next_stage].send(msg)

    async def send_complete(self, msg: CompleteMessage) -> None:
        """Send completion notification to coordinator."""
        if self._coordinator_socket is None:
            raise RuntimeError("Control plane not started")
        await self._coordinator_socket.send(msg)

    async def recv_abort(self) -> AbortMessage:
        """Receive abort broadcast (blocking).

        This should be run in a separate task.
        """
        if self._abort_socket is None:
            raise RuntimeError("Control plane not started")
        return await self._abort_socket.recv()

    def close(self) -> None:
        """Close all sockets."""
        if self._recv_socket:
            self._recv_socket.close()
        if self._coordinator_socket:
            self._coordinator_socket.close()
        if self._abort_socket:
            self._abort_socket.close()
        for sock in self._next_stage_sockets.values():
            sock.close()
        self._next_stage_sockets.clear()


class CoordinatorControlPlane:
    """Control plane interface for the Coordinator.

    Handles:
    - Submitting work to entry stage (PUSH)
    - Receiving completions from stages (PULL)
    - Broadcasting abort signals (PUB)
    """

    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
    ):
        self.completion_endpoint = completion_endpoint
        self.abort_endpoint = abort_endpoint

        self._completion_socket: PullSocket | None = None
        self._abort_socket: PubSocket | None = None
        self._stage_sockets: dict[str, PushSocket] = {}

    async def start(self) -> None:
        """Initialize all sockets."""
        # Socket to receive completions
        self._completion_socket = PullSocket(self.completion_endpoint, bind=True)
        await self._completion_socket.start()

        # Socket to broadcast aborts
        self._abort_socket = PubSocket(self.abort_endpoint)
        await self._abort_socket.bind()

        logger.info("Coordinator control plane started")

    async def submit_to_stage(
        self, stage_name: str, stage_endpoint: str, msg: SubmitMessage
    ) -> None:
        """Submit a request to a stage."""
        if stage_name not in self._stage_sockets:
            sock = PushSocket(stage_endpoint)
            await sock.connect()
            self._stage_sockets[stage_name] = sock

        await self._stage_sockets[stage_name].send(msg)

    async def recv_completion(self) -> CompleteMessage:
        """Receive completion from a stage."""
        if self._completion_socket is None:
            raise RuntimeError("Control plane not started")
        msg = await self._completion_socket.recv()
        if isinstance(msg, CompleteMessage):
            return msg
        raise ValueError(f"Expected CompleteMessage, got {type(msg)}")

    async def broadcast_abort(self, msg: AbortMessage) -> None:
        """Broadcast abort to all stages."""
        if self._abort_socket is None:
            raise RuntimeError("Control plane not started")
        await self._abort_socket.publish(msg)

    async def send_shutdown(self, stage_name: str, stage_endpoint: str) -> None:
        """Send shutdown message to a stage."""
        if stage_name not in self._stage_sockets:
            sock = PushSocket(stage_endpoint)
            await sock.connect()
            self._stage_sockets[stage_name] = sock

        await self._stage_sockets[stage_name].send(ShutdownMessage())

    def close(self) -> None:
        """Close all sockets."""
        if self._completion_socket:
            self._completion_socket.close()
        if self._abort_socket:
            self._abort_socket.close()
        for sock in self._stage_sockets.values():
            sock.close()
        self._stage_sockets.clear()
