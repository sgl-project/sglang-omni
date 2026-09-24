# SPDX-License-Identifier: Apache-2.0
"""Control plane for inter-stage communication via ZMQ."""

import asyncio
import logging

import msgpack
import zmq
import zmq.asyncio

from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminResultMessage,
    CompleteMessage,
    DataAckMessage,
    DataReadyMessage,
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
    StreamMessage,
    SubmitMessage,
    parse_message,
)

logger = logging.getLogger(__name__)

ControlMessage = (
    AdminMessage
    | AdminResultMessage
    | DataAckMessage
    | DataReadyMessage
    | KVTransferPrepareMessage
    | KVTransferReadyMessage
    | AbortMessage
    | CompleteMessage
    | StreamMessage
    | ShutdownMessage
    | SubmitMessage
    | ProfilerStartMessage
    | ProfilerStopMessage
)


def serialize_message(msg: ControlMessage) -> bytes:
    """Serialize a message to bytes."""
    return msgpack.packb(msg.to_dict(), use_bin_type=True)


def deserialize_message(data: bytes) -> ControlMessage:
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
        if cls.context is None:
            cls.context = zmq.asyncio.Context()
        return cls.context

    @classmethod
    def close(cls) -> None:
        """Close the shared context."""
        if cls.context is not None:
            cls.context.term()
            cls.context = None


class PushSocket:
    """Async PUSH socket for sending messages to a single destination."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.socket: zmq.asyncio.Socket | None = None

    async def connect(self) -> None:
        """Connect to the endpoint."""
        ctx = ControlPlaneContext.get()
        self.socket = ctx.socket(zmq.PUSH)
        self.socket.connect(self.endpoint)
        logger.debug("PUSH socket connected to %s", self.endpoint)

    async def send(self, msg: ControlMessage) -> None:
        """Send a message."""
        if self.socket is None:
            raise RuntimeError("Socket not connected")
        data = serialize_message(msg)
        await self.socket.send(data)
        logger.debug("PUSH sent %s to %s", type(msg).__name__, self.endpoint)

    def close(self) -> None:
        """Close the socket."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None


async def send_to_endpoint(
    sockets: dict[str, PushSocket],
    endpoint: str,
    msg: ControlMessage,
) -> None:
    socket = sockets.get(endpoint)
    if socket is None:
        socket = PushSocket(endpoint)
        await socket.connect()
        sockets[endpoint] = socket
    await socket.send(msg)


class PullSocket:
    """Async PULL socket for receiving messages."""

    def __init__(self, endpoint: str, bind: bool = True):
        self.endpoint = endpoint
        self.bind = bind
        self.socket: zmq.asyncio.Socket | None = None

    async def start(self) -> None:
        """Bind or connect the socket."""
        ctx = ControlPlaneContext.get()
        self.socket = ctx.socket(zmq.PULL)
        if self.bind:
            self.socket.bind(self.endpoint)
            logger.debug("PULL socket bound to %s", self.endpoint)
        else:
            self.socket.connect(self.endpoint)
            logger.debug("PULL socket connected to %s", self.endpoint)

    async def recv(self) -> ControlMessage:
        """Receive a message (blocking)."""
        if self.socket is None:
            raise RuntimeError("Socket not started")
        data = await self.socket.recv()
        msg = deserialize_message(data)
        logger.debug("PULL received %s", type(msg).__name__)
        return msg

    async def recv_nowait(self) -> ControlMessage | None:
        """Try to receive a message (non-blocking)."""
        if self.socket is None:
            raise RuntimeError("Socket not started")
        try:
            data = await asyncio.wait_for(self.socket.recv(), timeout=0)
            return deserialize_message(data)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Close the socket."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None


class PubSocket:
    """Async PUB socket for broadcasting messages (e.g., abort)."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.socket: zmq.asyncio.Socket | None = None

    async def bind(self) -> None:
        """Bind the socket."""
        ctx = ControlPlaneContext.get()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind(self.endpoint)
        # Give subscribers time to connect
        await asyncio.sleep(0.1)
        logger.debug("PUB socket bound to %s", self.endpoint)

    async def publish(self, msg: AbortMessage) -> None:
        """Publish a message to all subscribers."""
        if self.socket is None:
            raise RuntimeError("Socket not bound")
        data = serialize_message(msg)
        await self.socket.send(data)
        logger.debug("PUB published %s", type(msg).__name__)

    def close(self) -> None:
        """Close the socket."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None


class SubSocket:
    """Async SUB socket for receiving broadcast messages."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.socket: zmq.asyncio.Socket | None = None

    async def connect(self) -> None:
        """Connect to the publisher."""
        ctx = ControlPlaneContext.get()
        self.socket = ctx.socket(zmq.SUB)
        self.socket.connect(self.endpoint)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        logger.debug("SUB socket connected to %s", self.endpoint)

    async def recv(self) -> AbortMessage:
        """Receive a broadcast message (blocking)."""
        if self.socket is None:
            raise RuntimeError("Socket not connected")
        data = await self.socket.recv()
        msg = deserialize_message(data)
        if not isinstance(msg, AbortMessage):
            raise ValueError(f"Expected AbortMessage, got {type(msg)}")
        logger.debug("SUB received %s", type(msg).__name__)
        return msg

    def poll(self, timeout_ms: int = 0) -> bool:
        """Check if a message is available."""
        if self.socket is None:
            raise RuntimeError("Socket not connected")
        return self.socket.poll(timeout_ms, zmq.POLLIN) != 0

    def close(self) -> None:
        """Close the socket."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None


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

        self.recv_socket: PullSocket | None = None
        self.coordinator_socket: PushSocket | None = None
        self.abort_socket: SubSocket | None = None
        self.next_stage_sockets: dict[str, PushSocket] = {}

    async def start(self) -> None:
        """Initialize all sockets."""
        # Socket to receive work
        self.recv_socket = PullSocket(self.recv_endpoint, bind=True)
        await self.recv_socket.start()

        # Socket to send completions to coordinator
        self.coordinator_socket = PushSocket(self.coordinator_endpoint)
        await self.coordinator_socket.connect()

        # Socket to receive abort broadcasts
        self.abort_socket = SubSocket(self.abort_endpoint)
        await self.abort_socket.connect()

        logger.info("Stage %s control plane started", self.stage_name)

    async def recv(
        self,
    ) -> (
        AdminMessage
        | DataAckMessage
        | DataReadyMessage
        | SubmitMessage
        | ShutdownMessage
        | ProfilerStartMessage
        | ProfilerStopMessage
    ):
        """Receive work from previous stage or coordinator."""
        if self.recv_socket is None:
            raise RuntimeError("Control plane not started")
        msg = await self.recv_socket.recv()
        if isinstance(
            msg,
            (
                DataReadyMessage,
                DataAckMessage,
                SubmitMessage,
                ShutdownMessage,
                ProfilerStartMessage,
                ProfilerStopMessage,
                AdminMessage,
            ),
        ):
            return msg
        raise ValueError(f"Unexpected message type: {type(msg)}")

    async def send_to_stage(
        self,
        next_stage: str,
        next_stage_endpoint: str,
        msg: DataReadyMessage | DataAckMessage,
    ) -> None:
        """Send a stage-to-stage control message."""
        await send_to_endpoint(self.next_stage_sockets, next_stage_endpoint, msg)

    async def send_complete(self, msg: CompleteMessage) -> None:
        """Send completion notification to coordinator."""
        if self.coordinator_socket is None:
            raise RuntimeError("Control plane not started")
        await self.coordinator_socket.send(msg)

    async def send_stream(self, msg: StreamMessage) -> None:
        """Send a stream chunk to coordinator."""
        if self.coordinator_socket is None:
            raise RuntimeError("Control plane not started")
        await self.coordinator_socket.send(msg)

    async def send_admin_result(self, msg: AdminResultMessage) -> None:
        """Send an administrative result to coordinator."""
        if self.coordinator_socket is None:
            raise RuntimeError("Control plane not started")
        await self.coordinator_socket.send(msg)

    async def recv_abort(self) -> AbortMessage:
        """Receive abort broadcast (blocking).

        This should be run in a separate task.
        """
        if self.abort_socket is None:
            raise RuntimeError("Control plane not started")
        return await self.abort_socket.recv()

    def close(self) -> None:
        """Close all sockets."""
        if self.recv_socket:
            self.recv_socket.close()
        if self.coordinator_socket:
            self.coordinator_socket.close()
        if self.abort_socket:
            self.abort_socket.close()
        for sock in self.next_stage_sockets.values():
            sock.close()
        self.next_stage_sockets.clear()


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

        self.completion_socket: PullSocket | None = None
        self.abort_socket: PubSocket | None = None
        self.stage_sockets: dict[str, PushSocket] = {}

    async def start(self) -> None:
        """Initialize all sockets."""
        # Socket to receive completions
        self.completion_socket = PullSocket(self.completion_endpoint, bind=True)
        await self.completion_socket.start()

        # Socket to broadcast aborts
        self.abort_socket = PubSocket(self.abort_endpoint)
        await self.abort_socket.bind()

        logger.info("Coordinator control plane started")

    async def submit_to_stage(
        self,
        stage_name: str,
        stage_endpoint: str,
        msg: SubmitMessage | AdminMessage | ShutdownMessage,
    ) -> None:
        """Submit a request to a stage."""
        if stage_name not in self.stage_sockets:
            sock = PushSocket(stage_endpoint)
            await sock.connect()
            self.stage_sockets[stage_name] = sock

        await self.stage_sockets[stage_name].send(msg)

    async def recv_event(self) -> CompleteMessage | StreamMessage | AdminResultMessage:
        """Receive completion or stream event from a stage."""
        if self.completion_socket is None:
            raise RuntimeError("Control plane not started")
        msg = await self.completion_socket.recv()
        if isinstance(msg, (CompleteMessage, StreamMessage, AdminResultMessage)):
            return msg
        raise ValueError(
            "Expected CompleteMessage, StreamMessage, or AdminResultMessage, "
            f"got {type(msg)}"
        )

    async def send_admin(
        self, stage_name: str, stage_endpoint: str, msg: AdminMessage
    ) -> None:
        """Send an administrative operation to a stage."""
        await self.submit_to_stage(stage_name, stage_endpoint, msg)

    async def broadcast_abort(self, msg: AbortMessage) -> None:
        """Broadcast abort to all stages."""
        if self.abort_socket is None:
            raise RuntimeError("Control plane not started")
        await self.abort_socket.publish(msg)

    async def send_shutdown(self, stage_name: str, stage_endpoint: str) -> None:
        """Send shutdown message to a stage."""
        if stage_name not in self.stage_sockets:
            sock = PushSocket(stage_endpoint)
            await sock.connect()
            self.stage_sockets[stage_name] = sock

        await self.stage_sockets[stage_name].send(ShutdownMessage())

    def close(self) -> None:
        """Close all sockets."""
        if self.completion_socket:
            self.completion_socket.close()
        if self.abort_socket:
            self.abort_socket.close()
        for sock in self.stage_sockets.values():
            sock.close()
        self.stage_sockets.clear()
