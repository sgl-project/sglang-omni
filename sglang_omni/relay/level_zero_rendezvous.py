# SPDX-License-Identifier: Apache-2.0
"""Hand a Level-Zero IPC handle and its dma_buf descriptor to importing peers."""
from __future__ import annotations

import logging
import os
import selectors
import socket
import struct
import threading
import time
import uuid

from sglang_omni.platforms.level_zero import (
    ZE_MAX_IPC_HANDLE_SIZE,
    ExportedSegment,
    LevelZeroError,
)

logger = logging.getLogger(__name__)

RENDEZVOUS_BACKLOG = 16
RENDEZVOUS_POLL_S = 0.2
RENDEZVOUS_MAX_WAITING_PEERS = 64
HANDLE_REQUEST_TIMEOUT_S = 5.0
HANDLE_REQUEST_LIMIT_BYTES = 512
# struct ucred from SO_PEERCRED: a signed pid, then the unsigned uid and gid.
PEER_CREDENTIALS = struct.Struct("iII")


class HandleRendezvous:
    """Serve one exported segment over an abstract SOCK_SEQPACKET Unix socket."""

    def __init__(self, pool_id: str, segment: ExportedSegment) -> None:
        self.pool_id = pool_id
        self.segment = segment
        self.socket_name = (
            f"sglang-omni-level-zero-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.bind("\0" + self.socket_name)
        self.listener.listen(RENDEZVOUS_BACKLOG)
        self.listener.setblocking(False)
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self.serve,
            name=f"level-zero-ipc-{self.socket_name}",
            daemon=True,
        )
        self.thread.start()

    def serve(self) -> None:
        # Accepted peers whose request has not arrived, and when it is due by.
        waiting: dict[socket.socket, float] = {}
        with selectors.DefaultSelector() as selector:
            selector.register(self.listener, selectors.EVENT_READ)
            try:
                while not self.stopped.is_set():
                    for key, _events in selector.select(timeout=RENDEZVOUS_POLL_S):
                        if key.fileobj is self.listener:
                            self.admit_peer(selector, waiting)
                        elif key.fileobj in waiting:
                            connection = key.fileobj
                            forget_peer(selector, waiting, connection)
                            self.answer_peer(connection)
                        else:
                            # Dropped and closed while admitting a peer in this batch.
                            pass
                    now = time.monotonic()
                    for connection, deadline in list(waiting.items()):
                        if deadline <= now:
                            forget_peer(selector, waiting, connection)
                            connection.close()
                        else:
                            pass
            finally:
                for connection in list(waiting):
                    forget_peer(selector, waiting, connection)
                    connection.close()

    def admit_peer(
        self,
        selector: selectors.BaseSelector,
        waiting: dict[socket.socket, float],
    ) -> None:
        try:
            connection, _address = self.listener.accept()
        except BlockingIOError:
            # The peer gave up between the readiness report and the accept.
            return
        except OSError as error:
            if self.stopped.is_set():
                return
            else:
                raise LevelZeroError(
                    f"Level-Zero IPC rendezvous {self.socket_name} failed: {error}"
                ) from error
        if not self.is_own_user(connection):
            connection.close()
            return
        else:
            pass
        if len(waiting) >= RENDEZVOUS_MAX_WAITING_PEERS:
            oldest = next(iter(waiting))
            forget_peer(selector, waiting, oldest)
            oldest.close()
            logger.warning(
                f"Level-Zero IPC rendezvous {self.socket_name} dropped a peer that "
                f"had not sent its request"
            )
        else:
            pass
        connection.settimeout(HANDLE_REQUEST_TIMEOUT_S)
        selector.register(connection, selectors.EVENT_READ)
        waiting[connection] = time.monotonic() + HANDLE_REQUEST_TIMEOUT_S

    def is_own_user(self, connection: socket.socket) -> bool:
        try:
            _peer_pid, peer_uid, _peer_gid = PEER_CREDENTIALS.unpack(
                connection.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, PEER_CREDENTIALS.size
                )
            )
        except OSError as error:
            logger.warning(
                f"Level-Zero IPC rendezvous {self.socket_name} could not identify a "
                f"peer: {error}"
            )
            return False
        if peer_uid != os.geteuid():
            logger.warning(
                f"Level-Zero IPC rendezvous {self.socket_name} turned away a peer "
                f"running as uid {peer_uid}"
            )
            return False
        else:
            return True

    def answer_peer(self, connection: socket.socket) -> None:
        with connection:
            try:
                self.send_handle(connection)
            except (OSError, LevelZeroError) as error:
                # One importer giving up must not stop the pool reaching the rest.
                logger.warning(
                    f"Level-Zero IPC rendezvous {self.socket_name} could not serve "
                    f"a peer: {error}"
                )

    def send_handle(self, connection: socket.socket) -> None:
        request = connection.recv(HANDLE_REQUEST_LIMIT_BYTES + 1)
        if len(request) > HANDLE_REQUEST_LIMIT_BYTES:
            raise LevelZeroError(
                f"Level-Zero IPC request exceeds the {HANDLE_REQUEST_LIMIT_BYTES} "
                f"byte limit"
            )
        elif request != self.pool_id.encode("utf-8"):
            raise LevelZeroError(
                f"Level-Zero IPC rendezvous {self.socket_name} has no pool "
                f"{request.decode('utf-8', errors='replace')!r}"
            )
        else:
            pass
        sent = socket.send_fds(
            connection, [self.segment.handle_blob], [self.segment.file_descriptor]
        )
        if sent != len(self.segment.handle_blob):
            raise LevelZeroError(
                f"Level-Zero IPC rendezvous {self.socket_name} sent {sent} of "
                f"{len(self.segment.handle_blob)} handle bytes"
            )
        else:
            pass

    def close(self) -> None:
        # The thread answers every peer, so once it exits nothing sends the descriptor.
        self.stopped.set()
        try:
            self.listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            # A listener that cannot shut down also fails the thread's next accept.
            pass
        self.thread.join()
        self.listener.close()


def forget_peer(
    selector: selectors.BaseSelector,
    waiting: dict[socket.socket, float],
    connection: socket.socket,
) -> None:
    selector.unregister(connection)
    waiting.pop(connection)


def fetch_peer_segment(socket_name: str, pool_id: str) -> tuple[bytes, int]:
    """Collect an exported handle and its descriptor from the owning process."""
    request = pool_id.encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(HANDLE_REQUEST_TIMEOUT_S)
        connection.connect("\0" + socket_name)
        if connection.send(request) != len(request):
            raise LevelZeroError(
                f"Level-Zero IPC rendezvous {socket_name} took part of the request "
                f"for pool {pool_id!r}"
            )
        else:
            pass
        handle_blob, descriptors, flags, _address = socket.recv_fds(
            connection, ZE_MAX_IPC_HANDLE_SIZE, 1
        )
    truncated = flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
    if truncated or len(descriptors) != 1 or len(handle_blob) != ZE_MAX_IPC_HANDLE_SIZE:
        for descriptor in descriptors:
            os.close(descriptor)
        raise LevelZeroError(
            f"Level-Zero IPC rendezvous {socket_name} returned "
            f"{len(handle_blob)} handle bytes and {len(descriptors)} descriptors "
            f"(truncated: {bool(truncated)}) for pool {pool_id!r}"
        )
    else:
        pass
    return handle_blob, descriptors[0]
