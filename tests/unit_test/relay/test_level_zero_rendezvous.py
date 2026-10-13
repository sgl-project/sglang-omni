# SPDX-License-Identifier: Apache-2.0
"""How a Level-Zero IPC handle and its dma_buf descriptor reach a peer."""
from __future__ import annotations

import os
import resource
import selectors
import socket
import struct
import threading
import time
from types import SimpleNamespace

import pytest

import sglang_omni.relay.level_zero_rendezvous as level_zero_rendezvous
from sglang_omni.platforms.level_zero import (
    ZE_MAX_IPC_HANDLE_SIZE,
    ExportedSegment,
    LevelZeroError,
    importable_descriptor,
)
from sglang_omni.relay.level_zero_rendezvous import HandleRendezvous, fetch_peer_segment

PEER_TIMEOUT_S = 60
# Taken before any test patches selectors.DefaultSelector.
REAL_SELECTOR = selectors.DefaultSelector
# Well under HANDLE_REQUEST_TIMEOUT_S, which a silent peer used to hold everyone for.
PROMPT_S = 1.0


def serving_rendezvous() -> tuple[HandleRendezvous, int]:
    exported_fd = os.open("/dev/null", os.O_RDONLY)
    segment = ExportedSegment(
        base_pointer=0,
        segment_bytes=64,
        handle_blob=bytes(range(64)),
        file_descriptor=exported_fd,
    )
    return HandleRendezvous("pool", segment), exported_fd


def connect_silently(rendezvous: HandleRendezvous) -> socket.socket:
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    peer.connect("\0" + rendezvous.socket_name)
    return peer


def open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_the_rendezvous_hands_over_the_handle_and_a_working_descriptor() -> None:
    blob = bytes(range(64))
    exported_fd = os.open("/dev/null", os.O_RDONLY)
    segment = ExportedSegment(
        base_pointer=0, segment_bytes=64, handle_blob=blob, file_descriptor=exported_fd
    )
    rendezvous = HandleRendezvous("pool", segment)
    # The server shares this process's descriptor table, and closes its side of a
    # connection only after the peer has its reply; count once it has.
    answered = threading.Semaphore(0)
    answer_peer = rendezvous.answer_peer

    def answer_then_signal(connection: socket.socket) -> None:
        try:
            answer_peer(connection)
        finally:
            answered.release()

    rendezvous.answer_peer = answer_then_signal
    try:
        received_blob, received_fd = fetch_peer_segment(rendezvous.socket_name, "pool")
        try:
            assert received_blob == blob
            assert received_fd != exported_fd
            assert os.fstat(received_fd).st_ino == os.fstat(exported_fd).st_ino
        finally:
            os.close(received_fd)
        assert answered.acquire(timeout=PEER_TIMEOUT_S)

        before = open_descriptors()
        with pytest.raises(LevelZeroError):
            fetch_peer_segment(rendezvous.socket_name, "another-pool")
        assert answered.acquire(timeout=PEER_TIMEOUT_S)
        assert open_descriptors() == before

        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(PEER_TIMEOUT_S)
            connection.connect("\0" + rendezvous.socket_name)
            connection.send(b"\xff")
            assert connection.recv(ZE_MAX_IPC_HANDLE_SIZE) == b""
        # Neither rejected request may stop the rendezvous serving the next peer.
        received_blob, received_fd = fetch_peer_segment(rendezvous.socket_name, "pool")
        os.close(received_fd)
        assert received_blob == blob
    finally:
        rendezvous.close()
        os.close(exported_fd)


def test_closing_the_rendezvous_waits_for_the_peer_it_is_serving() -> None:
    """The relay releases the segment as soon as close returns, so the server must
    not be able to send its descriptor any more by then."""
    exported_fd = os.open("/dev/null", os.O_RDONLY)
    segment = ExportedSegment(
        base_pointer=0,
        segment_bytes=64,
        handle_blob=bytes(64),
        file_descriptor=exported_fd,
    )
    rendezvous = HandleRendezvous("pool", segment)
    serving = threading.Event()
    events: list[str] = []

    def serve_slowly(connection: socket.socket) -> None:
        serving.set()
        time.sleep(0.5)
        events.append("served")

    rendezvous.send_handle = serve_slowly
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as peer:
            peer.connect("\0" + rendezvous.socket_name)
            peer.send(b"pool")
            assert serving.wait(PEER_TIMEOUT_S)
            rendezvous.close()
            events.append("closed")
        assert events == ["served", "closed"]
        assert not rendezvous.thread.is_alive()
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as late_peer:
            with pytest.raises(ConnectionRefusedError):
                late_peer.connect("\0" + rendezvous.socket_name)
    finally:
        os.close(exported_fd)


def test_a_received_descriptor_is_duplicated_only_when_it_collides() -> None:
    received = os.open("/dev/null", os.O_RDONLY)
    try:
        assert importable_descriptor(received, received + 1) == received
        # An exporter can hold more descriptors than this process may open.
        _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert importable_descriptor(received, hard_limit + 1) == received

        duplicate = importable_descriptor(received, received)
        try:
            assert duplicate != received
            assert os.fstat(duplicate).st_ino == os.fstat(received).st_ino
        finally:
            os.close(duplicate)
    finally:
        os.close(received)


def test_a_silent_peer_does_not_hold_up_an_importer() -> None:
    rendezvous, exported_fd = serving_rendezvous()
    # More silent peers than any fixed set of workers could have held.
    silent_peers = [connect_silently(rendezvous) for _ in range(8)]
    try:
        start = time.monotonic()
        _blob, received_fd = fetch_peer_segment(rendezvous.socket_name, "pool")
        os.close(received_fd)
        assert time.monotonic() - start < PROMPT_S
    finally:
        rendezvous.close()
        for peer in silent_peers:
            peer.close()
        os.close(exported_fd)


def test_closing_the_rendezvous_does_not_wait_for_a_silent_peer() -> None:
    rendezvous, exported_fd = serving_rendezvous()
    silent_peer = connect_silently(rendezvous)
    try:
        # Let the rendezvous accept the peer and start waiting on its request.
        time.sleep(0.3)
        start = time.monotonic()
        rendezvous.close()
        assert time.monotonic() - start < PROMPT_S
        assert not rendezvous.thread.is_alive()
    finally:
        silent_peer.close()
        os.close(exported_fd)


def test_a_peer_of_another_user_is_turned_away_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendezvous, exported_fd = serving_rendezvous()
    monkeypatch.setattr(level_zero_rendezvous.os, "geteuid", lambda: os.getuid() + 1)
    try:
        start = time.monotonic()
        # The hang-up reads as a reset or an empty reply, depending on timing.
        with pytest.raises((ConnectionError, LevelZeroError)):
            fetch_peer_segment(rendezvous.socket_name, "pool")
        assert time.monotonic() - start < PROMPT_S
    finally:
        monkeypatch.undo()
        rendezvous.close()
        os.close(exported_fd)


def serve_one_reply(
    handle_blob: bytes, descriptor_count: int
) -> tuple[str, threading.Thread]:
    """A stand-in rendezvous that answers one request with the given reply."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    socket_name = f"sglang-omni-level-zero-test-{os.getpid()}-{time.monotonic_ns()}"
    listener.bind("\0" + socket_name)
    listener.listen(1)

    def reply() -> None:
        with listener:
            connection, _address = listener.accept()
            with connection:
                connection.recv(1024)
                descriptors = [
                    os.open("/dev/null", os.O_RDONLY) for _ in range(descriptor_count)
                ]
                try:
                    socket.send_fds(connection, [handle_blob], descriptors)
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)

    replier = threading.Thread(target=reply, daemon=True)
    replier.start()
    return socket_name, replier


@pytest.mark.parametrize(
    ("handle_bytes", "descriptor_count"),
    [
        (ZE_MAX_IPC_HANDLE_SIZE // 2, 1),
        (ZE_MAX_IPC_HANDLE_SIZE, 0),
        (ZE_MAX_IPC_HANDLE_SIZE, 2),
        (2 * ZE_MAX_IPC_HANDLE_SIZE, 1),
    ],
    ids=["short-handle", "no-descriptor", "extra-descriptor", "oversized-handle"],
)
def test_a_malformed_reply_is_refused_without_keeping_a_descriptor(
    handle_bytes: int, descriptor_count: int
) -> None:
    """An extra descriptor or an oversized handle arrives truncated; whatever did
    arrive with a refused reply must be closed rather than leaked."""
    # Counted before the stand-in opens its sockets and after it closes them.
    before = open_descriptors()
    socket_name, replier = serve_one_reply(bytes(handle_bytes), descriptor_count)

    with pytest.raises(LevelZeroError):
        fetch_peer_segment(socket_name, "pool")
    replier.join(PEER_TIMEOUT_S)
    assert open_descriptors() == before


class BatchingSelector:
    """A real selector that holds the serve loop once two peers wait, then hands it
    one batch with a new connection ahead of the oldest peer's request."""

    def __init__(self, both_ready: threading.Event, full: threading.Event) -> None:
        self.inner = REAL_SELECTOR()
        self.both_ready = both_ready
        self.full = full
        self.registered: list[object] = []
        self.batched = False

    def __enter__(self) -> BatchingSelector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.inner.close()

    def register(self, fileobj: object, events: int) -> selectors.SelectorKey:
        key = self.inner.register(fileobj, events)
        self.registered.append(fileobj)
        # The listener comes first, then the peers it admits.
        if len(self.registered) == 3:
            self.full.set()
        else:
            pass
        return key

    def unregister(self, fileobj: object) -> selectors.SelectorKey:
        return self.inner.unregister(fileobj)

    def select(
        self, timeout: float | None = None
    ) -> list[tuple[selectors.SelectorKey, int]]:
        if not self.full.is_set() or self.batched:
            return self.inner.select(timeout)
        else:
            pass
        assert self.both_ready.wait(PEER_TIMEOUT_S)
        listener, oldest = self.registered[0], self.registered[1]
        deadline = time.monotonic() + PEER_TIMEOUT_S
        while True:
            ready = self.inner.select(0.05)
            fileobjs = {key.fileobj for key, _events in ready}
            if listener in fileobjs and oldest in fileobjs:
                break
            else:
                assert time.monotonic() < deadline
        self.batched = True
        return sorted(ready, key=lambda item: item[0].fileobj is not listener)


def test_a_peer_dropped_while_its_request_is_pending_does_not_stop_the_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admitting a peer into a full waiting set drops the oldest, whose request may
    already be in the same batch of ready sockets; handling that stale entry must
    not end the only thread serving the pool."""
    both_ready = threading.Event()
    full = threading.Event()
    selectors_made: list[BatchingSelector] = []

    def make_selector() -> BatchingSelector:
        selectors_made.append(BatchingSelector(both_ready, full))
        return selectors_made[-1]

    monkeypatch.setattr(level_zero_rendezvous, "RENDEZVOUS_MAX_WAITING_PEERS", 2)
    monkeypatch.setattr(
        level_zero_rendezvous.selectors, "DefaultSelector", make_selector
    )
    rendezvous, exported_fd = serving_rendezvous()
    oldest = connect_silently(rendezvous)
    filling = connect_silently(rendezvous)
    newest = None
    try:
        assert full.wait(PEER_TIMEOUT_S)
        # The serve thread has its selector; nothing else should get one.
        monkeypatch.setattr(
            level_zero_rendezvous.selectors, "DefaultSelector", REAL_SELECTOR
        )
        newest = connect_silently(rendezvous)
        oldest.send(b"pool")
        both_ready.set()

        # The oldest peer was dropped with its request unread, which resets it.
        oldest.settimeout(PEER_TIMEOUT_S)
        with pytest.raises(ConnectionResetError):
            oldest.recv(ZE_MAX_IPC_HANDLE_SIZE)
        assert selectors_made[0].batched
        assert rendezvous.thread.is_alive()
        _blob, received_fd = fetch_peer_segment(rendezvous.socket_name, "pool")
        os.close(received_fd)
    finally:
        rendezvous.close()
        for peer in (oldest, filling, newest):
            if peer is not None:
                peer.close()
            else:
                pass
        os.close(exported_fd)


def test_a_peer_whose_uid_is_past_the_signed_range_is_its_own_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """uid_t is unsigned, so a uid above 2**31 must not read as negative and turn
    away a peer of the same user."""
    uid = 3_000_000_000
    rendezvous = HandleRendezvous.__new__(HandleRendezvous)
    rendezvous.socket_name = "unused"
    peer = SimpleNamespace(
        getsockopt=lambda level, option, buffer_bytes: struct.pack(
            "iII", 4242, uid, uid
        )
    )
    monkeypatch.setattr(level_zero_rendezvous.os, "geteuid", lambda: uid)

    assert rendezvous.is_own_user(peer)
