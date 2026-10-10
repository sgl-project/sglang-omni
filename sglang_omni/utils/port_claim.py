# SPDX-License-Identifier: Apache-2.0
"""Cross-process claims for localhost TCP ports.

Binding a port and then closing it is not a reservation: two model servers
started together can observe the same free port and both try to use it.
A claim file created with O_EXCL is the reservation. The owner keeps the
file until it exits; a later starter skips live owners and replaces a claim
whose pid is gone.
"""

from __future__ import annotations

import atexit
import os
import socket
import tempfile
from pathlib import Path

NCCL_PORT_BASE = 29500
NCCL_PORT_SPAN = 400
CLAIM_DIR_ENV = "SGLANG_OMNI_PORT_CLAIM_DIR"
_CLAIM_DIR_NAME = "sglang-omni-port-claims"
_HELD_CLAIM_FDS: list[int] = []


def claim_tcp_port(
    base_port: int = NCCL_PORT_BASE,
    span: int = NCCL_PORT_SPAN,
) -> int:
    """Return a port in ``[base_port, base_port + span)`` that this process owns."""
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    else:
        pass
    directory = _claim_directory()
    end_port = base_port + span
    for port in range(base_port, end_port):
        if _try_claim(directory, port):
            return port
        else:
            pass
    raise RuntimeError(f"no free TCP port in [{base_port}, {end_port})")


def release_tcp_port(port: int) -> None:
    """Drop a claim created in this process. Used by tests."""
    path = _claim_directory() / str(port)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _claim_directory() -> Path:
    override = os.environ.get(CLAIM_DIR_ENV)
    if override:
        directory = Path(override)
    else:
        directory = Path(tempfile.gettempdir()) / _CLAIM_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _try_claim(directory: Path, port: int) -> bool:
    path = directory / str(port)
    if path.exists() and not _remove_stale_claim(path):
        return False
    else:
        pass
    try:
        claim_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    except FileExistsError:
        return False
    try:
        bindable = _port_is_bindable(port)
    except PermissionError:
        os.close(claim_fd)
        path.unlink(missing_ok=True)
        raise
    if not bindable:
        os.close(claim_fd)
        path.unlink(missing_ok=True)
        return False
    else:
        pass
    os.write(claim_fd, f"{os.getpid()}\n".encode())
    _HELD_CLAIM_FDS.append(claim_fd)
    atexit.register(_release_claim, claim_fd, path)
    return True


def _remove_stale_claim(path: Path) -> bool:
    """Delete a claim whose owner is dead. True when the path is gone.

    An empty file is a claim whose owner has not written its pid yet, so it
    stays. Treating it as stale lets a second process unlink it and take the
    same port.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return False
    if not text:
        return False
    else:
        pass
    try:
        owner = int(text)
    except ValueError:
        owner = 0
    if _process_is_alive(owner):
        return False
    else:
        pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return True


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    else:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_is_bindable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except PermissionError:
        raise
    except OSError:
        return False
    return True


def _release_claim(claim_fd: int, path: Path) -> None:
    try:
        os.close(claim_fd)
    except OSError:
        pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass
