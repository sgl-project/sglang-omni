# SPDX-License-Identifier: Apache-2.0
"""SGLANG_OMNI_STRICT_PORT turns the port fallback into a hard error."""

from __future__ import annotations

import errno
import socket
import sys
from pathlib import Path

import pytest

from sglang_omni.serve.launcher import _find_available_port


def test_free_port_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert _find_available_port("127.0.0.1", free) == free


def test_busy_port_falls_back_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy = holder.getsockname()[1]
        assert _find_available_port("127.0.0.1", busy) != busy


def test_busy_port_hard_errors_under_strict(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy = holder.getsockname()[1]
        with pytest.raises(RuntimeError, match="STRICT_PORT"):
            _find_available_port("127.0.0.1", busy)


@pytest.mark.skipif(sys.platform != "linux", reason="Inspect Linux TCP TIME_WAIT state")
@pytest.mark.parametrize("strict", [False, True])
def test_closed_server_port_is_reused_after_connection(monkeypatch, strict):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1" if strict else "0")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(2)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(2)
                # note(wenyao): Closing the server side first leaves its port in TIME_WAIT.
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
                client.shutdown(socket.SHUT_WR)
                assert connection.recv(1) == b""

    states = {
        fields[3]
        for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
        if (fields := line.split())[1] == f"0100007F:{port:04X}"
    }
    assert states == {"06"}, states
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        with pytest.raises(OSError) as busy:
            probe.bind(("127.0.0.1", port))
        assert busy.value.errno == errno.EADDRINUSE
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
        probe.listen(1)

    assert _find_available_port("127.0.0.1", port) == port
