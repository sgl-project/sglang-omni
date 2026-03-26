# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities."""

from __future__ import annotations

import socket


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
