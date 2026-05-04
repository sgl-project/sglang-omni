# SPDX-License-Identifier: Apache-2.0
"""Network connection utilities."""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger(__name__)


def find_available_port(host: str = "0.0.0.0", port: int | None = None) -> int:
    """
    Find a free port on the host if the port is not provided. If the port is provided, check if it is available.

    Args:
        host: The host to bind to. Default is "0.0.0.0".
        port: The port to bind to. Default is None, which means a free port will be found.

    Returns:
        The available port.
    """
    if port is not None:
        # check if the given port can be used
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            logger.warning(f"Port {port} is already in use on {host}.")

    # find the available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]

    if port is not None:
        logger.warning(f"Using port {free_port} instead.")
    else:
        logger.info(f"Found available port {free_port}.")
    return free_port
