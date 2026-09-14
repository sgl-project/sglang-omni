# SPDX-License-Identifier: Apache-2.0
"""Data-plane child entry point (spawned by the supervisor).

Wraps the inherited listen socket (shared accept queue across all DPs) and
serves the data-plane app; snapshot refresh, heartbeats (with fencing), and
failure reporting are owned by the app's lifespan tasks.
"""

from __future__ import annotations

import os
import socket

import uvicorn

from sglang_omni_router.python.app_factory import load_config_from_env
from sglang_omni_router.python.supervisor import (
    LOG_LEVEL_ENV,
    SOCKET_FD_ENV,
    watch_supervisor_liveness,
)


def build_server_config() -> uvicorn.Config:
    config = load_config_from_env()
    return uvicorn.Config(
        "sglang_omni_router.python.data_plane:create_dp_app_from_env",
        factory=True,
        log_level=os.environ.get(LOG_LEVEL_ENV, "warning").lower(),
        access_log=False,
        # Note (Jiaxin Deng): a routine SIGTERM must drain, not truncate:
        # TTS/Omni requests routinely outlive a short fixed deadline, and the
        # single-process router sets no graceful deadline at all.
        timeout_graceful_shutdown=config.effective_shutdown_drain_secs,
    )


def main() -> None:
    watch_supervisor_liveness()
    fd = int(os.environ[SOCKET_FD_ENV])
    listen_socket = socket.socket(fileno=fd)
    server = uvicorn.Server(build_server_config())
    server.run(sockets=[listen_socket])


if __name__ == "__main__":
    main()
