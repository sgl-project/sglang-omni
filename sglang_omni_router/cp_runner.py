# SPDX-License-Identifier: Apache-2.0
"""Control-plane child entry point (spawned by the supervisor).

Serves the control-plane app (registry owner, snapshot publisher, admin
surface, internal DP channel) on the supervisor-owned UDS (the 0700 workdir
is the access control) or, on platforms without UDS, on a localhost port
guarded by the per-run internal token.
"""

from __future__ import annotations

import os

import uvicorn

from sglang_omni_router.app_factory import load_config_from_env
from sglang_omni_router.control_plane import create_control_plane_app
from sglang_omni_router.internal_channel import INTERNAL_TOKEN_ENV
from sglang_omni_router.supervisor import (
    ADMISSION_SHM_ENV,
    CHILD_GRACEFUL_SHUTDOWN_SECS,
    CP_EPOCH_ENV,
    EXPECTED_DPS_ENV,
    INTERNAL_TCP_URL_ENV,
    INTERNAL_UDS_ENV,
    LOG_LEVEL_ENV,
    SNAPSHOT_PATH_ENV,
    watch_supervisor_liveness,
)


def main() -> None:
    watch_supervisor_liveness()
    config = load_config_from_env()
    expected = os.environ.get(EXPECTED_DPS_ENV)
    app = create_control_plane_app(
        config,
        snapshot_path=os.environ[SNAPSHOT_PATH_ENV],
        cp_epoch=os.environ[CP_EPOCH_ENV],
        internal_token=os.environ.get(INTERNAL_TOKEN_ENV),
        expected_data_planes=int(expected) if expected else None,
        admission_shm_path=os.environ.get(ADMISSION_SHM_ENV),
    )
    log_level = os.environ.get(LOG_LEVEL_ENV, "warning").lower()
    uds = os.environ.get(INTERNAL_UDS_ENV)
    if uds:
        uvicorn.run(
            app,
            uds=uds,
            log_level=log_level,
            timeout_graceful_shutdown=CHILD_GRACEFUL_SHUTDOWN_SECS,
        )
        return
    tcp_url = os.environ.get(INTERNAL_TCP_URL_ENV)
    if not tcp_url:
        raise RuntimeError(
            f"neither {INTERNAL_UDS_ENV} nor {INTERNAL_TCP_URL_ENV} is set; "
            "cp_runner is only meant to be spawned by the router supervisor"
        )
    port = int(tcp_url.rsplit(":", 1)[1])
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=CHILD_GRACEFUL_SHUTDOWN_SECS,
    )


if __name__ == "__main__":
    main()
