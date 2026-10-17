#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# mps.sh start|stop  -- per-user CUDA MPS daemon (unprivileged; GPU compute mode = Default).
# start uses a FRESH pipe dir ($MPS_PIPE); stale pipe dirs accumulate crash state so that
# later workers fail with cudaErrorMpsRpcFailure -- always start clean. NEVER SIGKILL an
# MPS *client* (a server/worker): it corrupts the MPS server. Stop clients gracefully
# (SIGTERM) BEFORE `mps.sh stop`.
set -u
source "$(dirname "$0")/common.sh"
case "${1:-}" in
  start)
    rm -rf "$MPS_PIPE" "$MPS_LOG"; mkdir -p "$MPS_PIPE" "$MPS_LOG"
    CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG nvidia-cuda-mps-control -d
    sleep 2
    echo "MPS up, thread%: $(echo get_default_active_thread_percentage | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE nvidia-cuda-mps-control 2>&1)"
    ;;
  stop)
    echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE nvidia-cuda-mps-control 2>/dev/null
    sleep 2
    # -9 here matches only the MPS control daemon + server, never a client (clients are the
    # served workers, e.g. "sgl-omni serve", and must already be stopped gracefully). With the
    # pipe dir removed below and a FRESH one on the next `start`, this leaves no stale MPS
    # state: the "never -9 a client" rule is about clients, not the daemon. pkill is scoped to
    # this process namespace, so it does not touch another user's MPS.
    pkill -9 -f nvidia-cuda-mps 2>/dev/null
    rm -rf "$MPS_PIPE" "$MPS_LOG"
    echo "MPS stopped"
    ;;
  *) echo "usage: mps.sh start|stop"; exit 1 ;;
esac
