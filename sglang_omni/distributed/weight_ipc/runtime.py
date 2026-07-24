# SPDX-License-Identifier: Apache-2.0
"""Engine-init helpers: export (leader) / import (follower) around model load."""

from __future__ import annotations

import logging

import torch

from sglang_omni.distributed.weight_ipc.export import export_shared_weights
from sglang_omni.distributed.weight_ipc.import_ import import_and_alias
from sglang_omni.distributed.weight_ipc.lifecycle import (
    LeaderLivenessMonitor,
    pid_is_alive,
)
from sglang_omni.distributed.weight_ipc.store import WeightIpcStore
from sglang_omni.distributed.weight_ipc.types import WeightIpcConfig

logger = logging.getLogger(__name__)


def _require_live_leader(leader_pid: int) -> None:
    if not pid_is_alive(leader_pid):
        raise RuntimeError(
            f"weight IPC bundle owner pid={leader_pid} is not alive; "
            "refusing to open stale CUDA handles"
        )


def export_leader_weights(
    model: torch.nn.Module,
    config: WeightIpcConfig,
    *,
    model_path: str,
    model_revision: str | None,
) -> None:
    assert config.store_dir is not None
    store = WeightIpcStore(config.store_dir)
    store.prepare_for_write()
    bundle = export_shared_weights(
        model,
        model_path=model_path or config.model_path,
        model_revision=(
            model_revision if model_revision is not None else config.model_revision
        ),
    )
    store.write_bundle(bundle)
    logger.info(
        "weight_ipc: role=leader status=exported n=%s digest=%s store=%s",
        len(bundle.tensors),
        bundle.name_digest[:16],
        config.store_dir,
    )
    logger.info("weight_ipc: READY")


def materialize_follower_weights(
    model: torch.nn.Module,
    config: WeightIpcConfig,
    *,
    model_path: str,
    model_revision: str | None,
) -> LeaderLivenessMonitor:
    assert config.store_dir is not None
    store = WeightIpcStore(config.store_dir)
    store.wait_ready(timeout_s=config.timeout_s)
    bundle = store.load_bundle()
    _require_live_leader(bundle.leader_pid)
    imported = import_and_alias(
        model,
        bundle,
        model_path=model_path or config.model_path,
        model_revision=(
            model_revision if model_revision is not None else config.model_revision
        ),
    )
    _require_live_leader(bundle.leader_pid)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    monitor = LeaderLivenessMonitor(bundle.leader_pid)
    monitor.start()
    logger.info(
        "weight_ipc: role=follower status=aliased n=%s digest=%s "
        "checkpoint_load_skipped=true owner_monitor_started=true",
        len(imported),
        bundle.name_digest[:16],
    )
    return monitor
