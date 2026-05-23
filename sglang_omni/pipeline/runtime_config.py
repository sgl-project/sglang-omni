# SPDX-License-Identifier: Apache-2.0
"""Runtime preparation helpers shared by pipeline runners."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sglang_omni.config.placement import StagePlacementPlan, build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan, build_process_topology_plan

logger = logging.getLogger(__name__)


class IpcRuntimeDir:
    """Runtime-owned IPC directory for one pipeline instance."""

    def __init__(self, path: Path):
        self.path = path
        self._closed = False

    def __enter__(self) -> IpcRuntimeDir:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IpcRuntimeDir(path={self.path!r}, closed={self._closed})"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove IPC runtime dir %s: %s", self.path, exc)


@dataclass(frozen=True)
class PipelineRuntimePrep:
    """Prepared stage, endpoint, placement, and topology state."""

    stages_cfg: list[StageConfig]
    name_map: dict[str, str]
    entry_stage: str
    endpoints: dict[str, str]
    placement_plan: StagePlacementPlan
    process_plan: ProcessTopologyPlan
    runtime_dir: IpcRuntimeDir
    runtime_dir_created_here: bool


def create_ipc_runtime_dir(config: PipelineConfig) -> IpcRuntimeDir:
    """Create a per-run IPC namespace for one pipeline instance."""
    base_root = Path(config.endpoints.base_path)
    base_root.mkdir(parents=True, exist_ok=True)

    namespace_prefix = re.sub(r"[^0-9a-z]+", "-", config.name.lower()).strip("-")
    if not namespace_prefix:
        namespace_prefix = "pipeline"
    path = Path(tempfile.mkdtemp(prefix=f"{namespace_prefix}-", dir=base_root))
    return IpcRuntimeDir(path)


def prepare_pipeline_runtime(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> PipelineRuntimePrep:
    """Prepare fused stages, endpoint allocation, and process topology."""
    runtime_dir = ipc_runtime_dir
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config)
        runtime_dir_created_here = True
    else:
        runtime_dir_created_here = False

    try:
        stages_cfg, name_map, entry_stage = config.apply_fusion()
        placement_plan = build_stage_placement_plan(config, stages_cfg=stages_cfg)
        process_plan = build_process_topology_plan(
            config,
            placement_plan,
            stages_cfg=stages_cfg,
        )
        endpoints = allocate_endpoints(
            stages=stages_cfg,
            ipc_base_dir=runtime_dir.path,
        )
    except Exception:
        if runtime_dir_created_here:
            runtime_dir.close()
        raise

    return PipelineRuntimePrep(
        stages_cfg=stages_cfg,
        name_map=name_map,
        entry_stage=entry_stage,
        endpoints=endpoints,
        placement_plan=placement_plan,
        process_plan=process_plan,
        runtime_dir=runtime_dir,
        runtime_dir_created_here=runtime_dir_created_here,
    )


def build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": parse_gpu_id(relay_cfg.device),
        }

    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None
        elif isinstance(gpu, list):
            gpu_id = gpu[0]
        else:
            gpu_id = gpu

    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": 512,
        "credits": 2,
        "rank": None,
        "world_size": None,
        "gpu_id": gpu_id,
    }


def parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")


def allocate_endpoints(
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path,
) -> dict[str, str]:
    base_dir = ipc_base_dir
    endpoints = {
        "completion": f"ipc://{base_dir}/completion.sock",
        "abort": f"ipc://{base_dir}/abort.sock",
    }
    for stage in stages:
        endpoints[f"stage_{stage.name}"] = f"ipc://{base_dir}/stage_{stage.name}.sock"
    return endpoints
