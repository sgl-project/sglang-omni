# SPDX-License-Identifier: Apache-2.0
"""Runtime preparation helpers shared by pipeline runners."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import zmq

from sglang_omni.config.placement import StagePlacementPlan, build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan, build_process_topology_plan

logger = logging.getLogger(__name__)

# PyZMQ checks the filesystem path after ``ipc://`` against this budget.
_IPC_SUN_PATH_BUDGET = getattr(zmq, "IPC_PATH_MAX_LEN", 100)
_TEMPFILE_RANDOM_SUFFIX_LEN = 8


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


def create_ipc_runtime_dir(
    config: PipelineConfig,
    *,
    stages: list[StageConfig] | None = None,
) -> IpcRuntimeDir:
    """Create a per-run IPC namespace for one pipeline instance."""
    base_root = Path(config.endpoints.base_path)
    base_root.mkdir(parents=True, exist_ok=True)
    if stages is None:
        stages, _, _ = config.apply_fusion()

    namespace_prefix = re.sub(r"[^0-9a-z]+", "-", config.name.lower()).strip("-")
    if not namespace_prefix:
        namespace_prefix = "pipeline"
    namespace_prefix = _truncate_ipc_namespace_prefix(
        namespace_prefix,
        base_root=base_root,
        stages=stages,
    )
    dir_prefix = f"{namespace_prefix}-" if namespace_prefix else ""
    path = Path(tempfile.mkdtemp(prefix=dir_prefix, dir=base_root))
    return IpcRuntimeDir(path)


def prepare_pipeline_runtime(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> PipelineRuntimePrep:
    """Prepare fused stages, endpoint allocation, and process topology."""
    stages_cfg, name_map, entry_stage = config.apply_fusion()
    runtime_dir = ipc_runtime_dir
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config, stages=stages_cfg)
        runtime_dir_created_here = True
    else:
        runtime_dir_created_here = False

    try:
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
    _validate_ipc_endpoint_budget(stages=stages, ipc_base_dir=ipc_base_dir)
    base_dir = ipc_base_dir
    endpoints = {
        "completion": f"ipc://{base_dir}/completion.sock",
        "abort": f"ipc://{base_dir}/abort.sock",
    }
    for stage in stages:
        endpoints[f"stage_{stage.name}"] = f"ipc://{base_dir}/stage_{stage.name}.sock"
    return endpoints


def _truncate_ipc_namespace_prefix(
    namespace_prefix: str,
    *,
    base_root: Path,
    stages: list[StageConfig],
) -> str:
    endpoint_suffix_len = _longest_endpoint_suffix_len(stages)
    min_dir_len = _ipc_dir_len(
        base_root=base_root,
        namespace_prefix_len=0,
        endpoint_suffix_len=endpoint_suffix_len,
    )
    if min_dir_len > _IPC_SUN_PATH_BUDGET:
        _raise_ipc_path_budget_error(
            base_path=base_root,
            endpoint_suffix_len=endpoint_suffix_len,
            path_len=min_dir_len,
        )

    max_prefix_len = (
        _IPC_SUN_PATH_BUDGET
        - len(str(base_root))
        - len("/")
        - len("-")
        - _TEMPFILE_RANDOM_SUFFIX_LEN
        - endpoint_suffix_len
    )
    if max_prefix_len <= 0:
        return ""
    return namespace_prefix[:max_prefix_len]


def _validate_ipc_endpoint_budget(
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path,
) -> None:
    endpoint_suffix_len = _longest_endpoint_suffix_len(stages)
    path_len = len(str(ipc_base_dir)) + endpoint_suffix_len
    if path_len > _IPC_SUN_PATH_BUDGET:
        _raise_ipc_path_budget_error(
            base_path=ipc_base_dir,
            endpoint_suffix_len=endpoint_suffix_len,
            path_len=path_len,
        )


def _longest_endpoint_suffix_len(stages: list[StageConfig]) -> int:
    suffixes = [len("/completion.sock"), len("/abort.sock")]
    suffixes.extend(len(f"/stage_{stage.name}.sock") for stage in stages)
    return max(suffixes)


def _ipc_dir_len(
    *,
    base_root: Path,
    namespace_prefix_len: int,
    endpoint_suffix_len: int,
) -> int:
    dir_name_len = namespace_prefix_len + _TEMPFILE_RANDOM_SUFFIX_LEN
    if namespace_prefix_len:
        dir_name_len += len("-")
    return len(str(base_root)) + len("/") + dir_name_len + endpoint_suffix_len


def _raise_ipc_path_budget_error(
    *,
    base_path: Path,
    endpoint_suffix_len: int,
    path_len: int,
) -> None:
    raise ValueError(
        "IPC endpoint path would exceed the Unix-domain socket path limit "
        f"({_IPC_SUN_PATH_BUDGET} chars): base path {str(base_path)!r} plus "
        f"longest endpoint suffix length {endpoint_suffix_len} yields {path_len} "
        "chars. Shorten endpoints.base_path or stage names."
    )
