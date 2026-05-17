# SPDX-License-Identifier: Apache-2.0
"""Runtime preparation helpers shared by pipeline runners."""

from __future__ import annotations

import logging
import re
import shutil
import socket
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
    runtime_dir: IpcRuntimeDir | None
    runtime_dir_created_here: bool


def create_ipc_runtime_dir(config: PipelineConfig) -> IpcRuntimeDir | None:
    """Create a per-run IPC namespace for one pipeline instance."""
    if config.endpoints.scheme != "ipc":
        return None

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
    created_runtime_dir = None
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config)
        created_runtime_dir = runtime_dir
    runtime_dir_created_here = created_runtime_dir is not None

    try:
        stages_cfg, name_map, entry_stage = config.apply_fusion()
        placement_plan = build_stage_placement_plan(config, stages_cfg=stages_cfg)
        process_plan = build_process_topology_plan(
            config,
            placement_plan,
            stages_cfg=stages_cfg,
        )
        endpoints = allocate_endpoints(
            config,
            stages=stages_cfg,
            ipc_base_dir=runtime_dir.path if runtime_dir else None,
        )
    except Exception:
        if created_runtime_dir is not None:
            created_runtime_dir.close()
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


def find_free_tcp_ports(start: int, count: int) -> list[int]:
    """Find *count* available TCP ports starting from *start*."""
    ports: list[int] = []
    port = start
    while len(ports) < count:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                ports.append(port)
        except OSError:
            # Port is already bound or otherwise unbindable; advance to the
            # next candidate.
            pass
        port += 1
    return ports


def allocate_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path | None = None,
) -> dict[str, str]:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        if ipc_base_dir is None:
            raise ValueError("IPC endpoint allocation requires an IPC runtime dir")
        base_dir = ipc_base_dir
        endpoints.setdefault("completion", f"ipc://{base_dir}/completion.sock")
        endpoints.setdefault("abort", f"ipc://{base_dir}/abort.sock")
        for stage in stages:
            endpoints[f"stage_{stage.name}"] = (
                f"ipc://{base_dir}/stage_{stage.name}.sock"
            )
        return endpoints

    if config.endpoints.scheme == "tcp":
        needed = 2 + len(stages)
        ports = find_free_tcp_ports(config.endpoints.base_port, needed)
        idx = 0
        if "completion" not in endpoints:
            endpoints["completion"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        if "abort" not in endpoints:
            endpoints["abort"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        for stage in stages:
            endpoints[f"stage_{stage.name}"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        return endpoints

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")
