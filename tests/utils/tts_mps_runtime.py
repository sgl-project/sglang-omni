# SPDX-License-Identifier: Apache-2.0
"""Runtime and evidence boundary for the standalone TTS MPS CI stage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

REPLICA_COUNT = 2
RUN_ID_PATTERN = re.compile(r"run-[A-Za-z0-9_-]+")
# AF_UNIX sun_path is 108 bytes including the terminator on Linux.
SUN_PATH_LIMIT = 107
# Two pinned cores per replica.
MINIMUM_CORES = 2 * REPLICA_COUNT
HOST_RESERVE_RATIO = 0.25


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def new_summary(
    *,
    exact_sha: str,
    run_id: str,
    run_attempt: str,
    selected_model: str,
) -> dict[str, Any]:
    if selected_model not in {"higgs", "moss"}:
        raise ValueError(f"unsupported selected model {selected_model!r}")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be one safe run-<suffix> path component")
    return {
        "schema_version": 1,
        "exact_sha": exact_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "selected_model": selected_model,
        "topology": "mps_dp2_independent_weights",
        "runtime": None,
        "correctness": None,
        "cleanup": None,
        "evaluator": None,
        "timing": {},
    }


def validate_final_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != 1:
        raise ValueError("summary schema version mismatch")
    for section in ("runtime", "correctness", "cleanup", "evaluator"):
        value = summary.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"summary is incomplete: {section} is missing")
        if value.get("status") != "pass":
            raise ValueError(f"{section} status is not pass: {value.get('status')!r}")


def update_summary(path: str | Path, **sections: Any) -> dict[str, Any]:
    """Merge into each named section rather than replacing it.

    Callers update one field at a time, so whole-section replacement forced
    them to re-read the file and splice the previous contents back in.
    """
    summary_path = Path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in sections.items():
        if key not in payload:
            raise ValueError(f"unknown summary section {key!r}")
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    atomic_write_json(summary_path, payload)
    return payload


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.split(","):
        start, separator, end = item.strip().partition("-")
        if not start:
            continue
        first = int(start)
        last = int(end) if separator else first
        if last < first:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.extend(range(first, last + 1))
    return cpus


def format_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        raise ValueError("CPU block cannot be empty")
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def derive_core_blocks(
    gpu_id: int,
    *,
    pci_devices_root: Path = Path("/sys/bus/pci/devices"),
    numa_nodes_root: Path = Path("/sys/devices/system/node"),
) -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
            "-i",
            str(gpu_id),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    bus_id = result.stdout.strip().lower()
    try:
        domain, bus, device = bus_id.split(":")
        pci_device = pci_devices_root / f"{int(domain, 16):04x}:{bus}:{device}"
    except ValueError as exc:
        raise RuntimeError(f"invalid PCI bus ID for GPU {gpu_id}: {bus_id!r}") from exc
    if not pci_device.is_dir():
        raise RuntimeError(f"cannot resolve one PCI device for GPU {gpu_id}")
    numa_node = int((pci_device / "numa_node").read_text().strip())
    if numa_node < 0:
        raise RuntimeError(f"GPU {gpu_id} has no usable NUMA node")
    node_cpus = set(
        parse_cpu_list(
            (numa_nodes_root / f"node{numa_node}" / "cpulist").read_text().strip()
        )
    )
    allowed = sorted(node_cpus & set(os.sched_getaffinity(0)))
    if len(allowed) < MINIMUM_CORES:
        raise RuntimeError(
            f"MPS DP2 requires {MINIMUM_CORES} allowed CPU cores on the GPU "
            f"NUMA node, found {len(allowed)}"
        )
    # Reserve a slice for host-side work, but never below the hard minimum, so
    # a host that does have enough cores is not reported as insufficient.
    usable = allowed[: max(MINIMUM_CORES, int(len(allowed) * (1 - HOST_RESERVE_RATIO)))]
    split = len(usable) // 2
    return format_cpu_list(usable[:split]), format_cpu_list(usable[split:])


def require_exact_request_counts(
    summary: dict[str, Any],
    *,
    expected_requests: int,
) -> dict[str, int]:
    expected = {
        "total_requests": expected_requests,
        "completed_requests": expected_requests,
        "failed_requests": 0,
    }
    counts = {key: summary.get(key) for key in expected}
    if any(type(value) is not int for value in counts.values()) or counts != expected:
        raise RuntimeError(f"request counts do not match {expected}: {counts}")
    return {key: summary[key] for key in expected}


@dataclass(frozen=True)
class MpsLaunchSpec:
    repository_root: Path
    output_dir: Path
    state_root: Path
    run_id: str
    config_path: Path
    gpu_id: int
    base_port: int
    core_blocks: tuple[str, ...]
    python_bin: str
    serve_extra_args: str = ""

    def __post_init__(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe run-<suffix> path component")
        if len(self.core_blocks) != REPLICA_COUNT or any(
            not item.strip() for item in self.core_blocks
        ):
            raise ValueError("MPS DP2 requires two non-empty CPU core blocks")
        if self.gpu_id < 0 or not 1 <= self.base_port < 65535:
            raise ValueError("invalid GPU id or base port")
        if not self.config_path.is_file():
            raise ValueError(f"MPS config does not exist: {self.config_path}")
        if not (self.repository_root / "examples/mps_dp/launch.sh").is_file():
            raise ValueError("production MPS launcher does not exist")
        socket_bytes = len(str(self.control_socket).encode())
        if socket_bytes > SUN_PATH_LIMIT:
            # Note: (Jiaxin Deng) over the limit the daemon starts, fails to
            # bind, and exits, and the launcher only ever reports "Cannot find
            # MPS control daemon process". Name the real cause here instead.
            raise ValueError(
                f"MPS control socket path is {socket_bytes} bytes, over the "
                f"{SUN_PATH_LIMIT}-byte AF_UNIX sun_path limit: "
                f"{self.control_socket}. Use a shorter state root."
            )

    @property
    def command(self) -> tuple[str, ...]:
        return (
            "bash",
            str(self.repository_root / "examples/mps_dp/launch.sh"),
            "up",
        )

    @property
    def teardown_command(self) -> tuple[str, ...]:
        return (
            "bash",
            str(self.repository_root / "examples/mps_dp/launch.sh"),
            "down",
            self.run_id,
        )

    @property
    def state_dir(self) -> Path:
        return self.state_root / f"gpu-{self.gpu_id}" / self.run_id

    @property
    def control_socket(self) -> Path:
        return self.state_dir / "mps" / "pipe" / "control"

    @property
    def worker_urls(self) -> tuple[str, ...]:
        return tuple(
            f"http://127.0.0.1:{self.base_port + index}"
            for index in range(REPLICA_COUNT)
        )

    @property
    def environment(self) -> dict[str, str]:
        return {
            "BASE_PORT": str(self.base_port),
            "CONFIG": str(self.config_path),
            "CORE_BLOCKS": " ".join(self.core_blocks),
            "GPU_ID": str(self.gpu_id),
            "N": "2",
            "PYTHON_BIN": self.python_bin,
            "RUN_ID": self.run_id,
            "SERVE_EXTRA_ARGS": self.serve_extra_args,
            "STATE_ROOT": str(self.state_root),
            "WEIGHT_SHARE": "0",
        }


@dataclass(frozen=True)
class ReplicaState:
    index: int
    pid: int
    pgid: int
    port: int
    log_path: Path
    process_start: str
    kv_tokens: int


@dataclass(frozen=True)
class LauncherState:
    state_dir: Path
    manifest: dict[str, str]
    replicas: tuple[ReplicaState, ...]


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid launcher manifest line: {line!r}")
        values[key] = value
    return values


def read_launcher_state(state_dir: str | Path) -> LauncherState:
    state = Path(state_dir)
    manifest = _read_key_values(state / "manifest")
    required = {
        "run_id",
        "gpu_id",
        "gpu_uuid",
        "numa_node",
        "config",
        "n",
        "base_port",
        "core_blocks",
        "max_total_tokens",
        "weight_share",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"launcher manifest is missing fields: {missing}")
    if (
        manifest["n"] != "2"
        or manifest["weight_share"] != "0"
        or not manifest["max_total_tokens"].isdigit()
    ):
        raise ValueError("launcher did not establish independent-weight DP2")

    expected_kv = int(manifest["max_total_tokens"])
    replicas: list[ReplicaState] = []
    for line in (state / "replicas.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid replicas.tsv line: {line!r}")
        index, pid, pgid, port, raw_log, process_start = fields
        log_path = Path(raw_log)
        if not log_path.is_absolute():
            log_path = state / log_path
        replicas.append(
            ReplicaState(
                int(index),
                int(pid),
                int(pgid),
                int(port),
                log_path,
                process_start,
                expected_kv,
            )
        )
    if [item.index for item in replicas] != [0, 1]:
        raise ValueError("replicas.tsv must contain ordered replicas 0 and 1")
    if [item.port for item in replicas] != [
        int(manifest["base_port"]),
        int(manifest["base_port"]) + 1,
    ]:
        raise ValueError("replica ports do not match the launcher manifest")
    attachment = (state / "mps_attach.txt").read_text(encoding="utf-8")
    if "RESULT: PASS" not in attachment or any(
        f"replica {item.index} (port {item.port}): attached clients:" not in attachment
        for item in replicas
    ):
        raise ValueError("both replicas are not proven attached to private MPS")
    return LauncherState(state, manifest, tuple(replicas))


def _copy_raw_state(state: Path, output_dir: Path) -> Path:
    target = output_dir / "raw"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest",
        "replicas.tsv",
        "mps_attach.txt",
        "mps_ctl.err",
    ):
        source = state / name
        if source.is_file() and not source.is_symlink():
            shutil.copy2(source, target / name)
    logs = state / "logs"
    if logs.is_dir() and not logs.is_symlink():
        shutil.copytree(logs, target / "logs", dirs_exist_ok=True)
    mps_logs = state / "mps" / "log"
    if mps_logs.is_dir() and not mps_logs.is_symlink():
        shutil.copytree(mps_logs, target / "mps" / "log", dirs_exist_ok=True)
    for activity in state.glob("activity-*"):
        if activity.is_dir() and not activity.is_symlink():
            shutil.copytree(
                activity,
                target / activity.name,
                dirs_exist_ok=True,
            )
    return target


def reconcile_stale_launcher_states(spec: MpsLaunchSpec) -> None:
    gpu_state_root = spec.state_root / f"gpu-{spec.gpu_id}"
    if not gpu_state_root.exists():
        return
    environment = os.environ.copy()
    environment.update(spec.environment)
    for state in sorted(gpu_state_root.glob("run-*")):
        if state == spec.state_dir:
            continue
        if state.is_symlink() or not state.is_dir():
            raise RuntimeError(f"invalid stale launcher state: {state}")
        if not RUN_ID_PATTERN.fullmatch(state.name):
            raise RuntimeError(f"unsafe stale launcher run id: {state.name}")
        _copy_raw_state(
            state,
            spec.output_dir / "recovered-stale-launcher-state" / state.name,
        )
        result = subprocess.run(
            (
                "bash",
                str(spec.repository_root / "examples/mps_dp/launch.sh"),
                "down",
                state.name,
            ),
            cwd=spec.repository_root,
            env=environment,
            check=False,
        )
        if result.returncode != 0 or state.exists():
            raise RuntimeError(f"stale launcher state remains after teardown: {state}")


def launch_replicas(spec: MpsLaunchSpec) -> LauncherState:
    reconcile_stale_launcher_states(spec)
    environment = os.environ.copy()
    environment.update(spec.environment)
    try:
        subprocess.run(
            spec.command,
            cwd=spec.repository_root,
            env=environment,
            check=True,
        )
        snapshot = read_launcher_state(spec.state_dir)
        _copy_raw_state(spec.state_dir, spec.output_dir)
        return snapshot
    except BaseException:
        if spec.state_dir.exists():
            _copy_raw_state(spec.state_dir, spec.output_dir)
            subprocess.run(
                spec.teardown_command,
                cwd=spec.repository_root,
                env=environment,
                check=False,
            )
        raise


def _pid_alive(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def _port_open(port: int) -> bool:
    with socket.socket() as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def capture_gpu_clients() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    clients: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 2)]
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        clients.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
            }
        )
    return clients


def new_gpu_clients(
    baseline: list[dict[str, Any]],
    post: list[dict[str, Any]],
    *,
    gpu_uuid: str,
) -> list[dict[str, Any]]:
    existing = {
        (item.get("gpu_uuid"), item.get("pid"), item.get("process_name"))
        for item in baseline
    }
    return [
        item
        for item in post
        if item.get("gpu_uuid") == gpu_uuid
        and (item.get("gpu_uuid"), item.get("pid"), item.get("process_name"))
        not in existing
    ]


def verify_active_gpu_visibility(
    baseline: list[dict[str, Any]],
    active: list[dict[str, Any]],
    *,
    gpu_uuid: str,
) -> list[dict[str, Any]]:
    visible = new_gpu_clients(baseline, active, gpu_uuid=gpu_uuid)
    if not visible:
        raise RuntimeError(
            "GPU client visibility is unavailable: no new target-GPU client "
            "was visible while the MPS replicas were active"
        )
    return visible


def require_clean_cleanup(verdict: dict[str, Any]) -> None:
    if verdict.get("status") != "pass":
        raise RuntimeError(f"MPS teardown is dirty: {verdict}")


def record_cleanup_recovery(
    existing: Any,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(existing, dict) and existing.get("status") == "dirty":
        preserved = dict(existing)
        preserved["recovery_attempt"] = recovery
        return preserved
    return recovery


def wait_for_gpu_clients_to_settle(
    baseline: list[dict[str, Any]],
    *,
    gpu_uuid: str,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.5,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while True:
        post = capture_gpu_clients()
        if not new_gpu_clients(baseline, post, gpu_uuid=gpu_uuid):
            return post
        if time.monotonic() >= deadline:
            return post
        time.sleep(poll_interval_s)


def teardown_replicas(
    spec: MpsLaunchSpec,
    snapshot: LauncherState,
    *,
    baseline_gpu_clients: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    _copy_raw_state(snapshot.state_dir, spec.output_dir)
    environment = os.environ.copy()
    environment.update(spec.environment)
    result = subprocess.run(
        spec.teardown_command,
        cwd=spec.repository_root,
        env=environment,
        check=False,
    )
    live_pids = [item.pid for item in snapshot.replicas if _pid_alive(item.pid)]
    occupied_ports = [item.port for item in snapshot.replicas if _port_open(item.port)]
    post_gpu_clients = (
        wait_for_gpu_clients_to_settle(
            baseline_gpu_clients,
            gpu_uuid=snapshot.manifest["gpu_uuid"],
        )
        if baseline_gpu_clients is not None
        else capture_gpu_clients()
    )
    leaked_clients = (
        new_gpu_clients(
            baseline_gpu_clients,
            post_gpu_clients,
            gpu_uuid=snapshot.manifest["gpu_uuid"],
        )
        if baseline_gpu_clients is not None
        else []
    )
    checks = {
        "launch_down_succeeded": result.returncode == 0,
        "tracked_processes_exited": not live_pids,
        "ports_released": not occupied_ports,
        "mps_state_removed": not snapshot.state_dir.exists(),
        "gpu_baseline_available": baseline_gpu_clients is not None,
        "no_new_gpu_clients": not leaked_clients,
    }
    verdict = {
        "status": "pass" if all(checks.values()) else "dirty",
        "checks": checks,
        "live_pids": live_pids,
        "occupied_ports": occupied_ports,
        "new_gpu_clients": leaked_clients,
        "post_gpu_clients": post_gpu_clients,
    }
    return verdict


def start_request_profiles(snapshot: LauncherState, run_id: str) -> None:
    for replica in snapshot.replicas:
        event_dir = snapshot.state_dir / f"activity-{replica.index}"
        response = requests.post(
            f"http://127.0.0.1:{replica.port}/start_request_profile",
            json={"run_id": run_id, "event_dir": str(event_dir)},
            timeout=30,
        )
        response.raise_for_status()
    wait_for_profile_activity_files(snapshot)


def wait_for_profile_activity_files(
    snapshot: LauncherState,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
) -> None:
    engine_process_stages = ("preprocessing", "audio_encoder", "tts_engine")
    deadline = time.monotonic() + timeout_s
    while True:
        pending = [
            replica.index
            for replica in snapshot.replicas
            if not any(
                path
                for stage in engine_process_stages
                for path in (snapshot.state_dir / f"activity-{replica.index}").glob(
                    f"events_{stage}_*.jsonl"
                )
            )
        ]
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "TTS engine-process request profilers did not start for "
                f"replicas {pending}"
            )
        time.sleep(poll_interval_s)


def stop_request_profiles(snapshot: LauncherState, run_id: str) -> None:
    errors: list[str] = []
    for replica in snapshot.replicas:
        try:
            response = requests.post(
                f"http://127.0.0.1:{replica.port}/stop_request_profile",
                json={"run_id": run_id},
                timeout=30,
            )
            response.raise_for_status()
        except Exception as exc:
            errors.append(f"replica {replica.index}: {exc!r}")
    if errors:
        raise RuntimeError(f"request profiler stop failed: {errors}")


def _read_model_path_activity_once(snapshot: LauncherState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for replica in snapshot.replicas:
        directory = snapshot.state_dir / f"activity-{replica.index}"
        for path in sorted(directory.glob("events_*.jsonl")):
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for line_index, line in enumerate(lines):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    if line_index == len(lines) - 1 and not text.endswith("\n"):
                        continue
                    raise
                if raw.get("event_name") not in {
                    "model_path_start",
                    "model_path_end",
                }:
                    continue
                metadata = raw.get("metadata") or {}
                events.append(
                    {
                        "run_id": raw.get("run_id"),
                        "replica_id": replica.index,
                        "request_id": raw.get("request_id"),
                        "event": raw["event_name"],
                        "clock": metadata.get("clock"),
                        "monotonic_ns": metadata.get("monotonic_ns"),
                        "host_boot_id": metadata.get("host_boot_id"),
                        "status": metadata.get("status"),
                    }
                )
    return events


def read_model_path_activity(
    snapshot: LauncherState,
    *,
    min_terminal_events: int = 0,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while True:
        events = _read_model_path_activity_once(snapshot)
        terminal_count = sum(
            item.get("event") == "model_path_end" and item.get("status") == "success"
            for item in events
        )
        if terminal_count >= min_terminal_events:
            return events
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "request profiler did not flush all successful terminal events: "
                f"{terminal_count} < {min_terminal_events}"
            )
        time.sleep(poll_interval_s)


def finalize_from_cli(summary_path: Path, spec: MpsLaunchSpec) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if spec.state_dir.exists():
        existing_cleanup = payload.get("cleanup")
        try:
            snapshot = read_launcher_state(spec.state_dir)
            runtime = payload.get("runtime")
            baseline = (
                runtime.get("baseline_gpu_clients")
                if isinstance(runtime, dict)
                else None
            )
            cleanup = teardown_replicas(
                spec,
                snapshot,
                baseline_gpu_clients=baseline,
            )
        except BaseException as exc:
            payload["cleanup"] = record_cleanup_recovery(
                existing_cleanup,
                {"status": "dirty", "error": repr(exc)},
            )
            atomic_write_json(summary_path, payload)
            raise
        payload["cleanup"] = record_cleanup_recovery(existing_cleanup, cleanup)
        atomic_write_json(summary_path, payload)
        require_clean_cleanup(payload["cleanup"])
    validate_final_summary(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--summary", type=Path, required=True)
    initialize.add_argument("--exact-sha", required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--run-attempt", required=True)
    initialize.add_argument("--selected-model", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--repo", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--state-root", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--gpu-id", type=int, default=0)
    finalize.add_argument("--base-port", type=int, default=18801)
    args = parser.parse_args()
    if args.command == "initialize":
        atomic_write_json(
            args.summary,
            new_summary(
                exact_sha=args.exact_sha,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
                selected_model=args.selected_model,
            ),
        )
    elif args.command == "finalize":
        spec = MpsLaunchSpec(
            repository_root=args.repo.resolve(),
            output_dir=args.output.resolve(),
            state_root=args.state_root.resolve(),
            run_id=args.run_id,
            config_path=args.config.resolve(),
            gpu_id=args.gpu_id,
            base_port=args.base_port,
            core_blocks=("unused-0", "unused-1"),
            python_bin="python",
        )
        finalize_from_cli(args.summary, spec)


if __name__ == "__main__":
    main()
