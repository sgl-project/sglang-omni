# SPDX-License-Identifier: Apache-2.0
"""End-to-end CI for colocated Qwen3-Omni replicas behind the router."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from sglang_omni.utils import find_available_port
from tests.utils import (
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_partitioned,
    disable_proxy,
    no_proxy_env,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 20
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_TAIL_LINES = 120
CONCURRENCY = 16
MAX_SAMPLES = 50
DATASET_CACHE_ENV = "SGLANG_SEEDTTS50_DIR"
WER_TIMEOUT = 600
ROUTED_WORKER_MIN_SHARE = 0.10
ROUTER_POLICY = "least_request"
COLOCATED_WORKER_ARGS = "--config examples/configs/qwen3_omni_colocated.yaml --colocate"
ROUTER_CLEANUP_MANIFEST_ENV = "SGLANG_OMNI_ROUTER_CLEANUP_MANIFEST"

_ROUTER_COLOCATED_SEEDTTS_REFERENCE = {
    16: {
        "throughput_qps": 5.657,
        "tok_per_s_agg": 5.7,
        "latency_mean_s": 2.585,
        "latency_p95_s": 4.154,
        "rtf_mean": 0.8431,
    },
}
ROUTER_COLOCATED_SEEDTTS_THRESHOLDS = apply_slack(
    _ROUTER_COLOCATED_SEEDTTS_REFERENCE,
    slack_higher=0.75,
    slack_lower=1.50,
)
ROUTER_SEEDTTS_LATENCY_P95_MAX = round(
    _ROUTER_COLOCATED_SEEDTTS_REFERENCE[CONCURRENCY]["latency_p95_s"] * 1.50,
    1,
)
ROUTER_SEEDTTS_WER_BELOW_50_CORPUS_MAX = 0.014184397163120567
ROUTER_SEEDTTS_N_ABOVE_50_MAX = 0


@dataclass
class RouterTopology:
    router_proc: subprocess.Popen
    router_port: int
    worker_ports: list[int]
    router_log: Path | None
    stopped: bool = False

    def stop(self) -> None:
        if self.stopped:
            return
        stop_server(self.router_proc)
        self.stopped = True


@pytest.fixture(scope="module")
def router_topology(tmp_path_factory: pytest.TempPathFactory):
    worker_base_port = _find_available_port_range(2)
    worker_ports = [worker_base_port, worker_base_port + 1]
    router_port = _find_available_port_excluding(worker_ports)
    router_proc: subprocess.Popen | None = None
    router_log: Path | None = None
    topology: RouterTopology | None = None
    cleanup_manifest: Path | None = None

    try:
        launcher_config = _write_ci_launcher_config(
            tmp_path_factory,
            worker_base_port=worker_base_port,
        )
        router_log = server_log_file(tmp_path_factory, "omni_router_logs")
        cleanup_manifest = (
            tmp_path_factory.mktemp("omni_router_cleanup") / "router_pgids.txt"
        )
        router_cmd = [
            sys.executable,
            "-m",
            "sglang_omni_router.serve",
            "--host",
            "0.0.0.0",
            "--port",
            str(router_port),
            "--launcher-config",
            str(launcher_config),
            "--policy",
            ROUTER_POLICY,
            "--health-success-threshold",
            "1",
            "--health-failure-threshold",
            "2",
            "--health-check-interval-secs",
            "2",
            "--log-level",
            "info",
        ]
        router_proc = start_server_from_cmd(
            router_cmd,
            router_log,
            router_port,
            timeout=STARTUP_TIMEOUT + 60,
            env={ROUTER_CLEANUP_MANIFEST_ENV: str(cleanup_manifest)},
        )
        with cleanup_manifest.open("a", encoding="utf-8") as handle:
            handle.write(f"{os.getpgid(router_proc.pid)}\n")
        _wait_for_all_router_workers(router_port, expected_workers=len(worker_ports))
        print(
            "[Omni Router CI] topology "
            f"router_port={router_port} worker_ports={worker_ports} "
            f"launcher_config={launcher_config} policy={ROUTER_POLICY}"
        )
        topology = RouterTopology(
            router_proc=router_proc,
            router_port=router_port,
            worker_ports=worker_ports,
            router_log=router_log,
        )
        yield topology
    finally:
        if topology is not None:
            topology.stop()
        elif router_proc is not None:
            stop_server(router_proc)
        if cleanup_manifest is not None:
            _cleanup_process_groups_from_manifest(cleanup_manifest)


def _cleanup_process_groups_from_manifest(manifest: Path) -> None:
    if not manifest.exists():
        return
    process_group_ids: set[int] = set()
    for line in manifest.read_text().splitlines():
        try:
            process_group_ids.add(int(line.strip()))
        except ValueError:
            continue
    for sig, wait_seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        remaining: set[int] = set()
        for process_group_id in process_group_ids:
            try:
                os.killpg(process_group_id, sig)
                remaining.add(process_group_id)
            except ProcessLookupError:
                continue
        if not remaining:
            return
        time.sleep(wait_seconds)
        process_group_ids = {
            process_group_id
            for process_group_id in remaining
            if _process_group_exists(process_group_id)
        }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _write_ci_launcher_config(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    worker_base_port: int,
) -> Path:
    config_path = tmp_path_factory.mktemp("omni_router_launcher") / "launcher.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "backend": "local",
                    "model_path": MODEL_PATH,
                    "model_name": MODEL_NAME,
                    "num_workers": 2,
                    "num_gpus_per_worker": 1,
                    "worker_host": "127.0.0.1",
                    "worker_base_port": worker_base_port,
                    "worker_extra_args": COLOCATED_WORKER_ARGS,
                    "wait_timeout": STARTUP_TIMEOUT,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _router_get_json(port: int, path: str) -> dict:
    with disable_proxy():
        response = requests.get(
            f"http://127.0.0.1:{port}{path}",
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def _wait_for_all_router_workers(
    port: int,
    *,
    expected_workers: int,
    timeout: int = 120,
) -> None:
    deadline = time.monotonic() + timeout
    last_payload: dict | None = None
    while time.monotonic() < deadline:
        last_payload = _router_get_json(port, "/workers")
        if (
            last_payload["total_workers"] == expected_workers
            and last_payload["healthy_workers"] == expected_workers
            and last_payload["routable_workers"] == expected_workers
        ):
            return
        time.sleep(1)
    raise TimeoutError(f"router workers did not become fully routable: {last_payload}")


def _print_worker_snapshot(label: str, snapshot: dict) -> None:
    worker_states = [
        (
            worker["display_id"],
            worker["health_state"],
            worker["active_requests"],
            worker.get("routed_requests", 0),
            worker.get("successful_requests", 0),
            worker.get("failed_requests", 0),
            worker["routable"],
        )
        for worker in snapshot["workers"]
    ]
    print(
        f"[Omni Router CI] {label} "
        f"healthy={snapshot['healthy_workers']} "
        f"routable={snapshot['routable_workers']} "
        f"workers=(id, state, active, routed, successful, failed, routable) "
        f"{worker_states}"
    )


def _print_log_tail(label: str, log_file: Path | None) -> None:
    if log_file is None:
        print(f"[Omni Router CI] {label} log is streamed to terminal outside CI")
        return
    if not log_file.exists():
        print(f"[Omni Router CI] {label} log missing: {log_file}")
        return
    with log_file.open("r", encoding="utf-8", errors="replace") as log_handle:
        lines = deque(log_handle, maxlen=LOG_TAIL_LINES)
    print(f"\n[Omni Router CI] {label} log tail ({log_file})")
    for line in lines:
        print(line.rstrip())


def _print_diagnostics(topology: RouterTopology) -> None:
    try:
        _print_worker_snapshot(
            "failure /workers snapshot",
            _router_get_json(topology.router_port, "/workers"),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[Omni Router CI] failed to fetch /workers during diagnostics: {exc}")
    _print_log_tail("router", topology.router_log)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override_dir = os.environ.get(DATASET_CACHE_ENV)
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-50"], str(root), quiet=True)
    return root


def _run_seedtts_generate(
    *,
    router_port: int,
    meta_path: Path,
    output_dir: Path,
) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model=MODEL_NAME,
        base_url=f"http://127.0.0.1:{router_port}",
        port=router_port,
        meta=str(meta_path),
        output_dir=str(output_dir),
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        voice_clone=True,
    )
    results = asyncio.run(run_omni_seedtts_benchmark(config))
    assert "summary" in results
    assert "per_request" in results
    return results


def _run_seedtts_transcribe(
    *,
    meta_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--transcribe-only",
        "--meta",
        str(meta_path),
        "--output-dir",
        str(output_dir),
        "--model",
        MODEL_NAME,
        "--lang",
        "en",
        "--device",
        device,
    ]
    env = no_proxy_env()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=WER_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"SeedTTS transcribe failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    results_path = output_dir / "wer_results.json"
    assert results_path.exists(), f"WER results file not found: {results_path}"
    with results_path.open() as f:
        return json.load(f)


def _assert_both_workers_served_requests(snapshot: dict) -> None:
    workers = snapshot["workers"]
    routed_counts = [int(worker.get("routed_requests", 0)) for worker in workers]
    successful_counts = [
        int(worker.get("successful_requests", 0)) for worker in workers
    ]
    failed_counts = [int(worker.get("failed_requests", 0)) for worker in workers]
    total_routed = sum(routed_counts)
    min_expected = max(1, int(total_routed * ROUTED_WORKER_MIN_SHARE))

    assert len(workers) == 2
    assert total_routed >= MAX_SAMPLES, (
        f"Expected at least {MAX_SAMPLES} routed requests, got {total_routed}: "
        f"{routed_counts}"
    )
    assert all(count >= min_expected for count in routed_counts), (
        f"Both router workers must serve traffic. routed={routed_counts}, "
        f"minimum_per_worker={min_expected}"
    )
    assert sum(successful_counts) == total_routed, (
        f"All routed requests should succeed. successful={successful_counts}, "
        f"routed={routed_counts}"
    )
    assert sum(failed_counts) == 0, f"Router recorded request failures: {failed_counts}"


@pytest.mark.benchmark
def test_colocated_router_seedtts_uses_both_workers(
    router_topology: RouterTopology,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    try:
        workers = _router_get_json(router_topology.router_port, "/workers")
        _print_worker_snapshot("initial /workers snapshot", workers)
        assert workers["total_workers"] == 2
        assert workers["healthy_workers"] == 2
        assert workers["routable_workers"] == 2

        models = _router_get_json(router_topology.router_port, "/v1/models")
        assert {card["id"] for card in models["data"]} == {MODEL_NAME}

        output_dir = tmp_path / "qwen3_omni_colocated_router_seedtts50"
        meta_path = dataset_dir / "en" / "meta.lst"
        speed_results = _run_seedtts_generate(
            router_port=router_topology.router_port,
            meta_path=meta_path,
            output_dir=output_dir,
        )
        print_speed_summary(
            speed_results["summary"],
            MODEL_NAME,
            CONCURRENCY,
            title="Colocated Router SeedTTS Speed",
        )
        assert_summary_metrics(speed_results["summary"])
        assert_per_request_fields(speed_results["per_request"])
        assert_speed_thresholds(
            speed_results["summary"],
            ROUTER_COLOCATED_SEEDTTS_THRESHOLDS,
            CONCURRENCY,
        )
        assert (
            speed_results["summary"]["latency_p95_s"] <= ROUTER_SEEDTTS_LATENCY_P95_MAX
        ), (
            f"latency_p95_s {speed_results['summary']['latency_p95_s']} > "
            f"{ROUTER_SEEDTTS_LATENCY_P95_MAX} at concurrency {CONCURRENCY}"
        )

        final_workers = _router_get_json(router_topology.router_port, "/workers")
        _print_worker_snapshot("final /workers snapshot", final_workers)
        assert final_workers["routable_workers"] == 2
        assert all(
            worker["active_requests"] == 0 for worker in final_workers["workers"]
        )
        _assert_both_workers_served_requests(final_workers)

        router_topology.stop()
        wer_results = _run_seedtts_transcribe(
            meta_path=meta_path,
            output_dir=output_dir,
        )
        print_wer_summary(wer_results["summary"], MODEL_NAME)
        assert_wer_partitioned(
            wer_results,
            max_wer_below_50_corpus=ROUTER_SEEDTTS_WER_BELOW_50_CORPUS_MAX,
            max_n_above_50=ROUTER_SEEDTTS_N_ABOVE_50_MAX,
        )
        _print_log_tail("router", router_topology.router_log)
    except Exception:
        _print_diagnostics(router_topology)
        raise


def _find_available_port_excluding(excluded: list[int]) -> int:
    excluded_ports = set(excluded)
    while True:
        port = find_available_port()
        if port not in excluded_ports:
            return port


def _find_available_port_range(count: int) -> int:
    for _ in range(100):
        base_port = find_available_port()
        candidates = [base_port + offset for offset in range(count)]
        if all(_port_is_available(port) for port in candidates):
            return base_port
    raise RuntimeError(f"failed to find {count} consecutive available ports")


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True
