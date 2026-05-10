# SPDX-License-Identifier: Apache-2.0
"""End-to-end CI for the Omni router with two Qwen3-Omni V1 replicas."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 120
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGE_PATH = DATA_DIR / "cars.jpg"


@dataclass
class RouterTopology:
    router_proc: subprocess.Popen
    worker_procs: list[subprocess.Popen]
    router_port: int
    worker_ports: list[int]


@pytest.fixture(scope="module")
def router_topology(tmp_path_factory: pytest.TempPathFactory):
    ports = _find_available_ports(3)
    worker_ports = ports[:2]
    router_port = ports[2]
    worker_procs: list[subprocess.Popen] = []
    router_proc: subprocess.Popen | None = None

    try:
        for gpu_id, port in enumerate(worker_ports):
            log_file = server_log_file(tmp_path_factory, f"router_worker_{gpu_id}")
            cmd = [
                sys.executable,
                "examples/run_qwen3_omni_server.py",
                "--model-path",
                MODEL_PATH,
                "--model-name",
                MODEL_NAME,
                "--port",
                str(port),
            ]
            proc = start_server_from_cmd(
                cmd,
                log_file,
                port,
                timeout=STARTUP_TIMEOUT,
                env={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
            )
            worker_procs.append(proc)

        worker_urls = [f"http://127.0.0.1:{port}" for port in worker_ports]
        router_log = server_log_file(tmp_path_factory, "omni_router_logs")
        router_cmd = [
            sys.executable,
            "-m",
            "sglang_omni_router.serve",
            "--host",
            "0.0.0.0",
            "--port",
            str(router_port),
            "--worker-urls",
            *worker_urls,
            "--policy",
            "round_robin",
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
            timeout=120,
        )
        _wait_for_all_router_workers(router_port, expected_workers=len(worker_ports))
        yield RouterTopology(
            router_proc=router_proc,
            worker_procs=worker_procs,
            router_port=router_port,
            worker_ports=worker_ports,
        )
    finally:
        if router_proc is not None:
            stop_server(router_proc)
        for proc in worker_procs:
            stop_server(proc)


def _router_get_json(port: int, path: str) -> dict:
    with disable_proxy():
        response = requests.get(
            f"http://127.0.0.1:{port}{path}",
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def _router_chat(port: int, request_id: str) -> requests.Response:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "How many cars are there in the image? Answer briefly.",
            }
        ],
        "images": [str(IMAGE_PATH)],
        "modalities": ["text"],
        "max_tokens": 8,
    }
    with disable_proxy():
        return requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"x-request-id": request_id},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )


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


@pytest.mark.benchmark
def test_router_round_robin_uses_both_qwen3_omni_workers(
    router_topology: RouterTopology,
) -> None:
    workers = _router_get_json(router_topology.router_port, "/workers")
    assert workers["total_workers"] == 2
    assert workers["healthy_workers"] == 2
    assert workers["routable_workers"] == 2

    models = _router_get_json(router_topology.router_port, "/v1/models")
    assert {card["id"] for card in models["data"]} == {MODEL_NAME}

    selected_workers: list[str] = []
    for index in range(4):
        response = _router_chat(
            router_topology.router_port,
            request_id=f"router-ci-{index}",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0
        selected_workers.append(response.headers["x-sglang-omni-worker"])

    assert len(set(selected_workers)) == 2
    assert selected_workers[0] == selected_workers[2]
    assert selected_workers[1] == selected_workers[3]
    assert selected_workers[0] != selected_workers[1]


def _find_available_ports(count: int) -> list[int]:
    ports: list[int] = []
    while len(ports) < count:
        port = find_available_port()
        if port not in ports:
            ports.append(port)
    return ports
