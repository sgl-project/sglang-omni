# SPDX-License-Identifier: Apache-2.0
"""End-to-end CI for the Omni router with two Qwen3-Omni V1 replicas."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

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
REQUEST_TIMEOUT = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGE_PATH = DATA_DIR / "cars.jpg"
AUDIO_PATH = DATA_DIR / "query_to_cars.wav"
VIDEO_PATH = DATA_DIR / "draw.mp4"
VIDEO_AUDIO_PATH = DATA_DIR / "query_to_draw.wav"
LOG_TAIL_LINES = 120


@dataclass
class RouterTopology:
    router_proc: subprocess.Popen
    router_port: int
    worker_ports: list[int]
    router_log: Path | None


@pytest.fixture(scope="module")
def router_topology(tmp_path_factory: pytest.TempPathFactory):
    worker_base_port = _find_available_port_range(2)
    worker_ports = [worker_base_port, worker_base_port + 1]
    router_port = _find_available_port_excluding(worker_ports)
    router_proc: subprocess.Popen | None = None
    router_log: Path | None = None

    try:
        launcher_config = _write_ci_launcher_config(
            tmp_path_factory,
            worker_base_port=worker_base_port,
        )
        router_log = server_log_file(tmp_path_factory, "omni_router_logs")
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
            timeout=STARTUP_TIMEOUT + 60,
        )
        _wait_for_all_router_workers(router_port, expected_workers=len(worker_ports))
        print(
            "[Omni Router CI] topology "
            f"router_port={router_port} worker_ports={worker_ports} "
            f"launcher_config={launcher_config} policy=round_robin"
        )
        yield RouterTopology(
            router_proc=router_proc,
            router_port=router_port,
            worker_ports=worker_ports,
            router_log=router_log,
        )
    finally:
        if router_proc is not None:
            stop_server(router_proc)


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
                    "worker_extra_args": "--text-only",
                    "worker_capabilities": [
                        "chat",
                        "streaming",
                        "image_input",
                        "audio_input",
                        "video_input",
                    ],
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


def _router_chat(
    port: int,
    request_id: str,
    payload: dict[str, Any],
) -> requests.Response:
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


def _chat_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "modalities": ["text"],
        "max_tokens": 8,
    }
    payload.update(overrides)
    return payload


def _assert_chat_response(
    response: requests.Response,
    *,
    scenario: str,
) -> str:
    assert (
        response.status_code == 200
    ), f"{scenario} failed with status={response.status_code}: {response.text}"
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content) > 0
    worker = response.headers["x-sglang-omni-worker"]
    print(
        "[Omni Router CI] request "
        f"scenario={scenario} status={response.status_code} "
        f"worker={worker} preview={content[:80]!r}"
    )
    return worker


def _run_streaming_text_request(port: int) -> str:
    payload = _chat_payload(
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
        max_tokens=16,
    )
    with disable_proxy():
        with requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"x-request-id": "router-ci-streaming-text"},
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            chunks = [
                chunk.decode("utf-8", errors="replace")
                for chunk in response.iter_content(chunk_size=None)
                if chunk
            ]
            status_code = response.status_code
            response_headers = dict(response.headers)
    assert (
        status_code == 200
    ), f"streaming_text failed with status={status_code}: {''.join(chunks)}"
    body = "".join(chunks)
    assert body.strip(), "streaming_text returned an empty stream"
    worker = response_headers["x-sglang-omni-worker"]
    print(
        "[Omni Router CI] request "
        f"scenario=streaming_text status={status_code} "
        f"worker={worker} chunks={len(chunks)} preview={body[:80]!r}"
    )
    return worker


def _print_worker_snapshot(label: str, snapshot: dict) -> None:
    worker_states = [
        (
            worker["display_id"],
            worker["health_state"],
            worker["active_requests"],
            worker["routable"],
        )
        for worker in snapshot["workers"]
    ]
    print(
        f"[Omni Router CI] {label} "
        f"healthy={snapshot['healthy_workers']} "
        f"routable={snapshot['routable_workers']} "
        f"workers={worker_states}"
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


@pytest.mark.benchmark
def test_router_routes_common_qwen3_omni_requests_across_both_workers(
    router_topology: RouterTopology,
) -> None:
    try:
        workers = _router_get_json(router_topology.router_port, "/workers")
        _print_worker_snapshot("initial /workers snapshot", workers)
        assert workers["total_workers"] == 2
        assert workers["healthy_workers"] == 2
        assert workers["routable_workers"] == 2

        models = _router_get_json(router_topology.router_port, "/v1/models")
        assert {card["id"] for card in models["data"]} == {MODEL_NAME}

        scenarios: list[tuple[str, dict[str, Any]]] = [
            (
                "text_only",
                _chat_payload(
                    messages=[
                        {
                            "role": "user",
                            "content": "Answer with exactly one word: ready",
                        }
                    ],
                ),
            ),
            (
                "image_text",
                _chat_payload(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "How many cars are there in the image? "
                                "Answer briefly."
                            ),
                        }
                    ],
                    images=[str(IMAGE_PATH)],
                ),
            ),
            (
                "audio_image",
                _chat_payload(
                    messages=[{"role": "user", "content": ""}],
                    images=[str(IMAGE_PATH)],
                    audios=[str(AUDIO_PATH)],
                ),
            ),
            (
                "video_audio",
                _chat_payload(
                    messages=[{"role": "user", "content": ""}],
                    videos=[str(VIDEO_PATH)],
                    audios=[str(VIDEO_AUDIO_PATH)],
                ),
            ),
        ]

        selected_workers: list[str] = []
        for scenario, payload in scenarios:
            response = _router_chat(
                router_topology.router_port,
                request_id=f"router-ci-{scenario}",
                payload=payload,
            )
            selected_workers.append(_assert_chat_response(response, scenario=scenario))

        assert len(set(selected_workers)) == 2
        assert selected_workers[0] == selected_workers[2]
        assert selected_workers[1] == selected_workers[3]
        assert selected_workers[0] != selected_workers[1]

        _run_streaming_text_request(router_topology.router_port)
        final_workers = _router_get_json(router_topology.router_port, "/workers")
        _print_worker_snapshot("final /workers snapshot", final_workers)
        assert final_workers["routable_workers"] == 2
        assert all(
            worker["active_requests"] == 0 for worker in final_workers["workers"]
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
