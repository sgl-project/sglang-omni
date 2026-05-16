from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import sglang_omni_router.launcher.local as local_launcher
import sglang_omni_router.serve as serve_module
from sglang_omni_router.config import DEFAULT_CAPABILITIES, RouterConfig, WorkerConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.launcher import LocalLauncher, LocalLauncherConfig
from sglang_omni_router.launcher.config import load_launcher_config
from sglang_omni_router.launcher.utils import build_gpu_assignments
from sglang_omni_router.selector import NoEligibleWorkerError, WorkerSelector
from sglang_omni_router.serve import (
    build_config_from_args,
    build_parser,
    resolve_managed_worker_capabilities,
)
from sglang_omni_router.worker import build_workers


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:8101",
        "http://user:pass@127.0.0.1:8101",
        "http://127.0.0.1:8101/path",
        "http://127.0.0.1:8101?x=1",
        "http://169.254.169.254",
        "http://169.254.1.1",
    ],
)
def test_worker_url_validation_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        WorkerConfig(url=url)


def test_router_config_rejects_duplicate_urls_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate worker URLs"):
        RouterConfig(
            workers=[
                WorkerConfig(url="HTTP://LOCALHOST:8101/"),
                WorkerConfig(url="http://localhost:8101"),
            ]
        )


def test_worker_config_defaults_to_complete_omni_v1_replica_capabilities() -> None:
    worker = WorkerConfig(url="http://127.0.0.1:8101")

    assert worker.capabilities == DEFAULT_CAPABILITIES


@pytest.mark.parametrize("model", ["", "   "])
def test_worker_config_normalizes_blank_model_to_none(model: str) -> None:
    worker = WorkerConfig(url="http://127.0.0.1:8101", model=model)

    assert worker.model is None


def test_router_cli_worker_config_supports_heterogeneous_workers(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "url": "http://127.0.0.1:8101",
                        "model": "model-a",
                        "capabilities": ["chat", "image_input"],
                    },
                    {
                        "url": "http://127.0.0.1:8102",
                        "model": "model-b",
                        "capabilities": ["chat", "audio_input"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--worker-config",
            str(config_path),
            "--policy",
            "least_request",
        ]
    )

    config = build_config_from_args(args)

    assert config.policy == "least_request"
    assert [(worker.url, worker.model) for worker in config.workers] == [
        ("http://127.0.0.1:8101", "model-a"),
        ("http://127.0.0.1:8102", "model-b"),
    ]
    assert config.workers[0].capabilities == {"chat", "image_input"}
    assert config.workers[1].capabilities == {"chat", "audio_input"}


def test_router_worker_config_rejects_unknown_worker_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "url": "http://127.0.0.1:8101",
                        "capabilites": ["chat"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--worker-config", str(config_path)])

    with pytest.raises(ValueError, match="capabilites"):
        build_config_from_args(args)


def test_router_cli_rejects_global_model_with_worker_config(tmp_path: Path) -> None:
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "url": "http://127.0.0.1:8101",
                        "model": "model-a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--worker-config",
            str(config_path),
            "--model",
            "model-a",
        ]
    )

    with pytest.raises(ValueError, match="--model cannot be used"):
        build_config_from_args(args)


@pytest.mark.parametrize(
    "extra_args, error",
    [
        (
            ["--worker-urls", "http://127.0.0.1:8101"],
            "--launcher-config cannot be used with --worker-urls",
        ),
        (
            ["--worker-config", "workers.json"],
            "--launcher-config cannot be used with --worker-config",
        ),
        (
            ["--model", "qwen3-omni"],
            "--model cannot be used with --launcher-config",
        ),
    ],
)
def test_router_cli_rejects_launcher_config_with_other_worker_sources(
    extra_args: list[str],
    error: str,
) -> None:
    args = build_parser().parse_args(
        ["--launcher-config", "launcher.yaml", *extra_args]
    )

    with pytest.raises(ValueError, match=error):
        build_config_from_args(
            args,
            managed_worker_urls=["http://127.0.0.1:8101"],
            managed_model="qwen3-omni",
        )


def test_router_cli_builds_config_from_managed_worker_urls() -> None:
    args = build_parser().parse_args(
        [
            "--launcher-config",
            "launcher.yaml",
            "--policy",
            "least_request",
        ]
    )

    config = build_config_from_args(
        args,
        managed_worker_urls=["http://127.0.0.1:8101", "http://127.0.0.1:8102"],
        managed_model="qwen3-omni",
    )

    assert config.policy == "least_request"
    assert [(worker.url, worker.model) for worker in config.workers] == [
        ("http://127.0.0.1:8101", "qwen3-omni"),
        ("http://127.0.0.1:8102", "qwen3-omni"),
    ]
    assert [worker.capabilities for worker in config.workers] == [
        DEFAULT_CAPABILITIES,
        DEFAULT_CAPABILITIES,
    ]


def test_router_cli_uses_managed_worker_capabilities() -> None:
    args = build_parser().parse_args(["--launcher-config", "launcher.yaml"])

    config = build_config_from_args(
        args,
        managed_worker_urls=["http://127.0.0.1:8101"],
        managed_model="qwen3-omni",
        managed_worker_capabilities={"chat", "streaming", "image_input"},
    )

    assert config.workers[0].capabilities == {"chat", "streaming", "image_input"}


def test_router_cli_infers_text_only_managed_worker_capabilities() -> None:
    launcher_config = LocalLauncherConfig(
        model_path="model",
        worker_extra_args="--text-only --mem-fraction-static 0.7",
    )
    args = build_parser().parse_args(["--launcher-config", "launcher.yaml"])

    config = build_config_from_args(
        args,
        managed_worker_urls=["http://127.0.0.1:8101"],
        managed_model="qwen3-omni",
        managed_worker_capabilities=resolve_managed_worker_capabilities(
            launcher_config
        ),
    )

    assert config.workers[0].capabilities == {
        "chat",
        "streaming",
        "image_input",
        "audio_input",
        "video_input",
    }


def test_router_worker_config_requires_workers_object(tmp_path: Path) -> None:
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps([{"url": "http://127.0.0.1:8101"}]),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--worker-config", str(config_path)])

    with pytest.raises(ValueError, match="workers list"):
        build_config_from_args(args)


def test_launcher_config_passes_worker_extra_args_to_public_serve_command(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "launcher.yaml"
    config_path.write_text(
        """
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 2
  worker_host: 127.0.0.1
  worker_base_port: 8011
  worker_gpu_ids: ["0,1", "2,3"]
  wait_timeout: 600
  worker_extra_args: >-
    --mem-fraction-static 0.6 --thinker-tp-size 2
    --thinker-gpus '0,1'
""",
        encoding="utf-8",
    )

    config = load_launcher_config(config_path)
    command = LocalLauncher(config).build_worker_command(8011)

    assert command[:7] == [
        "sgl-omni",
        "serve",
        "--model-path",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
    ]
    assert command[7] == "8011"
    assert command[command.index("--model-name") + 1] == "qwen3-omni"
    assert command[command.index("--mem-fraction-static") + 1] == "0.6"
    assert command[command.index("--thinker-tp-size") + 1] == "2"
    assert command[command.index("--thinker-gpus") + 1] == "0,1"


def test_launcher_config_accepts_managed_worker_capabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "launcher.yaml"
    config_path.write_text(
        """
launcher:
  backend: local
  model_path: model
  worker_capabilities: ["chat", "streaming", "image_input"]
""",
        encoding="utf-8",
    )

    config = load_launcher_config(config_path)

    assert config.worker_capabilities == {"chat", "streaming", "image_input"}


@pytest.mark.parametrize(
    "yaml_text, error",
    [
        ("{}", "top-level launcher object"),
        ("launcher:\n  model_path: model\n  unknown: true\n", "unknown"),
        (
            "launcher:\n" "  model_path: model\n" "  mem_fraction_static: 0.6\n",
            "mem_fraction_static",
        ),
        ("launcher:\n  backend: local\n  num_workers: 1\n", "model_path"),
        (
            "launcher:\n"
            "  backend: local\n"
            "  model_path: model\n"
            "  num_workers: 2\n"
            "  worker_gpu_ids: ['0']\n",
            "worker_gpu_ids must contain exactly num_workers entries",
        ),
    ],
)
def test_launcher_config_rejects_invalid_yaml(
    tmp_path: Path,
    yaml_text: str,
    error: str,
) -> None:
    config_path = tmp_path / "launcher.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_launcher_config(config_path)


def test_launcher_gpu_assignment_uses_explicit_worker_gpu_ids() -> None:
    config = LocalLauncherConfig(
        model_path="model",
        num_workers=2,
        worker_gpu_ids=["0,1", "2,3"],
    )

    assert build_gpu_assignments(config) == ["0,1", "2,3"]


def test_launcher_gpu_assignment_groups_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    config = LocalLauncherConfig(
        model_path="model",
        num_workers=2,
        num_gpus_per_worker=2,
    )

    assert build_gpu_assignments(config) == ["4,5", "6,7"]


def test_launcher_gpu_assignment_allows_default_process_visibility(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "sglang_omni_router.launcher.utils.infer_available_cuda_devices",
        lambda: [],
    )
    config = LocalLauncherConfig(model_path="model", num_workers=1)

    assert build_gpu_assignments(config) is None


def test_launcher_cleans_up_managed_workers_on_health_timeout(monkeypatch) -> None:
    config = LocalLauncherConfig(
        model_path="model",
        num_workers=1,
        worker_gpu_ids=["7"],
        wait_timeout=1,
    )
    created_processes = []
    terminated_processes = []

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, env, start_new_session):
        process = FakeProcess()
        created_processes.append((process, command, env, start_new_session))
        return process

    def fail_health(**kwargs):
        raise TimeoutError("worker did not become healthy")

    monkeypatch.setattr(local_launcher.shutil, "which", lambda command: command)
    monkeypatch.setattr(local_launcher, "reserve_worker_ports", lambda config: [8011])
    monkeypatch.setattr(local_launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_launcher, "wait_for_worker_health", fail_health)
    monkeypatch.setattr(
        local_launcher,
        "terminate_processes",
        lambda processes: terminated_processes.extend(processes),
    )

    launcher = LocalLauncher(config)
    with pytest.raises(TimeoutError, match="worker did not become healthy"):
        launcher.launch_and_wait()

    process, command, env, start_new_session = created_processes[0]
    assert command[:2] == ["sgl-omni", "serve"]
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert start_new_session is True
    assert terminated_processes == [process]
    assert launcher.worker_urls == []


def test_launcher_cleans_up_managed_workers_on_startup_interrupt(monkeypatch) -> None:
    config = LocalLauncherConfig(
        model_path="model",
        num_workers=1,
        worker_gpu_ids=["7"],
        wait_timeout=10,
    )
    created_processes = []
    terminated_processes = []

    class StartupInterrupt(BaseException):
        pass

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, env, start_new_session):
        process = FakeProcess()
        created_processes.append((process, command, env, start_new_session))
        return process

    def wait_health(**kwargs):
        return None

    def interrupt_wait(*args, **kwargs):
        raise StartupInterrupt

    monkeypatch.setattr(local_launcher.shutil, "which", lambda command: command)
    monkeypatch.setattr(local_launcher, "reserve_worker_ports", lambda config: [8011])
    monkeypatch.setattr(local_launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_launcher, "wait_for_worker_health", wait_health)
    monkeypatch.setattr(local_launcher, "wait", interrupt_wait)
    monkeypatch.setattr(
        local_launcher,
        "terminate_processes",
        lambda processes: terminated_processes.extend(processes),
    )

    launcher = LocalLauncher(config)
    with pytest.raises(StartupInterrupt):
        launcher.launch_and_wait()

    assert terminated_processes == [created_processes[0][0]]
    assert launcher.worker_urls == []


def test_launcher_waits_for_managed_workers_in_parallel(monkeypatch) -> None:
    config = LocalLauncherConfig(
        model_path="model",
        num_workers=2,
        worker_gpu_ids=["0", "1"],
        wait_timeout=10,
    )
    created_processes = []
    terminated_processes = []
    waiting_workers = []
    waiting_lock = threading.Lock()
    both_waiting = threading.Event()

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return None

    def fake_popen(command, env, start_new_session):
        process = FakeProcess(12345 + len(created_processes))
        created_processes.append((process, command, env, start_new_session))
        return process

    def wait_health(**kwargs):
        worker_url = kwargs["worker_url"]
        with waiting_lock:
            waiting_workers.append(worker_url)
            if len(waiting_workers) == 2:
                both_waiting.set()

        assert both_waiting.wait(timeout=2)
        if worker_url.endswith(":8012"):
            raise RuntimeError("worker 2 failed")

    monkeypatch.setattr(local_launcher.shutil, "which", lambda command: command)
    monkeypatch.setattr(
        local_launcher, "reserve_worker_ports", lambda config: [8011, 8012]
    )
    monkeypatch.setattr(local_launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_launcher, "wait_for_worker_health", wait_health)
    monkeypatch.setattr(
        local_launcher,
        "terminate_processes",
        lambda processes: terminated_processes.extend(processes),
    )

    launcher = LocalLauncher(config)
    with pytest.raises(RuntimeError, match="worker 2 failed"):
        launcher.launch_and_wait()

    assert set(waiting_workers) == {
        "http://127.0.0.1:8011",
        "http://127.0.0.1:8012",
    }
    assert terminated_processes == [
        created_processes[0][0],
        created_processes[1][0],
    ]
    assert launcher.worker_urls == []


@pytest.mark.parametrize(
    "field",
    [
        "request_timeout_secs",
        "max_payload_size",
        "max_connections",
        "health_failure_threshold",
        "health_success_threshold",
        "health_check_timeout_secs",
        "health_check_interval_secs",
    ],
)
def test_router_config_rejects_non_positive_integer_knobs(field: str) -> None:
    with pytest.raises(ValidationError, match="value must be > 0"):
        RouterConfig(
            workers=[WorkerConfig(url="http://127.0.0.1:8101")],
            **{field: 0},
        )


def test_router_config_rejects_hyphenated_policy_aliases() -> None:
    with pytest.raises(ValidationError):
        RouterConfig(
            workers=[WorkerConfig(url="http://127.0.0.1:8101")],
            policy="round-robin",
        )


def test_router_console_script_entrypoint_resolves() -> None:
    script_target = None
    in_project_scripts = False
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_project_scripts = True
            continue
        if in_project_scripts and stripped.startswith("["):
            break
        if in_project_scripts and stripped.startswith("sgl-omni-router"):
            script_target = stripped.split("=", 1)[1].strip().strip('"')
            break

    assert script_target == "sglang_omni_router.serve:main"
    module_name, function_name = script_target.split(":")
    entrypoint = getattr(importlib.import_module(module_name), function_name)
    assert callable(entrypoint)


def test_router_main_shuts_down_managed_workers_on_startup_interrupt(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "launcher.yaml"
    config_path.write_text(
        """
launcher:
  backend: local
  model_path: model
  num_workers: 1
""",
        encoding="utf-8",
    )
    events: list[str] = []

    class FakeLauncher:
        def __init__(self, config) -> None:
            events.append(config.model_path)

        def launch_and_wait(self) -> list[str]:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(serve_module, "LocalLauncher", FakeLauncher)
    monkeypatch.setattr(serve_module.logging.config, "dictConfig", lambda config: None)

    with pytest.raises(SystemExit) as exc:
        serve_module.main(["--launcher-config", str(config_path)])

    assert exc.value.code == 130
    assert events == ["model", "shutdown"]


def test_selector_filters_by_health_and_capability() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101", capabilities={"speech"}),
            WorkerConfig(url="http://127.0.0.1:8102", capabilities={"chat"}),
            WorkerConfig(url="http://127.0.0.1:8103", capabilities={"speech"}),
        ]
    )
    workers[0].state = "unhealthy"
    workers[1].state = "healthy"
    workers[2].state = "healthy"

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8103"
    )


def test_selector_excludes_disabled_workers() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"
    workers[0].disabled = True

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(workers, required_capabilities={"chat"}).url
        == "http://127.0.0.1:8102"
    )


def test_selector_requires_all_capabilities() -> None:
    workers = build_workers(
        [
            WorkerConfig(
                url="http://127.0.0.1:8101",
                capabilities={"chat", "streaming"},
            ),
            WorkerConfig(
                url="http://127.0.0.1:8102",
                capabilities={"chat", "streaming", "video_input"},
            ),
        ]
    )
    for worker in workers:
        worker.state = "healthy"

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(
            workers,
            required_capabilities={"chat", "streaming", "video_input"},
        ).url
        == "http://127.0.0.1:8102"
    )


def test_selector_filters_by_requested_model_when_pool_declares_models() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101", model="model-a"),
            WorkerConfig(url="http://127.0.0.1:8102", model="model-b"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(
            workers,
            required_capabilities={"chat"},
            requested_model="model-a",
        ).url
        == "http://127.0.0.1:8101"
    )


def test_selector_preserves_unannotated_homogeneous_pool_behavior() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(
            workers,
            required_capabilities={"chat"},
            requested_model="qwen3-omni",
        ).url
        == "http://127.0.0.1:8101"
    )
    assert (
        selector.select(
            workers,
            required_capabilities={"chat"},
            requested_model="qwen3-omni",
        ).url
        == "http://127.0.0.1:8102"
    )


def test_round_robin_recomputes_candidates_after_health_change() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"

    selector = WorkerSelector("round_robin")

    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8101"
    )
    workers[0].state = "unhealthy"
    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8102"
    )
    workers[1].state = "unhealthy"
    with pytest.raises(NoEligibleWorkerError):
        selector.select(workers, required_capabilities={"speech"})


def test_least_request_selects_lowest_active_request_count() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"
    workers[0].active_requests = 2
    workers[1].active_requests = 2

    selector = WorkerSelector("least_request")

    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8101"
    )
    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8102"
    )

    workers[0].active_requests = 3
    assert (
        selector.select(workers, required_capabilities={"speech"}).url
        == "http://127.0.0.1:8102"
    )


def test_worker_request_guard_cleans_up_count() -> None:
    worker = build_workers([WorkerConfig(url="http://127.0.0.1:8101")])[0]

    with pytest.raises(RuntimeError):
        with worker.request_guard():
            assert worker.active_requests == 1
            raise RuntimeError("boom")

    assert worker.active_requests == 0


def test_worker_decrement_active_fails_on_unbalanced_cleanup() -> None:
    worker = build_workers([WorkerConfig(url="http://127.0.0.1:8101")])[0]

    with pytest.raises(AssertionError, match="active request count"):
        worker.decrement_active()


@pytest.mark.asyncio
async def test_health_checker_uses_failure_and_success_thresholds() -> None:
    statuses = iter([500, 500, 200, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), request=request)

    worker = build_workers([WorkerConfig(url="http://worker.local:8101")])[0]
    config = RouterConfig(
        workers=[WorkerConfig(url="http://worker.local:8101")],
        health_failure_threshold=2,
        health_success_threshold=2,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = HealthChecker(workers=[worker], config=config, client=client)

        await checker.check_all_workers_health()
        assert worker.state == "unknown"
        assert worker.last_error == "status=500"
        await checker.check_all_workers_health()
        assert worker.state == "unhealthy"
        await checker.check_all_workers_health()
        assert worker.state == "unhealthy"
        await checker.check_all_workers_health()
        assert worker.state == "healthy"


@pytest.mark.asyncio
async def test_health_checker_does_not_record_router_internal_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("router-side bug")

    worker = build_workers([WorkerConfig(url="http://worker.local:8101")])[0]
    config = RouterConfig(workers=[WorkerConfig(url="http://worker.local:8101")])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = HealthChecker(workers=[worker], config=config, client=client)

        with pytest.raises(RuntimeError, match="router-side bug"):
            await checker.check_worker_health(worker)

    assert worker.state == "unknown"
    assert worker.consecutive_failures == 0
    assert worker.last_error is None


@pytest.mark.asyncio
async def test_health_checker_isolates_unexpected_worker_check_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "worker-a":
            raise RuntimeError("router-side bug")
        return httpx.Response(200, request=request)

    workers = build_workers(
        [
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ]
    )
    config = RouterConfig(
        workers=[
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ],
        health_success_threshold=1,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = HealthChecker(workers=workers, config=config, client=client)

        await checker.check_all_workers_health()

    assert workers[0].state == "unknown"
    assert workers[0].consecutive_failures == 0
    assert workers[0].last_error is None
    assert workers[1].state == "healthy"
    assert workers[1].last_status_code == 200
