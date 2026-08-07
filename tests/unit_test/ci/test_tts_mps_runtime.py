# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SCRIPT = REPO_ROOT / "tests/utils/tts_mps_runtime.py"
OVERLAP_SCRIPT = REPO_ROOT / "tests/utils/tts_mps_overlap.py"


def _load(path: Path, name: str):
    assert path.is_file(), f"{path.name} is missing"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_contract_is_one_gpu_dp2_without_weight_sharing(tmp_path: Path) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime")
    config = tmp_path / "config.yaml"
    config.write_text("config_cls: HiggsTtsPipelineConfig\n", encoding="utf-8")
    launcher = tmp_path / "examples/mps_dp/launch.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    spec = runtime.MpsLaunchSpec(
        repository_root=tmp_path,
        output_dir=tmp_path / "output",
        state_root=tmp_path / "state",
        run_id="run-unit",
        config_path=config,
        gpu_id=0,
        base_port=8801,
        core_blocks=("0-3", "4-7"),
        python_bin="python",
    )
    assert spec.environment["N"] == "2"
    assert spec.environment["GPU_ID"] == "0"
    assert spec.environment["WEIGHT_SHARE"] == "0"
    assert "SGLANG_OMNI_WEIGHT_SHARE" not in spec.environment
    assert spec.worker_urls == (
        "http://127.0.0.1:8801",
        "http://127.0.0.1:8802",
    )


def test_launch_spec_rejects_a_control_socket_over_the_sun_path_limit(
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_sun_path")
    config = tmp_path / "config.yaml"
    config.write_text("config_cls: HiggsTtsPipelineConfig\n", encoding="utf-8")
    launcher = tmp_path / "examples/mps_dp/launch.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    deep_state_root = tmp_path.joinpath(*[f"segment{index:02d}" for index in range(12)])

    with pytest.raises(ValueError, match="sun_path"):
        runtime.MpsLaunchSpec(
            repository_root=tmp_path,
            output_dir=tmp_path / "output",
            state_root=deep_state_root,
            run_id="run-unit",
            config_path=config,
            gpu_id=0,
            base_port=8801,
            core_blocks=("0-3", "4-7"),
            python_bin="python",
        )


def test_core_blocks_preserve_the_pci_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_pci_domain")
    pci_devices = tmp_path / "pci"
    numa_nodes = tmp_path / "nodes"
    for domain, numa_node in (("0000", 0), ("0001", 1)):
        device = pci_devices / f"{domain}:08:00.0"
        device.mkdir(parents=True)
        (device / "numa_node").write_text(f"{numa_node}\n", encoding="utf-8")
        node = numa_nodes / f"node{numa_node}"
        node.mkdir(parents=True)
        (node / "cpulist").write_text(
            "0-7\n" if numa_node == 0 else "8-15\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="00000000:08:00.0\n"),
    )
    monkeypatch.setattr(
        runtime.os,
        "sched_getaffinity",
        lambda _pid: set(range(16)),
        raising=False,
    )

    assert runtime.derive_core_blocks(
        0,
        pci_devices_root=pci_devices,
        numa_nodes_root=numa_nodes,
    ) == ("0-2", "3-5")


@pytest.mark.parametrize("core_count", [4, 5])
def test_core_blocks_keep_the_hard_minimum_when_the_node_is_small(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    core_count: int,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, f"tts_mps_runtime_small_node_{core_count}")
    pci_devices = tmp_path / "pci"
    numa_nodes = tmp_path / "nodes"
    device = pci_devices / "0000:08:00.0"
    device.mkdir(parents=True)
    (device / "numa_node").write_text("0\n", encoding="utf-8")
    node = numa_nodes / "node0"
    node.mkdir(parents=True)
    (node / "cpulist").write_text(f"0-{core_count - 1}\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="00000000:08:00.0\n"),
    )
    monkeypatch.setattr(
        runtime.os,
        "sched_getaffinity",
        lambda _pid: set(range(core_count)),
        raising=False,
    )

    # The host reserve must not truncate below the minimum and report a real
    # host as insufficient.
    assert runtime.derive_core_blocks(
        0,
        pci_devices_root=pci_devices,
        numa_nodes_root=numa_nodes,
    ) == ("0-1", "2-3")


def test_stale_launcher_state_is_archived_and_reconciled_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_stale_reconcile")
    config = tmp_path / "config.yaml"
    config.write_text("config_cls: HiggsTtsPipelineConfig\n", encoding="utf-8")
    launcher = tmp_path / "examples/mps_dp/launch.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    spec = runtime.MpsLaunchSpec(
        repository_root=tmp_path,
        output_dir=tmp_path / "output",
        state_root=tmp_path / "state",
        run_id="run-current",
        config_path=config,
        gpu_id=0,
        base_port=8801,
        core_blocks=("0-3", "4-7"),
        python_bin="python",
    )
    stale = spec.state_root / "gpu-0" / "run-stale"
    stale.mkdir(parents=True)
    (stale / "mps_ctl.err").write_text("retained\n", encoding="utf-8")
    mps_logs = stale / "mps" / "log"
    mps_logs.mkdir(parents=True)
    (mps_logs / "control.log").write_text("daemon root cause\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        shutil.rmtree(stale)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    recovered = runtime.reconcile_stale_launcher_states(spec)

    assert recovered is None
    assert commands == [
        (
            "bash",
            str(launcher),
            "down",
            "run-stale",
        )
    ]
    assert not stale.exists()
    assert (
        spec.output_dir
        / "recovered-stale-launcher-state"
        / "run-stale"
        / "raw"
        / "mps_ctl.err"
    ).read_text(encoding="utf-8") == "retained\n"
    assert (
        spec.output_dir
        / "recovered-stale-launcher-state"
        / "run-stale"
        / "raw"
        / "mps"
        / "log"
        / "control.log"
    ).read_text(encoding="utf-8") == "daemon root cause\n"


def test_launcher_state_rejects_shared_weights_and_consumes_verified_kv(
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_state")
    state = tmp_path / "run-unit"
    logs = state / "logs"
    logs.mkdir(parents=True)
    manifest = {
        "run_id": "run-unit",
        "gpu_id": "0",
        "gpu_uuid": "GPU-test",
        "numa_node": "0",
        "config": "/tmp/config.yaml",
        "n": "2",
        "base_port": "8801",
        "core_blocks": "0-3 4-7",
        "max_total_tokens": "30000",
        "weight_share": "1",
    }
    (state / "manifest").write_text(
        "".join(f"{key}={value}\n" for key, value in manifest.items()),
        encoding="utf-8",
    )
    (state / "replicas.tsv").write_text(
        "0\t11\t11\t8801\tlogs/replica_0.log\t1\n"
        "1\t22\t22\t8802\tlogs/replica_1.log\t2\n",
        encoding="utf-8",
    )
    (logs / "replica_0.log").write_text("KV Pool is allocated. #tokens: 30000\n")
    (logs / "replica_1.log").write_text("KV Pool is allocated. #tokens: 20000\n")
    (state / "mps_attach.txt").write_text(
        "replica 0 (port 8801): attached clients: 11\n"
        "replica 1 (port 8802): attached clients: 22\n"
        "RESULT: PASS\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="independent-weight DP2"):
        runtime.read_launcher_state(state)

    text = (
        (state / "manifest")
        .read_text(encoding="utf-8")
        .replace("weight_share=1", "weight_share=0")
    )
    (state / "manifest").write_text(text, encoding="utf-8")
    snapshot = runtime.read_launcher_state(state)
    assert [replica.kv_tokens for replica in snapshot.replicas] == [30000, 30000]


def test_overlap_requires_repeated_server_monotonic_intervals() -> None:
    overlap = _load(OVERLAP_SCRIPT, "tts_mps_overlap")
    events = []
    for replica, intervals in {
        0: [(100, 500), (600, 1000)],
        1: [(200, 550), (700, 1100)],
    }.items():
        for index, (start, end) in enumerate(intervals):
            request_id = f"r{replica}-{index}"
            events.extend(
                [
                    {
                        "run_id": "run-unit",
                        "replica_id": replica,
                        "request_id": request_id,
                        "event": "model_path_start",
                        "clock": "CLOCK_MONOTONIC",
                        "monotonic_ns": start,
                        "host_boot_id": "boot",
                    },
                    {
                        "run_id": "run-unit",
                        "replica_id": replica,
                        "request_id": request_id,
                        "event": "model_path_end",
                        "clock": "CLOCK_MONOTONIC",
                        "monotonic_ns": end,
                        "host_boot_id": "boot",
                        "status": "success",
                    },
                ]
            )
    verdict = overlap.build_overlap_verdict(
        events,
        expected_run_id="run-unit",
        min_successes_per_replica=2,
        measurement_uncertainty_ns=10,
    )
    assert verdict["matched_overlap_count"] == 2
    assert verdict["aggregate_overlap_ns"] == 600

    with pytest.raises(ValueError, match="insufficient successful requests"):
        overlap.build_overlap_verdict(
            events[:2],
            expected_run_id="run-unit",
            min_successes_per_replica=2,
            measurement_uncertainty_ns=10,
        )
    with pytest.raises(ValueError, match="expected run"):
        overlap.build_overlap_verdict(
            events,
            expected_run_id="run-other",
            min_successes_per_replica=2,
            measurement_uncertainty_ns=10,
        )


def test_atomic_summary_validation_rejects_dirty_or_partial_evidence(
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_summary")
    summary = runtime.new_summary(
        exact_sha="a" * 40,
        run_id="run-unit",
        run_attempt="1",
        selected_model="higgs",
    )
    path = tmp_path / "tts_mps_summary.json"
    runtime.atomic_write_json(path, summary)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="incomplete"):
        runtime.validate_final_summary(loaded)

    loaded["runtime"] = {"status": "pass"}
    loaded["correctness"] = {"status": "pass"}
    loaded["cleanup"] = {"status": "dirty"}
    with pytest.raises(ValueError, match="cleanup"):
        runtime.validate_final_summary(loaded)


def test_scheduler_model_path_activity_is_monotonic_and_default_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sglang_omni.profiler import event_recorder

    recorder = event_recorder.get_recorder()
    recorder.stop()
    event_recorder.emit_model_path_start("request-off")
    assert recorder.is_active() is False

    path = recorder.start("run-unit", str(tmp_path), "tts_engine")
    monkeypatch.setattr(event_recorder.time, "monotonic_ns", lambda: 123456)
    monkeypatch.setattr(event_recorder, "_read_host_boot_id", lambda: "boot-test")
    event_recorder.emit_model_path_start("request-on")
    event_recorder.emit_model_path_end("request-on", status="success")
    recorder.stop()
    emitted = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    assert [item["event_name"] for item in emitted] == [
        "model_path_start",
        "model_path_end",
    ]
    assert all(item["metadata"]["monotonic_ns"] == 123456 for item in emitted)
    assert all(item["metadata"]["clock"] == "CLOCK_MONOTONIC" for item in emitted)
    assert all(item["metadata"]["host_boot_id"] == "boot-test" for item in emitted)
    assert emitted[-1]["metadata"]["status"] == "success"


def test_model_path_activity_preserves_the_profiler_wall_clock_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sglang_omni.profiler import event_recorder
    from sglang_omni.profiler.views import reconstruct_timelines

    recorder = event_recorder.get_recorder()
    recorder.stop()
    path = recorder.start("run-clock-domain", str(tmp_path), "tts_engine")
    monkeypatch.setattr(event_recorder.time, "time_ns", lambda: 2_000_000_000)
    monkeypatch.setattr(event_recorder.time, "monotonic_ns", lambda: 123456)
    event_recorder.emit(
        request_id="request-clock-domain",
        stage="tts_engine",
        event_name="request_started",
    )
    event_recorder.emit_model_path_start("request-clock-domain")
    event_recorder.emit_model_path_end(
        "request-clock-domain",
        status="success",
    )
    event_recorder.emit(
        request_id="request-clock-domain",
        stage="tts_engine",
        event_name="request_finished",
    )
    recorder.stop()

    emitted = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    model_path_events = [
        item for item in emitted if item["event_name"].startswith("model_path_")
    ]
    assert all(item["timestamp_ns"] == 2_000_000_000 for item in emitted)
    assert all(item["metadata"]["monotonic_ns"] == 123456 for item in model_path_events)
    assert reconstruct_timelines(path)["request-clock-domain"].total_ms == 0.0


def test_profile_start_waits_for_each_engine_process_recorder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_profile_start")
    snapshot = SimpleNamespace(
        state_dir=tmp_path,
        replicas=(SimpleNamespace(index=0), SimpleNamespace(index=1)),
    )
    first = tmp_path / "activity-0"
    first.mkdir()
    (first / "events_coordinator_9.jsonl").touch()
    (first / "events_preprocessing_10.jsonl").touch()

    def create_second(_seconds: float) -> None:
        second = tmp_path / "activity-1"
        second.mkdir(exist_ok=True)
        (second / "events_coordinator_19.jsonl").touch()
        (second / "events_preprocessing_20.jsonl").touch()

    monkeypatch.setattr(runtime.time, "sleep", create_second)
    runtime.wait_for_profile_activity_files(
        snapshot,
        timeout_s=1.0,
        poll_interval_s=0.0,
    )


def test_activity_reader_waits_for_all_terminal_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_profile_stop")
    snapshot = SimpleNamespace(
        state_dir=tmp_path,
        replicas=(SimpleNamespace(index=0), SimpleNamespace(index=1)),
    )
    for index in (0, 1):
        directory = tmp_path / f"activity-{index}"
        directory.mkdir()
        (directory / f"events_tts_engine_{index}.jsonl").touch()

    def flush_events(_seconds: float) -> None:
        for index in (0, 1):
            event = {
                "run_id": "run-unit",
                "request_id": f"request-{index}",
                "event_name": "model_path_end",
                "metadata": {
                    "clock": "CLOCK_MONOTONIC",
                    "monotonic_ns": 100 + index,
                    "host_boot_id": "boot-unit",
                    "status": "success",
                },
            }
            path = tmp_path / f"activity-{index}" / f"events_tts_engine_{index}.jsonl"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    monkeypatch.setattr(runtime.time, "sleep", flush_events)
    events = runtime.read_model_path_activity(
        snapshot,
        min_terminal_events=2,
        timeout_s=1.0,
        poll_interval_s=0.0,
    )

    assert len(events) == 2
    assert all(item["event"] == "model_path_end" for item in events)


def test_activity_reader_does_not_count_failed_terminal_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_failed_terminal")
    snapshot = SimpleNamespace(state_dir=tmp_path, replicas=())
    events = [
        {
            "event": "model_path_end",
            "status": "success" if index < 4 else "error",
        }
        for index in range(8)
    ]
    monkeypatch.setattr(
        runtime,
        "_read_model_path_activity_once",
        lambda _snapshot: events,
    )

    with pytest.raises(RuntimeError, match="successful terminal events"):
        runtime.read_model_path_activity(
            snapshot,
            min_terminal_events=8,
            timeout_s=0.0,
        )


def test_canary_request_counts_require_every_request_to_succeed() -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_request_counts")
    assert runtime.require_exact_request_counts(
        {
            "total_requests": 8,
            "completed_requests": 8,
            "failed_requests": 0,
        },
        expected_requests=8,
    ) == {
        "total_requests": 8,
        "completed_requests": 8,
        "failed_requests": 0,
    }
    with pytest.raises(RuntimeError, match="request counts"):
        runtime.require_exact_request_counts(
            {
                "total_requests": 8,
                "completed_requests": 4,
                "failed_requests": 4,
            },
            expected_requests=8,
        )


def test_managed_router_lifecycle_accepts_two_external_mps_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.test_model import omni_router_utils

    command: list[str] = []
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(
        omni_router_utils,
        "start_server_from_cmd",
        lambda cmd, *_args, **_kwargs: command.extend(cmd) or process,
    )
    monkeypatch.setattr(
        omni_router_utils.os,
        "getpgid",
        lambda _pid: 123,
        raising=False,
    )
    for name in (
        "_record_process_group",
        "wait_for_all_router_workers",
        "stop_server",
        "cleanup_process_groups_from_manifest",
    ):
        monkeypatch.setattr(omni_router_utils, name, lambda *_args, **_kwargs: None)

    factory = SimpleNamespace(mktemp=lambda _name: tmp_path)
    worker_urls = ["http://127.0.0.1:8801", "http://127.0.0.1:8802"]
    with omni_router_utils.launch_managed_router(
        tmp_path_factory=factory,
        model_path="unused-for-external-workers",
        model_name="model-under-test",
        worker_extra_args="",
        external_worker_urls=worker_urls,
        wait_timeout=30,
    ) as handle:
        assert handle.worker_ports == [8801, 8802]

    worker_index = command.index("--worker-urls")
    assert command[worker_index + 1 : worker_index + 3] == worker_urls
    assert command[command.index("--model") + 1] == "model-under-test"
    assert "--launcher-config" not in command


def _healthy_mps_summary() -> dict:
    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.metrics.performance import compute_speed_metrics

    return compute_speed_metrics(
        [
            RequestResult(
                is_success=True,
                latency_s=1.5,
                audio_duration_s=3.75,
                rtf=0.4,
                completion_tokens=100,
                engine_time_s=1.0,
            )
            for _ in range(12)
        ],
        wall_clock_s=1.0,
    )


def test_mps_performance_fails_closed_on_an_uncalibrated_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_ci import tts_mps_ci_config

    monkeypatch.setattr(
        tts_mps_ci_config,
        "MPS_PERFORMANCE_REFERENCES",
        {"higgs": {"throughput_qps": (None, "minimum", "req/s")}},
        raising=True,
    )
    verdict = tts_mps_ci_config.check_mps_performance(
        model="higgs",
        concurrency=tts_mps_ci_config.MPS_CONCURRENCY,
        summary=_healthy_mps_summary(),
    )

    assert verdict["status"] == "fail"
    assert verdict["uncalibrated"] == ["throughput_qps"]
    assert any("tune-ci-thresholds" in check for check in verdict["failed_checks"])


def test_shipped_mps_references_are_all_calibrated() -> None:
    from tests.test_ci import tts_mps_ci_config

    uncalibrated = [
        f"{model}.{metric}"
        for model, metrics in tts_mps_ci_config.MPS_PERFORMANCE_REFERENCES.items()
        for metric, (reference, _direction, _unit) in metrics.items()
        if reference is None
    ]
    assert uncalibrated == []


def test_mps_performance_applies_slack_to_calibrated_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_ci import tts_mps_ci_config

    healthy = _healthy_mps_summary()
    calibrated = {
        "higgs": {
            "throughput_qps": (healthy["throughput_qps"], "minimum", "req/s"),
            "latency_mean_s": (healthy["latency_mean_s"], "maximum", "s"),
        }
    }
    monkeypatch.setattr(
        tts_mps_ci_config, "MPS_PERFORMANCE_REFERENCES", calibrated, raising=True
    )

    verdict = tts_mps_ci_config.check_mps_performance(
        model="higgs",
        concurrency=tts_mps_ci_config.MPS_CONCURRENCY,
        summary=healthy,
    )
    assert verdict["status"] == "pass"
    assert verdict["checks"]["throughput_qps"]["threshold"] == pytest.approx(
        healthy["throughput_qps"] * tts_mps_ci_config.MPS_SLACK_HIGHER
    )

    # Below the slackened floor, so a real regression is still caught.
    regressed = {**healthy, "throughput_qps": healthy["throughput_qps"] * 0.5}
    verdict = tts_mps_ci_config.check_mps_performance(
        model="higgs",
        concurrency=tts_mps_ci_config.MPS_CONCURRENCY,
        summary=regressed,
    )
    assert verdict["status"] == "fail"
    assert verdict["checks"]["throughput_qps"]["pass"] is False
    assert any("throughput_qps" in check for check in verdict["failed_checks"])


def test_gpu_client_delta_flags_only_clients_created_by_the_mps_stage() -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_gpu")
    baseline = [
        {"gpu_uuid": "GPU-0", "pid": 10, "process_name": "foreign"},
    ]
    post = [
        {"gpu_uuid": "GPU-0", "pid": 10, "process_name": "foreign"},
        {"gpu_uuid": "GPU-0", "pid": 20, "process_name": "replica"},
        {"gpu_uuid": "GPU-1", "pid": 30, "process_name": "other-gpu"},
    ]
    assert runtime.new_gpu_clients(
        baseline,
        post,
        gpu_uuid="GPU-0",
    ) == [{"gpu_uuid": "GPU-0", "pid": 20, "process_name": "replica"}]
    with pytest.raises(RuntimeError, match="visibility"):
        runtime.verify_active_gpu_visibility(
            baseline,
            baseline,
            gpu_uuid="GPU-0",
        )
    assert runtime.verify_active_gpu_visibility(
        baseline,
        post,
        gpu_uuid="GPU-0",
    ) == [{"gpu_uuid": "GPU-0", "pid": 20, "process_name": "replica"}]


def test_dirty_cleanup_verdict_is_not_treated_as_success() -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_cleanup")
    dirty = {
        "status": "dirty",
        "checks": {"tracked_processes_exited": False},
    }
    with pytest.raises(RuntimeError, match="dirty"):
        runtime.require_clean_cleanup(dirty)
    runtime.require_clean_cleanup({"status": "pass", "checks": {}})
    merged = runtime.record_cleanup_recovery(
        dirty,
        {"status": "pass", "checks": {"tracked_processes_exited": True}},
    )
    assert merged["status"] == "dirty"
    assert merged["recovery_attempt"]["status"] == "pass"


def test_gpu_client_cleanup_waits_for_nvml_to_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load(RUNTIME_SCRIPT, "tts_mps_runtime_settle")
    baseline = [{"gpu_uuid": "GPU-0", "pid": 10, "process_name": "foreign"}]
    lingering = [
        *baseline,
        {"gpu_uuid": "GPU-0", "pid": 20, "process_name": "mps-server"},
    ]
    snapshots = iter([lingering, baseline])
    monkeypatch.setattr(runtime, "capture_gpu_clients", lambda: next(snapshots))
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)

    assert (
        runtime.wait_for_gpu_clients_to_settle(
            baseline,
            gpu_uuid="GPU-0",
            timeout_s=1.0,
            poll_interval_s=0.0,
        )
        == baseline
    )
