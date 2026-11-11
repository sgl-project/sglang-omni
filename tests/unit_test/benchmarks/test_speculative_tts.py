# SPDX-License-Identifier: Apache-2.0
"""TTS sweep lifecycle and result coverage across generation and scoring."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.eval import speculative_sweep, speculative_tts
from benchmarks.eval.speculative_sweep import parse_args
from benchmarks.eval.speculative_workloads import WORKLOADS, LoadSettings, TtsWorkload
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.speculative import Leg, compare, read_server_log, render
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer
from benchmarks.tasks.tts import save_speed_results, save_wer_results

Outcome = Literal[
    "ok",
    "generation",
    "all_failed",
    "transcription",
    "unscorable",
    "unscored",
    "ttfc",
    "legacy",
]


def write_point(
    directory: Path,
    *,
    concurrency: int,
    stream: bool,
    outcome: Outcome = "ok",
    language: Literal["en", "zh"] = "en",
) -> None:
    """Use the real result writers so changes to their schema break the adapter test."""
    directory.mkdir(parents=True, exist_ok=True)
    requests = [
        RequestResult(
            request_id=str(index),
            is_success=outcome != "all_failed"
            and not (outcome == "generation" and index == 1),
            latency_s=1.0 + index,
            audio_duration_s=4.0,
            audio_ttfp_s=(0.1 if stream and outcome != "ttfc" else None),
        )
        for index in range(2)
    ]
    reference, transcript = (
        ("red blue", "red") if language == "en" else ("你好世界", "你好世")
    )
    samples = []
    for request in requests:
        empty_reference = outcome == "unscored" or (
            outcome == "unscorable" and request.request_id == "1"
        )
        sample = SampleOutput(
            sample_id=request.request_id,
            target_text="" if empty_reference else reference,
        )
        if request.is_success and not (
            outcome == "transcription" and request.request_id == "1"
        ):
            apply_wer(sample, transcript, language)
        else:
            pass
        samples.append(sample)
    metrics = compute_speed_metrics(requests, wall_clock_s=4.0)
    if outcome != "legacy":
        metrics["wall_clock_s"] = 4.0
    else:
        pass
    save_speed_results(
        requests,
        metrics,
        {"stream": stream, "concurrency": concurrency},
        str(directory),
    )
    save_wer_results(
        samples, calculate_wer_metrics(samples, language), {}, str(directory)
    )


@pytest.mark.parametrize(
    ("workload_name", "language", "error_rate", "expected_rate", "stream", "repeats"),
    [
        ("tts-seedtts-en", "en", "WER", 0.5, False, 1),
        ("tts-seedtts-en", "en", "WER", 0.5, True, 3),
        ("tts-seedtts-zh", "zh", "CER", 0.25, False, 3),
        ("tts-seedtts-zh", "zh", "CER", 0.25, True, 1),
    ],
)
def test_tts_loads_read_every_repeat_and_keep_generation_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeats: int,
    stream: bool,
    workload_name: str,
    language: Literal["en", "zh"],
    error_rate: Literal["WER", "CER"],
    expected_rate: float,
) -> None:
    commands: list[list[str]] = []

    def transcribe(command: list[str], log_path: Path, *, check: bool) -> None:
        commands.append(command)
        assert command[0] == "benchmarks.eval.benchmark_tts_seedtts"
        assert "--transcribe-only" in command
        assert "--generate-only" not in command
        assert "--repeats" not in command
        assert command[command.index("--asr-model-path") + 1] == "test-asr"
        assert command[command.index("--lang") + 1] == language
        # note (swsqy): ASR latency must never replace the synthesis timings.
        (log_path.parent / "asr_speed_results.json").write_text(
            json.dumps({"throughput_qps": 999.0, "latency_p95_s": 999.0})
        )

    monkeypatch.setattr(speculative_tts, "run_client", transcribe)
    for concurrency in (1, 4):
        for repeat in range(1, repeats + 1):
            point_name = (
                f"c{concurrency}" if repeats == 1 else f"c{concurrency}_r{repeat}"
            )
            write_point(
                tmp_path / point_name,
                concurrency=concurrency,
                stream=stream,
                language=language,
            )
    workload = WORKLOADS[workload_name]
    assert isinstance(workload, TtsWorkload)
    loads = speculative_tts.measure_tts(
        workload,
        LoadSettings(
            concurrencies=[1, 4],
            repeats=repeats,
            warmup=True,
            max_samples=2,
            client_arguments=["--stream"] if stream else [],
        ),
        port=8700,
        model_path="test-tts",
        asr_model_path="test-asr",
        asr_concurrency=2,
        leg_dir=tmp_path,
    )
    assert len(commands) == 2 * repeats
    for measurement in loads:
        assert measurement.total_requests == 2 * repeats
        assert measurement.failed_requests == 0
        assert measurement.throughput_requests_per_s == 0.5
        assert measurement.latency_p95_s == 1.95
        assert measurement.first_output_latency_p95_s == (0.1 if stream else None)
        assert measurement.accuracy == {error_rate: expected_rate}
        assert measurement.incomplete_reasons == []


@pytest.mark.parametrize(
    ("outcome", "complete", "failed_requests", "reason"),
    [
        ("ok", True, 0, None),
        ("generation", False, 1, None),
        ("all_failed", False, 2, "no scored samples"),
        ("transcription", False, 0, "not transcribed"),
        ("unscorable", True, 0, None),
        ("unscored", False, 0, "no scored samples"),
        ("ttfc", False, 0, "first-output latency"),
        ("legacy", False, 0, "missing performance measurements"),
    ],
)
def test_incomplete_repeat_cannot_hide_behind_valid_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Outcome,
    complete: bool,
    failed_requests: int,
    reason: str | None,
) -> None:
    write_point(tmp_path / "c1_r1", concurrency=1, stream=True, outcome=outcome)
    write_point(tmp_path / "c1_r2", concurrency=1, stream=True)
    monkeypatch.setattr(
        speculative_tts, "run_client", lambda *arguments, **keywords: None
    )
    workload = WORKLOADS["tts-seedtts-en"]
    assert isinstance(workload, TtsWorkload)
    loads = speculative_tts.measure_tts(
        workload,
        LoadSettings(
            concurrencies=[1],
            repeats=2,
            warmup=False,
            max_samples=2,
            client_arguments=["--stream"],
        ),
        port=8700,
        model_path="test-tts",
        asr_model_path="test-asr",
        asr_concurrency=2,
        leg_dir=tmp_path,
    )
    assert loads[0].failed_requests == failed_requests
    legs = [
        Leg(
            name=side, side=side, startup_s=1.0, server=read_server_log(""), loads=loads
        )
        for side in ("baseline", "draft")
    ]
    comparison = compare(legs, k_gain=1.0)
    assert comparison.rows[0].is_complete is complete
    assert comparison.k == (1 if complete else None)
    if reason is not None:
        assert reason in render(comparison, legs, title="TTS")
    else:
        assert loads[0].incomplete_reasons == []


def test_mismatched_quality_samples_fail_instead_of_scoring_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_point(tmp_path / "c1", concurrency=1, stream=False)
    quality_path = tmp_path / "c1/wer_results.json"
    quality = json.loads(quality_path.read_text())
    quality["per_sample"][1]["id"] = "another-run"
    quality_path.write_text(json.dumps(quality))
    monkeypatch.setattr(
        speculative_tts, "run_client", lambda *arguments, **keywords: None
    )
    workload = WORKLOADS["tts-seedtts-en"]
    assert isinstance(workload, TtsWorkload)
    with pytest.raises(ValueError, match="inconsistent TTS result coverage"):
        speculative_tts.measure_tts(
            workload,
            LoadSettings(
                concurrencies=[1],
                repeats=1,
                warmup=False,
                max_samples=2,
                client_arguments=[],
            ),
            port=8700,
            model_path="test-tts",
            asr_model_path="test-asr",
            asr_concurrency=2,
            leg_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("order", "warmup", "failing_phase"),
    [
        ("AB", False, None),
        ("ABBA", True, None),
        ("ABBA", True, "generation"),
        ("ABBA", False, "transcription"),
    ],
)
def test_sweep_scores_all_legs_on_one_separate_asr_server_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    warmup: bool,
    failing_phase: Literal["generation", "transcription"] | None,
) -> None:
    """Scoring used to restart ASR between TTS legs and reuse a generation port."""
    events: list[str] = []
    active_models: list[str] = []
    server_ports: list[int] = []
    quality_results: dict[Path, str] = {}
    output_dir = tmp_path / "sweep"
    leg_names = [
        f"{index + 1}_{speculative_sweep.SIDE_BY_LETTER[letter]}"
        for index, letter in enumerate(order)
    ]

    @contextmanager
    def server(
        *,
        model_path: str,
        port: int,
        host: str,
        log_file: Path,
        server_config: str | None = None,
    ) -> Iterator[None]:
        assert active_models == []
        active_models.append(model_path)
        server_ports.append(port)
        events.append(f"start {model_path}")
        if model_path == "test-asr":
            assert log_file == output_dir / "asr_server.log"
        else:
            assert log_file.name == "server.log"
        log_file.write_text(
            f"KV Cache is allocated. #tokens: {100 if model_path == 'test-tts' else 999}"
        )
        try:
            yield
        finally:
            active_models.pop()
            events.append(f"stop {model_path}")

    def client(command: list[str], log_path: Path, *, check: bool) -> None:
        assert command[0] == "benchmarks.eval.benchmark_tts_seedtts"
        assert command[command.index("--lang") + 1] == "en"
        assert command[command.index("--model") + 1] == "test-tts"
        assert command[command.index("--port") + 1] == str(server_ports[-1])
        assert command[command.index("--output-dir") + 1] == str(log_path.parent)
        assert "--use-existing-server" in command
        assert "--stream" in command
        assert check
        if "--generate-only" in command:
            assert active_models == ["test-tts"]
            assert "--measure-gpu-memory" in command
            assert "--max-new-tokens" not in command
            assert ("--warmup" in command) is not warmup
            if not warmup:
                assert command[command.index("--warmup") + 1] == "0"
            else:
                pass
            events.append(f"generate {log_path.parent.name}")
            if failing_phase == "generation":
                raise RuntimeError("generation failed")
            else:
                for concurrency in (1, 2):
                    for repeat in (1, 2):
                        point_dir = log_path.parent / f"c{concurrency}_r{repeat}"
                        write_point(point_dir, concurrency=concurrency, stream=True)
                        quality_path = point_dir / "wer_results.json"
                        quality_results[quality_path] = quality_path.read_text()
                        quality_path.unlink()
        else:
            assert "--transcribe-only" in command
            assert "--measure-gpu-memory" not in command
            assert active_models == ["test-asr"]
            assert command[command.index("--asr-model-path") + 1] == "test-asr"
            assert command[command.index("--asr-concurrency") + 1] == "3"
            events.append(f"transcribe {log_path.parent.relative_to(output_dir)}")
            if failing_phase == "transcription":
                raise RuntimeError("transcription failed")
            else:
                quality_path = log_path.parent / "wer_results.json"
                quality_path.write_text(quality_results.pop(quality_path))

    monkeypatch.setattr(speculative_sweep, "managed_omni_server", server)
    monkeypatch.setattr(speculative_tts, "run_client", client)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "speculative_sweep",
            "--draft-config",
            "examples/configs/fun_cosyvoice3_dspark.yaml",
            "--model-path",
            "test-tts",
            "--workload",
            "tts-seedtts-en",
            "--output-dir",
            str(output_dir),
            "--order",
            order,
            "--port",
            "9100",
            "--concurrencies",
            "1,2",
            "--repeats",
            "2",
            "--asr-model-path",
            "test-asr",
            "--asr-concurrency",
            "3",
            *([] if warmup else ["--no-warmup"]),
            "--",
            "--stream",
        ],
    )
    if failing_phase is not None:
        with pytest.raises(RuntimeError, match=f"{failing_phase} failed"):
            speculative_sweep.main()
        assert not (output_dir / "summary.md").exists()
        assert not (output_dir / "sweep.json").exists()
    else:
        speculative_sweep.main()
        sweep = json.loads((output_dir / "sweep.json").read_text())
        assert [leg["name"] for leg in sweep["legs"]] == leg_names
        assert all(leg["kv_pool_tokens"] == 100 for leg in sweep["legs"])
        assert list(sweep["loads"]) == leg_names
        for loads in sweep["loads"].values():
            assert [load["concurrency"] for load in loads] == [1, 2]
            assert all(load["accuracy"] == {"WER": 0.5} for load in loads)
            assert all(load["throughput_requests_per_s"] == 0.5 for load in loads)
        assert quality_results == {}
        assert "WER" in (output_dir / "summary.md").read_text()
    if failing_phase == "generation":
        expected_events = ["start test-tts", "generate 1_baseline", "stop test-tts"]
        assert server_ports == [9100]
    else:
        expected_events = [
            event
            for name in leg_names
            for event in ("start test-tts", f"generate {name}", "stop test-tts")
        ]
        transcriptions = [
            f"transcribe {name}/c{concurrency}_r{repeat}"
            for name in leg_names
            for concurrency in (1, 2)
            for repeat in (1, 2)
        ]
        expected_events += (
            ["start test-asr"]
            + (
                transcriptions[:1]
                if failing_phase == "transcription"
                else transcriptions
            )
            + ["stop test-asr"]
        )
        assert server_ports == list(range(9100, 9100 + len(order) + 1))
    assert events == expected_events
    assert active_models == []


@pytest.mark.parametrize(
    "client_arguments",
    [
        ["--base-url", "http://another-server"],
        ["--base-u=http://another-server"],
        ["--model", "another-model"],
        ["--max-concurrency", "99"],
        ["--warmup", "0"],
        ["--transcribe-only"],
        ["--similarity-only"],
        ["--request-rate", "5"],
        ["--asr-model-path", "another-asr"],
        ["--testset", "another-dataset"],
    ],
)
def test_tts_passthrough_cannot_override_driver_owned_settings(
    monkeypatch: pytest.MonkeyPatch, client_arguments: list[str]
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--draft-config",
                "draft.yaml",
                "--workload",
                "tts-seedtts-en",
                "--output-dir",
                "results",
                "--",
                *client_arguments,
            ]
        )


@pytest.mark.parametrize(
    "client_arguments",
    [
        ["--quantization", "fp8"],
        ["--quantization=fp8"],
        ["--quant=fp8"],
        ["--max-running-requests", "8"],
        ["--max-queued-requests", "8"],
        ["--cuda-graph-max-bs", "8"],
        ["--server-config", "another.yaml"],
        ["--server-timeout", "30"],
        ["--skip-gpu-cleanup"],
    ],
)
def test_tts_ignored_server_options_are_rejected_with_yaml_guidance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    client_arguments: list[str],
) -> None:
    """Server options were recorded by the client without configuring the server."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit) as failure:
        parse_args(
            [
                "--draft-config",
                "draft.yaml",
                "--workload",
                "tts-seedtts-en",
                "--output-dir",
                "results",
                "--",
                *client_arguments,
            ]
        )
    assert failure.value.code == 2
    message = capsys.readouterr().err
    assert "--draft-config YAML" in message
    assert "stages.<stage>.engine" in message
    assert "timeout and GPU cleanup are controlled by the sweep" in message
