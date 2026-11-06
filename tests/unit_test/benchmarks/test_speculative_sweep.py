# SPDX-License-Identifier: Apache-2.0
"""The speculative sweep reads server logs and compares a draft against no draft."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from pydantic import JsonValue

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_asr_seedtts, speculative_workloads
from benchmarks.eval.benchmark_asr_transcribe_diarize import (
    RESULTS_FILE,
    SPEED_RESULTS_FILE,
)
from benchmarks.eval.speculative_sweep import parse_args, write_side_configs
from benchmarks.eval.speculative_workloads import (
    WORKLOADS,
    LoadSettings,
    TranscribeDiarizeWorkload,
    measure_asr,
    measure_transcribe_diarize,
)
from benchmarks.metrics.speculative import (
    AccuracyMetric,
    Leg,
    LoadMeasurement,
    ServerLogStats,
    compare,
    read_server_log,
    render,
)
from benchmarks.tasks.asr import build_asr_eval_results
from sglang_omni.config.manager import ConfigManager

# Lines from SGLang 0.5.22 servers on H100: MOSS-TD with a DFlash draft, and a Qwen3-ASR server without one that meets the same prompt twice.
DRAFT_SERVER_LOG = """\
2026-11-02 04:22:13,418 [INFO] sglang.srt.mem_cache.memory_pool: KV Cache is allocated. dtype: torch.bfloat16, #tokens: 481423, K size: 25.71 GB, V size: 25.71 GB
2026-11-02 04:22:13,422 [INFO] sglang.srt.mem_cache.memory_pool: KV Cache is allocated. dtype: torch.bfloat16, #tokens: 481423, K size: 4.59 GB, V size: 4.59 GB
2026-11-02 04:23:19,097 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Prefill batch, #new-seq: 1, #new-token: 99, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 300.04
2026-11-02 04:23:31,527 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Decode batch, #running-req: 14, #token: 425966, token usage: 0.88, accept len: 5.26, accept rate: 0.61, cuda graph: True, gen throughput (token/s): 220.06, #queue-req: 2
2026-11-02 04:23:33,259 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Decode batch, #running-req: 14, #token: 431312, token usage: 0.90, accept len: 4.83, accept rate: 0.55, cuda graph: True, gen throughput (token/s): 3111.62, #queue-req: 2
2026-11-02 04:23:49,015 [WARNING] sglang.srt.managers.scheduler: KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 3597, #new_token_ratio: 0.0980 -> 0.0645
"""
BASELINE_SERVER_LOG = """\
2026-11-02 18:58:01,875 [INFO] sglang.srt.mem_cache.memory_pool: KV Cache is allocated. dtype: torch.bfloat16, #tokens: 604785, K size: 32.30 GB, V size: 32.30 GB
2026-11-02 18:58:31,258 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Prefill batch, #new-seq: 1, #new-token: 69, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 10.11
2026-11-02 18:58:31,310 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 68, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 19.28
"""


def test_server_log_gives_the_pool_usage_retractions_acceptance_and_cached_share() -> (
    None
):
    assert read_server_log(DRAFT_SERVER_LOG) == ServerLogStats(
        kv_pool_tokens=481423,
        token_usage_peak=0.90,
        retracted_requests=1,
        accept_length_mean=pytest.approx(5.045),
        cached_prompt_token_share=0.0,
    )
    assert read_server_log(BASELINE_SERVER_LOG) == ServerLogStats(
        kv_pool_tokens=604785,
        token_usage_peak=0.0,
        retracted_requests=0,
        accept_length_mean=None,
        cached_prompt_token_share=pytest.approx(68 / 138),
    )


def load(
    concurrency: int,
    throughput: float,
    latency_p95_s: float,
    *,
    failed_requests: int = 0,
    first_output_latency_p95_s: float | None = None,
    accuracy: dict[AccuracyMetric, float] | None = None,
) -> LoadMeasurement:
    return LoadMeasurement(
        concurrency=concurrency,
        total_requests=100,
        failed_requests=failed_requests,
        throughput_requests_per_s=throughput,
        latency_p95_s=latency_p95_s,
        first_output_latency_p95_s=first_output_latency_p95_s,
        accuracy={"WER": 0.013} if accuracy is None else accuracy,
        gpu_memory_peak_mib=None,
    )


def leg(
    name: str, accept_length_mean: float | None, loads: list[LoadMeasurement]
) -> Leg:
    return Leg(
        name=name,
        side="draft" if accept_length_mean is not None else "baseline",
        startup_s=60.0,
        server=ServerLogStats(
            kv_pool_tokens=None,
            token_usage_peak=None,
            retracted_requests=0,
            accept_length_mean=accept_length_mean,
            cached_prompt_token_share=None,
        ),
        loads=loads,
    )


def abba(
    baseline_loads: list[LoadMeasurement], draft_loads: list[LoadMeasurement]
) -> list[Leg]:
    return [
        leg("1_baseline", None, baseline_loads),
        leg("2_draft", 5.0, draft_loads),
        leg("3_draft", 5.0, draft_loads),
        leg("4_baseline", None, baseline_loads),
    ]


def test_comparison_reads_gain_k_and_every_load_the_draft_makes_worse() -> None:
    """At c16 the draft still gains throughput but raises p95; at c64 it loses throughput."""
    baseline_loads = [load(1, 10.0, 0.5), load(16, 100.0, 1.5), load(64, 200.0, 3.0)]
    draft_loads = [load(1, 20.0, 0.3), load(16, 120.0, 2.5), load(64, 192.0, 3.0)]
    legs = [
        leg("1_baseline", None, baseline_loads),
        leg("2_draft", 5.0, draft_loads),
        leg("3_draft", 5.5, [load(1, 22.0, 0.3), *draft_loads[1:]]),
        leg("4_baseline", None, [load(1, 12.0, 0.5), *baseline_loads[1:]]),
    ]

    comparison = compare(legs, k_gain=1.2)

    assert [row.gain for row in comparison.rows] == pytest.approx(
        [21.0 / 11.0, 1.2, 0.96]
    )
    assert comparison.k == 16
    assert comparison.worse_concurrencies == [16, 64]
    assert comparison.incomplete_concurrencies == []
    assert comparison.accept_length_mean == pytest.approx(5.25)
    assert "First output" not in render(comparison, legs, title="draft")


def test_a_load_with_failed_requests_keeps_its_numbers_but_qualifies_nothing() -> None:
    """Throughput over the requests that finished can show 2x while half of them failed."""
    legs = abba(
        [load(1, 10.0, 0.5), load(16, 100.0, 1.0)],
        [load(1, 15.0, 0.3), load(16, 200.0, 0.5, failed_requests=50)],
    )

    comparison = compare(legs, k_gain=1.2)

    assert comparison.rows[1].gain == pytest.approx(2.0)
    assert comparison.rows[1].draft.failed_requests == 100
    assert comparison.incomplete_concurrencies == [16]
    assert comparison.k == 1
    assert comparison.worse_concurrencies == []
    roadmap_row = render(comparison, legs, title="draft").splitlines()[-1]
    assert "| 1.50x | – | – | invalid | – | – |" in roadmap_row


@pytest.mark.parametrize("has_complete_load", [True, False])
def test_partial_transcription_is_only_reported_in_the_per_load_detail(
    has_complete_load: bool,
) -> None:
    baseline = [load(4, 10, 1, accuracy={"CER": 0.08})]
    draft = [load(4, 12, 1, accuracy={"CER": 0.00})]
    draft[0].incomplete_reasons.append("99 generated samples were not transcribed")
    if has_complete_load:
        baseline.insert(0, load(1, 10, 1, accuracy={"CER": 0.20}))
        draft.insert(0, load(1, 12, 1, accuracy={"CER": 0.18}))
    else:
        pass
    legs = abba(baseline, draft)
    report = render(compare(legs, k_gain=1.2), legs, title="TTS")
    detail = next(
        line for line in report.splitlines() if line.startswith("| 4 (incomplete)")
    )
    assert "8.00%, 8.00% | 0.00%, 0.00%" in detail
    assert "99 generated samples were not transcribed" in report
    expected_baseline, expected_draft = (
        ("20.00%", "18.00%") if has_complete_load else ("–", "–")
    )
    assert (
        f"no draft {expected_baseline} to {expected_baseline}, draft {expected_draft} to {expected_draft}"
        in report
    )
    assert report.splitlines()[-1].endswith(
        f"| CER {expected_baseline} → {expected_draft} | |"
    )


def test_a_slower_first_output_makes_a_load_worse_even_when_throughput_gains() -> None:
    legs = abba(
        [load(16, 100.0, 1.0, first_output_latency_p95_s=0.20)],
        [load(16, 150.0, 1.0, first_output_latency_p95_s=0.22)],
    )

    comparison = compare(legs, k_gain=1.2)
    summary = render(comparison, legs, title="draft")

    assert comparison.k == 16
    assert comparison.worse_concurrencies == [16]
    row = next(line for line in summary.splitlines() if line.startswith("| 16 "))
    assert "| 0.200 | 0.220 | 10.0% |" in row


def test_the_report_names_each_accuracy_metric_the_workload_measures() -> None:
    """MOSS-TD reports CER and DER, which the table and roadmap row label as such rather than as WER."""
    legs = abba(
        [load(1, 4.0, 0.4, accuracy={"CER": 0.0587, "DER": 0.2111})],
        [load(1, 8.0, 0.2, accuracy={"CER": 0.0582, "DER": 0.2101})],
    )

    summary = render(compare(legs, k_gain=1.2), legs, title="draft")

    header = next(line for line in summary.splitlines() if line.startswith("| c |"))
    assert "| CER, no draft | CER, draft | DER, no draft | DER, draft |" in header
    assert "WER" not in summary
    assert "- DER across legs and concurrencies: no draft 21.11% to 21.11%" in summary
    assert summary.splitlines()[-1].endswith(
        "| CER 5.87% → 5.82%, DER 21.11% → 21.01% | |"
    )


QWEN3_ASR_DFLASH_SETTINGS: dict[str, str | int] = {
    "speculative_algorithm": "DFLASH",
    "speculative_draft_model_path": "wilsonzheng0327/Qwen3-ASR-1.7B-dflash",
    "speculative_draft_model_revision": "d62bd4051e68282e02f93de774036a35f22f1373",
    "speculative_num_draft_tokens": 8,
}


@pytest.mark.parametrize("keep_prefix_cache", [False, True])
@pytest.mark.parametrize(
    "draft_settings",
    [
        pytest.param(
            {
                "stages": {
                    "asr": {
                        "engine": {
                            **QWEN3_ASR_DFLASH_SETTINGS,
                            "mem_fraction_static": 0.7,
                        }
                    }
                }
            },
            id="stage-engine",
        ),
        pytest.param(
            {
                "shared": [
                    {
                        "select": {"stages": ["asr"]},
                        "engine": QWEN3_ASR_DFLASH_SETTINGS,
                    }
                ],
                "stages": {
                    "asr": {
                        "engine": {
                            "speculative_algorithm": "DFLASH",
                            "mem_fraction_static": 0.7,
                        }
                    }
                },
            },
            id="shared-selector",
        ),
        pytest.param(
            {
                "shared": [
                    {
                        "select": {"stages": ["asr"]},
                        "engine": {
                            **QWEN3_ASR_DFLASH_SETTINGS,
                            "mem_fraction_static": 0.7,
                        },
                    }
                ]
            },
            id="shared-selector-without-stages",
        ),
    ],
)
def test_only_the_draft_side_serves_the_draft(
    tmp_path: Path,
    draft_settings: dict[str, JsonValue],
    keep_prefix_cache: bool,
) -> None:
    """A draft set through a shared selector stayed on in the no-draft config, so the sweep compared the draft against itself."""
    draft_config_path = tmp_path / "qwen3_asr_dflash.yaml"
    draft_config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3ASRPipelineConfig",
                "model_path": "Qwen/Qwen3-ASR-1.7B",
                **draft_settings,
            }
        )
    )
    output_dir = tmp_path / "sweep"
    output_dir.mkdir()

    config_paths = write_side_configs(
        draft_config_path, output_dir, keep_prefix_cache=keep_prefix_cache
    )

    served = {
        side: ConfigManager.from_file(str(path))
        .config.stage_named("asr")
        .engine.overrides()
        for side, path in config_paths.items()
    }
    assert served["draft"]["speculative_algorithm"] == "DFLASH"
    assert served["draft"]["speculative_num_draft_tokens"] == 8
    assert served["baseline"] == {
        setting: value
        for setting, value in served["draft"].items()
        if not setting.startswith("speculative_")
    }
    assert served["baseline"]["mem_fraction_static"] == 0.7
    assert ("disable_radix_cache" in served["draft"]) is not keep_prefix_cache


def test_a_config_without_a_draft_is_refused(tmp_path: Path) -> None:
    draft_config_path = tmp_path / "qwen3_asr.yaml"
    draft_config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3ASRPipelineConfig",
                "model_path": "Qwen/Qwen3-ASR-1.7B",
                "stages": {"asr": {"engine": {"mem_fraction_static": 0.7}}},
            }
        )
    )

    with pytest.raises(ValueError, match="runs no stage with a draft"):
        write_side_configs(draft_config_path, tmp_path, keep_prefix_cache=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("payload", "status", "reference", "failed_per_repeat"),
    [
        ({"text": "hello"}, 200, "hello", 0),
        ({"text": ""}, 200, "hello", 0),
        ({"text": "Mhm."}, 200, "Mhm.", 0),
        ({}, 200, "hello", 2),
        ({"text": None}, 200, "hello", 2),
        ({"error": "unavailable"}, 503, "hello", 2),
    ],
)
async def test_scoring_skips_do_not_fail_a_successful_http_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, str | None],
    status: int,
    reference: str,
    failed_per_repeat: int,
    stream: bool,
) -> None:
    """An unscorable reference must not invalidate successful loads or remove their timings."""
    audio_path = tmp_path / "clip.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(320))
    samples = [
        SampleInput(str(index), text, str(audio_path), text)
        for index, text in enumerate(["Mhm.", reference])
    ]

    async def respond(request: web.Request) -> web.Response:
        await request.read()
        if stream and status == 200:
            event = {"type": "transcript.text.done", **payload}
            return web.Response(
                text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )
        else:
            return web.json_response(payload, status=status)

    application = web.Application()
    application.router.add_post("/v1/audio/transcriptions", respond)
    server = web.AppRunner(application)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", 0).start()
    monkeypatch.setattr(
        benchmark_asr_seedtts, "_fetch_worker_snapshot", lambda host, port: {}
    )
    arguments = Namespace(
        host="127.0.0.1",
        port=server.addresses[0][1],
        model_path="test-asr",
        lang="en",
        stream=stream,
        disable_resource_monitor=True,
        sample_util=False,
        save_raw_dir=str(tmp_path / "raw"),
    )
    try:
        repeats = [
            await benchmark_asr_seedtts._run_repeat(
                arguments, samples, 1, repeat
            )  # noqa: leading-underscore  # production entry point
            for repeat in range(3)
        ]
    finally:
        await server.cleanup()
    for repeat in repeats:
        assert repeat["throughput_samples_per_s"] * repeat[
            "wall_clock_s"
        ] == pytest.approx(2 - failed_per_repeat)
        assert repeat["failed"] == failed_per_repeat
        if failed_per_repeat:
            assert repeat["latency_p95_s"] is None
        else:
            assert repeat["latency_p95_s"] > 0
    aggregate = benchmark_asr_seedtts._aggregate(
        repeats
    )  # noqa: leading-underscore  # production entry point
    assert aggregate["failed"] == 3 * failed_per_repeat
    assert aggregate["skipped"] == (
        6 if failed_per_repeat or reference == "Mhm." else 3
    )
    if reference == "Mhm.":
        assert aggregate["corpus_wer"]["mean"] is None
    elif payload.get("text") == "":
        assert aggregate["corpus_wer"]["mean"] == 1.0
    else:
        pass

    def write_client_payload(
        command: list[str], log_path: Path, *, check: bool
    ) -> None:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"results": [aggregate]}))

    monkeypatch.setattr(speculative_workloads, "run_client", write_client_payload)
    [measurement] = measure_asr(
        WORKLOADS["seedtts-en"],
        LoadSettings(
            concurrencies=[1],
            repeats=3,
            warmup=False,
            max_samples=None,
            client_arguments=[],
        ),
        port=8800,
        model_path="test-asr",
        leg_dir=tmp_path,
    )
    comparison = compare(abba([measurement], [measurement]), k_gain=1.0)
    assert comparison.rows[0].baseline.failed_requests == 6 * failed_per_repeat
    assert comparison.rows[0].is_complete is (failed_per_repeat == 0)
    assert comparison.k == (None if failed_per_repeat else 1)
    raw_records = [
        json.loads(line)
        for line in (tmp_path / "raw/conc1_rep0.jsonl").read_text().splitlines()
    ]
    assert raw_records[0]["is_transcribed"] is (failed_per_repeat == 0)
    assert raw_records[0]["is_success"] is False


def test_missing_results_and_request_errors_are_not_scoring_skips() -> None:
    """Missing responses and request exceptions must remain failures beside an unscorable success."""
    samples = [SampleInput(str(index), "Mhm.", "", "") for index in range(3)]
    report = build_asr_eval_results(
        samples,
        [
            RequestResult(request_id="0", text="Mhm.", is_success=True, latency_s=0.2),
            RequestResult(request_id="1", error="Request timed out"),
        ],
        wall_clock_s=1.0,
        lang="en",
    )
    assert report["summary"]["skipped"] == 3
    assert report["speed"]["throughput_samples_per_s"] == 1.0
    assert report["speed"]["latency_p95_s"] == 0.2
    assert report["speed"]["failed_requests"] == 2


def test_meeting_throughput_comes_from_the_wall_clock_not_the_rounded_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The client rounds throughput_qps to 3 decimals, so two aishell4_long runs 0.4% apart both read 0.028 requests/s."""
    wall_clocks_s = iter([716.0, 719.0])

    def write_client_files(command: list[str], log_path: Path, *, check: bool) -> None:
        run_dir = Path(command[command.index("--output-dir") + 1])
        run_dir.mkdir()
        (run_dir / SPEED_RESULTS_FILE).write_text(
            json.dumps(
                {
                    "config": {"wall_clock_s": next(wall_clocks_s)},
                    "speed": {
                        "total_requests": 20,
                        "completed_requests": 20,
                        "failed_requests": 0,
                        "throughput_qps": 0.028,
                        "latency_p95_s": 43.5,
                    },
                }
            )
        )
        (run_dir / RESULTS_FILE).write_text(
            json.dumps(
                {
                    "diarization_metrics": {
                        "cer": 0.1395,
                        "speaker_timestamp_der": 0.0999,
                    }
                }
            )
        )

    monkeypatch.setattr(speculative_workloads, "run_client", write_client_files)

    [measurement] = measure_transcribe_diarize(
        TranscribeDiarizeWorkload(dataset="aishell4_long"),
        LoadSettings(
            concurrencies=[1],
            repeats=2,
            warmup=False,
            max_samples=None,
            client_arguments=[],
        ),
        port=8800,
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        leg_dir=tmp_path,
    )

    assert measurement.throughput_requests_per_s == pytest.approx(
        (20 / 716.0 + 20 / 719.0) / 2
    )
    assert measurement.total_requests == 40
    assert measurement.accuracy == {"CER": 0.1395, "DER": 0.0999}


REQUIRED = [
    "--draft-config",
    "d.yaml",
    "--workload",
    "seedtts-en",
    "--output-dir",
    "out",
]


def test_sweep_help_does_not_import_task_or_dataset_executors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.modules['benchmarks.tasks.asr'] = None\n"
            "sys.modules['benchmarks.dataset.prepare'] = None\n"
            "sys.argv = ['speculative_sweep', '--help']\n"
            "runpy.run_module('benchmarks.eval.speculative_sweep', run_name='__main__')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "tts-seedtts-zh" in result.stdout


def test_arguments_after_the_separator_go_to_the_client(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    args = parse_args([*REQUIRED, "--concurrencies", "1,16", "--", "--stream"])

    assert args.concurrencies == [1, 16]
    assert args.client_args == ["--stream"]
    assert args.asr_model_path is None and args.asr_concurrency is None


@pytest.mark.parametrize("workload", ["seedtts-en", "movies800times"])
@pytest.mark.parametrize(
    "option", [["--asr-model-path", "Qwen/Qwen3-ASR-1.7B"], ["--asr-concurrency=32"]]
)
def test_only_tts_workloads_accept_evaluator_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workload: str,
    option: list[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        parse_args([*REQUIRED, "--workload", workload, *option])
    assert "apply only to TTS workloads" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("visible_gpus", "argv"),
    [
        pytest.param("0,1", REQUIRED, id="two-gpus"),
        pytest.param("", REQUIRED, id="no-gpu-named"),
        pytest.param("0", [*REQUIRED, "--order", "AA"], id="no-draft-leg"),
        pytest.param("0", [*REQUIRED, "--order", "ABC"], id="unknown-leg"),
        pytest.param("0", [*REQUIRED, "--", "--port=9000"], id="client-port"),
        pytest.param("0", [*REQUIRED, "--", "--repeats", "1"], id="client-repeats"),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--max-concurrency", "16"],
            id="client-concurrency-alias",
        ),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--max-concurrency=16"],
            id="client-concurrency-alias-equals",
        ),
        pytest.param(
            "0", [*REQUIRED, "--", "--max-conc", "16"], id="client-flag-abbreviation"
        ),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--base-url", "http://127.0.0.1:8000"],
            id="client-base-url",
        ),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--base-url=http://127.0.0.1:8000"],
            id="client-base-url-equals",
        ),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--reuse-asr-results", "/tmp/old-asr.json"],
            id="client-reused-results",
        ),
        pytest.param(
            "0",
            [*REQUIRED, "--", "--reuse", "/tmp/old-asr.json"],
            id="client-reused-results-abbreviation",
        ),
        pytest.param("0", [*REQUIRED, "--concurrencies", "0"], id="zero-concurrency"),
        pytest.param("0", [*REQUIRED, "--concurrencies", "1,1"], id="duplicate-load"),
        pytest.param("0", [*REQUIRED, "--repeats", "0"], id="no-measured-pass"),
        pytest.param("0", [*REQUIRED, "--max-samples", "0"], id="no-samples"),
        pytest.param(
            "0",
            [*REQUIRED, "--workload", "tts-seedtts-en", "--asr-concurrency", "0"],
            id="no-asr-slots",
        ),
    ],
)
def test_the_sweep_refuses_to_start_on_a_bad_setup(
    monkeypatch, visible_gpus: str, argv: list[str]
) -> None:
    """Between legs the managed server ends every process on every visible GPU, and the sweep owns the load settings and where each leg's requests go."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_gpus)

    with pytest.raises(SystemExit):
        parse_args(argv)
