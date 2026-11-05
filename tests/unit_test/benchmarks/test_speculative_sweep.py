# SPDX-License-Identifier: Apache-2.0
"""The speculative sweep reads server logs and compares a draft against no draft."""

from __future__ import annotations

import json
import wave
from argparse import Namespace
from pathlib import Path
from typing import Literal

import pytest
from aiohttp import web

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_asr_seedtts
from benchmarks.eval.speculative_sweep import parse_args
from benchmarks.metrics.speculative import (
    ClientResult,
    Leg,
    ServerLogStats,
    compare,
    read_server_log,
    render,
)
from benchmarks.tasks.asr import build_asr_eval_results

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


def client_result(
    concurrency: int,
    throughput: float,
    latency_p95_s: float | None,
    skipped: int = 0,
    failed: int = 0,
) -> ClientResult:
    return ClientResult.model_validate(
        {
            "concurrency": concurrency,
            "total": 100,
            "skipped": skipped,
            "failed": failed,
            "throughput_samples_per_s": {"mean": throughput},
            "latency_p95_s": {"mean": latency_p95_s},
            "corpus_wer": {"mean": 0.013},
            "resources": {"gpu_memory_used_peak_mib": None},
        }
    )


def leg(
    name: str, accept_length_mean: float | None, results: list[ClientResult]
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
        results=results,
    )


def test_comparison_reads_gain_k_and_every_load_the_draft_makes_worse() -> None:
    """Scoring skips must not hide the qualifying c16 gain or the c16/c64 regressions."""
    baseline_results = [
        client_result(1, 10.0, 0.5, skipped=1),
        client_result(16, 100.0, 1.5, skipped=1),
        client_result(64, 200.0, 3.0, skipped=1),
    ]
    draft_results = [
        client_result(1, 20.0, 0.3, skipped=1),
        client_result(16, 120.0, 2.5, skipped=1),
        client_result(64, 192.0, 3.0, skipped=1),
    ]
    legs = [
        leg("1_baseline", None, baseline_results),
        leg("2_draft", 5.0, draft_results),
        leg("3_draft", 5.5, [client_result(1, 22.0, 0.3), *draft_results[1:]]),
        leg("4_baseline", None, [client_result(1, 12.0, 0.5), *baseline_results[1:]]),
    ]

    comparison = compare(legs, k_gain=1.2)

    assert [row.gain for row in comparison.rows] == pytest.approx(
        [21.0 / 11.0, 1.2, 0.96]
    )
    assert comparison.k == 16
    assert comparison.worse_concurrencies == [16, 64]
    assert comparison.incomplete_concurrencies == []
    assert comparison.accept_length_mean == pytest.approx(5.25)


def test_a_load_with_failed_requests_keeps_its_numbers_but_qualifies_nothing() -> None:
    """Throughput over the requests that finished can show 2x while half of them failed."""
    baseline_results = [client_result(1, 10.0, 0.5), client_result(16, 100.0, 1.0)]
    draft_results = [
        client_result(1, 15.0, 0.3),
        client_result(16, 200.0, 0.5, skipped=50, failed=50),
    ]
    legs = [
        leg("1_baseline", None, baseline_results),
        leg("2_draft", 5.0, draft_results),
        leg("3_draft", 5.0, draft_results),
        leg("4_baseline", None, baseline_results),
    ]

    comparison = compare(legs, k_gain=1.2)

    assert comparison.rows[1].gain == pytest.approx(2.0)
    assert comparison.rows[1].draft.failed_requests == 100
    assert comparison.incomplete_concurrencies == [16]
    assert comparison.k == 1
    assert comparison.worse_concurrencies == []
    roadmap_row = render(comparison, legs, title="draft").splitlines()[-1]
    assert "| 1.50x | – | – | invalid | – | – |" in roadmap_row


@pytest.mark.parametrize("failed_side", ["baseline", "draft", "both"])
def test_all_failed_legs_still_render_an_incomplete_load(
    failed_side: Literal["baseline", "draft", "both"],
) -> None:
    """Two zero-throughput legs previously prevented the entire sweep summary from rendering."""
    legs = []
    for index, side in enumerate(("baseline", "draft", "draft", "baseline"), 1):
        failed = failed_side in (side, "both")
        results = [
            client_result(1, 15.0 if side == "draft" else 10.0, 0.5),
            client_result(
                16,
                0.0 if failed else 100.0,
                None if failed else 1.0,
                skipped=100 if failed else 0,
                failed=100 if failed else 0,
            ),
        ]
        legs.append(leg(f"{index}_{side}", 5.0 if side == "draft" else None, results))

    comparison = compare(legs, k_gain=1.2)
    summary = render(comparison, legs, title="failed load")

    assert comparison.k == 1
    assert comparison.incomplete_concurrencies == [16]
    assert comparison.worse_concurrencies == []
    row = next(line for line in summary.splitlines() if line.startswith("| 16 "))
    cells = [cell.strip() for cell in row.split("|")[1:-1]]
    assert cells[0] == "16 (incomplete)"
    for side, gap_column, failed_column in (("baseline", 4, 11), ("draft", 5, 12)):
        failed = failed_side in (side, "both")
        assert cells[gap_column] == ("–" if failed else "0.0%")
        assert cells[failed_column] == ("200" if failed else "0")
    assert "| 1.50x | – | – | invalid | – | – |" in summary.splitlines()[-1]


REQUIRED = ["--model-path", "m", "--draft-config", "d.yaml", "--output-dir", "out"]


def test_arguments_after_the_separator_go_to_the_client(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    args = parse_args([*REQUIRED, "--", "--concurrencies", "1,16", "--repeats", "1"])

    assert args.client_args == ["--concurrencies", "1,16", "--repeats", "1"]


@pytest.mark.parametrize(
    ("visible_gpus", "argv"),
    [
        pytest.param("0,1", REQUIRED, id="two-gpus"),
        pytest.param("", REQUIRED, id="no-gpu-named"),
        pytest.param("0", [*REQUIRED, "--order", "AA"], id="no-draft-leg"),
        pytest.param("0", [*REQUIRED, "--order", "ABC"], id="unknown-leg"),
        pytest.param("0", [*REQUIRED, "--", "--port=9000"], id="client-port"),
    ],
)
def test_the_sweep_refuses_to_start_on_a_bad_setup(
    monkeypatch, visible_gpus: str, argv: list[str]
) -> None:
    """Between legs the managed server ends every process on every visible GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_gpus)

    with pytest.raises(SystemExit):
        parse_args(argv)


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
    result = ClientResult.model_validate_json(json.dumps(aggregate))
    comparison = compare(
        [
            leg("A1", None, [result]),
            leg("B1", 5, [result]),
            leg("B2", 5, [result]),
            leg("A2", None, [result]),
        ],
        k_gain=1.0,
    )
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
