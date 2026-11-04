# SPDX-License-Identifier: Apache-2.0
"""The speculative sweep reads server logs and compares a draft against no draft."""

from __future__ import annotations

from typing import Literal

import pytest

from benchmarks.eval.speculative_sweep import parse_args
from benchmarks.metrics.speculative import (
    ClientResult,
    Leg,
    ServerLogStats,
    compare,
    read_server_log,
    render,
)

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
    concurrency: int, throughput: float, latency_p95_s: float | None, skipped: int = 0
) -> ClientResult:
    return ClientResult.model_validate(
        {
            "concurrency": concurrency,
            "total": 100,
            "skipped": skipped,
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
    """At c16 the draft still gains throughput but raises p95; at c64 it loses throughput."""
    baseline_results = [
        client_result(1, 10.0, 0.5),
        client_result(16, 100.0, 1.5),
        client_result(64, 200.0, 3.0),
    ]
    draft_results = [
        client_result(1, 20.0, 0.3),
        client_result(16, 120.0, 2.5),
        client_result(64, 192.0, 3.0),
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
        client_result(16, 200.0, 0.5, skipped=50),
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
