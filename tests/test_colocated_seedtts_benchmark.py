# SPDX-License-Identifier: Apache-2.0
"""Tests for colocated SeedTTS benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from benchmarks.eval.benchmark_omni_seedtts_colocated import (
    ColocatedSeedttsConfig,
    _build_aggregate_summary,
    _build_per_server_summaries,
    _config_from_args,
    _make_server_config,
    _validate_config,
    run_colocated_seedtts_benchmark,
)


def test_make_server_config_sets_unique_output_dir() -> None:
    config = ColocatedSeedttsConfig(
        base_urls=["http://localhost:8000", "http://localhost:8001"],
        model="qwen3-omni",
        meta="seedtts_testset/en/meta.lst",
        output_dir="results/out",
    )

    server_config = _make_server_config(
        config,
        base_url="http://localhost:8001",
        server_index=1,
        round_index=2,
    )

    assert server_config.base_url == "http://localhost:8001"
    assert server_config.output_dir.endswith("n2/round_3/server_2")


def test_build_per_server_summaries_groups_by_server() -> None:
    runs = [
        {
            "server_index": 1,
            "base_url": "http://localhost:8000",
            "summary": {
                "completed_requests": 10,
                "failed_requests": 0,
                "throughput_qps": 2.0,
                "latency_mean_s": 5.0,
                "latency_p95_s": 8.0,
                "tok_per_s_agg": 1.5,
                "rtf_mean": 1.2,
            },
        },
        {
            "server_index": 1,
            "base_url": "http://localhost:8000",
            "summary": {
                "completed_requests": 8,
                "failed_requests": 1,
                "throughput_qps": 4.0,
                "latency_mean_s": 7.0,
                "latency_p95_s": 10.0,
                "tok_per_s_agg": 2.5,
                "rtf_mean": 1.4,
            },
        },
    ]

    per_server = _build_per_server_summaries(runs)

    assert len(per_server) == 1
    assert per_server[0]["server_index"] == 1
    assert per_server[0]["rounds"] == 2
    assert per_server[0]["completed_requests_total"] == 18
    assert per_server[0]["failed_requests_total"] == 1
    assert per_server[0]["throughput_qps_mean"] == 3.0
    assert per_server[0]["latency_mean_s_mean"] == 6.0


def test_build_aggregate_summary_averages_metrics() -> None:
    runs = [
        {
            "server_index": 1,
            "base_url": "http://localhost:8000",
            "summary": {
                "completed_requests": 10,
                "failed_requests": 0,
                "throughput_qps": 2.0,
                "latency_mean_s": 5.0,
                "latency_p95_s": 8.0,
                "tok_per_s_agg": 1.5,
                "rtf_mean": 1.2,
            },
        },
        {
            "server_index": 2,
            "base_url": "http://localhost:8001",
            "summary": {
                "completed_requests": 10,
                "failed_requests": 1,
                "throughput_qps": 4.0,
                "latency_mean_s": 7.0,
                "latency_p95_s": 10.0,
                "tok_per_s_agg": 2.5,
                "rtf_mean": 1.4,
            },
        },
    ]
    per_server = _build_per_server_summaries(runs)
    aggregate = _build_aggregate_summary(runs, per_server=per_server)

    assert aggregate["num_servers"] == 2
    assert aggregate["total_runs"] == 2
    assert aggregate["completed_requests_total"] == 20
    assert aggregate["failed_requests_total"] == 1
    assert aggregate["throughput_qps_mean"] == 3.0
    assert aggregate["latency_mean_s_mean"] == 6.0
    assert aggregate["latency_p95_s_mean"] == 9.0
    assert aggregate["tok_per_s_agg_mean"] == 2.0
    assert aggregate["rtf_mean_mean"] == 1.3


def test_config_from_args_parses_repeated_base_urls() -> None:
    args = argparse.Namespace(
        base_url=["http://localhost:8000", "http://localhost:8001"],
        model="qwen3-omni",
        meta="seedtts_testset/en/meta.lst",
        output_dir="results/out",
        lang="en",
        speaker="Ethan",
        voice_clone=True,
        no_ref_audio=False,
        max_samples=50,
        max_new_tokens=256,
        temperature=0.7,
        warmup=1,
        max_concurrency=16,
        request_rate=float("inf"),
        disable_tqdm=False,
        rounds=2,
        server_timeout=1200,
    )

    config = _config_from_args(args)

    assert config.base_urls == ["http://localhost:8000", "http://localhost:8001"]
    assert config.voice_clone is True
    assert config.rounds == 2


def test_validate_config_rejects_duplicate_base_urls(tmp_path: Path) -> None:
    meta = tmp_path / "meta.lst"
    meta.write_text("sample|ref text|ref.wav|target text\n")

    config = ColocatedSeedttsConfig(
        base_urls=["http://localhost:8000", "http://localhost:8000"],
        model="qwen3-omni",
        meta=str(meta),
        output_dir=str(tmp_path / "results"),
    )

    with pytest.raises(ValueError, match="Duplicate --base-url"):
        _validate_config(config)


def test_validate_config_rejects_rounds_below_one(tmp_path: Path) -> None:
    meta = tmp_path / "meta.lst"
    meta.write_text("sample|ref text|ref.wav|target text\n")

    config = ColocatedSeedttsConfig(
        base_urls=["http://localhost:8000"],
        model="qwen3-omni",
        meta=str(meta),
        output_dir=str(tmp_path / "results"),
        rounds=0,
    )

    with pytest.raises(ValueError, match="--rounds must be >= 1"):
        _validate_config(config)


def test_run_colocated_seedtts_benchmark_runs_all_servers_and_rounds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    meta = tmp_path / "meta.lst"
    meta.write_text("sample|ref text|ref.wav|target text\n")

    waited = []
    invoked = []

    def _fake_wait_for_service(base_url: str, timeout: int = 1200, **kwargs) -> None:
        del kwargs
        waited.append((base_url, timeout))

    async def _fake_run_omni_seedtts_benchmark(config):
        invoked.append((config.base_url, config.output_dir))
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 0,
                "throughput_qps": 2.0,
                "latency_mean_s": 5.0,
                "latency_p95_s": 8.0,
                "tok_per_s_agg": 1.5,
                "rtf_mean": 1.2,
            },
            "config": {"base_url": config.base_url},
        }

    monkeypatch.setattr(
        "benchmarks.eval.benchmark_omni_seedtts_colocated.wait_for_service",
        _fake_wait_for_service,
    )
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_omni_seedtts_colocated.run_omni_seedtts_benchmark",
        _fake_run_omni_seedtts_benchmark,
    )

    config = ColocatedSeedttsConfig(
        base_urls=["http://localhost:8000", "http://localhost:8001"],
        model="qwen3-omni",
        meta=str(meta),
        output_dir=str(tmp_path / "results"),
        rounds=2,
    )

    results = asyncio.run(run_colocated_seedtts_benchmark(config))

    assert waited == [
        ("http://localhost:8000", 1200),
        ("http://localhost:8001", 1200),
    ]
    assert len(invoked) == 4
    assert results["aggregate"]["num_servers"] == 2
    assert results["aggregate"]["total_runs"] == 4
    assert len(results["per_server"]) == 2
    assert len(results["runs"]) == 4
