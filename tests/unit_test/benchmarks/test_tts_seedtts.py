# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark entry-point: model profiles, server lifecycle, WER filter."""

import sys
from contextlib import contextmanager

import pytest

from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics


@pytest.mark.parametrize(
    "model, is_auk",
    [
        ("tencent/AuK", True),
        ("tencent/AuK-Flash", True),
        ("tencent/AuK@revision", True),
        ("/ckpt/auk-flash", True),
        ("fishaudio/s2-pro", False),
    ],
)
def test_cli_defaults_follow_checkpoint_name(monkeypatch, model, is_auk):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", model])
    args, profile = tts._parse_args(tts._build_arg_parser())
    config = tts._config_from_args(args)
    assert profile.forward_sglang_engine is not is_auk
    if is_auk:
        assert config.concurrency == config.warmup == 1
        assert config.seed == 1234
        assert config.output_dir == "results/auk_seedtts"
    else:
        assert profile.argument_defaults == {}


@pytest.mark.parametrize("model", ["tencent/AuK", "fishaudio/s2-pro"])
def test_evaluation_releases_tts_server_before_starting_asr(monkeypatch, model):
    events = []
    servers = []

    @contextmanager
    def server(**kwargs):
        servers.append(kwargs)
        events.append("start")
        yield
        events.append("stop")

    async def generate(config):
        assert config.port == 18280
        assert config.max_samples == 2
        assert config.model == model
        events.append("generate")

    def transcribe(config, **kwargs):
        assert kwargs["asr_router_port"] == 18280
        events.append("transcribe")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            model,
            "--port",
            "18280",
            "--max-samples",
            "2",
        ],
    )
    monkeypatch.setattr(tts, "managed_omni_server", server)
    monkeypatch.setattr(tts, "benchmark", generate)
    monkeypatch.setattr(tts, "run_tts_seedtts_transcribe", transcribe)
    tts.main()

    assert events == ["start", "generate", "stop", "start", "transcribe", "stop"]
    assert servers[0]["model_path"] == model
    assert servers[1]["model_path"] != model
    if model == "tencent/AuK":
        assert "max_running_requests" not in servers[0]
        assert "cuda_graph_max_bs" not in servers[0]
        assert servers[0]["server_config"] is None
    else:
        assert servers[0]["max_running_requests"] == 64
        assert servers[0]["cuda_graph_max_bs"] == 64


def test_filtered_wer_mean_keeps_exactly_50_percent_and_excludes_failures():
    metrics = calculate_wer_metrics(
        [
            SampleOutput(is_success=True, wer=0, hits=10),
            SampleOutput(is_success=True, wer=0.5, hits=1, deletions=1),
            SampleOutput(is_success=True, wer=0.75, hits=1, deletions=3),
            SampleOutput(is_success=False),
        ],
        "en",
    )
    assert metrics["wer_below_50_per_sample_mean"] == 0.25
    assert metrics["wer_below_50_corpus"] == pytest.approx(1 / 12)
    assert metrics["n_above_50_pct_wer"] == 1
    assert metrics["evaluated"] == 3
    assert metrics["skipped"] == 1


def test_explicit_cli_overrides_model_profile_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "tencent/AuK",
            "--max-concurrency",
            "3",
            "--warmup",
            "0",
            "--seed",
            "7",
            "--output-dir",
            "custom-results",
            "--server-config",
            "custom.yaml",
        ],
    )

    args, _ = tts._parse_args(tts._build_arg_parser())
    config = tts._config_from_args(args)
    assert config.concurrency == 3
    assert config.warmup == 0
    assert config.seed == 7
    assert config.output_dir == "custom-results"
    assert config.server_config == "custom.yaml"
