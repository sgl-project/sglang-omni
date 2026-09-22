# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark entry-point: model profiles, server lifecycle, WER filter."""

import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pytest
import requests

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks import asr
from benchmarks.tasks.tts import _build_tts_payload
from tests.utils import QWEN3_ASR_WER_CONCURRENCY, assert_wer_partitioned

SEEDTTS_SAMPLE = SampleInput(
    sample_id="sample-1",
    ref_text="reference",
    ref_audio="ref.wav",
    target_text="hello world",
)


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
    args, profile = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
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

    args, _ = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
    assert config.concurrency == 3
    assert config.warmup == 0
    assert config.seed == 7
    assert config.output_dir == "custom-results"
    assert config.server_config == "custom.yaml"


def test_fun_cosyvoice3_cli_default_omits_max_new_tokens(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--model", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"]
    )
    args, _ = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
    payload = _build_tts_payload(
        SEEDTTS_SAMPLE,
        config.model,
        **tts._build_generation_kwargs(
            config
        ),  # noqa: leading-underscore  # production name
    )
    assert "max_new_tokens" not in payload


@pytest.mark.parametrize(
    "model, max_new_tokens",
    [
        ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", None),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", 2048),
    ],
)
def test_max_new_tokens_default_applies_without_cli(model, max_new_tokens):
    # note (Yucheng Hu): TTS CI builds the config directly, so the per-model
    # default has to resolve at request-build time, not only in _parse_args.
    config = tts.TtsSeedttsBenchmarkConfig(model=model, meta="meta.lst")
    payload = _build_tts_payload(
        SEEDTTS_SAMPLE,
        model,
        **tts._build_generation_kwargs(
            config
        ),  # noqa: leading-underscore  # production name
    )
    assert payload.get("max_new_tokens") == max_new_tokens


def test_explicit_max_new_tokens_overrides_fun_cosyvoice3_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "--max-new-tokens",
            "2048",
        ],
    )
    args, _ = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
    payload = _build_tts_payload(
        SEEDTTS_SAMPLE,
        config.model,
        **tts._build_generation_kwargs(
            config
        ),  # noqa: leading-underscore  # production name
    )
    assert payload["max_new_tokens"] == 2048


def test_wer_fanout_preserves_all_twenty_samples_at_long_audio_admission_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # note (wenyao): routing can send every request to one four-slot worker.
    slots = threading.BoundedSemaphore(4)
    cohort = threading.Barrier(min(QWEN3_ASR_WER_CONCURRENCY, 20))
    uploaded: list[str] = []

    def post(
        url: str, *, files: dict[str, tuple[str, BinaryIO, str]], **kwargs: object
    ) -> requests.Response:
        admitted = slots.acquire(blocking=False)
        try:
            # note (wenyao): all requests attempt admission before slots reopen.
            cohort.wait(timeout=5)
            uploaded.append(files["file"][0])
            response = requests.Response()
            response.url = url
            response.status_code = 200 if admitted else 503
            response._content = json.dumps(  # noqa: leading-underscore  # upstream name
                {"text": "hello world"}
                if admitted
                else {
                    "detail": "Too many long-audio transcriptions in flight "
                    "(limit 4); retry later"
                }
            ).encode()
            return response
        finally:
            if admitted:
                slots.release()

    monkeypatch.setattr(asr.requests, "post", post)
    records: list[dict[str, str | bool | int]] = []
    for index in range(20):
        path = tmp_path / f"sample-{index}.wav"
        path.write_bytes(b"saved audio for mocked transcription service")
        records.append(
            {
                "sample_id": f"sample-{index}",
                "raw_response": "hello world",
                "is_success": True,
                "wav_path": str(path),
                "audio_duration_s": 31,
            }
        )
    result = asr.compute_text_audio_consistency_from_records(
        records,
        "en",
        "cuda:0",
        asr_router_port=12345,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )

    assert len(uploaded) == len(set(uploaded)) == 20
    assert result["summary"]["evaluated"] == 20
    assert result["summary"]["skipped"] == 0
    assert_wer_partitioned(result, max_wer_below_50_corpus=0, max_n_above_50=0)
    assert asr.DEFAULT_ASR_TRANSCRIBE_CONCURRENCY == 32
