from __future__ import annotations

from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
)
from benchmarks.tasks.asr import normalize_text


def _config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    return _config_from_args(parser.parse_args(list(args)))


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = _config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = _config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128


def test_seedtts_benchmark_accepts_arabic() -> None:
    config = _config_from_cli(
        "--transcribe-only",
        "--skip-gpu-cleanup",
        "--meta",
        "google/fleurs",
        "--lang",
        "ar",
    )

    assert config.lang == "ar"
    assert config.meta == "google/fleurs"


def test_arabic_wer_normalization() -> None:
    assert normalize_text("إِنَّ ٱلْعَرَبِيَّةَ — جَمِيلَةٌ! ١٢٣", "ar") == ("ان العربية جميلة 123")
