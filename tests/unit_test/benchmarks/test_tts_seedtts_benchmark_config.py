from __future__ import annotations

from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
)


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


def test_seedtts_benchmark_records_quantization() -> None:
    config = _config_from_cli("--quantization", "fp8")
    assert config.quantization == "fp8"
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] == "fp8"


def test_seedtts_benchmark_quantization_defaults_to_none() -> None:
    config = _config_from_cli()
    assert config.quantization is None
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] is None
