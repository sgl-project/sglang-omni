from __future__ import annotations

from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_generation_kwargs,
    _build_results_config,
    _config_from_args,
)


def test_tts_seedtts_defaults_to_seeded_server_sampling() -> None:
    config = TtsSeedttsBenchmarkConfig(model="model", meta="meta")
    generation_kwargs = _build_generation_kwargs(config)

    assert config.temperature is None
    assert generation_kwargs["seed"] == 0
    assert "temperature" not in generation_kwargs


def test_tts_seedtts_cli_default_uses_seed_without_overriding_temperature() -> None:
    parser = _build_arg_parser()

    config = _config_from_args(
        parser.parse_args(["--model", "model", "--meta", "meta"])
    )

    assert config.seed == 0
    assert config.temperature is None


def test_tts_seedtts_cli_can_omit_seed_for_stochastic_sweeps() -> None:
    parser = _build_arg_parser()

    config = _config_from_args(
        parser.parse_args(["--model", "model", "--meta", "meta", "--seed", "-1"])
    )

    assert config.seed is None
    assert "seed" not in _build_generation_kwargs(config)


def test_tts_seedtts_results_config_records_sampling_controls() -> None:
    config = TtsSeedttsBenchmarkConfig(
        model="model",
        meta="meta",
        seed=123,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        repetition_penalty=1.1,
    )

    results_config = _build_results_config(config, base_url="http://server")

    assert results_config["seed"] == 123
    assert results_config["temperature"] == 0.8
    assert results_config["top_p"] == 0.8
    assert results_config["top_k"] == 30
    assert results_config["repetition_penalty"] == 1.1
