# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)


@pytest.mark.parametrize(
    "config_type",
    [
        Qwen3OmniPipelineConfig,
        Qwen3OmniSpeechPipelineConfig,
        Qwen3OmniSpeechColocatedPipelineConfig,
    ],
)
@pytest.mark.parametrize("cpus,workers,threads", [(192, 1, 8), (4, 1, 4), (4, 4, 1)])
def test_thread_defaults_follow_cpu_capacity_and_preprocessing_workers(
    monkeypatch, config_type, cpus, workers, threads
):
    from sglang_omni.utils import cpu

    monkeypatch.setattr(cpu, "effective_cpu_count", lambda: cpus)
    config = ConfigManager(config_type(model_path="model")).merge_config(
        [("preprocessing.factory.max_concurrency", str(workers))]
    )
    env = config.resolved_env_defaults()
    assert env["OMP_NUM_THREADS"] == str(threads)
    assert env["TOKENIZERS_PARALLELISM"] == "false"


def test_explicit_thread_and_tokenizer_settings_win():
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="model",
        env_defaults={"OMP_NUM_THREADS": "3", "TOKENIZERS_PARALLELISM": "true"},
    )
    assert config.resolved_env_defaults()["OMP_NUM_THREADS"] == "3"
    assert config.resolved_env_defaults()["TOKENIZERS_PARALLELISM"] == "true"


@pytest.mark.parametrize("initial,expected_prefix", [(0, 10), (2, 12), (4, 14)])
def test_talker_prefix_follows_first_audio_window(initial, expected_prefix):
    config = ConfigManager(
        Qwen3OmniSpeechPipelineConfig(model_path="model")
    ).merge_config([("code2wav.factory.initial_codec_chunk_frames", str(initial))])
    kwargs = config.stage_factory_kwargs("talker_ar")
    assert kwargs["codec_coalesce_early_frames"] == expected_prefix
    assert kwargs["codec_coalesce_frames"] == 10


@pytest.mark.parametrize(
    "config_type",
    [Qwen3OmniSpeechPipelineConfig, Qwen3OmniSpeechColocatedPipelineConfig],
)
def test_speech_first_chunk_defaults_to_four_frames(config_type):
    config = config_type(model_path="model")
    assert (
        config.stage_named("code2wav").factory.model_extra["initial_codec_chunk_frames"]
        == 4
    )
    assert config.stage_factory_kwargs("talker_ar")["codec_coalesce_early_frames"] == 14


def test_custom_chunk_size_and_explicit_talker_override():
    config = ConfigManager(
        Qwen3OmniSpeechPipelineConfig(model_path="model")
    ).merge_config(
        [
            ("code2wav.factory.stream_chunk_size", "6"),
            ("talker_ar.factory.codec_coalesce_early_frames", "20"),
        ]
    )
    resolved = resolve_stage_factory_args(config.stage_named("talker_ar"), config)
    assert resolved["codec_coalesce_frames"] == 6
    assert resolved["codec_coalesce_early_frames"] == 20
