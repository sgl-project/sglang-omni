# SPDX-License-Identifier: Apache-2.0
"""Round-trip coverage for the shared user-facing config export."""

from __future__ import annotations

from pathlib import Path

import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import dump_user_config
from sglang_omni.models.ming3_audio.config import Ming3AudioPipelineConfig
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils.json import JsonValue


def export_and_reload(
    config: PipelineConfig, tmp_path: Path
) -> tuple[dict[str, JsonValue], PipelineConfig]:
    exported = dump_user_config(config)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(exported, sort_keys=False),
        encoding="utf-8",
    )

    reloaded = ConfigManager.from_file(str(config_path)).config
    assert dump_user_config(reloaded) == exported
    return exported, reloaded


def test_ming_segmentation_default_off_survives_export(tmp_path: Path) -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")

    exported, reloaded = export_and_reload(config, tmp_path)

    assert exported["stages"]["talker"]["factory"]["enable_text_segmentation"] is False
    talker = reloaded.stage_named("talker")
    assert talker.factory.enable_text_segmentation is False


def test_ming_explicit_segmentation_on_survives_cli_export(tmp_path: Path) -> None:
    manager = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy"))
    config = manager.merge_config(
        manager.parse_extra_args(["--talker.factory.enable_text_segmentation", "true"])
    )

    exported, reloaded = export_and_reload(config, tmp_path)

    assert exported["stages"]["talker"]["factory"]["enable_text_segmentation"] is True
    assert reloaded.stage_named("talker").factory.enable_text_segmentation is True


def test_moss_tts_export_filters_topology_and_keeps_public_overrides(
    tmp_path: Path,
) -> None:
    manager = ConfigManager(MossTTSPipelineConfig(model_path="dummy"))
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--tts_engine.engine.mem_fraction_static",
                "0.55",
                "--tts_engine.engine.max_running_requests",
                "12",
                "--tts_engine.engine.disable_radix_cache",
                "true",
                "--tts_engine.factory.max_new_tokens",
                "256",
                "--preprocessing.factory.ref_audio_cache",
                "false",
                "--preprocessing.gpu_memory_fraction",
                "0.09",
            ]
        )
    )

    exported, reloaded = export_and_reload(config, tmp_path)

    assert "audio_chunking" not in exported
    assert "entry_stage" not in exported
    for stage in exported["stages"].values():
        assert not {
            "name",
            "factory_path",
            "next",
            "terminal",
            "route_fn",
            "stream_to",
            "stream_done_to_fn",
            "wait_for",
            "wait_for_fn",
            "merge_fn",
            "project_payload",
        }.intersection(stage)

    engine_stage = exported["stages"]["tts_engine"]
    assert engine_stage["engine"]["mem_fraction_static"] == 0.55
    assert engine_stage["engine"]["max_running_requests"] == 12
    assert engine_stage["engine"]["disable_radix_cache"] is True
    assert engine_stage["factory"]["max_new_tokens"] == 256
    assert exported["stages"]["preprocessing"]["factory"]["ref_audio_cache"] is False
    assert exported["stages"]["preprocessing"]["gpu_memory_fraction"] == 0.09

    engine = reloaded.stage_named("tts_engine")
    assert engine.engine.mem_fraction_static == 0.55
    assert engine.engine.max_running_requests == 12
    assert engine.engine.model_extra["disable_radix_cache"] is True
    assert engine.factory.max_new_tokens == 256
    preprocessing = reloaded.stage_named("preprocessing")
    assert preprocessing.factory.ref_audio_cache is False
    assert preprocessing.gpu_memory_fraction == 0.09


def test_whisper_audio_chunking_policy_survives_export(tmp_path: Path) -> None:
    manager = ConfigManager(WhisperASRPipelineConfig(model_path="dummy"))
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--audio_chunking.max_audio_clip_s",
                "20",
                "--audio_chunking.max_total_audio_s",
                "120",
                "--audio_chunking.max_concurrent_chunks",
                "4",
                "--audio_chunking.max_concurrent_long_audio_requests",
                "2",
            ]
        )
    )

    exported, reloaded = export_and_reload(config, tmp_path)

    assert exported["audio_chunking"] == {
        "max_audio_clip_s": 20.0,
        "max_total_audio_s": 120.0,
        "max_concurrent_chunks": 4,
        "max_concurrent_long_audio_requests": 2,
    }
    assert reloaded.audio_chunking == config.audio_chunking
