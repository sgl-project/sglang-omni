# SPDX-License-Identifier: Apache-2.0
"""Unit tests for user configuration dumping and round-trip consistency."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import (
    AudioChunkingConfig,
    EngineStageConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.config.sources import dump_user_config
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


class DummyChunkingPipelineConfig(PipelineConfig):
    """Pipeline with audio chunking disallowed and mixed stage types."""

    allow_audio_chunking: ClassVar[bool] = False
    audio_chunking: AudioChunkingConfig | None = AudioChunkingConfig()
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "engine_node": EngineStageConfig,
    }


@pytest.fixture
def registered_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> DummyChunkingPipelineConfig:
    config = DummyChunkingPipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="preprocess",
                factory_path="tests.dummy:create_pre",
                next="engine_node",
                stream_to=["engine_node"],
                gpu=0,
            ),
            EngineStageConfig(
                name="engine_node",
                factory_path="tests.dummy:create_engine",
                terminal=True,
                engine=None,
            ),
        ],
    )
    monkeypatch.setitem(
        PIPELINE_CONFIG_REGISTRY.configs,
        "DummyChunkingPipelineConfig",
        DummyChunkingPipelineConfig,
    )
    return config


def test_dump_user_config_omits_internal_topology_and_unsupported_chunking(
    registered_pipeline: DummyChunkingPipelineConfig,
) -> None:
    dumped = dump_user_config(registered_pipeline)

    # 1. Internal topology fields are stripped
    assert "factory_path" not in dumped["stages"]["preprocess"]
    assert "next" not in dumped["stages"]["preprocess"]
    assert "stream_to" not in dumped["stages"]["preprocess"]
    assert "factory_path" not in dumped["stages"]["engine_node"]
    assert "terminal" not in dumped["stages"]["engine_node"]

    # 2. Public fields are preserved
    assert dumped["stages"]["preprocess"]["gpu"] == 0

    # 3. Unsupported audio chunking is omitted
    assert "audio_chunking" not in dumped

    # 4. Non-engine stage drops engine placeholder, while engine stage preserves explicit null
    assert "engine" not in dumped["stages"]["preprocess"]
    assert dumped["stages"]["engine_node"]["engine"] is None


def test_dump_user_config_roundtrips_through_config_manager(
    registered_pipeline: DummyChunkingPipelineConfig, tmp_path: Path
) -> None:
    dumped = dump_user_config(registered_pipeline)
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(yaml.safe_dump(dumped, sort_keys=False))

    reloaded = ConfigManager.from_file(str(config_file)).config
    assert reloaded.stages[0].name == "preprocess"
    assert reloaded.stages[0].gpu == 0
    assert reloaded.stages[1].name == "engine_node"
    assert reloaded.stages[1].engine is None
