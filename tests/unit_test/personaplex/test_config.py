# SPDX-License-Identifier: Apache-2.0
"""PersonaPlex code2wav chunk ramp resolves from the CLI and YAML stage configuration."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.sources import patches_from_stages_mapping
from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig


@pytest.mark.parametrize("chunk_ramp_frames", [[1, 2, 4], [4], [1, 8, 2]])
@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_personaplex_chunk_ramp_resolves_from_user_configuration(
    chunk_ramp_frames: list[int], source: Literal["cli", "yaml"]
) -> None:
    config = PersonaPlexPipelineConfig(model_path="dummy")
    if source == "cli":
        config_manager = ConfigManager(config)
        resolved = config_manager.merge_config(
            config_manager.parse_extra_args(
                ["--code2wav.factory.chunk_ramp_frames", json.dumps(chunk_ramp_frames)]
            )
        )
    else:
        patches = patches_from_stages_mapping(
            {"code2wav": {"factory": {"chunk_ramp_frames": chunk_ramp_frames}}},
            type(config),
            ["code2wav"],
        )
        resolved = ConfigResolver(config).resolve(patches).config
    assert (
        resolved.stage_named("code2wav").factory.model_dump()["chunk_ramp_frames"]
        == chunk_ramp_frames
    )


def test_personaplex_chunk_ramp_default_is_in_the_stage_configuration() -> None:
    config = PersonaPlexPipelineConfig(model_path="dummy")
    assert config.stage_named("code2wav").factory.model_dump()["chunk_ramp_frames"] == [
        1,
        4,
    ]
    assert config.stage_named("code2wav").factory.max_graph_frames is None


@pytest.mark.parametrize(
    "option_text",
    ["[]", "[0,4]", "[1,-2]", "[1.5,4]", "[true,4]", '["4"]', "[4.0]", "4", "null"],
)
@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_personaplex_chunk_ramp_rejects_invalid_user_configuration(
    option_text: str, source: Literal["cli", "yaml"]
) -> None:
    config = PersonaPlexPipelineConfig(model_path="dummy")
    with pytest.raises(ValueError, match="chunk_ramp_frames"):
        if source == "cli":
            config_manager = ConfigManager(config)
            config_manager.merge_config(
                config_manager.parse_extra_args(
                    ["--code2wav.factory.chunk_ramp_frames", option_text]
                )
            )
        else:
            patches = patches_from_stages_mapping(
                {
                    "code2wav": {
                        "factory": {"chunk_ramp_frames": json.loads(option_text)}
                    }
                },
                type(config),
                ["code2wav"],
            )
            ConfigResolver(config).resolve(patches)
