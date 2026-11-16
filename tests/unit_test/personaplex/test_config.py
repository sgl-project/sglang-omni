# SPDX-License-Identifier: Apache-2.0
"""PersonaPlex code2wav chunk ramp and decode capacity resolve from the stage configuration."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.sources import patches_from_stages_mapping
from sglang_omni.models.personaplex.config import (
    DEFAULT_LM_MAX_RUNNING_REQUESTS,
    PersonaPlexPipelineConfig,
)


@pytest.mark.parametrize("chunk_ramp_frames", [[1, 2, 4]])
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


@pytest.mark.parametrize(
    "option_text",
    ["[]", "[0,4]", "[true,4]", "4", "null"],
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


def test_async_decode_is_opt_in_from_cli() -> None:
    config = PersonaPlexPipelineConfig(model_path="model")
    assert not config.stage_named("lm").factory.enable_async_decode
    manager = ConfigManager(config)
    updated = manager.merge_config(
        manager.parse_extra_args(["--lm.factory.enable_async_decode", "true"])
    )
    assert updated.stage_named("lm").factory.enable_async_decode


def rebuild(
    config: PersonaPlexPipelineConfig, *arguments: str
) -> PersonaPlexPipelineConfig:
    manager = ConfigManager(config)
    config = manager.merge_config(manager.parse_extra_args(list(arguments)))
    return PersonaPlexPipelineConfig.model_validate(config.model_dump())


def code2wav_arguments(config: PersonaPlexPipelineConfig) -> dict[str, object]:
    return resolve_stage_factory_args(config.stage_named("code2wav"), config)


def test_code2wav_decode_capacity_follows_the_lm_until_pinned() -> None:
    config = PersonaPlexPipelineConfig(model_path="dummy")
    arguments = code2wav_arguments(config)
    assert arguments["num_decode_slots"] == DEFAULT_LM_MAX_RUNNING_REQUESTS
    assert arguments["max_graph_frames"] == 4
    assert arguments["max_batch_size"] == 8

    # Derived values are never written back, so they follow later overrides.
    for concurrency, ramp in [(16, [1, 8, 2]), (4, [2]), (1, [1, 4])]:
        config = rebuild(
            config,
            "--lm.engine.max_running_requests",
            str(concurrency),
            "--code2wav.factory.chunk_ramp_frames",
            json.dumps(ramp),
        )
        factory = config.stage_named("code2wav").factory
        assert factory.num_decode_slots is None
        assert factory.max_graph_frames is None
        arguments = code2wav_arguments(config)
        assert arguments["num_decode_slots"] == concurrency
        assert arguments["max_graph_frames"] == max(ramp)
        assert arguments["chunk_ramp_frames"] == ramp

    # Pinned values win, and pinned slots may not fall below the LM's concurrency.
    config = rebuild(
        config,
        "--code2wav.factory.max_graph_frames",
        "0",
        "--code2wav.factory.num_decode_slots",
        "24",
        "--lm.engine.max_running_requests",
        "16",
    )
    arguments = code2wav_arguments(config)
    assert arguments["num_decode_slots"] == 24
    assert arguments["max_graph_frames"] == 0
    rebuild(config, "--lm.engine.max_running_requests", "24")
    with pytest.raises(ValueError, match="Omit num_decode_slots to derive it"):
        rebuild(config, "--lm.engine.max_running_requests", "25")
