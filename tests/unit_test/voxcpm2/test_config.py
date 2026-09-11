# SPDX-License-Identifier: Apache-2.0
"""Pipeline config and registry wiring for VoxCPM2."""

from __future__ import annotations

import importlib
import json

from sglang_omni.config import manager
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxcpm2.config import (
    ENGINE_STAGE,
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    VOCODER_STAGE,
    VoxCPM2PipelineConfig,
)


def test_registered_pipeline_is_connected():
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("voxcpm2")
    assert config_cls is VoxCPM2PipelineConfig

    config = config_cls(model_path="openbmb/VoxCPM2")
    preprocessing, reference_encode, engine, vocoder = config.stages
    assert config.resolved_entry_stage == preprocessing.name == PREPROCESSING_STAGE
    assert preprocessing.next == reference_encode.name == REFERENCE_ENCODE_STAGE
    assert reference_encode.next == engine.name == ENGINE_STAGE
    assert engine.next == vocoder.name == VOCODER_STAGE
    assert config.terminal_stages == [vocoder.name]

    for stage in config.stages:
        module, name = stage.factory_path.rsplit(".", 1)
        assert callable(getattr(importlib.import_module(module), name))


def test_engine_streams_to_a_vocoder_that_accepts_early_chunks():
    config = VoxCPM2PipelineConfig(model_path="openbmb/VoxCPM2")
    _, _, engine, vocoder = config.stages
    assert engine.stream_to == [VOCODER_STAGE]
    assert vocoder.can_accept_stream_before_payload is True


def test_singular_architecture_key_resolves_the_pipeline(tmp_path):
    """VoxCPM2 ships "architecture", not the usual "architectures" list."""
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}))
    assert (
        manager.resolve_config_cls_for_model_path(str(tmp_path))
        is VoxCPM2PipelineConfig
    )
