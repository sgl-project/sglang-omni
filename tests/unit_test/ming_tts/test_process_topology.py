# SPDX-License-Identifier: Apache-2.0
"""Ming TTS process-boundary contracts."""

import pytest

from sglang_omni.models.ming_tts.config import (
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


@pytest.mark.parametrize(
    ("frontend_stages", "edge"),
    [
        (
            {PREPROCESSING_STAGE},
            (PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE),
        ),
        (
            {PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE},
            (REFERENCE_ENCODE_STAGE, TTS_ENGINE_STAGE),
        ),
    ],
)
def test_frontend_edges_remain_process_local(
    frontend_stages: set[str],
    edge: tuple[str, str],
) -> None:
    config_data = MingTTSPipelineConfig(model_path="model").model_dump()
    for stage in config_data["stages"]:
        if stage["name"] in frontend_stages:
            stage["process"] = "ming_frontend"
    config = MingTTSPipelineConfig(**config_data)

    with pytest.raises(ValueError, match="Cross-process edge") as exc_info:
        build_compiled_process_topology(config)
    assert f"{edge[0]!r} -> {edge[1]!r}" in str(exc_info.value)


def test_engine_to_audio_decode_remains_cross_process_safe() -> None:
    config_data = MingTTSPipelineConfig(model_path="model").model_dump()
    config_data["stages"][-1]["process"] = "ming_audio_decode"
    fractions = {
        "reference_encode": 0.08,
        "tts_engine": 0.72,
        "audio_decode": 0.12,
    }
    for stage in config_data["stages"]:
        if stage["name"] in fractions:
            stage["gpu_memory_fraction"] = fractions[stage["name"]]
    config = MingTTSPipelineConfig(**config_data)

    plan = build_compiled_process_topology(config)

    assert plan.stage_to_process["tts_engine"] == "pipeline"
    assert plan.stage_to_process["audio_decode"] == "ming_audio_decode"
