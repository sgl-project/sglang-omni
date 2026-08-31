# SPDX-License-Identifier: Apache-2.0
"""Ming TTS process-boundary contracts."""

from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.sources import sources_from_config_file
from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_example_process_topology_compiles() -> None:
    config_path = _REPO_ROOT / "examples/configs/ming_omni_tts.yaml"
    config, patches = sources_from_config_file(str(config_path))
    config = ConfigManager(config).merge_config([], extra_patches=patches)
    assert isinstance(config, MingTTSPipelineConfig)

    plan = build_compiled_process_topology(config)

    assert plan.stage_to_process == {
        PREPROCESSING_STAGE: "preprocessing",
        REFERENCE_ENCODE_STAGE: "ming_tts_aux",
        TTS_ENGINE_STAGE: "tts_engine",
        AUDIO_DECODE_STAGE: "ming_tts_aux",
    }
