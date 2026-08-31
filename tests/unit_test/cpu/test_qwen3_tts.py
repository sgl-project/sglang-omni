# SPDX-License-Identifier: Apache-2.0
"""CPU placement contracts for MOSS-TTS."""

from __future__ import annotations


from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.platforms import current_platform


def test_qwen3_tts_uses_cpu_placement() -> None:
    assert current_platform.is_cpu()

    config = Qwen3TTSPipelineConfig(model_path="model")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["tts_engine"].gpu is None
    assert stages["vocoder"].gpu is None
    assert config.gpu_placement == {}