# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    THINKER_STAGE,
    LLaDA2UniTextPipelineConfig,
    Variants,
)


def test_text_variant_omits_image_encoder() -> None:
    config = LLaDA2UniTextPipelineConfig(model_path="/models/llada2")
    stages = {stage.name: stage for stage in config.stages}

    assert Variants["text"] is LLaDA2UniTextPipelineConfig
    assert IMAGE_STAGE not in stages
    assert stages[PREPROCESSING_STAGE].next == THINKER_STAGE
    assert stages[THINKER_STAGE].next == DECODE_STAGE
    assert stages[DECODE_STAGE].terminal is True


def test_text_variant_disables_custom_all_reduce_for_tp() -> None:
    updates = LLaDA2UniTextPipelineConfig.tensor_parallel_server_args_overrides(
        stage_name=THINKER_STAGE,
        tp_size=2,
    )

    assert updates == {"disable_custom_all_reduce": True}
