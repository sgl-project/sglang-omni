# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shlex

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager
from tests.test_model.conftest import (
    QWEN3_OMNI_BF16_COLOCATED_VIDEO_ARGS,
    QWEN3_OMNI_FP8_COLOCATED_VIDEO_ARGS,
    QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN,
)


def _speech_config() -> PipelineConfig:
    names = [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]
    stages = []
    for index, name in enumerate(names):
        stages.append(
            StageConfig(
                name=name,
                process=name,
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                factory_args=(
                    {"thinker_max_seq_len": 8192}
                    if name in {"preprocessing", "thinker"}
                    else {}
                ),
                next=names[index + 1] if index + 1 < len(names) else None,
                terminal=index + 1 == len(names),
            )
        )
    return PipelineConfig(model_path="dummy", stages=stages)


@pytest.mark.parametrize(
    "fixture_args",
    [
        QWEN3_OMNI_BF16_COLOCATED_VIDEO_ARGS,
        QWEN3_OMNI_FP8_COLOCATED_VIDEO_ARGS,
    ],
)
def test_full_model_fixture_context_override_targets_thinker_not_decode(
    fixture_args: str,
) -> None:
    tokens = shlex.split(fixture_args)
    stage_tokens = []
    for index, token in enumerate(tokens):
        if token.startswith("--stages."):
            stage_tokens.extend((token, tokens[index + 1]))

    manager = ConfigManager(_speech_config())
    merged = manager.merge_config(manager.parse_extra_args(stage_tokens))
    stages = {stage.name: stage for stage in merged.stages}

    assert (
        stages["preprocessing"].factory_args["thinker_max_seq_len"]
        == QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN
    )
    assert (
        stages["thinker"].factory_args["thinker_max_seq_len"]
        == QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN
    )
    assert "thinker_max_seq_len" not in stages["decode"].factory_args
