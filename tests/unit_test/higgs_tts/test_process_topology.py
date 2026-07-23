# SPDX-License-Identifier: Apache-2.0

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig


def test_higgs_preprocessing_and_audio_encoder_share_tts_frontend_process() -> None:
    config = HiggsTtsPipelineConfig(model_path="fake-model")
    placement_plan = build_stage_placement_plan(config)
    process_plan = build_process_topology_plan(config, placement_plan)

    assert process_plan.stage_to_process == {
        "preprocessing": "tts_frontend",
        "audio_encoder": "tts_frontend",
        "tts_engine": "pipeline",
        "vocoder": "vocoder",
    }
    assert [
        (group.name, group.stage_names, group.gpu_id) for group in process_plan.groups
    ] == [
        ("tts_frontend", ("preprocessing", "audio_encoder"), 0),
        ("pipeline", ("tts_engine",), 0),
        ("vocoder", ("vocoder",), 0),
    ]
