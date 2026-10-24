# SPDX-License-Identifier: Apache-2.0
"""The Qwen Session profile assembles without a cascade package."""

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import SessionNodeConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSessionPipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.sessions import admit_input
from sglang_omni.proto.session import ControlEdge, TimedChunk
from sglang_omni.proto.streams import (
    CONTROL,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    TEXT_DELTA,
)


def profile_config() -> Qwen3OmniSessionPipelineConfig:
    path = Path(__file__).resolve().parents[3] / "examples/qwen3_omni/session_h200.yaml"
    return ConfigManager.from_file(str(path)).config


def test_qwen3_omni_profile_declares_persistent_stage_routes_and_feedback() -> None:
    config = profile_config()
    assert config.resolved_entry_stage == "dialog"
    graph = config.session_graph()
    assert graph.input_nodes() == ("vad", "dialog")
    assert graph.reading_nodes("vad", SESSION_CONTROL) == ("dialog",)
    assert graph.reading_nodes("input", PCM16) == ("vad", "dialog")
    assert graph.reading_nodes("input", IMAGE_FRAME) == ("dialog",)
    assert graph.reading_nodes("dialog", "qwen.unit") == ("qwen_omni",)
    assert config.session_ports().input_modalities == ("audio", "image")
    assert graph.publish == {
        "vad": (SESSION_CONTROL,),
        "dialog": (SESSION_CONTROL,),
        "qwen_omni": (PCM_F32, SESSION_CONTROL, TEXT_DELTA),
    }
    assert "cascade" not in {stage.name for stage in config.stages}
    node = graph.nodes["qwen_omni"]
    assert node.stages == (
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    )
    assert node.controls == (
        ControlEdge("input", "configure"),
        ControlEdge("dialog", INTERRUPT, should_preempt=True),
        ControlEdge("dialog", "qwen.truncate"),
    )
    assert graph.reading_nodes("qwen_omni", SESSION_CONTROL) == ("dialog",)
    thinker = config.stage_named("thinker")
    talker = config.stage_named("talker_ar")
    code2wav = config.stage_named("code2wav")
    for stage in (thinker, talker):
        assert resolve_stage_factory_args(stage, config)["execution_mode"] == "session"
    assert code2wav.factory_path.endswith(
        "code2wav_scheduler.create_code2wav_scheduler"
    )
    assert thinker.stream_to == ["talker_ar"] and talker.stream_to == ["code2wav"]
    assert (
        talker.can_accept_stream_before_payload
        and code2wav.can_accept_stream_before_payload
    )
    assert talker.factory.enable_partial_start is True
    assert config.max_sessions == 16
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    placement = build_stage_placement_plan(
        config, stages_cfg=expanded, replica_instances=topology.replicas
    )
    qwen_stages = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")
    assert {placement.stages[name].gpu_ids for name in qwen_stages} == {(0,)}


@pytest.mark.parametrize("admission", [True, False])
def test_ar_stages_take_continuation_aware_admission_from_their_factory_settings(
    admission: bool,
) -> None:
    manager = ConfigManager(profile_config())
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.factory.continuation_aware_admission",
                str(admission).lower(),
                "--talker_ar.factory.continuation_aware_admission",
                str(admission).lower(),
            ]
        )
    )
    for stage_name in ("thinker", "talker_ar"):
        assert "continuation_aware_admission" not in resolve_stage_factory_args(
            profile_config().stage_named(stage_name), profile_config()
        )
        kwargs = resolve_stage_factory_args(config.stage_named(stage_name), config)
        assert kwargs["continuation_aware_admission"] is admission


def test_a_node_reading_every_input_format_must_take_camera_frames():
    config = profile_config()
    blind_vad = SessionNodeConfig(stages=["vad"], inputs=["input"])
    with pytest.raises(ValueError, match=r"nodes \['vad'\] read every format"):
        config.model_copy(
            update={"session_nodes": {**config.session_nodes, "vad": blind_vad}}
        ).session_ports()


def test_session_profile_round_trip_keeps_model_factory_arguments() -> None:
    config = profile_config()
    rebuilt = Qwen3OmniSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_ports() == config.session_ports()
    assert rebuilt.session_graph() == config.session_graph()
    assert rebuilt.session_limits().max_output_chunks == 1024
    for stage in config.stages:
        assert rebuilt.stage_factory_kwargs(stage.name) == config.stage_factory_kwargs(
            stage.name
        )
    assert config.realtime_deployment_factory == (
        "sglang_omni.serve.realtime.session_deployment.build_realtime_deployment"
    )


def test_session_limit_admits_a_complete_maximum_duration_utterance() -> None:
    config = profile_config()
    unit = TimedChunk(
        CONTROL,
        0,
        0,
        0,
        {
            "pcm": bytes(int(config.max_utterance_seconds * 32000)),
            "images": [],
            "carried_turns": [],
        },
        format="qwen.unit",
        turn=1,
    )
    admitted, encoded_bytes = admit_input(unit, config.session_limits(), {}, set())
    assert admitted.payload["pcm"] == unit.payload["pcm"]
    assert encoded_bytes > 1048576
