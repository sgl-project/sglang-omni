# SPDX-License-Identifier: Apache-2.0
"""VoiceChat stages: caller audio under the media policy, engine graph defaults."""

import numpy as np
import pytest

from sglang_omni.models.nemotron_voicechat import engine_builder, stages
from sglang_omni.models.nemotron_voicechat.engine_builder import (
    NemotronVoiceChatEngineBuilder,
    NemotronVoiceChatTalkerEngineBuilder,
)
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.preprocessing import resource_connector
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def test_caller_audio_follows_the_server_media_policy(monkeypatch, tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setenv(resource_connector.ALLOWED_LOCAL_MEDIA_PATH_ENV, str(allowed))
    monkeypatch.setattr(resource_connector, "_global_connector", None)
    reads = []

    def load_audio(source, **_):
        reads.append(source)
        return np.zeros((1, 1920), dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", load_audio)
    preprocess = stages.create_preprocessing_executor("unused").fn
    payload = StagePayload(
        "r", request=OmniRequest(inputs={"audio_path": str(outside)}), data={}
    )

    with pytest.raises(ValueError, match="not within allowed directory"):
        preprocess(payload)
    assert reads == []


def test_only_the_talker_captures_decode_graphs_by_default_on_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(engine_builder, "current_platform", CUDAOmniPlatform())
    thinker = NemotronVoiceChatEngineBuilder().generation_defaults(dtype="bfloat16")
    talker = NemotronVoiceChatTalkerEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert thinker["disable_cuda_graph"] is True
    assert talker["disable_cuda_graph"] is False


def test_talker_decodes_eagerly_by_default_off_cuda(monkeypatch) -> None:
    monkeypatch.setattr(engine_builder, "current_platform", XPUOmniPlatform())
    talker = NemotronVoiceChatTalkerEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert talker["disable_cuda_graph"] is True


def test_stage_engine_config_overrides_the_graph_default() -> None:
    overrides = build_generation_batch_overrides(
        server_args_overrides={"disable_cuda_graph": False},
        **NemotronVoiceChatEngineBuilder().generation_defaults(dtype="bfloat16"),
    )

    assert overrides["disable_cuda_graph"] is False
