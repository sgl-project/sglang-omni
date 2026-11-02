# SPDX-License-Identifier: Apache-2.0
"""Preprocessing reads the caller recording under the server's media policy."""

import numpy as np
import pytest

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.preprocessing import resource_connector
from sglang_omni.proto import OmniRequest, StagePayload


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
