# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr import request_builders, stages
from sglang_omni.proto import OmniRequest, StagePayload


def test_mlx_backend_requires_apple_platform(monkeypatch):
    from sglang.srt.hardware_backend.mlx import runtime

    monkeypatch.setattr(runtime, "use_mlx", lambda: True)
    with pytest.raises(RuntimeError, match="Apple Metal"):
        stages.create_nemotron3_5_asr_executor("unused", device="cpu")


def test_mlx_duration_limit_reuses_shared_preparation(monkeypatch):
    from sglang_omni.preprocessing import transcription

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda *a, **k: np.zeros(61 * 16000, dtype=np.float32),
    )
    builder = request_builders.make_nemotron3_5_asr_request_builder(
        prompt_dictionary={"auto": 101}, max_duration_s=60
    )
    payload = StagePayload(
        request_id="too-long", request=OmniRequest(inputs=b"audio"), data=None
    )
    with pytest.raises(ValueError, match="up to 60"):
        builder(payload)


def test_mlx_factory_clamps_batch_and_releases_runner(monkeypatch):
    pytest.importorskip("mlx.core")
    from sglang.srt.hardware_backend.mlx import runtime

    from sglang_omni import platforms
    from sglang_omni.models.nemotron3_5_asr.mlx import runner

    calls = {}

    class FakeRunner:
        prompt_dictionary = {"auto": 101}

        def __init__(self, *a, **kwargs):
            calls["load"] = kwargs

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(runtime, "use_mlx", lambda: True)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_mps=lambda: True)
    )
    monkeypatch.setattr(stages, "resolve_device_spec", lambda *a: "mps")
    monkeypatch.setattr(runner, "Nemotron3_5ASRMLXRunner", FakeRunner)
    monkeypatch.setattr(
        stages,
        "make_nemotron3_5_asr_request_builder",
        lambda **kwargs: calls.update(build=kwargs),
    )
    scheduler = stages.create_nemotron3_5_asr_executor("checkpoint", max_batch_size=8)
    assert scheduler._max_batch_size == 1
    assert calls["build"]["max_duration_s"] == 60
    scheduler.stop()
    assert calls["closed"]


def test_mlx_runner_rejects_unqualified_dtype_before_loading():
    pytest.importorskip("mlx.core")
    from sglang_omni.models.nemotron3_5_asr.mlx.runner import Nemotron3_5ASRMLXRunner

    with pytest.raises(ValueError, match="float32"):
        Nemotron3_5ASRMLXRunner("unused", dtype="float16")
