# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_model import conftest as model_conftest
from tests.test_model import test_qwen3_omni_videoamme_talker_tp2_ci as stage11


def test_stage11_fp8_tp2_fixture_uses_safe_startup_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield SimpleNamespace(port=8000)

    monkeypatch.setattr(model_conftest, "_start_qwen3_omni_speech_server", fake_start)

    fixture_fn = model_conftest.qwen3_omni_fp8_talker_server_tp2.__wrapped__
    generator = fixture_fn(None)
    assert next(generator).port == 8000
    with pytest.raises(StopIteration):
        next(generator)

    extra_args = captured["extra_args"]
    assert (
        extra_args[extra_args.index("--thinker-mem-fraction-static") + 1]
        == model_conftest.QWEN3_OMNI_FP8_TP2_THINKER_MEM_FRACTION_STATIC
    )
    assert (
        extra_args[extra_args.index("--talker-mem-fraction-static") + 1]
        == model_conftest.QWEN3_OMNI_FP8_TP2_TALKER_MEM_FRACTION_STATIC
    )
    assert (
        extra_args[extra_args.index("--thinker-cuda-graph") + 1]
        == model_conftest.QWEN3_OMNI_FP8_TP2_THINKER_CUDA_GRAPH
    )
    assert (
        extra_args[extra_args.index("--partial-start-min-chunks") + 1]
        == model_conftest.QWEN3_OMNI_FP8_TP2_PARTIAL_START_MIN_CHUNKS
    )


def test_stage11_videoamme_config_reduces_payload(monkeypatch, tmp_path_factory) -> None:
    monkeypatch.delenv(stage11.STAGE11_VIDEO_MIN_PIXELS_ENV, raising=False)
    monkeypatch.delenv(stage11.STAGE11_VIDEO_MAX_PIXELS_ENV, raising=False)
    monkeypatch.delenv(
        stage11.STAGE11_TALKER_PREFILL_USER_CONTEXT_ENV,
        raising=False,
    )
    captured: dict[str, object] = {}

    async def fake_run_videoamme_eval(config, compute_wer: bool):
        captured["config"] = config
        captured["compute_wer"] = compute_wer
        return {"summary": {}, "speed": {}, "per_sample": []}

    monkeypatch.setattr(stage11, "run_videoamme_eval", fake_run_videoamme_eval)

    artifacts = stage11.talker_eval_artifacts.__wrapped__(
        SimpleNamespace(port=8000),
        tmp_path_factory,
    )

    config = captured["config"]
    assert captured["compute_wer"] is False
    assert artifacts.per_sample == []
    assert config.video_min_pixels == 6_272
    assert config.video_max_pixels == 6_272
    assert config.warmup == 0
    assert config.extra_request_params == {
        stage11.STAGE11_TALKER_PREFILL_USER_CONTEXT_PARAM: False
    }


def test_stage11_videoamme_config_sweep_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv(stage11.STAGE11_VIDEO_MIN_PIXELS_ENV, "12544")
    monkeypatch.setenv(stage11.STAGE11_VIDEO_MAX_PIXELS_ENV, "25088")
    monkeypatch.setenv(stage11.STAGE11_TALKER_PREFILL_USER_CONTEXT_ENV, "true")

    video_min_pixels, video_max_pixels, extra_request_params = (
        stage11._stage11_video_request_options()
    )

    assert video_min_pixels == 12_544
    assert video_max_pixels == 25_088
    assert extra_request_params == {
        stage11.STAGE11_TALKER_PREFILL_USER_CONTEXT_PARAM: True
    }
