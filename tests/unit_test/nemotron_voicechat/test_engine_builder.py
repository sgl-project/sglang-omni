# SPDX-License-Identifier: Apache-2.0
"""The talker defaults to an attention backend that runs its head_dim of 72."""

from __future__ import annotations

import pytest

import sglang_omni.models.nemotron_voicechat.engine_builder as engine_builder
from sglang_omni.models.nemotron_voicechat.duplex_stages import TalkerBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def generation_defaults_on(
    monkeypatch: pytest.MonkeyPatch, builder, *, is_cuda: bool, sm_version: int | None
) -> dict:
    monkeypatch.setattr(engine_builder.current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(
        engine_builder, "get_visible_gpu_sm_version", lambda gpu_id: sm_version
    )
    builder.gpu_id = 0
    return builder.generation_defaults(dtype="bfloat16")


@pytest.mark.parametrize(
    ("is_cuda", "sm_version", "expected_backend"),
    [
        pytest.param(True, 90, None, id="sm90-fa3"),
        pytest.param(True, 100, "triton", id="sm100-trtllm-mha"),
        pytest.param(True, 120, "triton", id="sm120-flashinfer"),
        pytest.param(False, None, None, id="non-cuda"),
    ],
)
def test_talker_default_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    sm_version: int | None,
    expected_backend: str | None,
) -> None:
    builder = engine_builder.NemotronVoiceChatTalkerEngineBuilder()
    defaults = generation_defaults_on(
        monkeypatch, builder, is_cuda=is_cuda, sm_version=sm_version
    )

    assert defaults.get("attention_backend") == expected_backend


def test_operator_attention_backend_overrides_talker_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = engine_builder.NemotronVoiceChatTalkerEngineBuilder()
    defaults = generation_defaults_on(
        monkeypatch, builder, is_cuda=True, sm_version=120
    )

    overrides = build_generation_batch_overrides(
        **defaults, server_args_overrides={"attention_backend": "flashinfer"}
    )

    assert overrides["attention_backend"] == "flashinfer"


def test_duplex_talker_keeps_triton_on_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = generation_defaults_on(
        monkeypatch, TalkerBuilder(), is_cuda=True, sm_version=90
    )

    assert defaults["attention_backend"] == "triton"
