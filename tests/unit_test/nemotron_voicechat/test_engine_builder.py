# SPDX-License-Identifier: Apache-2.0
"""Engine defaults: the talker's attention backend and the decode graph per stage."""

from __future__ import annotations

import pytest

import sglang_omni.models.nemotron_voicechat.engine_builder as engine_builder
from sglang_omni.models.nemotron_voicechat.duplex_stages import (
    TalkerBuilder,
    ThinkerBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
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
    builder = engine_builder.NemotronVoiceChatTalkerEngineBuilder(
        enable_cuda_graph=True
    )
    defaults = generation_defaults_on(
        monkeypatch, builder, is_cuda=is_cuda, sm_version=sm_version
    )

    assert defaults.get("attention_backend") == expected_backend


def test_operator_attention_backend_overrides_talker_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = engine_builder.NemotronVoiceChatTalkerEngineBuilder(
        enable_cuda_graph=True
    )
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


@pytest.mark.parametrize(
    "sm_version", [pytest.param(90, id="sm90"), pytest.param(120, id="sm120")]
)
def test_only_the_talker_captures_decode_graphs_on_cuda(
    monkeypatch: pytest.MonkeyPatch, sm_version: int
) -> None:
    thinker = generation_defaults_on(
        monkeypatch,
        engine_builder.NemotronVoiceChatEngineBuilder(),
        is_cuda=True,
        sm_version=sm_version,
    )
    talker = generation_defaults_on(
        monkeypatch,
        engine_builder.NemotronVoiceChatTalkerEngineBuilder(enable_cuda_graph=True),
        is_cuda=True,
        sm_version=sm_version,
    )

    assert thinker["disable_cuda_graph"] is True
    assert talker["disable_cuda_graph"] is False


def test_talker_decodes_eagerly_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    talker = generation_defaults_on(
        monkeypatch,
        engine_builder.NemotronVoiceChatTalkerEngineBuilder(enable_cuda_graph=True),
        is_cuda=False,
        sm_version=None,
    )

    assert talker["disable_cuda_graph"] is True


@pytest.mark.parametrize("builder_class", [ThinkerBuilder, TalkerBuilder])
def test_duplex_stages_capture_decode_graphs_only_on_cuda(
    monkeypatch: pytest.MonkeyPatch, builder_class
) -> None:
    on_cuda = generation_defaults_on(
        monkeypatch, builder_class(), is_cuda=True, sm_version=90
    )
    off_cuda = generation_defaults_on(
        monkeypatch, builder_class(), is_cuda=False, sm_version=None
    )

    assert on_cuda["disable_cuda_graph"] is False
    assert on_cuda["cuda_graph_backend_prefill"] == CudaGraphBackend.DISABLED
    assert off_cuda["disable_cuda_graph"] is True
