# SPDX-License-Identifier: Apache-2.0
"""Distributed backend policy tests for the Ming image encoder."""

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt import server_args
from sglang.srt.distributed import parallel_state
from sglang.srt.layers import dp_attention

from sglang_omni.models.ming_omni.components import image_encoder


def test_tp_initialization_uses_platform_backend(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        image_encoder,
        "current_platform",
        SimpleNamespace(get_torch_distributed_backend_str=lambda: "hccl"),
    )
    monkeypatch.setattr(dp_attention, "_ATTN_TP_SIZE", None, raising=False)
    monkeypatch.setattr(dp_attention, "_ATTN_TP_RANK", None, raising=False)
    monkeypatch.setattr(
        parallel_state, "model_parallel_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        lambda **kwargs: calls.setdefault("distributed", kwargs),
    )
    monkeypatch.setattr(
        parallel_state,
        "initialize_model_parallel",
        lambda **kwargs: calls.setdefault("model_parallel", kwargs),
    )
    monkeypatch.setattr(server_args, "ServerArgs", lambda **_kwargs: object())
    monkeypatch.setattr(
        server_args, "set_global_server_args_for_scheduler", lambda _args: None
    )
    monkeypatch.setattr(image_encoder.MingImageEncoder, "_did_init_tp", False)

    image_encoder.MingImageEncoder._init_sglang_tp(tp_rank=1, tp_size=2)

    distributed = calls["distributed"]
    assert distributed["backend"] == "hccl"
    assert distributed["world_size"] == 2
    assert distributed["rank"] == 1
    assert calls["model_parallel"] == {"tensor_model_parallel_size": 2}
