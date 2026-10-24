# SPDX-License-Identifier: Apache-2.0
"""Distributed backend policy tests for the Ming image encoder."""

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt import runtime_context
from sglang.srt.distributed import parallel_state

from sglang_omni.models.ming_omni.components import image_encoder


def test_tp_initialization_uses_platform_backend(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        image_encoder,
        "current_platform",
        SimpleNamespace(get_torch_distributed_backend_str=lambda: "hccl"),
    )
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        lambda **kwargs: calls.setdefault("distributed", kwargs),
    )
    monkeypatch.setattr(
        parallel_state,
        "initialize_model_parallel",
        lambda: calls.setdefault("model_parallel", True),
    )
    monkeypatch.setattr(
        runtime_context,
        "publish",
        lambda record, *, role, ranks: calls.setdefault(
            "published", (record.tp_size, ranks.world_rank)
        ),
    )
    monkeypatch.setattr(image_encoder.MingImageEncoder, "did_init_tp", False)

    image_encoder.MingImageEncoder.init_sglang_tp(tp_rank=1, tp_size=2)

    distributed = calls["distributed"]
    assert distributed["backend"] == "hccl"
    assert distributed["world_size"] == 2
    assert distributed["rank"] == 1
    assert calls["published"] == (2, 1)
    assert calls["model_parallel"] is True
