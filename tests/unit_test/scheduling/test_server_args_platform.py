# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.scheduling.sglang_backend import server_args_builder


def test_server_args_receive_worker_platform_device(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _ServerArgs:
        enable_dp_attention = False

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(server_args_builder, "ServerArgs", _ServerArgs)
    monkeypatch.setattr(
        server_args_builder,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )

    server_args_builder.build_sglang_server_args("model", context_length=1024)

    assert captured["device"] == "cuda"
