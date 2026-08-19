# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from sglang_omni.vendor.sglang.server_args import override_server_args


def test_override_server_args_uses_legacy_server_args_api() -> None:
    server_args = SimpleNamespace(override=Mock())

    override_server_args(server_args, "test-source", disable_cuda_graph=True)

    server_args.override.assert_called_once_with("test-source", disable_cuda_graph=True)


def test_override_server_args_declares_unpublished_current_config(monkeypatch) -> None:
    server_args = SimpleNamespace()
    declare_late_resolution = Mock()

    class _UnpublishedContext:
        @property
        def server_args(self):
            raise ValueError("Global server args is not set yet!")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_context=lambda: _UnpublishedContext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.arg_groups.overrides",
        SimpleNamespace(declare_late_resolution=declare_late_resolution),
    )

    override_server_args(server_args, "test-source", disable_cuda_graph=True)

    declare_late_resolution.assert_called_once_with(
        server_args, "test-source", disable_cuda_graph=True
    )


def test_override_server_args_uses_current_published_context(monkeypatch) -> None:
    server_args = SimpleNamespace()
    context = SimpleNamespace(server_args=server_args, override=Mock())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_context=lambda: context),
    )

    override_server_args(server_args, "test-source", disable_cuda_graph=True)

    context.override.assert_called_once_with("test-source", disable_cuda_graph=True)
