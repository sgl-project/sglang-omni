# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.runtime_context import get_context

from sglang_omni.vendor.sglang.server_args import override_server_args


def test_override_server_args_resolves_an_unpublished_record_in_place() -> None:
    server_args = SimpleNamespace(disable_cuda_graph=False)

    override_server_args(server_args, "test-source", disable_cuda_graph=True)

    assert server_args.disable_cuda_graph is True
    assert server_args._runtime_mutations == [
        ("test-source", {"disable_cuda_graph": True})
    ]


def test_override_server_args_writes_the_bags_of_the_published_record() -> None:
    with get_context().override_server_args(disable_cuda_graph=False) as published:
        override_server_args(published, "test-source", disable_cuda_graph=True)

        assert get_context().config_leaf("disable_cuda_graph") is True
        assert get_context().overrides_log() == [
            ("test-source", {"disable_cuda_graph": True})
        ]
        assert published.disable_cuda_graph is False
