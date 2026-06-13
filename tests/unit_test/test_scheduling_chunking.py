# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.scheduling.chunking import (
    decrement_inflight_middle_chunks,
    get_inflight_middle_chunks,
    increment_inflight_middle_chunks,
    set_inflight_middle_chunks,
)


def test_chunking_helpers_prefer_current_sglang_counter() -> None:
    req = SimpleNamespace(inflight_middle_chunks=2)

    assert get_inflight_middle_chunks(req) == 2
    decrement_inflight_middle_chunks(req)

    assert req.inflight_middle_chunks == 1
    assert not hasattr(req, "is_chunked")


def test_chunking_helpers_support_legacy_counter_only() -> None:
    req = SimpleNamespace(is_chunked=0)

    increment_inflight_middle_chunks(req)
    set_inflight_middle_chunks(req, 3)

    assert get_inflight_middle_chunks(req) == 3
    assert req.is_chunked == 3
