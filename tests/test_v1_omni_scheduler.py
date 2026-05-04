# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler


def test_append_stream_chunk_default_buffers_raw_chunks() -> None:
    req_data = SimpleNamespace()
    chunk = SimpleNamespace(data="chunk-data", metadata={"token_id": 1})

    OmniScheduler._append_stream_chunk_default(req_data, chunk)

    assert list(req_data.stream_chunks) == [chunk]


def test_mark_stream_done_default_sets_generic_flag() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler._stream_done_handler = None
    req_data = SimpleNamespace()

    scheduler._mark_stream_done(req_data)

    assert req_data.stream_done is True
