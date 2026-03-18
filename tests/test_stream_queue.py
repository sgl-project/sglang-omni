# SPDX-License-Identifier: Apache-2.0
"""Tests for StreamQueue."""
from __future__ import annotations

import pytest

from sglang_omni.pipeline.stage.stream_queue import (
    StreamItem,
    StreamQueue,
    StreamSignal,
)

REQ = "req-1"


def _make_item(
    chunk_id: int = 0, data: str = "hello", stage: str = "enc"
) -> StreamItem:
    return StreamItem(chunk_id=chunk_id, data=data, from_stage=stage)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_put_get():
    sq = StreamQueue()
    sq.open(REQ)
    item = _make_item(chunk_id=0)
    sq.put(REQ, item)
    result = await sq.get(REQ)
    assert result is item
    assert result.chunk_id == 0
    assert result.data == "hello"
    sq.close(REQ)


@pytest.mark.asyncio
async def test_put_done_returns_none():
    sq = StreamQueue()
    sq.open(REQ)
    sq.put_done(REQ, from_stage="enc")
    result = await sq.get(REQ)
    assert result is None


@pytest.mark.asyncio
async def test_put_error_raises():
    sq = StreamQueue()
    sq.open(REQ)
    sq.put_error(REQ, ValueError("boom"), from_stage="enc")
    with pytest.raises(ValueError, match="boom"):
        await sq.get(REQ)


@pytest.mark.asyncio
async def test_get_with_source_returns_signal():
    sq = StreamQueue()
    sq.open(REQ)

    # done signal
    sq.put_done(REQ, from_stage="enc")
    sig = await sq.get_with_source(REQ)
    assert isinstance(sig, StreamSignal)
    assert sig.is_done is True
    assert sig.from_stage == "enc"

    # error signal
    err = RuntimeError("fail")
    sq.put_error(REQ, err, from_stage="dec")
    sig = await sq.get_with_source(REQ)
    assert isinstance(sig, StreamSignal)
    assert sig.error is err
    assert sig.from_stage == "dec"

    sq.close(REQ)


@pytest.mark.asyncio
async def test_close_cleans_up():
    sq = StreamQueue()
    sq.open(REQ)
    assert sq.has(REQ)
    sq.close(REQ)
    assert not sq.has(REQ)


@pytest.mark.asyncio
async def test_put_to_closed_queue_raises():
    sq = StreamQueue()
    # never opened
    with pytest.raises(KeyError):
        sq.put(REQ, _make_item())

    # opened then closed
    sq.open(REQ)
    sq.close(REQ)
    with pytest.raises(KeyError):
        sq.put(REQ, _make_item())


@pytest.mark.asyncio
async def test_has():
    sq = StreamQueue()
    assert not sq.has(REQ)
    sq.open(REQ)
    assert sq.has(REQ)
    sq.close(REQ)
    assert not sq.has(REQ)


@pytest.mark.asyncio
async def test_done_after_many_items():
    """Done signal enqueued after many items, delivered in order."""
    sq = StreamQueue()
    sq.open(REQ)
    sq.put(REQ, _make_item(chunk_id=0))
    sq.put(REQ, _make_item(chunk_id=1))
    sq.put_done(REQ, from_stage="enc")

    item0 = await sq.get(REQ)
    assert item0.chunk_id == 0
    item1 = await sq.get(REQ)
    assert item1.chunk_id == 1
    result = await sq.get(REQ)
    assert result is None  # done
    sq.close(REQ)


@pytest.mark.asyncio
async def test_error_after_items():
    """Error signal enqueued after items, raised in order."""
    sq = StreamQueue()
    sq.open(REQ)
    sq.put(REQ, _make_item(chunk_id=0))
    sq.put_error(REQ, ValueError("boom"), from_stage="enc")

    item0 = await sq.get(REQ)
    assert item0.chunk_id == 0
    with pytest.raises(ValueError, match="boom"):
        await sq.get(REQ)
    sq.close(REQ)


@pytest.mark.asyncio
async def test_get_with_source_signal_after_items():
    """get_with_source() delivers signals in order after data items."""
    sq = StreamQueue()
    sq.open(REQ)
    sq.put(REQ, _make_item(chunk_id=0))
    sq.put_done(REQ, from_stage="enc")

    # Drain data
    item0 = await sq.get_with_source(REQ)
    assert isinstance(item0, StreamItem)
    # Done signal
    sig = await sq.get_with_source(REQ)
    assert isinstance(sig, StreamSignal)
    assert sig.is_done is True
    sq.close(REQ)


@pytest.mark.asyncio
async def test_unbounded_put_preserves_all_chunks():
    """Unbounded queue preserves all chunks (no drop-oldest)."""
    sq = StreamQueue()
    sq.open(REQ)
    for i in range(100):
        sq.put(REQ, _make_item(chunk_id=i))

    for i in range(100):
        item = await sq.get(REQ)
        assert item.chunk_id == i
    sq.close(REQ)
