# SPDX-License-Identifier: Apache-2.0
"""Cancelling part of a burst leaves the rest correct and the server serving."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from benchmarks.offload.check_offload_safety import (
    SafetyPlan,
    check_server,
    output_mismatches,
    run_burst,
)

AUDIO = b"\x01\x02" * 64
URL = "http://server.test/v1/audio/speech"
BODY: dict[str, object] = {"model": "tts", "input": "hi", "response_format": "pcm"}


def plan(**overrides: object) -> SafetyPlan:
    fields = {
        "requests": 3,
        "concurrency": 2,
        "cancelled_requests": 1,
        "cancel_after_s": 0.01,
    }
    return SafetyPlan(**{**fields, **overrides})


def transport(
    *, chunk_delay_s: float = 0.0, audio: bytes = AUDIO
) -> httpx.MockTransport:
    """A server that streams eight audio chunks per request, ending on its own."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})

        async def stream() -> object:
            for _ in range(8):
                if chunk_delay_s:
                    await asyncio.sleep(chunk_delay_s)
                yield audio

        return httpx.Response(200, content=stream(), headers={"X-Sample-Rate": "24000"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_cancelled_tail_leaves_the_finished_requests_and_a_live_server() -> (
    None
):
    async with httpx.AsyncClient(transport=transport(chunk_delay_s=0.05)) as client:
        report = await check_server(
            client, URL, [dict(BODY) for _ in range(3)], plan(cancel_after_s=0.02)
        )
    assert report.planned == 3
    assert report.completed + report.cancelled == 3
    assert report.cancelled >= 1
    assert report.server_alive and report.is_safe


@pytest.mark.asyncio
async def test_a_burst_without_cancels_completes_every_request() -> None:
    async with httpx.AsyncClient(transport=transport()) as client:
        outcomes = await run_burst(
            client,
            URL,
            [dict(BODY) for _ in range(3)],
            plan(cancelled_requests=0, cancel_after_s=0.0),
        )
    assert list(outcomes.values()) == [AUDIO * 8] * 3


def test_audio_that_differs_is_named_and_equal_audio_is_not() -> None:
    baseline = {"req-0": b"a", "req-1": b"b"}
    assert output_mismatches(baseline, {"req-0": b"a", "req-1": b"b"}) == []
    assert output_mismatches(baseline, {"req-0": b"a", "req-1": b"c"}) == ["req-1"]
    assert output_mismatches(baseline, {"req-0": b"a"}) == ["req-1"]


def test_a_plan_that_cannot_be_sent_is_refused() -> None:
    with pytest.raises(ValueError):
        plan(requests=0)
    with pytest.raises(ValueError):
        plan(requests=2, concurrency=4)
    with pytest.raises(ValueError):
        plan(cancelled_requests=1, cancel_after_s=0)
    with pytest.raises(ValueError):
        plan(requests=2, cancelled_requests=3)
