from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval.benchmark_omni_seedtts import make_send_fn


def _sample() -> SampleInput:
    return SampleInput(
        sample_id="s1",
        ref_text="reference text",
        ref_audio="ref.wav",
        target_text="hello world",
    )


async def _call_send_fn(url: str, save_audio_dir: str):
    send_fn = make_send_fn(
        "qwen3-omni",
        url,
        lang="en",
        voice_clone=False,
        speaker="Ethan",
        max_tokens=8,
        temperature=0.7,
        stream=False,
        save_audio_dir=save_audio_dir,
    )
    async with aiohttp.ClientSession() as session:
        return await send_fn(session, _sample())


@pytest.mark.asyncio
async def test_send_fn_returns_failed_result_on_http_500(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="deterministic bad input")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        # Note (Jiaxin Deng): before the fix this raised RuntimeError out of
        # send_fn, killing every in-flight client task of a benchmark run.
        result = await _call_send_fn(
            str(server.make_url("/v1/chat/completions")), str(tmp_path)
        )
    finally:
        await server.close()

    assert result.is_success is False
    assert "HTTP 500" in result.error
    assert result.latency_s > 0


@pytest.mark.asyncio
async def test_send_fn_returns_failed_result_on_missing_audio(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"choices": [{"message": {"content": "text only"}}], "usage": {}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await _call_send_fn(
            str(server.make_url("/v1/chat/completions")), str(tmp_path)
        )
    finally:
        await server.close()

    assert result.is_success is False
    assert "No audio in response" in result.error
