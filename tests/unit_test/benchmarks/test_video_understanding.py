# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from benchmarks.tasks.video_understanding import make_video_send_fn

_TALKER_PREFILL_USER_CONTEXT_PARAM = "talker_prefill_user_context"


class _FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "Answer: A"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }


class _FakeSession:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, api_url: str, *, json: dict):
        del api_url
        self.payload = json
        return _FakeResponse()


def test_video_send_fn_merges_extra_request_params() -> None:
    async def _run() -> dict:
        session = _FakeSession()
        send_fn = make_video_send_fn(
            "qwen3-omni",
            "http://localhost/v1/chat/completions",
            video_min_pixels=6_272,
            video_max_pixels=6_272,
            extra_request_params={_TALKER_PREFILL_USER_CONTEXT_PARAM: False},
        )
        result = await send_fn(
            session,
            SimpleNamespace(
                sample_id="sample-1",
                prompt="Question?",
                video_path="/tmp/video.mp4",
            ),
        )
        assert result.is_success is True
        assert session.payload is not None
        return session.payload

    payload = asyncio.run(_run())
    assert payload["video_min_pixels"] == 6_272
    assert payload["video_max_pixels"] == 6_272
    assert payload[_TALKER_PREFILL_USER_CONTEXT_PARAM] is False
