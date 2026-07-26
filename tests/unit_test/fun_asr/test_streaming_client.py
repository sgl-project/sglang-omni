# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.proto import CompleteMessage, StreamMessage


class _FakeCoordinator:
    async def stream(self, request_id, omni_request):
        del omni_request
        yield StreamMessage(
            request_id=request_id,
            from_stage="asr",
            chunk={"text": "A", "modality": "text", "stage_name": "asr"},
            stage_name="asr",
            modality="text",
        )
        yield StreamMessage(
            request_id=request_id,
            from_stage="asr",
            chunk={"text": "B", "modality": "text", "stage_name": "asr"},
            stage_name="asr",
            modality="text",
        )
        yield CompleteMessage(
            request_id=request_id,
            from_stage="asr",
            success=True,
            result={
                "text": "AB",
                "finish_reason": "stop",
                "modality": "text",
            },
        )


def test_completion_stream_does_not_repeat_terminal_fun_asr_text() -> None:
    client = Client(coordinator=_FakeCoordinator())

    async def collect():
        return [
            chunk
            async for chunk in client.completion_stream(
                GenerateRequest(prompt="ignored", stream=True),
                request_id="req-fun-asr",
            )
        ]

    chunks = asyncio.run(collect())

    assert "".join(chunk.text or "" for chunk in chunks) == "AB"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].text in (None, "")
