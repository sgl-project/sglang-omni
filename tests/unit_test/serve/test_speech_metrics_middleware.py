# SPDX-License-Identifier: Apache-2.0
"""Generated audio HTTP measurement tests."""

import asyncio
import base64
import json
import unittest

import pytest
from prometheus_client import generate_latest
from starlette.types import Message, Receive, Scope, Send

from sglang_omni.serve.request_metrics import RequestMetrics
from sglang_omni.serve.speech_metrics import SpeechMetricsMiddleware


class SpeechMetricsMiddlewareTest(unittest.TestCase):
    def test_pcm_chunks_are_measured_through_response_completion(self) -> None:
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            scope["state"]["audio_streaming"] = True
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send(
                {"type": "http.response.body", "body": bytes(4800), "more_body": True}
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        metrics = RequestMetrics("test")
        sent: list[Message] = []

        async def send(message: Message) -> None:
            sent.append(message)

        async def receive() -> Message:
            return {"type": "http.request", "body": b""}

        asyncio.run(
            SpeechMetricsMiddleware(app, metrics.audio_metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = generate_latest(metrics.registry).decode()
        self.assertEqual(len(sent), 3)
        self.assertIn("sglang_omni:audio_duration_s_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_chunk_interval_s_count 1.0", snapshot)

    def test_failed_send_does_not_record_completed_stream(self) -> None:
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body":
                raise ConnectionError("client disconnected")
            else:
                pass

        async def receive() -> Message:
            return {"type": "http.request", "body": b""}

        metrics = RequestMetrics("test")
        with self.assertRaises(ConnectionError):
            asyncio.run(
                SpeechMetricsMiddleware(app, metrics.audio_metrics)(
                    {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                    receive,
                    send,
                )
            )
        snapshot = generate_latest(metrics.registry).decode()
        self.assertIn("sglang_omni:audio_duration_s_count 0.0", snapshot)

    def test_nonstream_audio_reports_duration_and_engine_rtf(self) -> None:
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            scope["state"]["audio_duration_s"] = 0.5
            scope["state"]["audio_generation_s"] = 0.25
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"audio/wav")],
                }
            )
            await send({"type": "http.response.body", "body": bytes(10)})

        async def send(message: Message) -> None:
            return None

        async def receive() -> Message:
            return {"type": "http.request", "body": b""}

        metrics = RequestMetrics("test")
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics.audio_metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = generate_latest(metrics.registry).decode()
        self.assertIn("sglang_omni:audio_duration_s_sum 0.5", snapshot)
        self.assertIn("sglang_omni:audio_rtf_sum 0.5", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s_count 0.0", snapshot)
        self.assertNotIn("sglang:generation_tokens_total{", snapshot)

    def test_nonstream_pcm_does_not_emit_ttfp_or_continuity(self) -> None:
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            scope["state"]["audio_duration_s"] = 0.1
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        async def send(message: Message) -> None:
            return None

        async def receive() -> Message:
            return {"type": "http.request", "body": b""}

        metrics = RequestMetrics("test")
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics.audio_metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = generate_latest(metrics.registry).decode()
        self.assertIn("sglang_omni:audio_duration_s_sum 0.1", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s_count 0.0", snapshot)
        self.assertIn("sglang_omni:audio_underrun_s_count 0.0", snapshot)


@pytest.mark.parametrize("completed", [False, True])
def test_sse_completion_requires_audio_done(completed: bool) -> None:
    """An SSE error after audio must not report a completed generation."""
    metrics = RequestMetrics("test")
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        scope["state"]["audio_streaming"] = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"x-sample-rate", b"24000"),
                    (b"x-channels", b"1"),
                    (b"x-bit-depth", b"16"),
                ],
            }
        )
        delta = {
            "type": "speech.audio.delta",
            "audio": base64.b64encode(bytes(4800)).decode("ascii"),
        }
        await send(
            {
                "type": "http.response.body",
                "body": f"data: {json.dumps(delta)}\n\n".encode(),
                "more_body": True,
            }
        )
        terminal = {"type": "speech.audio.done" if completed else "error"}
        await send(
            {
                "type": "http.response.body",
                "body": f"data: {json.dumps(terminal)}\n\n".encode(),
            }
        )

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(
        SpeechMetricsMiddleware(app, metrics.audio_metrics)(
            {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
            receive,
            send,
        )
    )

    assert len(sent) == 3
    assert metrics.registry.get_sample_value("sglang_omni:audio_ttfp_s_count") == 1
    assert metrics.registry.get_sample_value(
        "sglang_omni:audio_duration_s_count"
    ) == int(completed)
    assert metrics.registry.get_sample_value(
        "sglang_omni:audio_duration_s_sum"
    ) == pytest.approx(0.1 if completed else 0)
