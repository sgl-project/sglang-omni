# SPDX-License-Identifier: Apache-2.0
"""Measure generated audio as it leaves the HTTP server."""

import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang_omni.metrics.runtime import RuntimeMetrics


class SpeechMetricsMiddleware:
    """Record timing and continuity for generated speech responses."""

    def __init__(self, app: ASGIApp, metrics: RuntimeMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] != "/v1/audio/speech"
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})
        self.metrics.record(
            "audio_response_start", request_id, "api", time.perf_counter_ns()
        )
        bytes_per_second = 0
        is_audio_response = False
        succeeded = False

        async def send_audio(message: Message) -> None:
            nonlocal bytes_per_second, is_audio_response, succeeded
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"")
                is_audio_response = message["status"] < 400 and content_type.startswith(
                    b"audio/"
                )
                if (
                    is_audio_response
                    and scope["state"].get("audio_streaming", False)
                    and content_type.startswith(b"audio/pcm")
                ):
                    sample_rate = int(headers[b"x-sample-rate"])
                    channels = int(headers[b"x-channels"])
                    bit_depth = int(headers[b"x-bit-depth"])
                    bytes_per_second = sample_rate * channels * bit_depth // 8
            elif message["type"] == "http.response.body" and bytes_per_second:
                body = message.get("body", b"")
                if body:
                    self.metrics.record(
                        "audio_chunk",
                        request_id,
                        "api",
                        time.perf_counter_ns(),
                        {"duration_s": len(body) / bytes_per_second},
                    )
            await send(message)
            if message["type"] == "http.response.body" and is_audio_response:
                if not message.get("more_body", False):
                    succeeded = True

        try:
            await self.app(scope, receive, send_audio)
        finally:
            if succeeded:
                state = scope["state"]
                details = {
                    name: state[name]
                    for name in ("audio_duration_s", "audio_generation_s")
                    if name in state
                }
                details["is_streaming"] = state.get("audio_streaming", False)
                self.metrics.record(
                    "audio_response_done",
                    request_id,
                    "api",
                    time.perf_counter_ns(),
                    details,
                )
            else:
                self.metrics.record(
                    "audio_response_aborted", request_id, "api", time.perf_counter_ns()
                )
