# SPDX-License-Identifier: Apache-2.0

import msgpack
import numpy as np
import pytest

from sglang_omni.client.types import GenerateChunk
from sglang_omni.realtime.backend import OmniResponseBackend, TurnContext


class _FakeClient:
    def __init__(self) -> None:
        self.cancelled: str | None = None
        self.last_request = None

    async def generate(self, request, request_id: str):
        self.last_request = request
        yield GenerateChunk(request_id=request_id, text="hello")
        yield GenerateChunk(
            request_id=request_id,
            modality="audio",
            audio_data=np.array([0.1, -0.1], dtype=np.float32),
            sample_rate=24000,
        )
        yield GenerateChunk(request_id=request_id, finish_reason="stop")

    async def abort(self, request_id: str) -> None:
        self.cancelled = request_id


class _SnapshotTextClient:
    async def generate(self, request, request_id: str):
        del request
        yield GenerateChunk(request_id=request_id, text="Hi")
        yield GenerateChunk(request_id=request_id, text="Hi there")
        yield GenerateChunk(request_id=request_id, text="Hi there")
        yield GenerateChunk(request_id=request_id, finish_reason="stop")

    async def abort(self, request_id: str) -> None:
        del request_id


class _TerminalPayloadClient:
    async def generate(self, request, request_id: str):
        del request
        yield GenerateChunk(
            request_id=request_id,
            text="terminal text",
            finish_reason="stop",
        )

    async def abort(self, request_id: str) -> None:
        del request_id


@pytest.mark.asyncio
async def test_omni_response_backend_normalizes_turn_output():
    client = _FakeClient()
    backend = OmniResponseBackend(
        client=client,
        model="qwen3-omni",
        max_new_tokens=32,
        output_modalities=("text", "audio"),
    )
    turn = TurnContext(
        session_id="session-1",
        history=[{"role": "assistant", "content": "previous"}],
        instructions="be concise",
        user_text="hi there",
        user_audio=np.zeros(32, dtype=np.float32),
        user_audio_sample_rate=16000,
        recent_video=None,
        recent_video_fps=None,
    )

    events = [event async for event in backend.stream_response(turn)]

    assert [event.type for event in events] == [
        "response_started",
        "text_delta",
        "audio_chunk",
        "done",
    ]
    assert events[1].text == "hello"
    assert events[2].sample_rate == 24000
    assert events[3].finish_reason == "stop"

    request = client.last_request
    assert request is not None
    assert request.model == "qwen3-omni"
    assert request.metadata["audio_target_sr"] == 16000
    assert len(request.metadata["audios"]) == 1
    audio_payload = request.metadata["audios"][0]
    assert isinstance(audio_payload["audio_waveform"], bytes)
    assert audio_payload["audio_waveform_dtype"] == "float32"
    assert audio_payload["audio_waveform_shape"] == [32]
    assert request.stage_params == {"talker_ar": {"max_new_tokens": 32}}
    msgpack.packb(request.metadata, use_bin_type=True)
    assert len(request.messages) == 3
    assert request.messages[-1].content == "hi there"

    await backend.cancel(events[0].response_id)
    assert client.cancelled == events[0].response_id


@pytest.mark.asyncio
async def test_omni_response_backend_coerces_snapshot_text_to_deltas():
    backend = OmniResponseBackend(
        client=_SnapshotTextClient(),
        model="qwen3-omni",
        output_modalities=("text",),
    )
    turn = TurnContext(
        session_id="session-1",
        history=[],
        instructions=None,
        user_text="hello",
        user_audio=None,
        user_audio_sample_rate=None,
        recent_video=None,
        recent_video_fps=None,
    )

    events = [event async for event in backend.stream_response(turn)]
    text_events = [event for event in events if event.type == "text_delta"]

    assert [event.text for event in text_events] == ["Hi", " there"]
    assert "".join(event.text for event in text_events) == "Hi there"


@pytest.mark.asyncio
async def test_omni_response_backend_keeps_terminal_payload_before_done():
    backend = OmniResponseBackend(
        client=_TerminalPayloadClient(),
        model="qwen3-omni",
        output_modalities=("text",),
    )
    turn = TurnContext(
        session_id="session-1",
        history=[],
        instructions=None,
        user_text="hello",
        user_audio=None,
        user_audio_sample_rate=None,
        recent_video=None,
        recent_video_fps=None,
    )

    events = [event async for event in backend.stream_response(turn)]

    assert [event.type for event in events] == [
        "response_started",
        "text_delta",
        "done",
    ]
    assert events[1].text == "terminal text"
    assert events[2].finish_reason == "stop"
