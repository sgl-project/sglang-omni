# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
import wave

from benchmarks.tasks.tts import (
    _build_streaming_wav_bytes,
    _collect_chat_streaming_audio,
)


def _wav_b64(samples: bytes, *, sample_rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


def test_collect_chat_streaming_audio_chunks_and_usage() -> None:
    pcm_chunks: list[bytes] = []
    stream_format = None
    usage: dict = {}

    stream_format = _collect_chat_streaming_audio(
        _sse_event(
            {
                "choices": [
                    {
                        "delta": {
                            "audio": {"id": "audio-1", "data": _wav_b64(b"\x01\x00")}
                        }
                    }
                ]
            }
        ),
        pcm_chunks,
        stream_format,
        usage,
    )
    stream_format = _collect_chat_streaming_audio(
        _sse_event(
            {
                "choices": [
                    {
                        "delta": {
                            "audio": {"id": "audio-1", "data": _wav_b64(b"\x02\x00")}
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }
        ),
        pcm_chunks,
        stream_format,
        usage,
    )

    assert stream_format == (16000, 1, 2)
    assert pcm_chunks == [b"\x01\x00", b"\x02\x00"]
    assert usage["completion_tokens"] == 5

    wav_bytes = _build_streaming_wav_bytes(pcm_chunks, stream_format)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.readframes(wf.getnframes()) == b"\x01\x00\x02\x00"
