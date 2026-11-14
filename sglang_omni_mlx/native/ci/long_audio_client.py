# SPDX-License-Identifier: Apache-2.0
"""Send long recordings through the server's HTTP and realtime APIs and score them."""

from __future__ import annotations

import base64
import http.client
import io
import json
import math
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rapidfuzz.distance import Levenshtein
from websockets.sync.client import connect

from sglang_omni_mlx.native.ci.long_audio_corpus import SAMPLE_BYTES, SAMPLE_RATE
from sglang_omni_mlx.native.ci.transcript_tokens import tokens
from sglang_omni_mlx.native.tests.native_server import multipart_form

HTTP_TRANSCRIPTION_FIELDS = {
    "stream": "true",
    "include_generation_metadata": "true",
    "stop_on_token_loop": "true",
    "stop_at_end_of_text": "true",
    "audio_layout": "voxt_swift",
}
REALTIME_MESSAGE_FRAMES = SAMPLE_RATE // 10
SESSION_SETUP_TIMEOUT_SECONDS = 10
SOCKET_CLOSE_TIMEOUT_SECONDS = 5
# The token budget of a Voxt Final request, from its Qwen3-ASR tuning.
VOXT_TOKENS_PER_SECOND = 28
VOXT_EXTRA_TOKENS = 64
VOXT_MIN_TOKENS = 256
VOXT_MAX_TOKENS = 8192


@dataclass(frozen=True, kw_only=True)
class GenerationMetadata:
    generated_token_count: int
    language: str | None
    finish_reason: str


@dataclass(frozen=True, kw_only=True)
class HttpTranscription:
    text: str
    wall_seconds: float
    generation_metadata: list[GenerationMetadata]


@dataclass(frozen=True, kw_only=True)
class RealtimeTranscription:
    text: str
    wall_seconds: float
    finalization_seconds: float
    final_segment_count: int
    preview_event_count: int


@dataclass(frozen=True, kw_only=True)
class TranscriptScore:
    metric: Literal["wer_en", "cer_zh"]
    errors: int
    reference_tokens: int
    hypothesis_tokens: int
    deleted_reference_tokens: int
    longest_deleted_reference_token_run: int
    error_rate: float


def score(reference: str, hypothesis: str, language: str) -> TranscriptScore:
    reference_tokens = tokens(reference, language)
    hypothesis_tokens = tokens(hypothesis, language)
    operations = Levenshtein.editops(reference_tokens, hypothesis_tokens)
    deleted_reference_token_run_lengths = [
        block.src_end - block.src_start
        for block in operations.as_opcodes()
        if block.tag == "delete"
    ]
    errors = len(operations)
    return TranscriptScore(
        metric="wer_en" if language == "en" else "cer_zh",
        errors=errors,
        reference_tokens=len(reference_tokens),
        hypothesis_tokens=len(hypothesis_tokens),
        deleted_reference_tokens=sum(deleted_reference_token_run_lengths),
        longest_deleted_reference_token_run=max(
            deleted_reference_token_run_lengths, default=0
        ),
        error_rate=errors / len(reference_tokens),
    )


def voxt_token_budget(audio_seconds: float) -> int:
    budget = math.ceil(audio_seconds * VOXT_TOKENS_PER_SECOND) + VOXT_EXTRA_TOKENS
    return min(max(budget, VOXT_MIN_TOKENS), VOXT_MAX_TOKENS)


def transcribe_http(
    port: int,
    wav: bytes,
    language: str,
    timeout_seconds: float,
    max_new_tokens: int | None = None,
) -> HttpTranscription:
    fields = {"language": language, **HTTP_TRANSCRIPTION_FIELDS}
    if max_new_tokens is not None:
        fields["max_new_tokens"] = str(max_new_tokens)
    else:
        pass
    boundary, body = multipart_form(fields, wav)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_seconds)
    started = time.monotonic()
    try:
        connection.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")
        else:
            pass
        try:
            while line := response.readline():
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("HTTP transcription deadline exceeded")
                elif line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    if event.get("type") == "transcript.text.done":
                        return HttpTranscription(
                            text=event["text"],
                            wall_seconds=time.monotonic() - started,
                            generation_metadata=[
                                GenerationMetadata(**event["generation_metadata"])
                            ],
                        )
                    else:
                        raise RuntimeError(f"Unexpected SSE event: {event}")
                else:
                    pass
        finally:
            response.close()
        raise RuntimeError("Stream ended without a final transcript")
    finally:
        connection.close()


def transcribe_chunks(
    port: int,
    path: Path,
    frame_counts: list[int],
    language: str,
    timeout_seconds: float,
) -> HttpTranscription:
    """Send a recording as consecutive requests, each with its own Voxt budget."""
    started = time.monotonic()
    texts: list[str] = []
    generation_metadata: list[GenerationMetadata] = []
    with wave.open(str(path), "rb") as reader:
        for frame_count in frame_counts:
            chunk = io.BytesIO()
            with wave.open(chunk, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(SAMPLE_BYTES)
                writer.setframerate(SAMPLE_RATE)
                writer.writeframes(reader.readframes(frame_count))
            transcription = transcribe_http(
                port,
                chunk.getvalue(),
                language,
                timeout_seconds,
                voxt_token_budget(frame_count / SAMPLE_RATE),
            )
            texts.append(transcription.text)
            generation_metadata.extend(transcription.generation_metadata)
    return HttpTranscription(
        text=" ".join(texts),
        wall_seconds=time.monotonic() - started,
        generation_metadata=generation_metadata,
    )


def stream_recording(
    port: int,
    path: Path,
    language: str,
    speed: float,
    timeout_seconds: float,
    events_path: Path,
) -> RealtimeTranscription:
    final_segment_ids: set[int] = set()
    previews = 0
    completed = ""
    errors: list[str] = []
    started = time.monotonic()
    with connect(
        f"ws://127.0.0.1:{port}/v1/realtime",
        open_timeout=SESSION_SETUP_TIMEOUT_SECONDS,
        close_timeout=SOCKET_CLOSE_TIMEOUT_SECONDS,
    ) as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": language},
                }
            )
        )
        if (
            json.loads(socket.recv(timeout=SESSION_SETUP_TIMEOUT_SECONDS))["type"]
            != "transcription_session.updated"
        ):
            raise RuntimeError("Session was not accepted")
        else:
            pass

        def receive() -> None:
            nonlocal previews, completed
            try:
                with events_path.open("w") as output:
                    while True:
                        event = json.loads(socket.recv())
                        output.write(
                            json.dumps(
                                {
                                    "elapsed_seconds": time.monotonic() - started,
                                    **event,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        output.flush()
                        if event["type"] == "error":
                            raise RuntimeError(str(event))
                        elif event["type"] == "transcription.completed":
                            completed = event["text"]
                            return
                        elif (
                            event["type"] == "transcription.segment"
                            and event["is_final"]
                        ):
                            final_segment_ids.add(event["segment_id"])
                        else:
                            previews += 1
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                socket.close()

        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()
        with wave.open(str(path), "rb") as reader:
            sent_frames = 0
            send_started = time.monotonic()
            while pcm := reader.readframes(REALTIME_MESSAGE_FRAMES):
                if errors:
                    raise RuntimeError(errors[0])
                else:
                    pass
                socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm).decode(),
                        }
                    )
                )
                sent_frames += len(pcm) // SAMPLE_BYTES
                time.sleep(
                    max(
                        0,
                        send_started
                        + sent_frames / SAMPLE_RATE / speed
                        - time.monotonic(),
                    )
                )
        finished_sending = time.monotonic()
        socket.send(json.dumps({"type": "transcription.done"}))
        receiver.join(timeout=timeout_seconds)
        if receiver.is_alive():
            socket.close()
            receiver.join(timeout=SOCKET_CLOSE_TIMEOUT_SECONDS)
            raise TimeoutError("No completed transcript after the final audio frame")
        elif errors:
            raise RuntimeError(errors[0])
        else:
            finished = time.monotonic()
            return RealtimeTranscription(
                text=completed,
                wall_seconds=finished - started,
                finalization_seconds=finished - finished_sending,
                final_segment_count=len(final_segment_ids),
                preview_event_count=previews,
            )
