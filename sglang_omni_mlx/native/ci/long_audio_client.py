# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3-ASR long recordings through its public HTTP and realtime APIs."""

from __future__ import annotations

import base64
import ctypes
import http.client
import io
import json
import math
import re
import subprocess
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import TypedDict

from rapidfuzz.distance import Levenshtein
from websockets.sync.client import connect

from sglang_omni_mlx.native.ci.long_audio_corpus import SAMPLE_BYTES, SAMPLE_RATE

HTTP_TRANSCRIPTION_FIELDS = {
    "stream": "true",
    "include_generation_metadata": "true",
    "stop_on_token_loop": "true",
    "stop_at_end_of_text": "true",
    "audio_layout": "voxt_swift",
}
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.25
RUSAGE_INFO_V4 = 4
# The token budget of a Voxt Final request, from its Qwen3-ASR tuning.
VOXT_TOKENS_PER_SECOND = 28
VOXT_EXTRA_TOKENS = 64
VOXT_MIN_TOKENS = 256
VOXT_MAX_TOKENS = 8192


class GenerationMetadata(TypedDict):
    generated_token_count: int
    language: str
    finish_reason: str


class Measurement(TypedDict, total=False):
    text: str
    type: str
    generation_metadata: GenerationMetadata
    chunk_generation_metadata: list[GenerationMetadata]
    wall_seconds: float
    finalization_seconds: float
    final_segments: int
    preview_events: int
    speed: float


class ProcessUsage(ctypes.Structure):
    """The layout of rusage_info_v4 in sys/resource.h."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "idle_wakeups",
            "interrupt_wakeups",
            "pageins",
            "wired_bytes",
            "resident_bytes",
            "footprint_bytes",
            "start_time",
            "exit_time",
            "child_user_time",
            "child_system_time",
            "child_idle_wakeups",
            "child_interrupt_wakeups",
            "child_pageins",
            "child_elapsed_time",
            "disk_bytes_read",
            "disk_bytes_written",
            "cpu_time_qos_default",
            "cpu_time_qos_maintenance",
            "cpu_time_qos_background",
            "cpu_time_qos_utility",
            "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive",
            "billed_system_time",
            "serviced_system_time",
            "logical_writes",
            "peak_footprint_bytes",
            "instructions",
            "cycles",
            "billed_energy",
            "serviced_energy",
            "interval_peak_footprint_bytes",
            "runnable_time",
        )
    ]


class MemorySampler:
    def __init__(
        self,
        process: subprocess.Popen[str],
        destination: Path,
        limit_bytes: int,
        limit_seconds: float,
    ) -> None:
        self.process = process
        self.destination = destination
        self.limit_bytes = limit_bytes
        self.limit_seconds = limit_seconds
        self.samples: list[dict[str, float | int | str]] = []
        self.phase = "startup"
        self.exceeded = False
        self.error = ""
        self.done = threading.Event()
        self.library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.library.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.proc_pid_rusage.restype = ctypes.c_int
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def stop_on_error(self, message: str) -> None:
        self.error = message
        if self.process.poll() is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        else:
            pass

    def sample(self) -> None:
        started = time.monotonic()
        try:
            with self.destination.open("w") as output:
                while not self.done.is_set() and self.process.poll() is None:
                    usage = ProcessUsage()
                    if (
                        self.library.proc_pid_rusage(
                            self.process.pid, RUSAGE_INFO_V4, ctypes.byref(usage)
                        )
                        != 0
                    ):
                        self.stop_on_error(
                            f"proc_pid_rusage failed: errno {ctypes.get_errno()}"
                        )
                        return
                    else:
                        row = {
                            "phase": self.phase,
                            "elapsed_seconds": time.monotonic() - started,
                            "footprint_bytes": usage.footprint_bytes,
                            "peak_footprint_bytes": usage.peak_footprint_bytes,
                            "resident_bytes": usage.resident_bytes,
                        }
                        self.samples.append(row)
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                    if time.monotonic() - started > self.limit_seconds:
                        self.stop_on_error("Case wall-clock deadline exceeded")
                        return
                    elif usage.peak_footprint_bytes > self.limit_bytes:
                        self.exceeded = True
                        self.process.kill()
                        return
                    else:
                        self.done.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)
        except OSError as error:
            self.stop_on_error(f"Memory trace failed: {error}")


def score(
    reference: str, hypothesis: str, language: str
) -> dict[str, float | int | str]:
    sequences = []
    for text in (reference, hypothesis):
        text = unicodedata.normalize("NFKC", text).lower()
        if language == "en":
            words = re.sub(r"[^a-z0-9' ]+", " ", text).split()
            sequences.append([word.strip("'") for word in words if word.strip("'")])
        else:
            sequences.append(re.findall(r"[㐀-鿿豈-﫿]|[a-z0-9']+", text))
    reference_tokens, hypothesis_tokens = sequences
    if not reference_tokens:
        raise ValueError("Cannot score an empty reference")
    else:
        errors = Levenshtein.distance(reference_tokens, hypothesis_tokens)
        return {
            "metric": "wer_en" if language == "en" else "cer_zh",
            "errors": errors,
            "reference_tokens": len(reference_tokens),
            "hypothesis_tokens": len(hypothesis_tokens),
            "error_rate": errors / len(reference_tokens),
        }


def voxt_token_budget(audio_seconds: float) -> int:
    budget = math.ceil(audio_seconds * VOXT_TOKENS_PER_SECOND) + VOXT_EXTRA_TOKENS
    return min(max(budget, VOXT_MIN_TOKENS), VOXT_MAX_TOKENS)


def transcribe_http(
    port: int,
    wav: bytes,
    language: str,
    timeout_seconds: float,
    max_new_tokens: int | None = None,
) -> Measurement:
    boundary = "omni-long-audio-benchmark"
    fields = {"language": language, **HTTP_TRANSCRIPTION_FIELDS}
    if max_new_tokens is not None:
        fields["max_new_tokens"] = str(max_new_tokens)
    else:
        pass
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    )
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + wav
        + f"\r\n--{boundary}--\r\n".encode()
    )
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
                elif line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    event = json.loads(line[6:])
                    if event.get("type") == "transcript.text.done":
                        return {**event, "wall_seconds": time.monotonic() - started}
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
) -> Measurement:
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
            measurement = transcribe_http(
                port,
                chunk.getvalue(),
                language,
                timeout_seconds,
                voxt_token_budget(frame_count / SAMPLE_RATE),
            )
            texts.append(measurement["text"])
            generation_metadata.append(measurement["generation_metadata"])
    return {
        "text": " ".join(texts),
        "chunk_generation_metadata": generation_metadata,
        "wall_seconds": time.monotonic() - started,
    }


def stream_recording(
    port: int,
    path: Path,
    language: str,
    speed: float,
    timeout_seconds: float,
    events_path: Path,
) -> Measurement:
    finals: dict[int, str] = {}
    previews = 0
    completed = ""
    errors: list[str] = []
    started = time.monotonic()
    with connect(
        f"ws://127.0.0.1:{port}/v1/realtime", open_timeout=10, close_timeout=5
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
            json.loads(socket.recv(timeout=10))["type"]
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
                            finals[event["segment_id"]] = event["text"]
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
            while pcm := reader.readframes(SAMPLE_RATE // 10):
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
            receiver.join(timeout=10)
            raise TimeoutError("No completed transcript after the final audio frame")
        elif errors:
            raise RuntimeError(errors[0])
        else:
            return {
                "text": completed,
                "wall_seconds": time.monotonic() - started,
                "finalization_seconds": time.monotonic() - finished_sending,
                "final_segments": len(finals),
                "preview_events": previews,
                "speed": speed,
            }
