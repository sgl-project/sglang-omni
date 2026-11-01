# SPDX-License-Identifier: Apache-2.0
"""Clients for the native TTS servers' /v1/audio/speech API, shared by their API
tests: requests, the server's request states, and transcribing returned speech
with qwen3_asr_server.
"""

from __future__ import annotations

import http.client
import io
import json
import re
import socket
import time
import wave

import numpy as np
from test_server_api import Server

ASR_SAMPLE_RATE = 16000


def speak(server: Server, body: dict[str, object]) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=300)
    connection.request("POST", "/v1/audio/speech", body=json.dumps(body))
    response = connection.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def send_speech_request(
    server: Server, body: dict[str, object], receive_buffer_bytes: int | None = None
) -> socket.socket:
    """Sends a speech request on a socket of its own, without reading the answer."""
    encoded_body = json.dumps(body)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # The receive buffer is sized before connecting, so the window follows it.
    if receive_buffer_bytes is not None:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes)
    else:
        pass
    client.settimeout(300)
    client.connect(("127.0.0.1", server.port))
    client.sendall(
        b"POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(encoded_body)}\r\n\r\n{encoded_body}".encode()
    )
    return client


def open_speech_stream(
    server: Server, body: dict[str, object], receive_buffer_bytes: int | None = None
) -> socket.socket:
    """Sends a streamed speech request; returns once its 200 headers arrive."""
    client = send_speech_request(
        server,
        {**body, "response_format": "pcm", "stream": True},
        receive_buffer_bytes,
    )
    received = b""
    while b"\r\n\r\n" not in received:
        chunk = client.recv(65536)
        if not chunk:
            client.close()
            raise ConnectionError(f"stream closed before its headers: {received!r}")
        else:
            received += chunk
    if not received.startswith(b"HTTP/1.1 200 "):
        client.close()
        raise AssertionError(f"stream was refused: {received!r}")
    else:
        pass
    return client


def request_states(server: Server) -> dict[str, int]:
    return json.loads(server.request("GET", "/health")[1])["request_states"]


def wait_until_running(server: Server, timeout_seconds: float) -> None:
    started_at = time.monotonic()
    while request_states(server) != {"running": 1}:
        assert time.monotonic() - started_at < timeout_seconds
        time.sleep(0.05)


def wait_until_idle(server: Server, timeout_seconds: float) -> None:
    started_at = time.monotonic()
    while request_states(server):
        assert (
            time.monotonic() - started_at < timeout_seconds
        ), "generation kept running after it was cancelled"
        time.sleep(0.05)


def transcript_words(asr_server: Server, pcm: bytes, sample_rate: int) -> list[str]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    resampled_count = len(samples) * ASR_SAMPLE_RATE // sample_rate
    # Band-limited resampling: keep the spectrum below the new Nyquist rate.
    spectrum = np.fft.rfft(samples)[: resampled_count // 2 + 1]
    resampled = np.fft.irfft(spectrum, resampled_count) * resampled_count / len(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(ASR_SAMPLE_RATE)
        writer.writeframes(np.clip(resampled, -32768, 32767).astype("<i2").tobytes())
    status, body = asr_server.post_form({"language": "en"}, buffer.getvalue())
    assert status == 200
    return words(json.loads(body)["text"])


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    distances = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        diagonal, distances[0] = distances[0], row
        for column, heard in enumerate(hypothesis, 1):
            diagonal, distances[column] = distances[column], min(
                distances[column] + 1,
                distances[column - 1] + 1,
                diagonal + (expected != heard),
            )
    return distances[-1]
