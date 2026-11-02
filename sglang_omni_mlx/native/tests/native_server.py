# SPDX-License-Identifier: Apache-2.0
"""A native server started the way Voxt starts it, and the requests the tests send it."""

from __future__ import annotations

import http.client
import io
import json
import os
import select
import socket
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

EVENT_TIMEOUT_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 30
EXIT_TIMEOUT_SECONDS = 10
STATE_POLL_SECONDS = 0.02
VOXT_SAMPLE_RATE = 16000


@dataclass(frozen=True, kw_only=True)
class HttpResponse:
    status: int
    headers: http.client.HTTPMessage
    body: bytes

    def json(self) -> dict[str, object]:
        return json.loads(self.body)


class SupervisedServer:
    """A server binary run with --supervised, read through its stdout events."""

    def __init__(
        self,
        binary: Path,
        model_kind: str,
        model_directory: Path,
        flags: tuple[str, ...] = (),
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(binary),
                "--supervised",
                "--model-kind",
                model_kind,
                "--model-directory",
                str(model_directory),
                *flags,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self.first_event = self.read_event()

    @property
    def port(self) -> int:
        return self.first_event["port"]

    def read_event(self) -> dict[str, object]:
        """The next stdout line, waiting at most EVENT_TIMEOUT_SECONDS for it."""
        line = b""
        while not line.endswith(b"\n"):
            readable, _, _ = select.select(
                [self.process.stdout], [], [], EVENT_TIMEOUT_SECONDS
            )
            if not readable:
                raise TimeoutError(f"no event after {line!r}")
            else:
                pass
            byte = os.read(self.process.stdout.fileno(), 1)
            if not byte:
                raise EOFError(f"stdout closed after {line!r}")
            else:
                pass
            line += byte
        return json.loads(line)

    def remaining_stdout(self) -> bytes:
        return self.process.stdout.read()

    def send_line(self, line: str) -> None:
        self.process.stdin.write(f"{line}\n".encode())

    def wait(self) -> int:
        return self.process.wait(timeout=EXIT_TIMEOUT_SECONDS)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.send_line('{"command": "shutdown"}')
            self.wait()
        else:
            pass

    def request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=REQUEST_TIMEOUT_SECONDS
        )
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return HttpResponse(
            status=response.status, headers=response.headers, body=response.read()
        )

    def post_form(self, fields: dict[str, str], wav: bytes | None) -> HttpResponse:
        boundary, body = multipart_form(fields, wav)
        return self.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def post_speech(self, request: dict[str, object]) -> HttpResponse:
        return self.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request).encode(),
            {"Content-Type": "application/json"},
        )

    def open_request(self, path: str, content_type: str, body: bytes) -> socket.socket:
        """A request whose client stays connected, without reading its answer."""
        client = socket.create_connection(
            ("127.0.0.1", self.port), timeout=REQUEST_TIMEOUT_SECONDS
        )
        client.sendall(
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        return client

    def request_states(self) -> dict[str, int]:
        return self.request("GET", "/health").json()["request_states"]

    def wait_for_request_states(self, expected: dict[str, int]) -> None:
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        while self.request_states() != expected:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"request states {self.request_states()}, not {expected}"
                )
            else:
                time.sleep(STATE_POLL_SECONDS)


def multipart_form(fields: dict[str, str], wav: bytes | None) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    if wav is not None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n"
        )
    else:
        pass
    return boundary, b"".join(parts) + f"--{boundary}--\r\n".encode()


def silent_wav(
    frame_count: int, sample_rate: int = VOXT_SAMPLE_RATE, channel_count: int = 1
) -> bytes:
    """16-bit PCM silence in a WAV file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channel_count)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\0\0" * frame_count * channel_count)
    return output.getvalue()


def sse_payloads(body: bytes) -> list[str]:
    """The data payloads of an event stream, without its heartbeat comments."""
    return [
        line.removeprefix("data: ")
        for line in body.decode().split("\n")
        if line.startswith("data: ")
    ]
