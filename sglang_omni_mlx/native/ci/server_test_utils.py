"""HTTP client and lifecycle helpers for real native ASR servers."""

from __future__ import annotations

import http.client
import io
import json
import os
import subprocess
import uuid
import wave
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
MODEL_REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"


def model_directory() -> Path:
    return Path(DATA_ROOT) / "models" / MODEL_REPO.replace("/", "_")


def clip(clip_id: str) -> bytes:
    return (Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav").read_bytes()


def pcm16(clip_id: str) -> bytes:
    with wave.open(io.BytesIO(clip(clip_id))) as reader:
        return reader.readframes(reader.getnframes())


def sse_events(body: bytes) -> list[dict[str, JsonValue] | str]:
    return [
        (
            line[len("data: ") :]
            if line == "data: [DONE]"
            else json.loads(line[len("data: ") :])
        )
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]


def long_wav(seconds: int) -> bytes:
    pcm = pcm16("0344_en_long")
    repeated = pcm * (seconds * 32000 // len(pcm) + 1)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(repeated[: seconds * 32000])
    return output.getvalue()


class Server:
    def __init__(
        self,
        binary: str = "qwen3_asr_server",
        model_kind: str = "qwen3_asr",
        directory: Path | None = None,
        flags: tuple[str, ...] = (),
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / binary),
                "--supervised",
                "--model-kind",
                model_kind,
                "--model-directory",
                str(directory or model_directory()),
                *flags,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.ready = json.loads(self.process.stdout.readline())
        self.port = self.ready.get("port")

    def request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def post_form(self, fields: dict[str, str], wav: bytes | None) -> tuple[int, bytes]:
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        if wav is not None:
            body += (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                    "Content-Type: audio/wav\r\n\r\n"
                ).encode()
                + wav
                + b"\r\n"
            )
        else:
            pass
        body += f"--{boundary}--\r\n".encode()
        return self.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write('{"command": "shutdown"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        else:
            pass
