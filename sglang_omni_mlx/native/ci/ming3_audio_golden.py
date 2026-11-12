# SPDX-License-Identifier: Apache-2.0
"""Runs Ming3 audio's frozen corpus through the native chat service."""

from __future__ import annotations

import base64
import http.client
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from check_golden import run


class ChatSettings(TypedDict):
    system_prompt: str
    prompt: str
    temperature: int
    max_completion_tokens: int


class Golden(TypedDict):
    model: str
    context_length: int
    request: ChatSettings


class CorpusClip(TypedDict):
    stratum: str


class Transcript(TypedDict):
    text: str
    language: str
    generated_token_count: int
    finish_reason: str


@contextmanager
def running_chat(
    runtime_bin: Path,
    model_directory: Path,
    model_name: str,
    log_directory: Path,
    extra_arguments: tuple[str, ...] = (),
) -> Iterator[tuple[int, subprocess.Popen[str]]]:
    with (log_directory / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [
                str(runtime_bin / "ming3_audio_server"),
                "--supervised",
                "--model-directory",
                str(model_directory),
                "--model-name",
                model_name,
                *extra_arguments,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            ready = process.stdout.readline()
            if not ready:
                log.seek(0)
                raise RuntimeError(log.read())
            else:
                event = json.loads(ready)
                if event["event"] != "ready":
                    raise RuntimeError(ready)
                else:
                    yield event["port"], process
        finally:
            if process.poll() is None:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("server did not shut down") from None
            else:
                pass
            if process.returncode != 0:
                log.seek(0)
                raise RuntimeError(log.read())
            else:
                pass


def request(
    port: int, body: str = "", method: str = "POST", path: str = "/v1/chat/completions"
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request(
            method,
            path,
            body=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def transcribe(
    runtime_bin: Path,
    data_root: Path,
    golden: Golden,
    manifest: dict[str, CorpusClip],
) -> dict[str, Transcript]:
    settings = golden["request"]
    results: dict[str, Transcript] = {}
    with tempfile.TemporaryDirectory(prefix="ming3-audio-golden-") as log_directory:
        with running_chat(
            runtime_bin,
            data_root / "models" / golden["model"].replace("/", "_"),
            golden["model"],
            Path(log_directory),
            ("--context-length", str(golden["context_length"])),
        ) as (port, _):
            for clip_id, clip in manifest.items():
                audio = (
                    data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
                ).read_bytes()
                content = [
                    {"type": "text", "text": settings["prompt"]},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "format": "wav",
                            "data": base64.b64encode(audio).decode(),
                        },
                    },
                ]
                status, response = request(
                    port,
                    json.dumps(
                        {
                            "model": golden["model"],
                            "messages": [
                                {
                                    "role": "system",
                                    "content": settings["system_prompt"],
                                },
                                {"role": "user", "content": content},
                            ],
                            "temperature": settings["temperature"],
                            "max_completion_tokens": settings["max_completion_tokens"],
                        }
                    ),
                )
                reply = json.loads(response)
                if status != 200:
                    raise RuntimeError(f"{clip_id}: HTTP {status}: {reply}")
                else:
                    choice = reply["choices"][0]
                if choice["finish_reason"] == "length" and clip["stratum"] not in (
                    "silence",
                    "noise",
                ):
                    raise RuntimeError(
                        f"{clip_id}: speech transcription reached the token limit"
                    )
                else:
                    pass
                language, separator, text = choice["message"]["content"].partition("\t")
                if not separator:
                    text, language = language, ""
                else:
                    pass
                results[clip_id] = {
                    "text": text,
                    "language": language,
                    "generated_token_count": reply["usage"]["completion_tokens"],
                    "finish_reason": choice["finish_reason"],
                }
                print(
                    f"{len(results)}/{len(manifest)} {clip_id}",
                    file=sys.stderr,
                    flush=True,
                )
    return results


if __name__ == "__main__":
    run(transcribe)
