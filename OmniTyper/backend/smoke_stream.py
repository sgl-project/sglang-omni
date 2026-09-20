#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify real MLX hypotheses arrive while PCM is still being sent, without an LLM."""

import base64
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# Note (Jiaxin Deng): PYTHONSAFEPATH omits the script directory needed for sibling imports.
BACKEND_DIRECTORY = str(Path(__file__).resolve().parent)
if BACKEND_DIRECTORY not in sys.path:
    sys.path.insert(0, BACKEND_DIRECTORY)

from server import NativeASRServer
from websockets.sync.client import connect


def main() -> None:
    server = NativeASRServer()
    try:
        server.start(lambda message: print(message, flush=True))
        with tempfile.TemporaryDirectory(prefix="omnityper-stream-") as directory:
            source = Path(directory) / "speech.aiff"
            audio = Path(directory) / "speech.wav"
            subprocess.run(
                [
                    "/usr/bin/say",
                    "-v",
                    "Samantha",
                    "-o",
                    str(source),
                    " ".join(
                        [
                            "The quick brown fox jumps over the lazy dog.",
                            "Please send the report tomorrow.",
                            "We are testing live speech recognition.",
                            "Words should appear on the screen while I am still speaking.",
                            "The final transcript should include this last sentence.",
                        ]
                    ),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    str(source),
                    str(audio),
                ],
                check=True,
            )
            with wave.open(str(audio)) as recording:
                pcm = recording.readframes(recording.getnframes())
            url = (
                server.url.replace("http://", "ws://")
                + "/v1/realtime?intent=transcription"
            )
            with connect(
                url, proxy=None, max_size=256 * 1024, open_timeout=30
            ) as socket:
                socket.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"language": "en", "turn_detection": None},
                        }
                    )
                )
                while True:
                    event = json.loads(socket.recv(timeout=30))
                    assert event["type"] != "error", event
                    if event["type"] == "session.updated":
                        break
                started = time.monotonic()
                partials = []
                for offset in range(0, len(pcm), 8000):
                    socket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(
                                    pcm[offset : offset + 8000]
                                ).decode(),
                            }
                        )
                    )
                    deadline = time.monotonic() + 0.25
                    while time.monotonic() < deadline:
                        try:
                            event = json.loads(
                                socket.recv(timeout=max(0, deadline - time.monotonic()))
                            )
                        except TimeoutError:
                            break
                        assert event["type"] != "error", event
                        if event["type"] == "transcription.segment" and event["text"]:
                            print(
                                json.dumps(
                                    {
                                        "elapsed": round(time.monotonic() - started, 3),
                                        **event,
                                    }
                                ),
                                flush=True,
                            )
                            if not event["is_final"] and offset + 8000 < len(pcm):
                                partials.append(event)
                assert (
                    partials
                ), "No nonempty hypothesis arrived before audio ingestion finished"
                socket.send(json.dumps({"type": "transcription.done"}))
                while True:
                    event = json.loads(socket.recv(timeout=180))
                    assert event["type"] != "error", event
                    if event["type"] == "transcription.completed":
                        print(json.dumps(event), flush=True)
                        assert "fox" in event["text"].lower(), event
                        assert "last sentence" in event["text"].lower(), event
                        break
    finally:
        server.close()


if __name__ == "__main__":
    main()
