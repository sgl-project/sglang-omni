#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real worker, native MLX speech server, and a configured text API.

Run with the prepared environment: python backend/smoke.py [--audio /path.wav]
Without --audio, macOS's built-in Samantha voice creates a known test phrase.
The first run downloads model snapshots. Output contains timings and results.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument(
        "--model", required=True, help="Model name exposed by your text API"
    )
    parser.add_argument(
        "--options", default="{}", help="Additional chat-completion JSON fields"
    )
    args = parser.parse_args()
    api = {
        "text_api_url": args.base_url,
        "text_model": args.model,
        "text_api_key": os.environ.get("OMNITYPER_API_KEY", ""),
        "text_api_options": json.loads(args.options),
    }
    with tempfile.TemporaryDirectory(prefix="omnityper-smoke-") as directory:
        audio = args.audio
        if audio is None:
            source = Path(directory) / "speech.aiff"
            audio = Path(directory) / "speech.wav"
            subprocess.run(
                [
                    "/usr/bin/say",
                    "-v",
                    "Samantha",
                    "-o",
                    str(source),
                    "The quick brown fox jumps over the lazy dog. Please send the report tomorrow.",
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
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            cases = [
                {
                    "id": "speech",
                    "op": "transcribe",
                    "audio_path": str(audio.resolve()),
                    "language": "en",
                    "style": "verbatim",
                },
                {
                    "id": "clean",
                    "op": "process",
                    "text": "um hello team uh please send the report tomorrow",
                },
                {
                    "id": "zh-clean",
                    "op": "process",
                    "text": "嗯你好啊明天我们用SGLang做性能优化然后发送报告",
                },
                {
                    "id": "translate",
                    "op": "process",
                    "mode": "translate",
                    "target_language": "fr",
                    "text": "Please send the report tomorrow.",
                },
                {
                    "id": "edit",
                    "op": "process",
                    "mode": "edit",
                    "selected_text": "The meeting starts at ten.",
                    "text": "Change ten to eleven.",
                },
                {
                    "id": "ask",
                    "op": "process",
                    "mode": "ask",
                    "text": "What is two plus two?",
                },
            ]
            for request in cases:
                request.update(api)
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                while True:
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError(
                            f"Worker exited before completing {request['id']}"
                        )
                    result = json.loads(line)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    if "ok" in result:
                        assert result["ok"], result
                        assert result["text"].strip(), result
                        assert not result.get("warning"), result
                        if request["id"] == "speech" and args.audio is None:
                            assert "fox" in result["text"].lower(), result
                            assert "report" in result["text"].lower(), result
                        if request["id"] == "clean":
                            assert not result["text"].lower().startswith("um "), result
                            assert " uh " not in result["text"].lower(), result
                        if request["id"] == "translate":
                            assert "demain" in result["text"].lower(), result
                        if request["id"] == "zh-clean":
                            assert (
                                "报告" in result["text"] and "明天" in result["text"]
                            ), result
                            assert "SGLang" in result["text"], result
                        if request["id"] == "edit":
                            assert (
                                "eleven" in result["text"].lower()
                                or "11" in result["text"]
                            ), result
                        if request["id"] == "ask":
                            assert (
                                "4" in result["text"]
                                or "four" in result["text"].lower()
                            ), result
                        break
            process.stdin.close()
            assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
