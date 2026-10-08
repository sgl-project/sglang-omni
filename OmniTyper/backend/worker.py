#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private, serial JSON-lines worker. stdout is reserved for the protocol."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import signal
import stat
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any, BinaryIO, TextIO

import numpy as np
from numpy.typing import NDArray

# Note (Codex): Checkout scripts use local sources; bundled workers use the installed package.
for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if (root / "sglang_omni").is_dir():
        sys.path.insert(0, str(root))
        break

# Note (Jiaxin Deng): PYTHONSAFEPATH omits the script directory needed for bundled sibling imports.
BACKEND_DIRECTORY = str(Path(__file__).resolve().parent)
if BACKEND_DIRECTORY not in sys.path:
    sys.path.insert(0, BACKEND_DIRECTORY)

import text_api
from server import DEFAULT_MODEL, MODELSCOPE_MODEL, NativeASRServer

import sglang_omni

# Note (Codex): Load the language helper without eager model configuration or runtime imports.
LANGUAGE_PATH = Path(sglang_omni.__file__).parent / "models/qwen3_asr/languages.py"
LANGUAGE_SPEC = importlib.util.spec_from_file_location("asr_languages", LANGUAGE_PATH)
LANGUAGE_MODULE = importlib.util.module_from_spec(LANGUAGE_SPEC)
LANGUAGE_SPEC.loader.exec_module(LANGUAGE_MODULE)
resolve_language = LANGUAGE_MODULE.resolve_language

DEFAULT_TEXT_API = "http://127.0.0.1:11434/v1"
MAX_LINE_BYTES = 256 * 1024
MAX_TEXT = 12000
MAX_AUDIO_SECONDS = 300
logger = logging.getLogger(__name__)
FIELDS = {
    "id",
    "op",
    "audio_path",
    "asr_model",
    "text_model",
    "text_api_url",
    "text_api_key",
    "text_api_options",
    "mode",
    "language",
    "target_language",
    "style",
    "instructions",
    "dictionary",
    "selected_text",
    "app_name",
    "text",
}


def validate_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Request must be a JSON object.")
    if value.keys() - FIELDS:
        raise ValueError(
            "Unknown request field(s): " + ", ".join(sorted(value.keys() - FIELDS))
        )
    request = value.copy()
    limits = {
        "id": 128,
        "op": 16,
        "audio_path": 4096,
        "asr_model": 256,
        "text_model": 256,
        "text_api_url": 2048,
        "text_api_key": 4096,
        "mode": 16,
        "language": 64,
        "target_language": 64,
        "style": 16,
        "instructions": 2000,
        "selected_text": MAX_TEXT,
        "app_name": 256,
        "text": MAX_TEXT,
    }
    for field, limit in limits.items():
        item = request.get(field, "")
        if not isinstance(item, str) or len(item) > limit or "\x00" in item:
            raise ValueError(
                f"{field} must be a string of at most {limit} characters without NUL."
            )
    if not request.get("id", "").strip():
        raise ValueError("id must be a nonempty string.")
    if request.get("op") not in {"prepare", "transcribe", "process", "models"}:
        raise ValueError("op must be prepare, transcribe, process, or models.")
    defaults = {
        "asr_model": DEFAULT_MODEL,
        "text_model": "",
        "text_api_url": DEFAULT_TEXT_API,
        "text_api_key": "",
        "mode": "dictate",
        "style": "clean",
        "language": "",
        "target_language": "",
        "instructions": "",
        "selected_text": "",
        "app_name": "",
        "text": "",
    }
    for field, default in defaults.items():
        request.setdefault(field, default)
    if request["asr_model"] not in {DEFAULT_MODEL, MODELSCOPE_MODEL}:
        raise ValueError(f"Supported ASR models: {DEFAULT_MODEL}, {MODELSCOPE_MODEL}.")
    options = request.setdefault("text_api_options", {})
    if not isinstance(options, dict) or any(
        not isinstance(key, str) for key in options
    ):
        raise ValueError("Text API options must be a JSON object.")
    if options.keys() & {"model", "messages", "stream"}:
        raise ValueError("Text API options cannot override model, messages, or stream.")
    if len(json.dumps(options, ensure_ascii=False, allow_nan=False).encode()) > 8192:
        raise ValueError("Text API options exceed 8 KiB.")
    if request["mode"] not in {"dictate", "translate", "edit", "ask"}:
        raise ValueError("Unsupported mode.")
    if request["style"] not in {"clean", "verbatim", "casual", "formal", "concise"}:
        raise ValueError("Unsupported style.")
    for field in ("language", "target_language"):
        if request[field]:
            request[field] = resolve_language(request[field])
    if request["mode"] == "translate" and not request["target_language"]:
        raise ValueError("Translation requires target_language.")
    if request["mode"] == "edit" and not request["selected_text"].strip():
        raise ValueError("Editing requires selected_text.")
    if request["op"] == "transcribe" and not request.get("audio_path"):
        raise ValueError("Transcription requires audio_path.")
    dictionary = request.setdefault("dictionary", [])
    if not isinstance(dictionary, list) or len(dictionary) > 200:
        raise ValueError("dictionary must be an array with at most 200 entries.")
    for entry in dictionary:
        if not isinstance(entry, dict) or entry.keys() != {"spoken", "written"}:
            raise ValueError("Dictionary entries require spoken and written strings.")
        for item in entry.values():
            if (
                not isinstance(item, str)
                or not item.strip()
                or len(item) > 200
                or "\x00" in item
            ):
                raise ValueError(
                    "Dictionary phrases must contain 1–200 characters without NUL."
                )
    return request


def read_audio(path: str) -> tuple[NDArray[np.float32], int]:
    """Validate before decoding or model loading; return mono float32 and rate."""
    location = Path(path).expanduser()
    if not location.is_absolute():
        raise ValueError("audio_path must be an absolute local WAV path.")
    # Note (Codex): Nonblocking open rejects named pipes without hanging the worker.
    descriptor = os.open(location, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 120 * 1024 * 1024:
            raise ValueError("Audio must be a regular WAV file no larger than 120 MiB.")
        with wave.open(source, "rb") as recording:
            channels, width, rate, frames, compression, _ = recording.getparams()
            if channels not in {1, 2} or width != 2 or compression != "NONE":
                raise ValueError(
                    "Audio must be uncompressed 16-bit mono or stereo WAV."
                )
            if not 8000 <= rate <= 96000 or frames > rate * MAX_AUDIO_SECONDS:
                raise ValueError("Audio must be 8–96 kHz and at most 300 seconds.")
            pcm = recording.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("WAV data is truncated.")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, rate


def is_silent(samples: NDArray[np.float32]) -> bool:
    # ponytail: energy gate rejects silence, not background speech; add VAD if needed.
    return samples.size == 0 or float(np.std(samples)) < 0.0003


def apply_dictionary(text: str, dictionary: list[dict[str, str]]) -> str:
    if not dictionary:
        return text
    entries = sorted(dictionary, key=lambda item: len(item["spoken"]), reverse=True)
    patterns = []
    for index, item in enumerate(entries):
        spoken = item["spoken"]
        left = r"(?<!\w)" if spoken[0].isascii() and spoken[0].isalnum() else ""
        right = r"(?!\w)" if spoken[-1].isascii() and spoken[-1].isalnum() else ""
        patterns.append(f"(?P<entry{index}>{left}{re.escape(spoken)}{right})")
    return re.sub(
        "|".join(patterns),
        lambda match: entries[int(match.lastgroup[5:])]["written"],
        text,
        flags=re.IGNORECASE,
    )


class Worker:
    def __init__(self) -> None:
        self.asr = NativeASRServer()

    def handle(
        self, value: object, progress: Callable[[str], None] = lambda message: None
    ) -> dict[str, Any]:
        started = time.monotonic()
        request = validate_request(value)
        if request["op"] == "models":
            progress("Connecting to the text API…")
            response = text_api.api_request(request, "/models")
            data = response.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Text API returned an invalid model list.")
            models = list(
                dict.fromkeys(
                    item["id"]
                    for item in data
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and 0 < len(item["id"]) <= 256
                    and not any(ord(c) < 32 for c in item["id"])
                )
            )[:200]
            return {"id": request["id"], "ok": True, "models": models}
        if request["op"] == "prepare":
            self.asr.start(progress, request["asr_model"])
            return {
                "id": request["id"],
                "ok": True,
                "realtime_url": self.asr.url.replace("http://", "ws://", 1)
                + "/v1/realtime?intent=transcription",
            }
        raw = request["text"]
        if request["op"] == "transcribe":
            samples, rate = read_audio(request["audio_path"])
            if is_silent(samples):
                raw = ""
            else:
                self.asr.start(progress, request["asr_model"])
                progress("Transcribing locally…")
                raw = self.asr.transcribe(
                    samples,
                    rate,
                    request["language"],
                    [entry["written"] for entry in request["dictionary"]],
                )
        if len(raw) > MAX_TEXT:
            raise ValueError(
                "Transcription exceeds the text limit; use a shorter clip."
            )
        text = apply_dictionary(raw, request["dictionary"])
        if len(text) > MAX_TEXT * 2:
            return {
                "id": request["id"],
                "ok": False,
                "raw_text": raw,
                "error": "Dictionary expansion exceeds the output limit. Shorten its replacements.",
            }
        warning = ""
        if text.strip() and not (
            request["mode"] == "dictate" and request["style"] == "verbatim"
        ):
            try:
                text = text_api.process_text(request, text, progress)
            except Exception as exc:
                if request["mode"] != "dictate":
                    return {
                        "id": request["id"],
                        "ok": False,
                        "error": str(exc)[:2000],
                        "raw_text": raw,
                    }
                warning = f"Text cleanup failed; the unpolished transcript was kept. {str(exc)[:2000]}"
        elif not text.strip():
            text = ""
        return {
            "id": request["id"],
            "ok": True,
            "text": text,
            "raw_text": raw,
            "warning": warning,
            "duration": round(time.monotonic() - started, 3),
        }


def serve(source: BinaryIO, output: TextIO, worker: Worker) -> None:
    def emit(value: dict[str, Any]) -> None:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()

    while True:
        line = source.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        request_id = ""
        try:
            if len(line) > MAX_LINE_BYTES:
                while line and not line.endswith(b"\n"):
                    line = source.readline(MAX_LINE_BYTES + 1)
                raise ValueError("Request exceeds the 256 KiB protocol limit.")
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                request_id = value["id"][:128]
            result = worker.handle(
                value,
                lambda message: emit(
                    {"id": request_id, "event": "progress", "message": message}
                ),
            )
            emit(result)
        except Exception as exc:
            logger.warning("Worker request failed: %s", type(exc).__name__)
            emit({"id": request_id, "ok": False, "error": str(exc)[:2000]})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Note (Codex): Redirect fd 1 so native library logs cannot corrupt the JSON protocol.
    protocol = os.fdopen(
        os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1
    )
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    worker = Worker()

    def terminate(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        serve(sys.stdin.buffer, protocol, worker)
    except (BrokenPipeError, KeyboardInterrupt):
        # Note (Jiaxin Deng): Pipe closure and interruption still release the owned server in finally.
        pass
    finally:
        worker.asr.close()
        protocol.close()


if __name__ == "__main__":
    main()
