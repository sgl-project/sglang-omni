# SPDX-License-Identifier: Apache-2.0
"""Own the native SGLang-Omni MLX deployment used by the private app worker."""

from __future__ import annotations

import fnmatch
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
MODEL_REVISION = "313d850181767edf09f00a9c289becca70e58cd0"
SERVED_MODEL = "Qwen/Qwen3-ASR-0.6B"
MODEL_FILES = ["*.json", "*.safetensors", "*.txt"]


def snapshot_bytes(model: str, revision: str) -> int:
    """Size of the files a first download fetches."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model, revision=revision, files_metadata=True, timeout=10)
    return sum(
        file.size or 0
        for file in info.siblings or []
        if any(fnmatch.fnmatch(file.rfilename, pattern) for pattern in MODEL_FILES)
    )


def download_reporter(progress: Callable[..., None], expected: int) -> type:
    """Build the tqdm class that reports a snapshot's byte progress instead of drawing it.

    Note (Yifei Leng): The cache folder cannot stand in for this. Xet transfers keep the bytes
    out of the blob until the file is whole, so its size stays near zero for the entire download.
    """
    from tqdm import tqdm

    class Reporter(tqdm):
        finished = False
        fraction = 0.0
        reported = 0.0

        def __init__(self, *args: object, **kwargs: object) -> None:
            # Note (Yifei Leng): huggingface_hub passes its own `name` keyword to some custom classes,
            # and tqdm would otherwise switch itself off because the worker's stderr is not a terminal.
            kwargs.pop("name", None)
            kwargs.update(disable=False, file=io.StringIO())
            super().__init__(*args, **kwargs)

        def display(self, *args: object, **kwargs: object) -> None:
            # Note (Yifei Leng): Recent releases keep one bar for network bytes and one for written
            # bytes, and both only grow, so the furthest bar is the honest figure.
            total = max(self.total or 0, expected)
            if Reporter.finished or self.unit != "B" or not total:
                return
            fraction = min(self.n / total, 0.99)
            now = time.monotonic()
            if fraction <= Reporter.fraction or now - Reporter.reported < 0.25:
                return
            Reporter.fraction, Reporter.reported = fraction, now
            progress(
                f"Downloading the speech model… {int(fraction * 100)}% of {total // 1_000_000} MB",
                fraction,
            )

    return Reporter


def model_snapshot(model: str, revision: str, progress: Callable[..., None]) -> str:
    """Use a complete pinned local snapshot offline; download on first use."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    required = {
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    if model == DEFAULT_MODEL:
        required.add("preprocessor_config.json")
    try:
        # Note (Yifei Leng): Recent huggingface_hub releases call a snapshot incomplete unless
        # every file of the repository is cached, so name the files this app downloads.
        cached = snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
            allow_patterns=MODEL_FILES,
        )
        if all((Path(cached) / name).is_file() for name in required):
            return cached
    except LocalEntryNotFoundError:
        # Note (Jiaxin Deng): Not cached yet, so fall through to the online download below.
        pass
    try:
        expected = snapshot_bytes(model, revision)
    except Exception:
        # Note (Yifei Leng): The percentage is best effort; the download below reports real failures.
        expected = 0
    reporter = download_reporter(progress, expected)
    try:
        return snapshot_download(
            model, revision=revision, allow_patterns=MODEL_FILES, tqdm_class=reporter
        )
    finally:
        # Note (Yifei Leng): A bar collected later must not write into the protocol stream.
        reporter.finished = True


class NativeASRServer:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""
        # Note (Codex): Local audio must not leave loopback through inherited proxy settings.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self, progress: Callable[..., None]) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        progress(
            "Loading the pinned local speech model; first use downloads model files…"
        )
        model_path = model_snapshot(DEFAULT_MODEL, MODEL_REVISION, progress)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment["SGLANG_USE_MLX"] = "1"
        environment["SGLANG_OMNI_STRICT_PORT"] = "1"
        environment["SGLANG_OMNI_ADMIN_KEY"] = secrets.token_urlsafe(32)
        ffmpeg = Path("/opt/homebrew/opt/ffmpeg@7/lib")
        if ffmpeg.is_dir():
            environment["DYLD_LIBRARY_PATH"] = str(ffmpeg) + (
                ":" + environment["DYLD_LIBRARY_PATH"]
                if environment.get("DYLD_LIBRARY_PATH")
                else ""
            )
        progress(
            "Starting SGLang-Omni MLX speech server; first use downloads model files…"
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                model_path,
                "--model-name",
                SERVED_MODEL,
                "--enable-realtime",
                "--asr.engine.max_running_requests",
                "1",
                "--audio_chunking.max_total_audio_s",
                "300",
                "--audio_chunking.max_concurrent_chunks",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + 1200
        next_progress = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "SGLang-Omni speech server exited. Check the backend log and run setup.sh."
                    )
                try:
                    with self.http.open(self.url + "/health", timeout=1) as response:
                        if response.status == 200:
                            progress("SGLang-Omni MLX speech server is ready.")
                            return
                except (urllib.error.URLError, TimeoutError, OSError):
                    # Note (Jiaxin Deng): Expected while the server boots, so keep polling
                    # until the deadline passes or the process exits.
                    pass
                if time.monotonic() >= next_progress:
                    progress("Waiting for the local speech model to finish loading…")
                    next_progress = time.monotonic() + 15
                time.sleep(0.25)
            raise TimeoutError(
                "The local speech server did not become ready within 20 minutes."
            )
        except BaseException:
            self.close()
            raise

    def transcribe(
        self,
        samples: NDArray[np.float32],
        rate: int,
        language: str,
        hotwords: Sequence[str] = (),
    ) -> str:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("The local speech server is not running.")
        audio = io.BytesIO()
        with wave.open(audio, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(rate)
            recording.writeframes(
                np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
            )
        boundary = "OmniTyper" + secrets.token_hex(16)
        fields = {"model": SERVED_MODEL, "response_format": "json"}
        if language:
            fields["language"] = language
        if hotwords:
            # Note (Codex): Keep vocabulary biasing within the native ASR prompt limit.
            fields["prompt"] = json.dumps(
                list(hotwords)[:20], ensure_ascii=False
            ).replace("<", "\\u003c")
        parts = []
        for key, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
        parts.extend(
            [
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="recording.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
                audio.getvalue(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            self.url + "/v1/audio/transcriptions",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with self.http.open(request, timeout=600) as response:
                payload = response.read(256 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Local transcription failed (HTTP {exc.code}): {detail}"
            ) from exc
        if len(payload) > 256 * 1024:
            raise RuntimeError("Local transcription returned an oversized response.")
        result = json.loads(payload)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError("Local transcription returned an invalid response.")
        return result["text"].strip()

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Note (Jiaxin Deng): The group is already gone, and teardown has to stay idempotent.
            pass
        try:
            process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            # Note (Jiaxin Deng): Graceful exit did not finish in time, so fall through to SIGKILL.
            pass
        # Note (Codex): Reap stage descendants even if the launcher exited first.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # Note (Jiaxin Deng): SIGTERM already reaped the group, so there is nothing to kill.
            pass
        process.wait(timeout=1)
