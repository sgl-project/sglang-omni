# SPDX-License-Identifier: Apache-2.0
"""Own the native SGLang-Omni MLX deployment used by the private app worker."""

from __future__ import annotations

import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
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
MODELSCOPE_MODEL = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
MODELSCOPE_REVISION = "3478f178e267548f04a6b616ff10beeb1e644e54"
SERVED_MODEL = "Qwen/Qwen3-ASR-0.6B"


def model_snapshot(model: str, revision: str) -> str:
    """Use a complete pinned local snapshot offline; download on first use."""
    if model == MODELSCOPE_MODEL:
        from modelscope_hub.compat.snapshot_download import snapshot_download
        from modelscope_hub.errors import CacheNotFound as CacheMiss
    elif model == DEFAULT_MODEL:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError as CacheMiss
    else:
        raise ValueError("Unsupported ASR model.")

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
    patterns = ["*.json", "*.safetensors", "*.txt"]
    cached = None
    try:
        cached = snapshot_download(
            model, revision=revision, local_files_only=True, allow_patterns=patterns
        )
    except CacheMiss:
        # Note (Jiaxin Deng): Not cached yet, so fall through to the online download below.
        pass
    if cached is None or not all((Path(cached) / name).is_file() for name in required):
        cached = snapshot_download(model, revision=revision, allow_patterns=patterns)
    if not all((Path(cached) / name).is_file() for name in required):
        raise RuntimeError(
            "The downloaded speech model is incomplete. Retry preparing ASR."
        )
    if (
        model == MODELSCOPE_MODEL
        and not (Path(cached) / "preprocessor_config.json").is_file()
    ):
        # Note (Codex): This conversion omits the Qwen3-ASR processor. Match the pinned HF model locally.
        processor = {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
            "chunk_length": 30,
            "n_samples": 480000,
            "nb_max_frames": 3000,
            "padding_side": "right",
            "padding_value": 0.0,
            "dither": 0.0,
            "return_attention_mask": True,
            "processor_class": "Qwen3ASRProcessor",
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=cached, delete=False) as output:
            json.dump(processor, output)
        Path(output.name).replace(Path(cached) / "preprocessor_config.json")
    return str(cached)


class NativeASRServer:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""
        self.model = ""
        # Note (Codex): Local audio must not leave loopback through inherited proxy settings.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(
        self, progress: Callable[[str], None], model: str = DEFAULT_MODEL
    ) -> None:
        if model not in {DEFAULT_MODEL, MODELSCOPE_MODEL}:
            raise ValueError("Unsupported ASR model.")
        if (
            self.process is not None
            and self.process.poll() is None
            and self.model == model
        ):
            return
        self.close()
        progress(
            "Loading the pinned local speech model; first use downloads model files…"
        )
        model_path = model_snapshot(
            model, MODELSCOPE_REVISION if model == MODELSCOPE_MODEL else MODEL_REVISION
        )
        self.model = model
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
