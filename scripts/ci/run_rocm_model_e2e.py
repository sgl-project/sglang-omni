#!/usr/bin/env python3
"""Run one reproducible model-weight E2E case on a ROCm accelerator."""

from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import wave
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "scripts/ci/rocm_model_e2e_cases.yaml"
REFERENCE_AUDIO = PROJECT_ROOT / "docs/_static/audio/male-voice.wav"
TRANSCRIPTION_AUDIO = PROJECT_ROOT / "tests/data/query_to_cars.wav"
REFERENCE_TEXT = (
    "Hey, Adam here. Let's create something that feels real, sounds human, "
    "and connects every time."
)
IGNORED_SNAPSHOT_FILES = (
    "*.onnx",
    "*.onnx_data",
    "*.tflite",
    "*.msgpack",
    "*.h5",
    "flax_model*",
    "tf_model*",
)
MODELS_WITH_RUNTIME_ONNX_ARTIFACTS = frozenset({"ming_omni", "ming_tts"})
MING_OMNI_SOURCE_REVISION = "2a0c02ae3130190160c215f89fce7de3005db483"
MING_OMNI_AUXILIARY_ASSETS = (
    {
        "source": "data/voice_name.json",
        "destination": "talker/data/voice_name.json",
        "sha256": "fb717d8940ba116fe6335779176ef51e285ca1427ef3b51a85f9868943d39de0",
    },
    {
        "source": "data/spks/prompt.wav",
        "destination": "talker/data/spks/prompt.wav",
        "sha256": "1611471cc6ebdda5a207802802ad12d2265b21e61e2ca43b98a2605cf981559c",
    },
    {
        "source": "data/lg/prompt_15014.wav",
        "destination": "talker/data/lg/prompt_15014.wav",
        "sha256": "c1e7e796f9e7798db98de29625cf1ae8acb5eaa271171f52223e47ea777b2802",
    },
)


@dataclass(frozen=True)
class ModelCase:
    id: str
    architecture: str
    checkpoint: str
    revision: str
    mode: str
    required_gpus: int = 1
    config: str | None = None
    served_model_name: str | None = None
    launcher: str | None = None

    @property
    def request_model(self) -> str:
        return self.served_model_name or self.checkpoint


def _load_cases(path: Path) -> dict[str, ModelCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(
            f"unsupported E2E manifest schema: {raw.get('schema_version')}"
        )
    cases = [ModelCase(**entry) for entry in raw["models"]]
    by_id = {case.id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("ROCm model E2E ids must be unique")
    if invalid := [case.id for case in cases if case.required_gpus < 1]:
        raise ValueError(
            f"ROCm model E2E cases require a positive GPU count: {invalid}"
        )
    return by_id


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _gpu_fingerprint() -> dict[str, Any]:
    import torch

    architectures: list[str] = []
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        architecture = getattr(properties, "gcnArchName", "")
        architectures.append(str(architecture).split(":", 1)[0])
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "architecture": architecture,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "accelerator": "rocm" if torch.version.hip else "cuda",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "architectures": sorted(set(architectures)),
        "devices": devices,
    }


def _probe(case: ModelCase, artifact_dir: Path) -> dict[str, Any]:
    info = HfApi().model_info(
        case.checkpoint, revision=case.revision, files_metadata=True
    )
    result = {
        "architecture": case.architecture,
        "checkpoint": case.checkpoint,
        "expected_revision": case.revision,
        "resolved_revision": info.sha,
        "repository_bytes": sum(sibling.size or 0 for sibling in info.siblings),
    }
    if info.sha != case.revision:
        raise RuntimeError(
            f"{case.id}: expected revision {case.revision}, resolved {info.sha}"
        )
    _write_json(artifact_dir / "probe.json", result)
    return result


def _snapshot_ignore_patterns(case: ModelCase) -> tuple[str, ...]:
    if case.id not in MODELS_WITH_RUNTIME_ONNX_ARTIFACTS:
        return IGNORED_SNAPSHOT_FILES
    return tuple(
        pattern
        for pattern in IGNORED_SNAPSHOT_FILES
        if pattern not in {"*.onnx", "*.onnx_data"}
    )


def _stage_auxiliary_assets(case: ModelCase, model_path: Path) -> dict[str, Any] | None:
    """Stage small, pinned upstream assets omitted from a model checkpoint."""
    if case.id != "ming_omni":
        return None

    base_url = (
        "https://raw.githubusercontent.com/inclusionAI/Ming/"
        f"{MING_OMNI_SOURCE_REVISION}/"
    )
    staged = []
    for asset in MING_OMNI_AUXILIARY_ASSETS:
        destination = model_path / asset["destination"]
        source_url = base_url + asset["source"]
        if destination.is_file():
            payload = destination.read_bytes()
            reused = True
        else:
            response = httpx.get(source_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            payload = response.content
            reused = False

        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != asset["sha256"]:
            raise RuntimeError(
                f"{case.id}: auxiliary asset checksum mismatch for {source_url}: "
                f"expected {asset['sha256']}, got {actual_sha256}"
            )
        if not reused:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        staged.append(
            {
                "source_url": source_url,
                "destination": asset["destination"],
                "sha256": actual_sha256,
                "bytes": len(payload),
                "reused": reused,
            }
        )

    return {
        "repository": "inclusionAI/Ming",
        "revision": MING_OMNI_SOURCE_REVISION,
        "assets": staged,
    }


def _download(case: ModelCase, cache_dir: Path, artifact_dir: Path) -> Path:
    started = time.time()
    model_path = Path(
        snapshot_download(
            repo_id=case.checkpoint,
            revision=case.revision,
            cache_dir=cache_dir,
            ignore_patterns=_snapshot_ignore_patterns(case),
        )
    )
    auxiliary_assets = _stage_auxiliary_assets(case, model_path)
    result = {
        "checkpoint": case.checkpoint,
        "revision": case.revision,
        "snapshot_path": str(model_path),
        "elapsed_seconds": time.time() - started,
    }
    if auxiliary_assets is not None:
        result["auxiliary_assets"] = auxiliary_assets
    _write_json(artifact_dir / "download.json", result)
    return model_path


def _server_command(case: ModelCase, model_path: Path, port: int) -> list[str]:
    if case.launcher == "ming":
        return [
            sys.executable,
            "examples/run_omni.py",
            "ming-speech-server",
            "--model-path",
            str(model_path),
            "--model-name",
            case.request_model,
            "--tp-size",
            "4",
            "--gpu-thinker",
            "0",
            "--gpu-talker",
            "4",
            "--port",
            str(port),
        ]

    command = [
        "sgl-omni",
        "serve",
        "--model-path",
        str(model_path),
        "--model-name",
        case.request_model,
        "--port",
        str(port),
    ]
    if case.config:
        command.extend(["--config", case.config])
    if case.mode == "reference_tts":
        command.extend(["--allowed-local-media-path", str(REFERENCE_AUDIO.parent)])
    return command


def _wait_for_server(process: subprocess.Popen[bytes], port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited before readiness with status {return_code}"
            )
        try:
            response = httpx.get(url, timeout=2.0)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"server did not become ready within {timeout} seconds")


def _validate_wav(path: Path) -> dict[str, Any]:
    with closing(wave.open(str(path), "rb")) as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        payload = wav_file.readframes(frames)
    if channels < 1 or sample_rate < 8000 or sample_width not in (2, 4):
        raise RuntimeError(
            f"invalid WAV metadata: channels={channels} rate={sample_rate} "
            f"width={sample_width}"
        )
    duration = frames / sample_rate
    if duration < 0.2:
        raise RuntimeError(f"generated WAV is too short: {duration:.3f}s")
    typecode = "h" if sample_width == 2 else "i"
    samples = array.array(typecode)
    samples.frombytes(payload)
    nonzero = sum(sample != 0 for sample in samples)
    if nonzero == 0:
        raise RuntimeError("generated WAV is silent")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "frames": frames,
        "duration_seconds": duration,
        "nonzero_samples": nonzero,
    }


def _tts_payload(case: ModelCase) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": case.request_model,
        "input": "ROCm inference is ready on AMD Instinct accelerators.",
        "voice": "default",
        "response_format": "wav",
        "max_new_tokens": 256,
    }
    if case.id not in {"ming_tts", "voxtral_tts", "zonos2"}:
        payload["seed"] = 42
    if case.mode == "preset_voice_tts":
        payload["voice"] = "cheerful_female"
    if case.mode == "reference_tts":
        payload["references"] = [
            {"audio_path": str(REFERENCE_AUDIO), "text": REFERENCE_TEXT}
        ]
    return payload


def _request(case: ModelCase, port: int, artifact_dir: Path) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{port}"
    started = time.time()
    if case.mode == "music":
        response = httpx.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": case.request_model,
                "input": (
                    "[Verse]\nROCm carries the song\n"
                    "[Chorus]\nTwo architectures strong"
                ),
                "instructions": "A gentle acoustic pop song at 90 BPM",
                "seed": 42,
                "max_new_tokens": 250,
            },
            timeout=900.0,
        )
        response.raise_for_status()
        output_path = artifact_dir / "output.wav"
        output_path.write_bytes(response.content)
        output = _validate_wav(output_path)
    elif case.mode in {"tts", "reference_tts", "preset_voice_tts"}:
        response = httpx.post(
            f"{base_url}/v1/audio/speech",
            json=_tts_payload(case),
            timeout=900.0,
        )
        response.raise_for_status()
        output_path = artifact_dir / "output.wav"
        output_path.write_bytes(response.content)
        output = _validate_wav(output_path)
    elif case.mode in {"transcription", "transcription_verbose"}:
        data = {
            "model": case.request_model,
            "response_format": (
                "verbose_json" if case.mode == "transcription_verbose" else "json"
            ),
        }
        with TRANSCRIPTION_AUDIO.open("rb") as audio_file:
            response = httpx.post(
                f"{base_url}/v1/audio/transcriptions",
                data=data,
                files={"file": (TRANSCRIPTION_AUDIO.name, audio_file, "audio/wav")},
                timeout=900.0,
            )
        response.raise_for_status()
        payload = response.json()
        _write_json(artifact_dir / "response.json", payload)
        text = payload.get("text")
        if not isinstance(text, str) or len(text.strip()) < 3:
            raise RuntimeError(f"transcription response has no usable text: {payload}")
        output = {"text": text, "characters": len(text)}
    else:
        body: dict[str, Any] = {
            "model": case.request_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe ROCm in one short sentence.",
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        if case.mode == "chat_audio":
            body["modalities"] = ["text", "audio"]
            body["talker_max_new_tokens"] = 256
        response = httpx.post(
            f"{base_url}/v1/chat/completions", json=body, timeout=900.0
        )
        response.raise_for_status()
        payload = response.json()
        _write_json(artifact_dir / "response.json", payload)
        message = payload["choices"][0]["message"]
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"chat response has no text: {payload}")
        output = {"text": text, "characters": len(text)}
        if case.mode == "chat_audio":
            audio = message.get("audio") or {}
            encoded = audio.get("data")
            if not isinstance(encoded, str) or not encoded:
                raise RuntimeError("chat response has no audio payload")
            audio_path = artifact_dir / "output.wav"
            audio_path.write_bytes(base64.b64decode(encoded))
            output["audio"] = _validate_wav(audio_path)
    output["request_elapsed_seconds"] = time.time() - started
    return output


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run(
    case: ModelCase,
    model_path: Path,
    artifact_dir: Path,
    expected_architecture: str,
    port: int,
    startup_timeout: int,
    visible_devices_override: str | None = None,
) -> dict[str, Any]:
    fingerprint = _gpu_fingerprint()
    expected_architecture = expected_architecture.strip().lower().split(":", 1)[0]
    if fingerprint["accelerator"] != "rocm":
        raise RuntimeError(f"expected ROCm, got {fingerprint['accelerator']}")
    if fingerprint["architectures"] != [expected_architecture]:
        raise RuntimeError(
            f"expected only {expected_architecture}, got {fingerprint['architectures']}"
        )
    if len(fingerprint["devices"]) < case.required_gpus:
        raise RuntimeError(
            f"{case.id} requires {case.required_gpus} GPU(s), but only "
            f"{len(fingerprint['devices'])} are visible"
        )
    command = _server_command(case, model_path, port)
    environment = dict(os.environ)
    if visible_devices_override is not None:
        physical_devices = [
            device.strip()
            for device in visible_devices_override.split(",")
            if device.strip()
        ]
        if len(physical_devices) < case.required_gpus:
            raise RuntimeError(
                f"{case.id} requires {case.required_gpus} GPU(s), but "
                f"--visible-devices supplied {len(physical_devices)}"
            )
        logical_devices = ",".join(str(index) for index in range(len(physical_devices)))
        environment["ROCR_VISIBLE_DEVICES"] = ",".join(physical_devices)
        environment["HIP_VISIBLE_DEVICES"] = logical_devices
        environment["CUDA_VISIBLE_DEVICES"] = logical_devices
    _write_json(
        artifact_dir / "launch.json",
        {
            "command": command,
            "environment": {
                name: environment.get(name)
                for name in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "gpu": fingerprint,
        },
    )

    server_log = (artifact_dir / "server.log").open("wb")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.time()
    try:
        _wait_for_server(process, port, startup_timeout)
        output = _request(case, port, artifact_dir)
        result = {
            "status": "passed",
            "model": case.id,
            "architecture": case.architecture,
            "checkpoint": case.checkpoint,
            "revision": case.revision,
            "gpu_architecture": expected_architecture,
            "startup_seconds": time.time()
            - started
            - output["request_elapsed_seconds"],
            "output": output,
        }
        _write_json(artifact_dir / "result.json", result)
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "model": case.id,
            "architecture": case.architecture,
            "checkpoint": case.checkpoint,
            "revision": case.revision,
            "gpu_architecture": expected_architecture,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(artifact_dir / "result.json", result)
        raise
    finally:
        _stop_process_group(process)
        server_log.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", help="Manifest model id")
    parser.add_argument("--all", action="store_true", help="Run every manifest case")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("/models"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("/artifacts"))
    parser.add_argument("--expected-gpu-arch")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument(
        "--visible-devices",
        help="Optional physical device list for an isolated run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = _load_cases(args.manifest)
    if args.list:
        for case in cases.values():
            print(f"{case.id}\t{case.architecture}\t{case.checkpoint}@{case.revision}")
        return 0
    if bool(args.model) == bool(args.all):
        raise SystemExit("select exactly one of --model or --all unless --list is used")
    if args.model is not None and args.model not in cases:
        raise SystemExit(
            f"unknown model {args.model!r}; choose from: {', '.join(cases)}"
        )
    selected_cases = list(cases.values()) if args.all else [cases[args.model]]
    if args.visible_devices is not None and len(selected_cases) != 1:
        raise SystemExit("--visible-devices can only be used with --model")
    results = []
    for case in selected_cases:
        artifact_dir = args.artifact_dir / case.id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _probe(case, artifact_dir)
        if args.probe_only:
            continue
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = _download(case, args.cache_dir, artifact_dir)
        if args.download_only:
            continue
        if not args.expected_gpu_arch:
            raise SystemExit("--expected-gpu-arch is required for an E2E run")
        results.append(
            _run(
                case,
                model_path,
                artifact_dir,
                args.expected_gpu_arch,
                args.port,
                args.startup_timeout,
                args.visible_devices,
            )
        )
    if results:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
