# SPDX-License-Identifier: Apache-2.0
"""Checks AuK PCM waveforms against exact outputs recorded for each Apple chip."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import NotRequired, TypedDict


class SynthesisCase(TypedDict):
    id: str
    instruction: str
    seconds: float
    seed: int
    reference_clip: NotRequired[str]


class AudioDigest(TypedDict):
    sha256: str
    sample_count: int
    sample_rate: int


class DeviceOutputs(TypedDict):
    metrics: dict[str, float]
    clips: dict[str, AudioDigest]


class Golden(TypedDict):
    model: str
    kind: str
    dtype: str
    cases: list[SynthesisCase]
    devices: dict[str, DeviceOutputs]


class NativeResult(TypedDict):
    id: str
    file: str
    sample_count: int
    sample_rate: int
    elapsed_s: float


def validate_cases(golden: Golden) -> None:
    identifiers = [request["id"] for request in golden["cases"]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("AuK golden requests require unique nonempty case IDs")
    else:
        pass
    for request in golden["cases"]:
        if (
            re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request["id"]) is None
            or not request["instruction"].strip()
            or not math.isfinite(request["seconds"])
            or not 0 < request["seconds"] <= 30
            or isinstance(request["seed"], bool)
            or not isinstance(request["seed"], int)
            or not 0 <= request["seed"] < 2**64
            or (
                "reference_clip" in request
                and re.fullmatch(r"[A-Za-z0-9_-]+", request["reference_clip"]) is None
            )
        ):
            raise ValueError(f"Invalid AuK golden request: {request['id']}")
        else:
            pass


def validate_outputs(golden: Golden, clips: dict[str, AudioDigest]) -> list[str]:
    validate_cases(golden)
    expected = {request["id"]: request for request in golden["cases"]}
    failures = []
    if set(clips) != set(expected):
        failures.append("generated case IDs do not match the golden request set")
    else:
        pass
    for request_id, result in clips.items():
        if request_id not in expected:
            continue
        else:
            pass
        sample_count = math.ceil(expected[request_id]["seconds"] * 50) * 480
        digest = result.get("sha256", "")
        if (
            result.get("sample_count") != sample_count
            or result.get("sample_rate") != 24000
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            failures.append(f"{request_id}: invalid waveform shape or SHA-256")
        else:
            pass
    return failures


def check(
    golden: Golden,
    runtime_bin: Path,
    data_root: Path,
    device: str,
    outputs: Path | None,
    write_path: Path | None,
) -> tuple[list[str], list[str]]:
    validate_cases(golden)
    clips: dict[str, AudioDigest] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="auk-golden-") as temporary:
        directory = Path(temporary)
        request_file = directory / "requests.jsonl"
        requests = []
        for case in golden["cases"]:
            request = {
                name: value for name, value in case.items() if name != "reference_clip"
            }
            if "reference_clip" in case:
                request["reference_wav"] = str(
                    data_root
                    / "corpus"
                    / "v1"
                    / "clips"
                    / f"{case['reference_clip']}.wav"
                )
            else:
                pass
            requests.append(json.dumps(request, ensure_ascii=False))
        request_file.write_text("\n".join(requests) + "\n")
        command = [
            str(runtime_bin / "auk_synthesize"),
            "--model-path", str(data_root / "native_models" / golden["model"].replace("/", "_")),
            "--requests", str(request_file),
            "--output-directory", str(directory / "audio"),
            "--dtype", golden["dtype"],
        ]  # fmt: skip
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "MLX_ENABLE_TF32": "0"},
        )
        if completed.returncode:
            failures.append(
                f"auk_synthesize exited {completed.returncode}: {completed.stderr[-2000:].strip()}"
            )
        else:
            for line in completed.stdout.splitlines():
                result: NativeResult = json.loads(line)
                request_id = result["id"]
                path = Path(result["file"])
                if request_id in clips:
                    failures.append(f"{request_id}: duplicate native result")
                    continue
                else:
                    pass
                with wave.open(str(path)) as audio:
                    sample_count = audio.getnframes()
                    sample_rate = audio.getframerate()
                    if (
                        audio.getnchannels() != 1
                        or audio.getsampwidth() != 2
                        or len(audio.readframes(sample_count)) != sample_count * 2
                        or sample_count != result["sample_count"]
                        or sample_rate != result["sample_rate"]
                    ):
                        failures.append(
                            f"{request_id}: invalid PCM16 WAV or CLI metadata"
                        )
                    else:
                        pass
                clips[request_id] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_count": sample_count,
                    "sample_rate": sample_rate,
                }
    failures.extend(validate_outputs(golden, clips))
    measured = {
        "model": golden["model"],
        "device": device,
        "metrics": {},
        "clips": clips,
    }
    if outputs is not None:
        outputs.mkdir(parents=True, exist_ok=True)
        name = "auk-flash.json" if golden["model"].endswith("AuK-Flash") else "auk.json"
        (outputs / name).write_text(
            json.dumps(measured, ensure_ascii=False, indent=1) + "\n"
        )
    else:
        pass
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Generated {len(clips)} waveform cases.",
    ]
    if write_path is not None and not failures:
        golden["devices"][device] = {"metrics": {}, "clips": clips}
        write_path.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")
        lines.append(f"Recorded exact waveforms in {write_path}.")
    elif device not in golden["devices"]:
        failures.append(
            "no AuK golden for this chip; import the golden-outputs artifact after review"
        )
    else:
        reference = golden["devices"][device]["clips"]
        failures.extend(validate_outputs(golden, reference))
        failures.extend(
            f"{request_id}: differs from this chip's golden waveform"
            for request_id, result in clips.items()
            if result != reference.get(request_id)
        )
    return lines, failures
