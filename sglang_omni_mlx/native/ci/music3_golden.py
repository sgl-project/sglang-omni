# SPDX-License-Identifier: Apache-2.0
"""Records and checks native Music3 PCM waveforms for each Apple chip."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TypedDict

import numpy as np


class SynthesisCase(TypedDict):
    id: str
    input: str
    instructions: str
    seed: int
    max_new_tokens: int
    dit_steps: int


class AudioDigest(TypedDict):
    sha256: str
    generated_frames: int
    sample_count: int
    sample_rate: int


class DeviceOutputs(TypedDict):
    metrics: dict[str, float]
    clips: dict[str, AudioDigest]


class Golden(TypedDict):
    model: str
    kind: str
    reference: str
    cases: list[SynthesisCase]
    devices: dict[str, DeviceOutputs]


def validate_outputs(golden: Golden, clips: dict[str, AudioDigest]) -> list[str]:
    cases = {case["id"]: case for case in golden["cases"]}
    if not cases or len(cases) != len(golden["cases"]):
        raise ValueError("Music3 requires unique nonempty golden case IDs")
    else:
        pass
    failures = []
    if set(clips) != set(cases):
        failures.append("Music3 output IDs differ from the golden case IDs")
    else:
        pass
    for identifier, result in clips.items():
        if identifier not in cases:
            continue
        else:
            pass
        frames = result["generated_frames"]
        if not 1 <= frames <= cases[identifier]["max_new_tokens"]:
            failures.append(f"{identifier}: invalid generated frame count")
            continue
        else:
            pass
        if cases[identifier]["max_new_tokens"] > 200 and frames <= 200:
            failures.append(
                f"{identifier}: the overlap case did not cross an acoustic window"
            )
        else:
            pass
        native_samples = 0
        for start in range(0, frames, 100):
            end = min(start + 200, frames)
            native_samples += ((end - start) * 441 // 128) * 512
            native_samples -= (86 * 512 if start else 0) + (
                258 * 512 if end != frames else 0
            )
            if end == frames:
                break
            else:
                pass
        expected_samples = (native_samples * 320 + 440) // 441
        if (
            result["sample_count"] != expected_samples
            or result["sample_rate"] != 32000
            or re.fullmatch(r"[0-9a-f]{64}", result["sha256"]) is None
        ):
            failures.append(f"{identifier}: invalid waveform shape or SHA-256")
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
    clips: dict[str, AudioDigest] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="music3-golden-") as temporary:
        directory = Path(temporary)
        request_file = directory / "requests.jsonl"
        request_file.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False) + "\n" for case in golden["cases"]
            )
        )
        audio_directory = outputs / "music3-audio" if outputs else directory / "audio"
        completed = subprocess.run(
            [
                str(runtime_bin / "music3_synthesize"),
                "--model-path",
                str(data_root / "models" / golden["model"].replace("/", "_")),
                "--requests",
                str(request_file),
                "--output-directory",
                str(audio_directory),
            ],  # fmt: skip
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            failures.append(
                f"music3_synthesize exited {completed.returncode}: {completed.stderr[-2000:]}"
            )
        else:
            for line in completed.stdout.splitlines():
                result = json.loads(line)
                identifier = result["id"]
                if identifier in clips:
                    failures.append(f"{identifier}: duplicate native output")
                    continue
                else:
                    pass
                path = Path(result["file"])
                with wave.open(str(path)) as audio:
                    sample_count = audio.getnframes()
                    sample_rate = audio.getframerate()
                    pcm = audio.readframes(sample_count)
                    if (
                        audio.getnchannels() != 2
                        or audio.getsampwidth() != 2
                        or len(pcm) != sample_count * 4
                        or sample_count != result["sample_count"]
                        or sample_rate != result["sample_rate"]
                        or not np.any(np.frombuffer(pcm, dtype="<i2"))
                    ):
                        failures.append(
                            f"{identifier}: invalid or silent stereo PCM16 WAV"
                        )
                    else:
                        pass
                clips[identifier] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_count": sample_count,
                    "sample_rate": sample_rate,
                    "generated_frames": result["generated_frames"],
                }
    failures.extend(validate_outputs(golden, clips))
    if outputs is not None:
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "minimax-music3.json").write_text(
            json.dumps(
                {
                    "model": golden["model"],
                    "device": device,
                    "metrics": {},
                    "clips": clips,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
    else:
        pass
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Generated {len(clips)} music cases.",
    ]
    if write_path is not None and not failures:
        golden["devices"][device] = {"metrics": {}, "clips": clips}
        write_path.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")
        lines.append(f"Recorded waveforms in {write_path}.")
    elif device not in golden["devices"]:
        failures.append(
            "no Music3 golden for this chip; review and import the golden-outputs artifact"
        )
    else:
        reference = golden["devices"][device]["clips"]
        failures.extend(validate_outputs(golden, reference))
        failures.extend(
            f"{identifier}: differs from this chip's golden waveform"
            for identifier, result in clips.items()
            if result != reference.get(identifier)
        )
    return lines, failures
