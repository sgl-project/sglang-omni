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
    waveforms: dict[str, AudioDigest] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="music3-golden-") as temporary_path:
        temporary_directory = Path(temporary_path)
        request_file = temporary_directory / "requests.jsonl"
        request_file.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False) + "\n" for case in golden["cases"]
            )
        )
        resident_waveforms: dict[str, AudioDigest] = {}
        for residency_mode, offload_flags in (
            ("resident", ()),
            ("offload", ("--stage-offload-components", "ar,dit")),
        ):
            audio_directory = (
                outputs or temporary_directory
            ) / f"music3-{residency_mode}-audio"
            waveforms = {}
            process_result = subprocess.run(
                [
                    str(runtime_bin / "music3_synthesize"),
                    "--model-path",
                    str(data_root / "models" / golden["model"].replace("/", "_")),
                    "--requests",
                    str(request_file),
                    "--output-directory",
                    str(audio_directory),
                    *offload_flags,
                ],  # fmt: skip
                capture_output=True,
                text=True,
                check=False,
            )
            if process_result.returncode:
                failures.append(
                    f"{residency_mode} music3_synthesize exited {process_result.returncode}: {process_result.stderr[-2000:]}"
                )
            else:
                for line in process_result.stdout.splitlines():
                    synthesis_result = json.loads(line)
                    case_id = synthesis_result["id"]
                    if case_id in waveforms:
                        failures.append(f"{case_id}: duplicate native output")
                        continue
                    else:
                        pass
                    audio_path = Path(synthesis_result["file"])
                    with wave.open(str(audio_path)) as audio_file:
                        sample_count = audio_file.getnframes()
                        sample_rate = audio_file.getframerate()
                        pcm_bytes = audio_file.readframes(sample_count)
                        if (
                            audio_file.getnchannels() != 2
                            or audio_file.getsampwidth() != 2
                            or len(pcm_bytes) != sample_count * 4
                            or sample_count != synthesis_result["sample_count"]
                            or sample_rate != synthesis_result["sample_rate"]
                            or not np.any(np.frombuffer(pcm_bytes, dtype="<i2"))
                        ):
                            failures.append(
                                f"{case_id}: invalid or silent stereo PCM16 WAV"
                            )
                        else:
                            pass
                    waveforms[case_id] = {
                        "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                        "sample_count": sample_count,
                        "sample_rate": sample_rate,
                        "generated_frames": synthesis_result["generated_frames"],
                    }
            failures.extend(validate_outputs(golden, waveforms))
            if residency_mode == "resident":
                resident_waveforms = waveforms
            elif waveforms != resident_waveforms:
                failures.append("Music3 offload differs from resident waveforms")
            else:
                pass
    if outputs is not None:
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "minimax-music3.json").write_text(
            json.dumps(
                {
                    "model": golden["model"],
                    "device": device,
                    "metrics": {},
                    "clips": waveforms,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
    else:
        pass
    summary_lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Generated {len(waveforms)} music cases in resident and offload modes.",
    ]
    if write_path is not None and not failures:
        golden["devices"][device] = {"metrics": {}, "clips": waveforms}
        write_path.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")
        summary_lines.append(f"Recorded waveforms in {write_path}.")
    elif device not in golden["devices"]:
        failures.append(
            "no Music3 golden for this chip; review and import the golden-outputs artifact"
        )
    else:
        reference_waveforms = golden["devices"][device]["clips"]
        failures.extend(validate_outputs(golden, reference_waveforms))
        failures.extend(
            f"{case_id}: differs from this chip's golden waveform"
            for case_id, waveform_digest in waveforms.items()
            if waveform_digest != reference_waveforms.get(case_id)
        )
    return summary_lines, failures
