# SPDX-License-Identifier: Apache-2.0
"""Silero VAD golden check for check_golden.py, against the original Voxt's outputs
(Swift on MLX 0.31.1). The runtime's MLX 0.32.3 kernels differ in the last bits, so
the checks allow for kernel differences and nothing more."""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_probe(
    runtime_bin: Path, model_directory: Path, clips: list[Path], profiles: dict
) -> Path:
    output = Path(tempfile.mkdtemp(prefix="silero-golden-"))
    (output / "profiles.json").write_text(json.dumps(profiles))
    subprocess.run(
        [
            str(runtime_bin / "silero_vad_probe"),
            "--model-path", str(model_directory),
            "--out", str(output),
            "--profiles", str(output / "profiles.json"),
        ]  # fmt: skip
        + [str(clip) for clip in clips],
        check=True,
    )
    return output


def speech_mismatch(expected: list, actual: list) -> float:
    """Samples covered by one side only, over samples covered by either."""

    def covered(ranges: list) -> list[tuple[int, int]]:
        return [(start, end) for start, end in ranges]

    events = sorted(
        [(start, 1, 0) for start, _ in covered(expected)]
        + [(end, -1, 0) for _, end in covered(expected)]
        + [(start, 1, 1) for start, _ in covered(actual)]
        + [(end, -1, 1) for _, end in covered(actual)]
    )
    depth = [0, 0]
    either = only_one = 0
    previous = None
    for position, step, side in events:
        if previous is not None and position > previous:
            inside = (depth[0] > 0, depth[1] > 0)
            span = position - previous
            either += span if any(inside) else 0
            only_one += span if inside[0] != inside[1] else 0
        else:
            pass
        depth[side] += step
        previous = position
    return only_one / either if either else 0.0


def check(
    golden: dict, runtime_bin: Path, data_root: Path, clip_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    model_directory = data_root / "models" / golden["model"].replace("/", "_")
    clips = [data_root / "corpus" / "v1" / "clips" / f"{clip}.wav" for clip in clip_ids]
    output = run_probe(runtime_bin, model_directory, clips, golden["profiles"])
    tolerance = golden["tolerance"]
    failures = []

    largest = 0.0
    flips = chunks = 0
    differences = []
    for clip, encoded in golden["stream_probabilities"].items():
        expected = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        actual = np.fromfile(output / f"{clip}.stream.f32", dtype="<f4")
        if expected.shape != actual.shape:
            failures.append(
                f"`{clip}`: {actual.size} stream chunks, expected {expected.size}"
            )
            continue
        else:
            pass
        difference = np.abs(expected.astype(np.float64) - actual)
        differences.append(difference)
        largest = max(largest, float(difference.max()) if difference.size else 0.0)
        flips += int(np.sum((expected >= 0.5) != (actual >= 0.5)))
        chunks += expected.size
        if difference.size and difference.max() > tolerance["max_abs_probability"]:
            failures.append(
                f"`{clip}`: stream probability off by {difference.max():.4f}"
            )
        else:
            pass
    mean = float(np.concatenate(differences).mean()) if differences else 0.0
    if not chunks:
        failures.append("the golden file has no stream probabilities")
    else:
        pass
    if mean > tolerance["mean_abs_probability"]:
        failures.append(f"mean stream probability difference {mean:.2e}")
    else:
        pass
    if flips > tolerance["max_flipped_decisions"] * chunks:
        failures.append(f"{flips} stream chunks fall on the other side of 0.5")
    else:
        pass

    exact = total = 0
    worst = 0.0
    for clip in clip_ids:
        actual = json.loads((output / f"{clip}.timestamps.json").read_text())
        for profile, expected in golden["timestamps"][clip].items():
            total += 1
            exact += int(actual[profile] == expected)
            mismatch = speech_mismatch(expected, actual[profile])
            worst = max(worst, mismatch)
            if mismatch > tolerance["max_speech_mismatch"]:
                failures.append(
                    f"`{clip}` ({profile}): speech differs by {mismatch:.2%}"
                )
            else:
                pass
    if not total:
        failures.append("the golden file has no speech timestamps")
    elif exact < tolerance["min_identical_timestamps"] * total:
        failures.append(f"only {exact}/{total} speech timestamp sets are identical")
    else:
        pass

    lines = [
        f"### {golden['model']}",
        "",
        f"Reference: {golden['reference']}",
        "",
        f"- Stream probabilities: {chunks} chunks on {len(golden['stream_probabilities'])} clips, "
        f"max |Δ| {largest:.2g}, mean {mean:.2g}, {flips} decisions flipped at 0.5",
        f"- Speech timestamps: {exact}/{total} identical; speech differs by at most {worst:.2%}",
    ]
    return lines, failures
