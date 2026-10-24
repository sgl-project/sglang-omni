# SPDX-License-Identifier: Apache-2.0
"""Sortformer golden check against the original Voxt's outputs, for check_golden.py.

Voxt runs MLX 0.31.1 and the runtime 0.32.3, so the tolerances allow kernel
differences only; the all-clip limits catch a computation that is slightly wrong.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SPEAKERS = 4
GRID_SECONDS = 0.01


def run_probe(runtime_bin: Path, model_directory: Path, clips: list[Path]) -> Path:
    output = Path(tempfile.mkdtemp(prefix="sortformer-golden-"))
    subprocess.run(
        [
            str(runtime_bin / "sortformer_probe"),
            "--model-path", str(model_directory),
            "--out", str(output),
        ]  # fmt: skip
        + [str(clip) for clip in clips],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return output


def speech(segments: list, speaker: int) -> set[int]:
    covered: set[int] = set()
    for start, end, owner in segments:
        if owner == speaker:
            covered.update(
                range(round(start / GRID_SECONDS), round(end / GRID_SECONDS))
            )
        else:
            pass
    return covered


def speech_mismatch(expected: list, actual: list) -> float:
    difference = union = 0
    for speaker in range(SPEAKERS):
        want, have = speech(expected, speaker), speech(actual, speaker)
        difference += len(want ^ have)
        union += len(want | have)
    return difference / union if union else 0.0


def as_float32(segments: list) -> list:
    return [
        (int(owner), np.float32(start), np.float32(end))
        for start, end, owner in segments
    ]


def within_one_frame(expected: list, actual: list, frame_seconds: float) -> bool:
    want, have = sorted(as_float32(expected)), sorted(as_float32(actual))
    return len(want) == len(have) and all(
        e[0] == a[0]
        and abs(e[1] - a[1]) <= frame_seconds + 1e-4
        and abs(e[2] - a[2]) <= frame_seconds + 1e-4
        for e, a in zip(want, have)
    )


def check(
    golden: dict, runtime_bin: Path, data_root: Path
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    model_directory = data_root / "models" / golden["model"].replace("/", "_")
    clip_ids = list(golden["clips"])
    clips = [data_root / "corpus" / "v1" / "clips" / f"{clip}.wav" for clip in clip_ids]
    output = run_probe(runtime_bin, model_directory, clips)
    tolerance = golden["tolerance"]
    frame_seconds = golden["feed"]["frame_seconds"]
    failures = []
    if not clip_ids:
        failures.append("the golden file has no clips")
    else:
        pass
    for clip in clip_ids:
        if (
            not golden["clips"][clip]["probabilities"]
            or not golden["clips"][clip]["state"]
        ):
            failures.append(f"`{clip}`: the golden file has no reference output")
        else:
            pass
    differences = []
    flips = decisions = exact = close = 0
    worst_mismatch = 0.0
    feed_ms = []
    for clip in clip_ids:
        reference = golden["clips"][clip]
        expected = np.frombuffer(
            base64.b64decode(reference["probabilities"]), dtype="<f4"
        ).reshape(-1, SPEAKERS)
        actual = np.fromfile(output / f"{clip}.probs.f32", dtype="<f4").reshape(
            -1, SPEAKERS
        )
        state = json.loads((output / f"{clip}.state.json").read_text())
        feed_ms += state["feed_ms"][1:]
        if expected.shape != actual.shape:
            failures.append(
                f"`{clip}`: {actual.shape[0]} frames, expected {expected.shape[0]}"
            )
            continue
        else:
            pass
        if any(state[key] != value for key, value in reference["state"].items()):
            failures.append(f"`{clip}`: final state differs")
        else:
            pass
        difference = np.abs(expected.astype(np.float64) - actual)
        differences.append(difference.ravel())
        largest = float(difference.max())
        p99 = float(np.quantile(difference, 0.99))
        flipped = int(np.sum((expected > 0.5) != (actual > 0.5)))
        flips += flipped
        decisions += expected.size
        if largest > tolerance["max_abs_probability"]:
            failures.append(f"`{clip}`: a probability is off by {largest:.4f}")
        else:
            pass
        if p99 > tolerance["p99_abs_probability"]:
            failures.append(f"`{clip}`: 99th percentile difference {p99:.4f}")
        else:
            pass
        if flipped > tolerance["max_flipped_decisions"] * expected.size:
            failures.append(f"`{clip}`: {flipped} decisions flipped at 0.5")
        else:
            pass

        segments = [
            [item["start"], item["end"], item["speaker"]]
            for item in json.loads((output / f"{clip}.segments.json").read_text())
        ]
        mismatch = speech_mismatch(reference["segments"], segments)
        worst_mismatch = max(worst_mismatch, mismatch)
        if mismatch > tolerance["max_speech_mismatch"]:
            failures.append(f"`{clip}`: speaker speech differs by {mismatch:.2%}")
        else:
            pass
        exact += int(as_float32(segments) == as_float32(reference["segments"]))
        close += int(within_one_frame(reference["segments"], segments, frame_seconds))

    everything = np.concatenate(differences) if differences else np.zeros(1)
    mean = float(everything.mean())
    over = float(np.mean(everything > 0.05))
    if mean > tolerance["mean_abs_probability"]:
        failures.append(f"mean probability difference {mean:.2e} over all clips")
    else:
        pass
    if over > tolerance["fraction_over_0_05"]:
        failures.append(f"{over:.2%} of probabilities are off by more than 0.05")
    else:
        pass
    lines = [
        f"### {golden['model']}",
        "",
        f"Reference: {golden['reference']}",
        "",
        f"- Probabilities: {decisions // SPEAKERS} frames on {len(clip_ids)} clips, "
        f"max |Δ| {everything.max():.2g}, median {np.median(everything):.2g}, "
        f"mean {mean:.2g}, {over:.2%} off by more than 0.05, "
        f"{flips} of {decisions} decisions flipped at 0.5",
        f"- Segments: {exact}/{len(clip_ids)} clips identical, {close}/{len(clip_ids)} "
        f"within one frame; speaker speech mismatch at most {worst_mismatch:.2%}",
        (
            f"- Feed latency: median {np.median(feed_ms):.0f} ms per 4.96 s feed"
            if feed_ms
            else "- Feed latency: no feeds"
        ),
    ]
    return lines, failures
