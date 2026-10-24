# SPDX-License-Identifier: Apache-2.0
"""dots.tts TTS golden check for check_golden.py.

Synthesis is deterministic per (checkpoint, seed): dots_tts_speech with a fixed
--seed produces identical WAV bytes on the same GPU, so a golden pins the exact
per-clip SHA-256 on the reference chip and lets other chips gate on how many
clips match that reference exactly. The golden file names a small en/zh test
subset of the speech corpus; each clip is synthesized from its reference text
(the transcript is the prompt).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def synthesize(
    runtime_bin: Path,
    model_directory: Path,
    text: str,
    seed: int,
    output: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
) -> Path:
    command = [
        str(runtime_bin / "dots_tts_speech"),
        "--model-path", str(model_directory),
        "--text", text,
        "--seed", str(seed),
        "--out", str(output),
    ]
    if ref_audio is not None:
        command += ["--ref-audio", str(ref_audio)]
        command += ["--ref-text", ref_text or ""]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


def check(
    golden: dict, runtime_bin: Path, data_root: Path, clip_ids: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures). Synthesizes every speech clip and
    compares the WAV SHA-256 against the golden per-device hashes."""
    model_directory = data_root / "models" / golden["model"].replace("/", "_")
    manifest = {
        clip["id"]: clip
        for row in (data_root / "corpus" / "manifest.jsonl").read_text().splitlines()
        for clip in [json.loads(row)]
    }
    # note (guozhihao-224): the golden file names the test set; the corpus has
    # hundreds of speech clips and every synthesis is ~seconds, so the golden
    # pins a small en/zh subset instead of the whole corpus.
    target_ids = golden.get("clips")
    if not target_ids:
        raise SystemExit("dots golden file needs a non-empty clips list")
    missing = [clip_id for clip_id in target_ids if clip_id not in manifest]
    if missing:
        raise SystemExit(f"dots golden lists unknown corpus clips: {missing}")
    seed = int(golden.get("seed", 42))
    devices = golden["devices"]
    chip = _chip()
    reference_device = chip if chip in devices else next(iter(devices))
    reference = devices[reference_device]

    work = Path(tempfile.mkdtemp(prefix="dots-golden-"))
    checked_ids = list(target_ids)
    hashes = {}
    failures = []
    for clip_id in target_ids:
        wav_path = synthesize(
            runtime_bin,
            model_directory,
            manifest[clip_id]["reference"],
            seed,
            work / f"{clip_id}.wav",
        )
        hashes[clip_id] = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    # note (guozhihao-224): the cloned clip covers the reference path (fbank,
    # x-vector and prompt prefill): first synthesize the prompt WAV, then clone
    # its speaker into a fixed target sentence.
    clone = golden.get("clone")
    if clone:
        prompt_clip = manifest[clone["prompt_id"]]
        prompt_wav = synthesize(
            runtime_bin,
            model_directory,
            prompt_clip["reference"],
            seed,
            work / f"{clone['prompt_id']}.prompt.wav",
        )
        target_wav = synthesize(
            runtime_bin,
            model_directory,
            clone["text"],
            seed,
            work / "clone.wav",
            ref_audio=prompt_wav,
            ref_text=prompt_clip["reference"],
        )
        hashes["clone"] = hashlib.sha256(target_wav.read_bytes()).hexdigest()
        checked_ids = target_ids + ["clone"]
    if chip in devices:
        failures += [
            f"`{clip_id}` differs from its golden WAV"
            for clip_id in checked_ids
            if hashes[clip_id] != reference.get(clip_id)
        ]
    else:
        identical = sum(
            hashes[clip_id] == reference.get(clip_id)
            for clip_id in checked_ids
        )
        if identical < golden["tolerance"]["min_identical"] * len(checked_ids):
            failures.append(
                f"only {identical}/{len(checked_ids)} WAVs match "
                f"{reference_device}'s outputs"
            )
    identical = sum(
        hashes[clip_id] == reference.get(clip_id)
        for clip_id in checked_ids
    )
    lines = [
        f"### {golden['model']}",
        "",
        f"{reference_device}: {identical}/{len(checked_ids)} WAVs identical "
        f"(seed {seed})",
        "",
        "exact mode"
        if chip in devices
        else f"compared against {reference_device} on {chip}",
    ]
    return lines, failures


def _chip() -> str:
    return subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()