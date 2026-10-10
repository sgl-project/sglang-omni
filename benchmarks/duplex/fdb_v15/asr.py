# SPDX-License-Identifier: Apache-2.0
"""Step 2: Parakeet ASR of all four v1.5 roles, official VAD timing intervals,
then Parakeet ASR of the v1.0 outputs."""

from __future__ import annotations

from pathlib import Path

from benchmarks.duplex.fdb_v15.common import (
    ENGINE_LABEL,
    PARAKEET_SHA256,
    V10_DIR,
    Settings,
    log,
    reference_command,
    run_command,
    step_command,
    v10_command,
)


def v10_asr(settings: Settings, tree: Path) -> bool:
    """Transcribe the v1.0 export once; the pinned path refuses a second pass."""
    if (tree / "asr.json").is_file():
        log(f"== v1.0 ASR done earlier: {tree / 'asr.json'}")
        return True
    else:
        pass
    log(f"== v1.0 ASR (Parakeet on GPU {settings.gpu})")
    return run_command(
        v10_command(
            settings,
            "reference-asr",
            "--tree",
            str(tree),
            "--reference-source",
            str(settings.fdb_source),
            "--nemo",
            str(settings.parakeet_nemo),
            "--nemo-sha256",
            PARAKEET_SHA256,
            "--device",
            "cuda",
        ),
        visible_gpus=settings.gpu,
    )


def asr(settings: Settings, repeat: int, retry_failed: bool) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    if not (repeat_dir / "reference-audio" / "reference-manifest.json").is_file():
        raise SystemExit(
            f"ERROR: run `{step_command('generate', settings, repeat)}` first."
        )
    else:
        pass
    scores = repeat_dir / "scores"
    common_arguments = [
        "--reference-source",
        str(settings.fdb_source),
        "--tree",
        f"{ENGINE_LABEL}={repeat_dir / 'reference-audio'}",
        "--out",
        str(scores),
        *(["--retry-failed"] if retry_failed else []),
    ]

    log(f"== ASR (Parakeet on GPU {settings.gpu})")
    is_asr_ok = run_command(
        reference_command(
            settings,
            "asr",
            *common_arguments,
            "--nemo",
            str(settings.parakeet_nemo),
            "--nemo-sha256",
            PARAKEET_SHA256,
            "--device",
            "cuda",
        ),
        visible_gpus=settings.gpu,
    )
    log("== Timing (official VAD intervals)")
    is_timing_ok = run_command(
        reference_command(
            settings, "timing", *common_arguments, "--audio-loader", "soundfile"
        ),
        visible_gpus="",
    )
    v10_tree = repeat_dir / V10_DIR / "reference"
    is_v10_ok = (
        v10_asr(settings, v10_tree) if (v10_tree / "manifest.json").is_file() else True
    )

    if not (is_asr_ok and is_timing_ok and is_v10_ok):
        raise SystemExit(
            f"WARNING: a phase reported failures; logs are in {scores / 'logs'}. "
            f"Rerun with: {step_command('asr', settings, repeat)} --retry-failed"
        )
    else:
        pass
    log(f"Step 2 done: {scores}")
