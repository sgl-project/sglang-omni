# SPDX-License-Identifier: Apache-2.0
"""Step 2: Parakeet ASR of all four roles, then official VAD timing intervals."""

from __future__ import annotations

from benchmarks.duplex.fdb_v15.common import (
    ENGINE_LABEL,
    PARAKEET_SHA256,
    Settings,
    log,
    reference_command,
    run_command,
    step_command,
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

    if not (is_asr_ok and is_timing_ok):
        raise SystemExit(
            f"WARNING: a phase reported failures; logs are in {scores / 'logs'}. "
            f"Rerun with: {step_command('asr', settings, repeat)} --retry-failed"
        )
    else:
        pass
    log(f"Step 2 done: {scores}")
