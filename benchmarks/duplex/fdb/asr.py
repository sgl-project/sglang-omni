# SPDX-License-Identifier: Apache-2.0
"""Step 2: Parakeet ASR, v1.5 timing, v1.0 unified scoring, and the listening
quality of every reply."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from benchmarks.duplex.fdb.common import (
    LISTENING_DIR,
    PARAKEET_SHA256,
    Settings,
    listening_command,
    log,
    reference_command,
    require_v10_export,
    run_command,
    selected_v15,
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


def listening(settings: Settings, manifests: list[Path], out: Path) -> bool:
    """Cut and transcribe the replies in the scoring venv, then score them in
    the serving venv, which holds the UTMOS and WavLM models."""
    log(f"== Listening quality: reply transcripts (Parakeet on GPU {settings.gpu})")
    is_transcribed = run_command(
        listening_command(
            settings.scoring_python,
            "transcribe",
            *(argument for path in manifests for argument in ("--manifest", str(path))),
            "--out",
            str(out),
            "--nemo",
            str(settings.parakeet_nemo),
            "--nemo-sha256",
            PARAKEET_SHA256,
            "--device",
            "cuda",
        ),
        visible_gpus=settings.gpu,
    )
    if not is_transcribed:
        return False
    else:
        pass
    log(
        f"== Listening quality: UTMOS, voice consistency, omitted words (GPU {settings.gpu})"
    )
    return run_command(
        listening_command(
            Path(sys.executable), "score", "--out", str(out), "--device", "cuda"
        ),
        visible_gpus=settings.gpu,
    )


def asr(settings: Settings, repeat: int, retry_failed: bool) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    has_v15 = selected_v15(settings, repeat)
    v15_manifest = repeat_dir / "reference-audio" / "reference-manifest.json"
    if has_v15 and not v15_manifest.is_file():
        raise SystemExit(
            f"ERROR: {v15_manifest} is missing: `generate` did not finish the v1.5 "
            f"export. Delete {repeat_dir} and run "
            f"`{step_command('generate', settings, repeat)}` again."
        )
    else:
        pass
    v10_tree = require_v10_export(settings, repeat)
    scores = repeat_dir / "scores"
    is_asr_ok = is_timing_ok = True
    if has_v15:
        common_arguments = [
            "--reference-source",
            str(settings.fdb_source),
            "--tree",
            f"{settings.engine.name}={repeat_dir / 'reference-audio'}",
            "--out",
            str(scores),
            *(["--retry-failed"] if retry_failed else []),
        ]
        log(f"== v1.5 ASR (Parakeet on GPU {settings.gpu})")
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
        log("== v1.5 timing (official VAD intervals)")
        is_timing_ok = run_command(
            reference_command(
                settings, "timing", *common_arguments, "--audio-loader", "soundfile"
            ),
            visible_gpus="",
        )
    else:
        pass
    is_v10_ok = v10_tree is None or v10_asr(settings, v10_tree)
    is_unified_ok = True
    if v10_tree is not None and is_v10_ok:
        log("== v1.0 unified scoring")
        is_unified_ok = run_command(
            v10_command(settings, "unified-score", "--tree", str(v10_tree)),
            visible_gpus="",
        )
    else:
        pass
    manifests = [
        *([v15_manifest] if has_v15 else []),
        *([v10_tree / "manifest.json"] if v10_tree is not None else []),
    ]
    is_listening_ok = listening(settings, manifests, repeat_dir / LISTENING_DIR)

    problems = []
    if not (is_asr_ok and is_timing_ok):
        problems.append(
            f"WARNING: a v1.5 phase reported failures; logs are in {scores / 'logs'}. "
            f"Rerun with: {step_command('asr', settings, repeat)} --retry-failed"
        )
    else:
        pass
    if not is_v10_ok:
        problems.append(
            f"WARNING: the v1.0 ASR failed; its log is {v10_tree / 'logs' / 'asr.log'}. "
            f"Rerun `{step_command('asr', settings, repeat)}`; the v1.0 ASR restarts "
            "in full."
        )
    else:
        pass
    if not is_unified_ok:
        problems.append(
            f"WARNING: v1.0 unified scoring failed for {v10_tree}. "
            "After fixing the cause, backfill without repeating ASR with "
            f'`CUDA_VISIBLE_DEVICES="" {shlex.quote(str(settings.scoring_python))} '
            "-m benchmarks.eval.benchmark_duplex_v10 unified-score "
            f"--tree {shlex.quote(str(v10_tree))}`."
        )
    else:
        pass
    if not is_listening_ok:
        problems.append(
            "WARNING: the listening-quality phase failed; its output is above. "
            f"Rerun `{step_command('asr', settings, repeat)}`; it recomputes the "
            "listening metrics in full."
        )
    else:
        pass
    if problems:
        raise SystemExit("\n".join(problems))
    else:
        pass
    log(f"Step 2 done: {repeat_dir}")
