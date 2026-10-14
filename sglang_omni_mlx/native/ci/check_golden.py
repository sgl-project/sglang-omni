# SPDX-License-Identifier: Apache-2.0
"""Checks the native runtime against a model's golden outputs on the frozen corpus.

Greedy decoding follows each Apple chip's GPU arithmetic, so a golden file keeps
exact outputs per chip; other chips are gated on error rates within tolerance.
Silero VAD and Sortformer golden files are checked within kernel tolerances.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Callable

# (runtime bin, data root, golden file, manifest of the clips to run) -> outputs
TranscribeCorpus = Callable[[Path, Path, dict, dict[str, dict]], dict[str, dict]]

import sortformer_golden
import vad_golden

CI_DIRECTORY = Path(__file__).resolve().parent
COMPARED_FIELDS = ("text", "language", "generated_token_count", "finish_reason")
QUALITY_GROUPS = {
    "wer_en": lambda clip: clip["lang"] == "en"
    and clip["stratum"] not in ("silence", "noise"),
    "cer_zh": lambda clip: clip["lang"] == "zh",
    "mer_mixed": lambda clip: clip["stratum"] == "mixed",
}
# MOSS-Transcribe-Diarize tags: timestamps, speaker labels and acoustic events,
# which Voxt's plain-text rendering drops before the text is shown.
MOSS_TAG = re.compile(r"\[\d+(?:[.,]\d+)?\]|\[S\d+\]|\[[a-z][a-z0-9 _-]{0,31}\]")


def tokens(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "en":
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return [token.strip("'") for token in text.split() if token.strip("'")]
    else:
        return re.findall(r"[㐀-鿿豈-﫿]|[a-z0-9']+", text)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def quality(manifest: dict[str, dict], texts: dict[str, str]) -> dict[str, float]:
    metrics = {}
    for name, keep in QUALITY_GROUPS.items():
        errors = total = 0
        for clip_id, text in texts.items():
            clip = manifest[clip_id]
            if keep(clip):
                lang = "en" if clip["lang"] == "en" else "zh"
                reference = tokens(clip["reference"], lang)
                errors += edit_distance(reference, tokens(text, lang))
                total += len(reference)
            else:
                pass
        metrics[name] = round(errors / total, 5) if total else 0.0
    return metrics


def spoken_text(text: str, model_kind: str | None) -> str:
    """The text error rates are scored on: what Voxt shows of a model's output."""
    if model_kind == "moss_transcribe_diarize":
        return " ".join(MOSS_TAG.sub(" ", text).split())
    else:
        return text


def transcribe(
    runtime_bin: Path, model_directory: Path, clips: list[Path], request: dict
) -> dict[str, dict]:
    command = [
        str(runtime_bin / "qwen3_asr_transcribe"),
        "--model-path", str(model_directory),
        "--layout", request["layout"],
        "--language", request["language"],
        "--max-new-tokens", str(request["max_new_tokens"]),
    ]  # fmt: skip
    if request["stop_at_end_of_text"]:
        command.append("--stop-at-end-of-text")
    else:
        pass
    if request["stop_on_token_loop"]:
        command.append("--stop-on-token-loop")
    else:
        pass
    completed = subprocess.run(
        command + [str(clip) for clip in clips],
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        results[Path(row["file"]).stem] = {
            field: row[field] for field in COMPARED_FIELDS
        }
    return results


def model_directory(data_root: Path, repo: str) -> Path:
    return data_root / "models" / repo.replace("/", "_")


def chip() -> str:
    return subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_golden(path: Path, golden: dict) -> None:
    path.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")


def report(lines: list[str], failures: list[str]) -> None:
    text = "\n".join(lines + [f"\n- {failure}" for failure in failures[:10]]) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(text)
    else:
        pass
    sys.exit(1 if failures else 0)


def check(
    golden: dict, device: str, results: dict, metrics: dict
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    devices = golden["devices"]
    tolerance = golden["tolerance"]
    if device in devices:
        reference_device = device
    else:
        reference_device = next(iter(devices))
    reference = devices[reference_device]
    identical = sum(
        results.get(clip_id) == outputs
        for clip_id, outputs in reference["clips"].items()
    )
    total = len(reference["clips"])
    failures = []
    if not total:
        failures.append("the golden file has no clips")
    elif device in devices:
        failures += [
            f"`{clip_id}` differs from its golden output"
            for clip_id, outputs in reference["clips"].items()
            if results.get(clip_id) != outputs
        ]
    else:
        if identical < tolerance["min_identical"] * total:
            failures.append(
                f"only {identical}/{total} clips match {reference_device}'s outputs"
            )
        else:
            pass
        for name, value in metrics.items():
            delta = (value - reference["metrics"][name]) * 100
            if abs(delta) > tolerance["metric_pp"]:
                failures.append(f"{name} is {delta:+.2f} pp from {reference_device}'s")
            else:
                pass
    mode = (
        "exact"
        if device in devices
        else f"no golden outputs for this chip; error rates within "
        f"{tolerance['metric_pp']} pp of {reference_device}'s"
    )
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden clips identical to {reference_device}: {identical}/{total} ({mode})",
        "",
        "| | Original Voxt (Swift) | Native runtime | Δ |",
        "|---|---|---|---|",
    ]
    for name, value in metrics.items():
        if golden["baseline"].get("source") == "pending":
            lines.append(f"| {name} | pending | {value:.2%} | n/a |")
        else:
            baseline = golden["baseline"][name]
            lines.append(
                f"| {name} | {baseline:.2%} | {value:.2%} | {(value - baseline) * 100:+.2f} pp |"
            )
    return lines, failures


def run(transcribe_corpus: TranscribeCorpus) -> None:
    """Checks, records or imports one golden file; transcribe_corpus runs the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--import", dest="import_path", type=Path)
    arguments = parser.parse_args()

    golden = json.loads(arguments.golden.read_text())
    if arguments.import_path:
        outputs = json.loads(arguments.import_path.read_text())
        if outputs["model"] != golden["model"]:
            raise SystemExit(f"{arguments.import_path} is not {golden['model']}")
        else:
            pass
        golden["devices"][outputs["device"]] = {
            "metrics": outputs["metrics"],
            "clips": outputs["clips"],
        }
        write_golden(arguments.golden, golden)
        print(f"added {outputs['device']} to {arguments.golden}")
        return
    elif arguments.runtime_bin is None or arguments.data_root is None:
        raise SystemExit("--runtime-bin and --data-root are required")
    else:
        pass

    manifest = {
        row["id"]: row
        for row in map(
            json.loads,
            (CI_DIRECTORY / "corpus" / "manifest.jsonl").read_text().splitlines(),
        )
        if row["duration"] > golden.get("clips_over_seconds", 0)
    }
    if golden.get("kind") == "silero_vad":
        report(
            *vad_golden.check(
                golden, arguments.runtime_bin, arguments.data_root, list(manifest)
            )
        )
    elif golden.get("kind") == "sortformer":
        report(
            *sortformer_golden.check(golden, arguments.runtime_bin, arguments.data_root)
        )
    else:
        pass
    results = transcribe_corpus(
        arguments.runtime_bin, arguments.data_root, golden, manifest
    )
    metrics = quality(
        manifest,
        {
            clip_id: spoken_text(row["text"], golden.get("model_kind"))
            for clip_id, row in results.items()
        },
    )
    device = chip()

    if arguments.outputs:
        arguments.outputs.mkdir(parents=True, exist_ok=True)
        (arguments.outputs / arguments.golden.name).write_text(
            json.dumps(
                {
                    "model": golden["model"],
                    "device": device,
                    "metrics": metrics,
                    "clips": results,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
    else:
        pass
    if arguments.write:
        golden["devices"][device] = {"metrics": metrics, "clips": results}
        write_golden(arguments.golden, golden)
        print(f"wrote {len(results)} clips for {device} to {arguments.golden}")
        return
    else:
        pass
    report(*check(golden, device, results, metrics))


def transcribe_qwen3_asr(
    runtime_bin: Path, data_root: Path, golden: dict, manifest: dict[str, dict]
) -> dict[str, dict]:
    clips = [
        data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav" for clip_id in manifest
    ]
    return transcribe(
        runtime_bin,
        model_directory(data_root, golden["model"]),
        clips,
        golden["request"],
    )


if __name__ == "__main__":
    run(transcribe_qwen3_asr)
