# SPDX-License-Identifier: Apache-2.0
"""Export, transcribe and evaluate Full-Duplex-Bench v1.0 sessions with pinned reference code."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import JsonValue

from benchmarks.duplex.reference_asr import (
    USE_CUDA_GRAPH_DECODER,
    ParakeetModel,
    load_nemo_model,
    local_model_hub,
)
from benchmarks.duplex.reference_audio import (
    POLICY,
    POLICY_VERSION,
    analyze_variant,
    diagnostics,
    load_runs,
    sha_bytes,
    write_wav,
)
from benchmarks.duplex.reference_capture import TraceFormat
from benchmarks.duplex.reference_core import (
    ASR_MODEL_ID,
    JUDGE_MAX_ATTEMPTS,
    REFERENCE_REVISION,
    V10_REFERENCE_FILES,
    atomic_write_json,
    canonical_hash,
    load_module,
    package_versions,
    phase_log,
    read_json,
    record_identity,
    utc_now,
)
from benchmarks.duplex.reference_export import send_receipts_required
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import DECLARED_COUNTS, SUBSET_TASKS, SUBSETS, Task
from benchmarks.duplex.v10_evaluation import RUN_KIND
from benchmarks.duplex.v15_audio import normalize_audio, write_json

MANIFEST_KIND = "full-duplex-bench-v1.0-reference-audio"
SUMMARY_KIND = "full-duplex-bench-v1.0-reference-summary"
EVALUATE_TASKS: dict[Task, str] = {
    "pause_handling": "pause_handling",
    "turn_taking": "smooth_turn_taking",
    "user_interruption": "user_interruption",
    "backchannel": "backchannel",
}
REFERENCE_FILES_RECORD = {
    key: {"path": path, "sha256": sha256}
    for key, (path, sha256) in V10_REFERENCE_FILES.items()
}
# note (luojiaxuan): the pinned evaluator sends neither a token cap nor a thinking
# switch; a self-hosted Qwen judge would spend SGLang's default 128 new tokens on
# reasoning, so the non-official path turns thinking off and caps the answer.
OFFICIAL_JUDGE_BASE_URL = "https://api.openai.com/v1/"
SELF_HOSTED_JUDGE_OPTIONS: dict[str, JsonValue] = {
    "max_tokens": 512,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
RESULT_MEAN_LINE = re.compile(r"(?P<label>.+) - Mean: (?P<mean>\S+) ± (?P<std>\S+)")
RESULT_VALUE_LINE = re.compile(r"(?P<label>[^:]+):\s+(?P<value>\S+)")


def export(runs: list[Path], dataset_root: Path, output: Path) -> dict[str, JsonValue]:
    """Write <output>/<subset>/<id>/output.wav and its dataset annotation per eligible session."""
    output = output.resolve()
    for run in runs:
        if output.is_relative_to(run.resolve()):
            raise ValueError("Reference output must be outside every source run")
        elif read_json(run / "manifest.json").get("kind") != RUN_KIND:
            raise ValueError(f"{run} is not a {RUN_KIND} run")
        else:
            pass
    chosen, sources, superseded = load_runs(runs, TraceFormat.PCM16)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample_id in sorted(chosen):
        run, entry = chosen[sample_id]
        state = entry["variants"]["input"]
        directory = (run / state["directory"]).resolve()
        if not directory.is_relative_to(run):
            raise ValueError(f"Variant directory escapes run: {directory}")
        else:
            pass
        record, pcm, audio = analyze_variant(
            directory,
            TraceFormat.PCM16,
            None if state["input"] is None else state["input"]["sha256"],
            receipts_required=send_receipts_required(run),
        )
        reasons = record["window"]["reasons"]
        annotation = entry["paths"].get("annotation")
        if pcm is not None:
            source = dataset_root / state["source"]["file"]
            if file_sha256(source) != state["source"]["sha256"]:
                reasons.append("dataset source sha256 differs from run.json")
            elif sha_bytes(normalize_audio(source)[0]) != sha_bytes(pcm):
                reasons.append("renormalized dataset source differs from input.pcm")
            elif (
                annotation is not None
                and file_sha256(dataset_root / annotation)
                != entry["sha256"]["annotation"]
            ):
                reasons.append("dataset annotation sha256 differs from run.json")
            else:
                pass
        else:
            pass
        record["window"]["valid"] = not reasons
        flags = [
            name for name, value in record.get("boundary", {}).items() if value is True
        ]
        files = {}
        if reasons:
            record.pop("output", None)
            record.pop("boundary", None)
        else:
            target = output / sample_id
            target.mkdir(parents=True)
            files["output.wav"] = write_wav(target / "output.wav", audio)
            if annotation is not None:
                annotation_name = Path(annotation).name
                shutil.copyfile(dataset_root / annotation, target / annotation_name)
                files[annotation_name] = file_sha256(target / annotation_name)
            else:
                pass
        rows.append(
            {
                "sample_id": sample_id,
                "subset": entry["subset"],
                "task": entry["task"],
                "run": str(run),
                **record,
                "eligible": not reasons,
                "reasons": reasons,
                "flags": flags,
                "protocol_diagnostics": diagnostics(run, state),
                "files": files,
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "trace_format": TraceFormat.PCM16.value,
        "created_utc": utc_now(),
        "policy_version": POLICY_VERSION,
        "policy": POLICY,
        "timeline": "FIFO receipt-time playout in [0,T_input], mono 16 kHz PCM16",
        "dataset_root": str(dataset_root.resolve()),
        "sources": sources,
        "superseded": superseded,
        "builder_sha256": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("reference_audio.py"),
                Path(__file__).with_name("reference_capture.py"),
                Path(__file__).with_name("reference_export.py"),
                Path(__file__).with_name("run_artifacts.py"),
            )
        },
        "counts": {
            subset: {
                "selected": sum(row["subset"] == subset for row in rows),
                "eligible": len(exported_names(rows, subset)),
            }
            for subset in SUBSETS
        },
        "samples": rows,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def exported_names(rows: list[dict[str, JsonValue]], subset: str) -> list[str]:
    return sorted(
        row["sample_id"].split("/")[1]
        for row in rows
        if row["subset"] == subset and row["eligible"]
    )


def exported_subsets(tree: Path) -> list[str]:
    rows = read_json(tree / "manifest.json")["samples"]
    return [subset for subset in SUBSETS if exported_names(rows, subset)]


def transcribe(
    tree: Path,
    paths: dict[str, Path],
    nemo_path: Path,
    nemo_sha256: str | None,
    device: str,
    model: ParakeetModel | None = None,
) -> dict[str, int]:
    """Run the pinned get_time_aligned_transcription on every exported subset.

    For user interruption it crops output.wav at interrupt.json[0]["timestamp"][1]
    before ASR and adds that offset back to every word (asr.py lines 36-48, 66-68).
    """
    receipt_path = tree / "asr.json"
    if receipt_path.exists():
        raise SystemExit(f"{receipt_path} exists; transcribe a new export instead")
    else:
        pass
    checkpoint_sha256 = file_sha256(nemo_path)
    if nemo_sha256 is not None and checkpoint_sha256 != nemo_sha256:
        raise SystemExit(f"{nemo_path} sha256 {checkpoint_sha256} != --nemo-sha256")
    else:
        pass
    rows = read_json(tree / "manifest.json")["samples"]
    official = load_module(paths["asr"], "fdb_v10_asr_3e799c4")
    official.nemo_asr = local_model_hub(
        load_nemo_model(nemo_path, device) if model is None else model, device
    )
    config = {
        "reference_asr_sha256": file_sha256(paths["asr"]),
        "model_id": ASR_MODEL_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "device_type": device,
        "use_cuda_graph_decoder": USE_CUDA_GRAPH_DECODER,
        "packages": package_versions(),
    }
    record_identity(
        tree,
        "asr",
        {
            "asr_config": config,
            "nemo_path": str(nemo_path.resolve()),
            "reference_files": REFERENCE_FILES_RECORD,
        },
    )
    subsets = {}
    with phase_log(tree, "asr"):
        for subset in exported_subsets(tree):
            task = (
                "user_interruption"
                if SUBSET_TASKS[subset] == "user_interruption"
                else "default"
            )
            official.get_time_aligned_transcription(str(tree / subset), task)
            subsets[subset] = {
                "task": task,
                "transcripts": {
                    name: file_sha256(tree / subset / name / "output.json")
                    for name in exported_names(rows, subset)
                },
            }
    atomic_write_json(
        receipt_path, {"created_utc": utc_now(), "config": config, "subsets": subsets}
    )
    return {subset: len(value["transcripts"]) for subset, value in subsets.items()}


class JudgeLedger:
    """Chat client for the pinned interruption judge, recording every exchange.

    The pinned evaluator resends an identical request until its rating parses;
    the JUDGE_MAX_ATTEMPTS-th identical request is the last one sent. base_url
    is the endpoint the client resolved, so the summary names where the
    requests went, whatever the CLI flag or OPENAI_BASE_URL said.
    """

    def __init__(
        self,
        client: OpenAI,
        ledger_path: Path,
        base_url: str,
        served_model: str | None,
    ) -> None:
        self.client: OpenAI = client
        self.ledger_path: Path = ledger_path
        self.base_url: str = base_url
        self.served_model: str | None = served_model
        self.attempts: Counter[str] = Counter()
        self.exchanges: list[dict[str, JsonValue]] = []
        self.chat: types.SimpleNamespace = types.SimpleNamespace(completions=self)

    def create(
        self, *, model: str, messages: list[dict[str, str]], seed: int
    ) -> ChatCompletion:
        request_sha256 = canonical_hash(
            {"model": model, "messages": messages, "seed": seed}
        )
        if self.attempts[request_sha256] == JUDGE_MAX_ATTEMPTS:
            raise RuntimeError(
                f"judge request {request_sha256[:12]} got no parsable rating "
                f"in {JUDGE_MAX_ATTEMPTS} attempts; see {self.ledger_path}"
            )
        else:
            pass
        self.attempts[request_sha256] += 1
        sent_model = model if self.served_model is None else self.served_model
        options = {} if self.served_model is None else SELF_HOSTED_JUDGE_OPTIONS
        started_utc, started_s = utc_now(), time.monotonic()
        response = self.client.chat.completions.create(
            model=sent_model, messages=messages, seed=seed, **options
        )
        exchange = {
            "request_sha256": request_sha256,
            "attempt": self.attempts[request_sha256],
            "requested_model": model,
            "sent_model": sent_model,
            "sent_options": options,
            "returned_model": response.model,
            "seed": seed,
            "started_utc": started_utc,
            "elapsed_s": round(time.monotonic() - started_s, 3),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
            "content": response.choices[0].message.content,
        }
        self.exchanges.append(exchange)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as ledger_file:
            ledger_file.write(json.dumps(exchange, sort_keys=True) + "\n")
        return response

    def summary(self) -> dict[str, JsonValue]:
        return {
            "official": self.served_model is None
            and self.base_url == OFFICIAL_JUDGE_BASE_URL,
            "base_url": self.base_url,
            "requested_models": sorted(
                {exchange["requested_model"] for exchange in self.exchanges}
            ),
            "sent_models": sorted(
                {exchange["sent_model"] for exchange in self.exchanges}
            ),
            "returned_models": dict(
                Counter(exchange["returned_model"] for exchange in self.exchanges)
            ),
            "requests": len(self.exchanges),
            "distinct_requests": len(self.attempts),
            "retries": len(self.exchanges) - len(self.attempts),
            "max_attempts_per_request": JUDGE_MAX_ATTEMPTS,
            "prompt_tokens": sum(
                exchange["prompt_tokens"] for exchange in self.exchanges
            ),
            "completion_tokens": sum(
                exchange["completion_tokens"] for exchange in self.exchanges
            ),
            "ledger": str(self.ledger_path),
        }


def parse_result_block(stdout: str) -> dict[str, float]:
    """Numbers printed between the "[Result]" line and the closing dashed line."""
    lines = [line.strip() for line in stdout.splitlines()]
    if lines.count("[Result]") != 1:
        raise ValueError(
            f"expected one [Result] block, found {lines.count('[Result]')}"
        )
    else:
        pass
    metrics = {}
    for line in lines[lines.index("[Result]") + 1 :]:
        mean_match = RESULT_MEAN_LINE.fullmatch(line)
        value_match = RESULT_VALUE_LINE.fullmatch(line)
        if line.startswith("---"):
            break
        elif mean_match is not None:
            metrics[f"{mean_match['label']} mean"] = float(mean_match["mean"])
            metrics[f"{mean_match['label']} std"] = float(mean_match["std"])
        elif value_match is not None:
            metrics[value_match["label"]] = float(value_match["value"])
        elif line in ("", "[Raw Counts]"):
            continue
        else:
            raise ValueError(f"unrecognized result line {line!r}")
    return metrics


def evaluate(
    tree: Path,
    paths: dict[str, Path],
    subsets: list[str],
    judge: JudgeLedger | None,
) -> dict[str, JsonValue]:
    """Run the pinned evaluator on each subset and merge its result into summary.json."""
    manifest_path, asr_path = tree / "manifest.json", tree / "asr.json"
    summary_path = tree / "summary.json"
    rows, asr = read_json(manifest_path)["samples"], read_json(asr_path)
    inputs = {
        "manifest_sha256": file_sha256(manifest_path),
        "asr_sha256": file_sha256(asr_path),
    }
    summary = (
        read_json(summary_path)
        if summary_path.exists()
        else {
            "schema_version": 1,
            "kind": SUMMARY_KIND,
            "reference_revision": REFERENCE_REVISION,
            "reference_files": REFERENCE_FILES_RECORD,
            **inputs,
            "subsets": {},
        }
    )
    unexported = sorted(set(subsets) - set(exported_subsets(tree)))
    repeated = sorted(set(subsets) & set(summary["subsets"]))
    if any(summary[key] != value for key, value in inputs.items()):
        raise SystemExit(f"{tree}: manifest or ASR receipt changed since summary.json")
    elif unexported or repeated:
        raise SystemExit(
            f"not exported: {unexported}; already evaluated: {repeated}; "
            "evaluate a new export to repeat a subset"
        )
    else:
        pass
    record_identity(
        tree,
        "evaluate",
        {
            "subsets": subsets,
            "interpreter": sys.executable,
            "reference_files": REFERENCE_FILES_RECORD,
            "judge": None if judge is None else judge.summary(),
        },
    )
    for subset in subsets:
        names = exported_names(rows, subset)
        subset_dir = tree / subset
        transcripts = {
            name: file_sha256(subset_dir / name / "output.json") for name in names
        }
        if (
            sorted(path.name for path in subset_dir.iterdir()) != names
            or transcripts != asr["subsets"][subset]["transcripts"]
        ):
            raise SystemExit(f"{subset_dir} differs from manifest.json or asr.json")
        else:
            pass
        task = EVALUATE_TASKS[SUBSET_TASKS[subset]]
        log_path = tree / "logs" / f"evaluate-{subset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if task == "user_interruption":
            module = load_module(
                paths["user_interruption"], "fdb_v10_eval_user_interruption_3e799c4"
            )
            with (
                open(log_path, "w", encoding="utf-8") as log_file,
                contextlib.redirect_stdout(log_file),
            ):
                module.eval_user_interruption(str(subset_dir), judge)
            stdout = log_path.read_text(encoding="utf-8")
        else:
            # note (luojiaxuan): eval_backchannel opens ./icc_gt_distribution.json.
            completed = subprocess.run(
                [sys.executable, str(paths["evaluate"].resolve()), "--task", task]
                + ["--root_dir", str(subset_dir.resolve())],
                cwd=paths["evaluate"].resolve().parent,
                capture_output=True,
                text=True,
                check=False,
            )
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise SystemExit(
                    f"evaluate.py --task {task} exited {completed.returncode}; see {log_path}"
                )
            else:
                pass
            stdout = completed.stdout
        summary["subsets"][subset] = {
            "task": task,
            "declared": DECLARED_COUNTS[subset],
            "selected": sum(row["subset"] == subset for row in rows),
            "evaluated": len(names),
            "result": parse_result_block(stdout),
            "log": str(log_path.relative_to(tree)),
            "evaluated_utc": utc_now(),
            "judge": judge.summary() if task == "user_interruption" else None,
        }
        atomic_write_json(summary_path, summary)
    return summary
