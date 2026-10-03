# SPDX-License-Identifier: Apache-2.0
"""Transcribe reference audio with a local Parakeet checkpoint and resumable receipts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import time
import types
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Protocol, TypedDict

from pydantic import JsonValue

from benchmarks.duplex.reference_core import (
    ASR_END_TOLERANCE_S,
    ASR_MODEL_ID,
    VARIANTS,
    Engine,
    HashCache,
    Progress,
    atomic_write_bytes,
    atomic_write_json,
    audio_duration,
    canonical_hash,
    finite,
    load_module,
    package_versions,
    phase_log,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)


class WordTimestamp(TypedDict):
    word: str
    start: float
    end: float


class TimestampedHypothesis(Protocol):
    timestamp: dict[str, list[WordTimestamp]]


class ParakeetModel(Protocol):
    def transcribe(
        self, paths: list[str], timestamps: bool
    ) -> list[TimestampedHypothesis]: ...

    def cuda(self) -> ParakeetModel: ...

    def eval(self) -> ParakeetModel: ...


def asr_config(asr_path: Path, nemo_sha: str, device: str) -> dict[str, JsonValue]:
    return {
        "reference_asr_sha256": sha256_file(asr_path),
        "task": "default",
        "timestamps": True,
        "model_id": ASR_MODEL_ID,
        "checkpoint_sha256": nemo_sha,
        "device_type": device,
        "entry": "get_time_aligned_transcription(root, 'default', 'audio.wav')",
        "model_bridge": "nemo_asr.models.ASRModel.from_pretrained -> preloaded restore_from(local .nemo)",
        "nemo_toolkit": package_versions()["nemo_toolkit"],
        "torch": package_versions()["torch"],
    }


def local_model_hub(model: ParakeetModel, device: str) -> types.SimpleNamespace:
    """Stand-in for the official module's nemo_asr: returns the one preloaded local model."""

    class Placement:
        def cuda(self) -> ParakeetModel:
            # Note (wenyao): official hardcodes .cuda(); --device cpu keeps the CPU model.
            return model.cuda() if device == "cuda" else model

    def from_pretrained(model_name: str) -> Placement:
        if model_name != ASR_MODEL_ID:
            raise RuntimeError(f"official ASR requested {model_name!r}")
        else:
            pass
        return Placement()

    return types.SimpleNamespace(
        models=types.SimpleNamespace(
            ASRModel=types.SimpleNamespace(from_pretrained=from_pretrained)
        )
    )


def load_nemo_model(nemo_path: Path, device: str) -> ParakeetModel:
    import torch

    nemo_asr = importlib.import_module("nemo.collections.asr")
    if device == "cuda":
        device_count = torch.cuda.device_count()
        if device_count != 1:
            raise SystemExit(
                f"--device cuda needs exactly one visible GPU, found {device_count}"
            )
        else:
            pass
    else:
        pass
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location=torch.device(device)
    )
    model.eval()
    return model


def validate_transcript(
    doc: JsonValue, duration_s: float, tolerance_s: float = ASR_END_TOLERANCE_S
) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != {"text", "chunks"}:
        return ["keys must be exactly text, chunks"]
    else:
        pass
    errors = []
    if not isinstance(doc["text"], str) or not isinstance(doc["chunks"], list):
        return ["text must be str and chunks list"]
    else:
        pass
    words = []
    for i, chunk in enumerate(doc["chunks"]):
        timestamps_s = chunk.get("timestamp") if isinstance(chunk, dict) else None
        if (
            not isinstance(chunk, dict)
            or set(chunk) != {"text", "timestamp"}
            or not isinstance(chunk["text"], str)
        ):
            errors.append(f"chunk {i} malformed")
            continue
        else:
            pass
        words.append(chunk["text"])
        if not (
            isinstance(timestamps_s, list)
            and len(timestamps_s) == 2
            and finite(*timestamps_s)
        ):
            errors.append(f"chunk {i} timestamp not two finite numbers")
        elif not 0 <= timestamps_s[0] <= timestamps_s[1] <= duration_s + tolerance_s:
            errors.append(
                f"chunk {i} timestamp {timestamps_s} outside [0, {duration_s}]"
            )
        else:
            pass
    if doc["text"] != " ".join(words).strip():
        errors.append("text differs from joined chunk words")
    else:
        pass
    return errors


def transcribe_one(official: types.ModuleType, audio: Path, stage_root: Path) -> bytes:
    """Run the unmodified official function on a single-file staging root.

    Returns the official JSON bytes verbatim; re-serializing would reorder keys
    and change the judge payload built from json.load of these files.
    """
    if stage_root.exists():
        shutil.rmtree(stage_root)
    else:
        pass
    staging_sample = stage_root / "item"
    staging_sample.mkdir(parents=True)
    os.symlink(audio.resolve(), staging_sample / "audio.wav")
    if str(staging_sample / "audio.wav").count("audio.wav") != 1:
        raise RuntimeError("staging path would break official path.replace")
    else:
        pass
    try:
        official.get_time_aligned_transcription(str(stage_root), "default", "audio.wav")
        return (staging_sample / "audio.json").read_bytes()
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def run_asr(
    args: Namespace,
    engines: list[Engine],
    paths: dict[str, Path],
    hashes: HashCache,
    official: types.ModuleType | None = None,
    model: ParakeetModel | None = None,
) -> Counter[str]:
    nemo_path = Path(args.nemo)
    checkpoint_sha256 = hashes.get(nemo_path)
    if args.nemo_sha256 and checkpoint_sha256 != args.nemo_sha256:
        raise SystemExit(f"{nemo_path} sha256 {checkpoint_sha256} != --nemo-sha256")
    else:
        pass
    config = asr_config(paths["asr"], checkpoint_sha256, args.device)
    config_hash = canonical_hash(config)
    cache_root = args.out / "asr-cache" / config_hash[:16]
    atomic_write_json(cache_root / "config.json", config)

    uses: list[tuple[Engine, str, str, Path, str]] = []
    for engine in engines:
        for sample_id in selected(engine, args.only):
            for variant, files in VARIANTS.items():
                if not engine.eligible(sample_id, variant):
                    continue
                else:
                    pass
                for file_name in files:
                    source_path = engine.source_audio(sample_id, file_name)
                    uses.append(
                        (
                            engine,
                            sample_id,
                            file_name,
                            source_path,
                            hashes.get(source_path) if source_path.exists() else "",
                        )
                    )
    hashes.save()
    pending = sorted(
        {
            (audio_sha256, source_path)
            for *_, source_path, audio_sha256 in uses
            if audio_sha256
            and not cache_ok(cache_root / audio_sha256, args.retry_failed)
        },
        key=lambda x: x[0],
    )
    pending = list(
        {audio_sha256: source_path for audio_sha256, source_path in pending}.items()
    )[: args.limit]
    counts: Counter[str] = Counter()
    pending_sha256s = {audio_sha256 for audio_sha256, _ in pending}
    failed_cache_sha256s = set()
    progress = Progress(args.out, "asr", len(pending))

    if pending:
        if official is None:
            official = load_module(paths["asr"], "fdb_v15_asr_3e799c4")
        else:
            pass
        if model is None:
            model = load_nemo_model(nemo_path, args.device)
        else:
            pass
        official.nemo_asr = local_model_hub(model, args.device)
        record_identity(
            args.out,
            "asr",
            {
                "asr_config": config,
                "asr_config_hash": config_hash,
                "nemo_path": str(nemo_path.resolve()),
            },
        )
    else:
        pass
    with phase_log(args.out, "asr"):
        for audio_sha256, source_path in pending:
            cache_directory = cache_root / audio_sha256
            started_s = time.monotonic()
            receipt = {
                "audio_sha256": audio_sha256,
                "first_source": str(source_path),
                "config_hash": config_hash,
                "duration_s": audio_duration(source_path),
                "started_at": utc_now(),
            }
            try:
                transcript_bytes = transcribe_one(
                    official, source_path, args.out / "asr-stage" / str(os.getpid())
                )
                atomic_write_bytes(cache_directory / "audio.json", transcript_bytes)
                transcript = json.loads(transcript_bytes)
                receipt.update(
                    status="ok",
                    words=len(transcript.get("chunks", [])),
                    output_sha256=sha256_file(cache_directory / "audio.json"),
                )
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            receipt["elapsed_s"] = round(time.monotonic() - started_s, 3)
            atomic_write_json(cache_directory / "receipt.json", receipt)
            counts[receipt["status"]] += 1
            progress.add(receipt["status"])

    for engine, sample_id, file_name, source_path, audio_sha256 in uses:
        sample = engine.sample_dir(sample_id)
        stem = file_name.rsplit(".", 1)[0]
        receipt_path = sample / "receipts" / f"asr-{stem}.json"
        if (
            receipt_path.exists()
            and read_json(receipt_path).get("config_hash", config_hash) != config_hash
        ):
            raise SystemExit(f"{receipt_path}: ASR config changed; use a new --out")
        else:
            pass
        if not audio_sha256:
            atomic_write_json(
                receipt_path, {"status": "missing_audio", "source": str(source_path)}
            )
            counts["missing_audio"] += 1
            continue
        else:
            pass
        engine.link(sample / file_name, source_path)
        cache_directory = cache_root / audio_sha256
        cached_receipt = (
            read_json(cache_directory / "receipt.json")
            if (cache_directory / "receipt.json").exists()
            else None
        )
        if cached_receipt is None or cached_receipt["status"] != "ok":
            if (
                cached_receipt
                and audio_sha256 not in pending_sha256s
                and audio_sha256 not in failed_cache_sha256s
            ):
                counts["reused_failed"] += 1
                failed_cache_sha256s.add(audio_sha256)
            else:
                pass
            atomic_write_json(
                receipt_path,
                {
                    "status": "failed" if cached_receipt else "not_run",
                    "audio_sha256": audio_sha256,
                    "config_hash": config_hash,
                },
            )
            continue
        else:
            pass
        transcript = read_json(cache_directory / "audio.json")
        errors = validate_transcript(transcript, cached_receipt["duration_s"])
        if errors:
            status = "invalid_transcript"
            counts[status] += 1
        else:
            status = "ok"
            transcript_bytes = (cache_directory / "audio.json").read_bytes()
            if (
                hashlib.sha256(transcript_bytes).hexdigest()
                != cached_receipt["output_sha256"]
            ):
                raise SystemExit(
                    f"{cache_directory}/audio.json changed after transcription"
                )
            else:
                pass
            atomic_write_bytes(sample / f"{stem}.json", transcript_bytes)
        atomic_write_json(
            receipt_path,
            {
                "status": status,
                "errors": errors,
                "audio_sha256": audio_sha256,
                "config_hash": config_hash,
                "cache": str(cache_directory),
                "transcript_sha256": cached_receipt["output_sha256"],
                "words": (
                    len(transcript["chunks"])
                    if isinstance(transcript, dict)
                    and isinstance(transcript.get("chunks"), list)
                    else None
                ),
            },
        )
    progress.write(finished=True)
    return counts


def cache_ok(unit: Path, retry_failed: bool) -> bool:
    receipt = unit / "receipt.json"
    if not receipt.exists():
        return False
    else:
        pass
    return read_json(receipt)["status"] == "ok" or not retry_failed
