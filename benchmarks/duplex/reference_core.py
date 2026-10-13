# SPDX-License-Identifier: Apache-2.0
"""Track immutable inputs, file identities and resumable scoring progress."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import types
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from pydantic import JsonValue
from typing_extensions import NotRequired

from benchmarks.duplex.run_artifacts import file_sha256

REFERENCE_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
REFERENCE_FILES = {
    "asr": (
        "v1_v1.5/get_transcript/asr.py",
        "aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf",
    ),
    "timing": (
        "v1_v1.5/evaluation/get_timing.py",
        "4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17",
    ),
    "behavior": (
        "v1_v1.5/evaluation/eval_behavior.py",
        "0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2",
    ),
    "instruction": (
        "v1_v1.5/evaluation/instruction/behavior.txt",
        "19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977",
    ),
}
V10_REFERENCE_FILES = {
    "asr": REFERENCE_FILES["asr"],
    "evaluate": (
        "v1_v1.5/evaluation/evaluate.py",
        "ae362d718f80428a6fe73d8a87e9e3b8935e86008267f0b2e34d1cc7573f1d4e",
    ),
    "pause_handling": (
        "v1_v1.5/evaluation/eval_pause_handling.py",
        "1e86519b4f4543f2f667ce60b9bea2f6148a3bc4255bb20c1ea111b45e0adbb5",
    ),
    "smooth_turn_taking": (
        "v1_v1.5/evaluation/eval_smooth_turn_taking.py",
        "cb7d6b987cd415e1b71b6eea06e67ea8fdc77a1167107e5c0552105bac05b2df",
    ),
    "user_interruption": (
        "v1_v1.5/evaluation/eval_user_interruption.py",
        "e701c14579c1477577b262265cfff75b28ac2e057beee5ba82572afa2a35963a",
    ),
    "backchannel": (
        "v1_v1.5/evaluation/eval_backchannel.py",
        "36371e3efb4a25e6b5afe317d70c81ad8f8815b95ef025baa06b11c378ecd0d9",
    ),
    "backchannel_distribution": (
        "v1_v1.5/evaluation/icc_gt_distribution.json",
        "92bcd0ff27246aaa9a0136737b476afd456cd393f003944c3394c140e590f2a8",
    ),
}
ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
JUDGE_MODEL = "gpt-4o-2024-08-06"
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
JUDGE_MAX_ATTEMPTS = 3
ASR_END_TOLERANCE_S = 0.08
TIMING_END_TOLERANCE_S = 1e-3
EOF_TOLERANCE_S = 0.05
VARIANTS = {
    "overlap": ("input.wav", "output.wav"),
    "clean": ("clean_input.wav", "clean_output.wav"),
}
AUDIO_FILES = tuple(name for pair in VARIANTS.values() for name in pair)
EVENT_SELECTION_RULE = (
    "campaign_event_selected_v1 (NOT official): stop = official stop intervals "
    "intersecting [event_start, event_end]; response = first official response "
    "interval whose start >= event_start"
)
PACKAGES = (
    "torch",
    "torchaudio",
    "numpy",
    "soundfile",
    "silero-vad",
    "nemo_toolkit",
    "openai",
    "tqdm",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary_path, "wb") as file_handle:
        file_handle.write(data)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, value: JsonValue) -> None:
    atomic_write_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def read_json(path: Path) -> JsonValue:
    with open(path, encoding="utf-8") as file_handle:
        return json.load(file_handle)


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def finite(*values: JsonValue) -> bool:
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    )


class HashCache:
    """sha256 by (resolved path, size, mtime_ns); trees are immutable after export."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.data: dict[str, str] = read_json(path) if path.exists() else {}
        self.dirty: bool = False

    def get(self, path: Path) -> str:
        resolved_path = path.resolve()
        file_stat = resolved_path.stat()
        key = f"{resolved_path}|{file_stat.st_size}|{file_stat.st_mtime_ns}"
        if key not in self.data:
            self.data[key] = file_sha256(resolved_path)
            self.dirty = True
        else:
            pass
        return self.data[key]

    def save(self) -> None:
        if self.dirty:
            atomic_write_json(self.path, self.data)
            self.dirty: bool = False
        else:
            pass


class ProgressState(TypedDict):
    phase: str
    pid: int
    started_at: str
    total_units: int
    counts: Counter[str]
    finished: bool
    updated_at: NotRequired[str]


class Progress:
    def __init__(self, out: Path, phase: str, total: int) -> None:
        self.path: Path = out / "progress" / f"{phase}.json"
        self.state: ProgressState = {
            "phase": phase,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "total_units": total,
            "counts": Counter(),
            "finished": False,
        }
        self.write()

    def add(self, status: str) -> None:
        self.state["counts"][status] += 1
        self.write()

    def write(self, finished: bool = False) -> None:
        self.state["updated_at"] = utc_now()
        self.state["finished"] = finished
        atomic_write_json(
            self.path, {**self.state, "counts": dict(self.state["counts"])}
        )


def load_module(path: Path, name: str) -> types.ModuleType:
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_manifest(path: Path, projection: Path | None) -> dict[str, JsonValue]:
    """Load reference-manifest.json into the canonical form, optionally via project(doc)."""
    manifest = read_json(path)
    if projection is not None:
        manifest = load_module(projection, "fdb_manifest_projection").project(manifest)
    else:
        pass
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: 'samples' must be a non-empty list")
    else:
        pass
    seen = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or sample_id.count("/") != 1
            or any(part in ("", ".", "..") for part in sample_id.split("/"))
            or sample_id in seen
            or ".." in sample_id
        ):
            raise ValueError(f"{path}: invalid or duplicate sample_id {sample_id!r}")
        else:
            pass
        seen.add(sample_id)
        category = sample_id.split("/")[0]
        if sample.setdefault("category", category) != category:
            raise ValueError(f"{sample_id}: category disagrees with sample_id")
        else:
            pass
        variants = sample.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValueError(
                f"{sample_id}: variants must be exactly {sorted(VARIANTS)}"
            )
        else:
            pass
        for name, variant in variants.items():
            if not isinstance(variant.get("eligible"), bool):
                raise ValueError(f"{sample_id}/{name}: 'eligible' must be a bool")
            else:
                pass
            if not variant["eligible"] and not variant.get("reasons"):
                raise ValueError(
                    f"{sample_id}/{name}: ineligible variant needs reasons"
                )
            else:
                pass
        span = sample.get("event_span_s")
        if span is not None and not (
            len(span) == 2 and finite(*span) and span[0] < span[1]
        ):
            raise ValueError(f"{sample_id}: invalid event_span_s {span}")
        else:
            pass
    return manifest


class Engine:
    def __init__(
        self,
        out: Path,
        name: str,
        tree: Path,
        manifest_path: Path,
        projection: Path | None,
    ) -> None:
        if not name.replace("-", "").replace("_", "").isalnum():
            raise SystemExit(f"invalid engine name {name!r}")
        else:
            pass
        self.name: str = name
        self.tree: Path = tree.resolve()
        self.root: Path = out / "engines" / name
        source_bytes = manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(source_bytes).hexdigest()
        receipt_path = self.root / "manifest-receipt.json"
        frozen = {
            "tree": str(self.tree),
            "source_manifest_sha256": manifest_sha,
            "projection": str(projection.resolve()) if projection else None,
            "projection_sha256": file_sha256(projection) if projection else None,
        }
        receipt = read_json(receipt_path) if receipt_path.exists() else None

        def check(keys: list[str]) -> None:
            changed = [key for key in keys if receipt.get(key) != frozen[key]]
            if changed:
                raise SystemExit(
                    f"{name}: {', '.join(changed)} changed since first phase; use a new --out"
                )
            else:
                pass

        if receipt is not None:
            check(list(frozen))
        else:
            pass
        self.manifest: dict[str, JsonValue] = load_manifest(manifest_path, projection)
        frozen["projected_manifest_sha256"] = canonical_hash(self.manifest)
        if receipt is not None:
            check(["projected_manifest_sha256"])
        else:
            atomic_write_bytes(self.root / "source-manifest.json", source_bytes)
            atomic_write_json(self.root / "projected-manifest.json", self.manifest)
            atomic_write_json(
                receipt_path,
                {
                    "engine": name,
                    "source_manifest": str(manifest_path.resolve()),
                    **frozen,
                    "samples": len(self.manifest["samples"]),
                    "created_at": utc_now(),
                },
            )
        self.samples: dict[str, dict[str, JsonValue]] = {
            sample["sample_id"]: sample for sample in self.manifest["samples"]
        }

    def sample_dir(self, sid: str) -> Path:
        return self.root / "samples" / sid

    def source_audio(self, sid: str, fname: str) -> Path:
        return self.tree / sid / fname

    def link(self, dst: Path, src: Path) -> None:
        """Read-only view of source audio inside the output tree."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.resolve() != src.resolve():
                raise SystemExit(f"{dst} points to {dst.resolve()}, expected {src}")
            else:
                pass
            return
        else:
            pass
        os.symlink(src.resolve(), dst)

    def eligible(self, sid: str, variant: str) -> bool:
        return self.samples[sid]["variants"][variant]["eligible"]


def selected(engine: Engine, only: list[str]) -> list[str]:
    sample_ids = sorted(engine.samples)
    if only:
        unknown = set(only) - set(sample_ids)
        if unknown:
            raise SystemExit(f"{engine.name}: unknown --only {sorted(unknown)}")
        else:
            pass
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in only]
    else:
        pass
    return sample_ids


def record_identity(
    out: Path, phase: str, extra: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    identity = {
        "phase": phase,
        "recorded_at": utc_now(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "wrapper_sha256": {
            path.name: file_sha256(path)
            for path in sorted(
                [
                    *Path(__file__).parent.glob("reference_*.py"),
                    Path(__file__).with_name("run_artifacts.py"),
                ]
            )
        },
        "reference_revision": REFERENCE_REVISION,
        "reference_files": {
            key: {"path": value[0], "sha256": value[1]}
            for key, value in REFERENCE_FILES.items()
        },
        **extra,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(out / "identity" / f"{phase}-{timestamp}.json", identity)
    return identity


@contextlib.contextmanager
def phase_log(out: Path, phase: str) -> Iterator[None]:
    """Official scripts print per file; keep that output with the run, not on the console."""
    path = out / "logs" / f"{phase}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(path, "a", encoding="utf-8") as file_handle,
        contextlib.redirect_stdout(file_handle),
    ):
        print(f"=== {phase} {utc_now()} pid={os.getpid()}")
        yield


def audio_duration(path: Path) -> float:
    import soundfile

    info = soundfile.info(str(path))
    return info.frames / info.samplerate
