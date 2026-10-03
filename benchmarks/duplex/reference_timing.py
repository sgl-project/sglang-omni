# SPDX-License-Identifier: Apache-2.0
"""Run pinned timing formulas and retain validated intervals and VAD evidence."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Literal

from pydantic import JsonValue

from benchmarks.duplex.reference_core import (
    EOF_TOLERANCE_S,
    TIMING_END_TOLERANCE_S,
    VARIANTS,
    Engine,
    HashCache,
    Progress,
    atomic_write_json,
    audio_duration,
    canonical_hash,
    finite,
    package_versions,
    phase_log,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)
from benchmarks.duplex.reference_source import (
    AudioTensor,
    SileroLoader,
    load_official_timing,
    soundfile_load_wav,
)


def timing_config(
    paths: dict[str, Path], bridge: dict[str, JsonValue], loader: str
) -> dict[str, JsonValue]:
    versions = package_versions()
    return {
        "reference_timing_sha256": sha256_file(paths["timing"]),
        "entry": "process_folder(folder)",
        "vad_branch": bridge["vad_branch"],
        "silero_vad": versions["silero-vad"],
        "silero_jit_sha256": bridge["silero_jit_sha256"],
        "audio_loader": loader,
        "torch": versions["torch"],
        "torchaudio": versions["torchaudio"],
        "constants": bridge["constants"],
    }


def select_loader(module: ModuleType, probe: Path, choice: str) -> str:
    if choice == "official":
        return "official torchaudio.load"
    else:
        pass
    if choice == "auto":
        try:
            module.load_wav(probe)
            return "official torchaudio.load"
        except (ImportError, RuntimeError):
            pass
    else:
        pass
    module.load_wav = soundfile_load_wav(module.SR)
    return "soundfile.read(float32)+torchaudio.functional.resample bridge"


def validate_intervals(doc: JsonValue, input_s: float, output_s: float) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != {
        "latency_stop_list",
        "latency_resp_list",
    }:
        return ["unexpected latency_intervals keys"]
    else:
        pass
    errors = []
    limit = max(input_s, output_s) + TIMING_END_TOLERANCE_S
    for key, intervals in doc.items():
        for interval in intervals:
            if not (
                len(interval) == 2
                and finite(*interval)
                and 0 <= interval[0] <= interval[1] <= limit
            ):
                errors.append(f"{key} interval {interval} invalid")
            else:
                pass
    return errors


def run_timing(
    args: Namespace,
    engines: list[Engine],
    paths: dict[str, Path],
    hashes: HashCache,
    silero_loader: SileroLoader | None = None,
) -> Counter[str]:
    units = []
    for engine in engines:
        for sample_id in selected(engine, args.only):
            for variant in VARIANTS:
                if engine.eligible(sample_id, variant):
                    units.append((engine, sample_id, variant))
                else:
                    pass
    counts: Counter[str] = Counter()
    if not units:
        return counts
    else:
        pass
    module, bridge = load_official_timing(paths["timing"], silero_loader)
    probe_engine, probe_sample_id, probe_variant = units[0]
    probe = probe_engine.source_audio(probe_sample_id, VARIANTS[probe_variant][0])
    loader = select_loader(module, probe, args.audio_loader)
    config = timing_config(paths, bridge, loader)
    config_hash = canonical_hash(config)
    record_identity(
        args.out,
        "timing",
        {"timing_config": config, "timing_config_hash": config_hash, "bridge": bridge},
    )

    captured: list[
        tuple[Literal["raw"], list[tuple[int, int]]]
        | tuple[Literal["merged"], float, list[tuple[float, float]]]
    ] = []
    segment_seconds, vad_timestamps = module.seg_sec, module._vad_ts

    def capture_raw(waveform: AudioTensor) -> list[tuple[int, int]]:
        timestamps = vad_timestamps(waveform)
        captured.append(("raw", timestamps))
        return timestamps

    def capture_merged(waveform: AudioTensor, gap: float) -> list[tuple[float, float]]:
        segments = segment_seconds(waveform, gap)
        captured.append(("merged", gap, segments))
        return segments

    module._vad_ts = capture_raw
    module.seg_sec = capture_merged

    pending_variants = []
    for engine, sample_id, variant in units:
        receipt_path = (
            engine.sample_dir(sample_id) / "receipts" / f"timing-{variant}.json"
        )
        if receipt_path.exists():
            old = read_json(receipt_path)
            if old.get("status") == "ok":
                input_name, output_name = VARIANTS[variant]
                for name, key in (
                    (input_name, "input_sha256"),
                    (output_name, "output_sha256"),
                ):
                    if hashes.get(engine.source_audio(sample_id, name)) != old[key]:
                        raise ValueError(
                            f"Audio changed after timing: {sample_id}/{name}"
                        )
                    else:
                        pass
                folder = engine.sample_dir(sample_id)
                if variant == "clean":
                    folder = folder / "clean"
                else:
                    pass
                if sha256_file(folder / module.OUT_FILENAME) != old["intervals_sha256"]:
                    raise ValueError(f"Timing intervals changed: {sample_id}/{variant}")
                else:
                    pass
            else:
                pass
            if old.get("status") != "ok" and args.retry_failed:
                pending_variants.append((engine, sample_id, variant, receipt_path))
            elif old.get("config_hash") != config_hash:
                raise SystemExit(
                    f"{receipt_path}: timing config changed; use a new --out"
                )
            else:
                counts["reused"] += 1
                if old.get("status") != "ok":
                    counts[f"reused_{old['status']}"] += 1
                else:
                    pass
            continue
        else:
            pass
        pending_variants.append((engine, sample_id, variant, receipt_path))
    pending_variants = pending_variants[: args.limit]
    progress = Progress(args.out, "timing", len(pending_variants))
    with phase_log(args.out, "timing"):
        for engine, sample_id, variant, receipt_path in pending_variants:
            input_name, output_name = VARIANTS[variant]
            sample = engine.sample_dir(sample_id)
            folder = sample if variant == "overlap" else sample / "clean"
            input_path, output_path = engine.source_audio(
                sample_id, input_name
            ), engine.source_audio(sample_id, output_name)
            receipt = {
                "variant": variant,
                "config_hash": config_hash,
                "folder": str(folder),
                "official_inputs": {
                    "input.wav": str(input_path),
                    "output.wav": str(output_path),
                },
            }
            if not (input_path.exists() and output_path.exists()):
                receipt["status"] = "missing_audio"
            else:
                try:
                    engine.link(sample / input_name, input_path)
                    engine.link(sample / output_name, output_path)
                    if variant == "clean":
                        engine.link(folder / "input.wav", input_path)
                        engine.link(folder / "output.wav", output_path)
                    else:
                        pass
                    receipt.update(
                        input_sha256=hashes.get(input_path),
                        output_sha256=hashes.get(output_path),
                        input_s=audio_duration(input_path),
                        output_s=audio_duration(output_path),
                    )
                    captured.clear()
                    module.process_folder(folder)
                    intervals = read_json(folder / module.OUT_FILENAME)
                    merged = [
                        captured_segment
                        for captured_segment in captured
                        if captured_segment[0] == "merged"
                    ]
                    raw = [
                        captured_segment[1]
                        for captured_segment in captured
                        if captured_segment[0] == "raw"
                    ]
                    user_segments, model_segments = [
                        list(map(list, merged_segment[2])) for merged_segment in merged
                    ]
                    errors = validate_intervals(
                        intervals, receipt["input_s"], receipt["output_s"]
                    )
                    receipt.update(
                        status="invalid_intervals" if errors else "ok",
                        errors=errors,
                        intervals_sha256=sha256_file(folder / module.OUT_FILENAME),
                        raw_vad_samples={"user": raw[0], "model": raw[1]},
                        user_segments=user_segments,
                        model_segments=model_segments,
                        labels=eof_labels(
                            user_segments,
                            model_segments,
                            receipt["input_s"],
                            receipt["output_s"],
                        ),
                        stop_n=len(intervals["latency_stop_list"]),
                        resp_n=len(intervals["latency_resp_list"]),
                    )
                except Exception as exc:
                    receipt.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
            receipt["finished_at"] = utc_now()
            atomic_write_json(receipt_path, receipt)
            counts[receipt["status"]] += 1
            progress.add(receipt["status"])
    hashes.save()
    progress.write(finished=True)
    return counts


def eof_labels(
    user: list[list[float]], model: list[list[float]], input_s: float, output_s: float
) -> dict[str, JsonValue]:
    """Diagnostic censoring labels from the official merged segments; no interval is altered."""
    starts = [start_s for start_s, _ in model]
    return {
        "model_speech_at_output_end": bool(model)
        and model[-1][1] >= output_s - EOF_TOLERANCE_S,
        "user_speech_at_input_end": bool(user)
        and user[-1][1] >= input_s - EOF_TOLERANCE_S,
        "user_ends_without_later_model_start": [
            end_s for _, end_s in user if not any(start_s > end_s for start_s in starts)
        ],
    }
