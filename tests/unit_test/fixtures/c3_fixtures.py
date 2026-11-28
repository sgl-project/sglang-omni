# SPDX-License-Identifier: Apache-2.0
"""Synthetic C3 dataset builders and audio fixtures shared by C3 benchmark unit tests."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.c3.containment import write_json_atomic
from benchmarks.c3.registry import C3_SUBTASK_SPECS, C3SubtaskSpec

SYNTHETIC_AUDIO_SAMPLE_RATE_HZ = 24000
SYNTHETIC_AUDIO_SAMPLE_COUNT = 2400
SYNTHETIC_AUDIO_AMPLITUDE = 0.15


def build_valid_wav_base64() -> str:
    waveform = np.full(
        SYNTHETIC_AUDIO_SAMPLE_COUNT, SYNTHETIC_AUDIO_AMPLITUDE, dtype=np.float32
    )
    wav_bytes_buffer = io.BytesIO()
    soundfile.write(
        wav_bytes_buffer,
        waveform,
        SYNTHETIC_AUDIO_SAMPLE_RATE_HZ,
        format="WAV",
        subtype="PCM_16",
    )
    return base64.b64encode(wav_bytes_buffer.getvalue()).decode("ascii")


def write_synthetic_audio_clip(audio_path: Path, *, use_real_mp3: bool = False) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.full(
        SYNTHETIC_AUDIO_SAMPLE_COUNT, SYNTHETIC_AUDIO_AMPLITUDE, dtype=np.float32
    )
    if use_real_mp3:
        soundfile.write(
            str(audio_path), waveform, SYNTHETIC_AUDIO_SAMPLE_RATE_HZ, format="MP3"
        )
    else:
        soundfile.write(
            str(audio_path),
            waveform,
            SYNTHETIC_AUDIO_SAMPLE_RATE_HZ,
            format="WAV",
            subtype="PCM_16",
        )


def make_synthetic_record_for_spec(
    subtask_spec: C3SubtaskSpec,
    source_index: int,
    *,
    reference_sentinel: str = "REF_SENTINEL",
) -> JsonValue:
    if subtask_spec.is_multiturn and subtask_spec.language == "en":
        return [
            f"First English question {source_index} {reference_sentinel}",
            f"Middle English follow-up {source_index} {reference_sentinel}",
            f"First English question {source_index} {reference_sentinel}",
        ]
    elif subtask_spec.is_multiturn and subtask_spec.language == "zh":
        return {
            "history": [
                f"第一轮中文问题 {source_index} {reference_sentinel}",
                f"中间追问 {source_index} {reference_sentinel}",
            ],
            "question": (
                f"第一轮中文问题 {source_index} {reference_sentinel}。"
                "如果之前出现过该问题，则直接给出之前的答案。"
            ),
        }
    elif subtask_spec.language == "en":
        if subtask_spec.subtask == "phonological_heterograph":
            return {
                "content": f"Homophone content {source_index} {reference_sentinel}",
                "problem": f"Which word {source_index}?",
                "notation": f"Standard heterograph answer {source_index} {reference_sentinel}",
            }
        elif subtask_spec.subtask == "semantic_syntactic":
            return {
                "content": f"Syntactic sentence {source_index} {reference_sentinel}",
                "notation": {
                    "ambiguity": f"Attachment ambiguity {source_index} {reference_sentinel}",
                    "interpretations": [
                        f"Meaning A {source_index}",
                        f"Meaning B {source_index}",
                    ],
                },
            }
        elif subtask_spec.phenomenon == "coreference":
            return {
                "content": f"Coreference sentence {source_index} {reference_sentinel}",
                "problem": f"Who does she refer to {source_index}?",
                "answer": f"Target entity {source_index} {reference_sentinel}",
            }
        else:
            return {
                "content": f"English sentence {source_index} {reference_sentinel}",
                "notation": f"Standard notation {source_index} {reference_sentinel}",
            }
    elif subtask_spec.phenomenon in ("omission", "coreference"):
        return {
            "history": [
                f"甲：第一句 {source_index} {reference_sentinel}",
                f"乙：第二句 {source_index} {reference_sentinel}",
                f"甲：最后一句 {source_index} {reference_sentinel}",
            ],
            "omission_complete": f"补全省略句 {source_index} {reference_sentinel}",
            "coreference_complete": f"替换指代句 {source_index} {reference_sentinel}",
        }
    else:
        return {
            "content": f"中文歧义句 {source_index} {reference_sentinel}",
            "annotation": f"中文标准解释 {source_index} {reference_sentinel}",
            "answer": f"参考读音 {source_index}",
        }


def build_synthetic_c3_dataset(
    dataset_root: Path,
    *,
    reference_records_per_subtask: int | None = None,
    instances_with_audio_per_subtask: int | None = None,
) -> Path:
    """Materialize synthetic C3 JSON metadata and audio clips (official JSON record counts by default)."""
    written_json_paths: set[str] = set()
    for subtask_spec in C3_SUBTASK_SPECS:
        json_record_count = (
            reference_records_per_subtask
            if reference_records_per_subtask is not None
            else subtask_spec.expected_instance_count
        )
        if subtask_spec.reference_json_relative_path not in written_json_paths:
            reference_records = [
                make_synthetic_record_for_spec(subtask_spec, source_index)
                for source_index in range(json_record_count)
            ]
            write_json_atomic(
                dataset_root / subtask_spec.reference_json_relative_path,
                reference_records,
            )
            written_json_paths.add(subtask_spec.reference_json_relative_path)
        else:
            pass
        materialized_audio_instances_count = (
            min(json_record_count, instances_with_audio_per_subtask)
            if instances_with_audio_per_subtask is not None
            else json_record_count
        )
        for source_index in range(materialized_audio_instances_count):
            if subtask_spec.is_multiturn:
                for turn_index in range(3):
                    write_synthetic_audio_clip(
                        dataset_root
                        / subtask_spec.speech_directory_relative_path
                        / str(source_index)
                        / f"{turn_index}.mp3"
                    )
            else:
                write_synthetic_audio_clip(
                    dataset_root
                    / subtask_spec.speech_directory_relative_path
                    / f"{source_index}.mp3"
                )
    return dataset_root


def build_pinned_synthetic_c3_dataset(
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    instances_with_audio_per_subtask: int | None = None,
) -> tuple[Path, dict[str, str]]:
    """Build a synthetic C3 dataset and pin its exact metadata and SHA256SUMS digests in module constants."""
    build_synthetic_c3_dataset(
        dataset_root,
        instances_with_audio_per_subtask=instances_with_audio_per_subtask,
    )
    metadata_digests: dict[str, str] = {}
    sum_lines: list[str] = []
    audio_digests: dict[str, str] = {}
    for language_directory_name in ("english", "chinese"):
        for file_path in sorted((dataset_root / language_directory_name).rglob("*")):
            if file_path.is_file():
                relative_posix_path = file_path.relative_to(dataset_root).as_posix()
                file_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
                sum_lines.append(f"{file_sha256}  {relative_posix_path}")
                if relative_posix_path.endswith(".json"):
                    metadata_digests[relative_posix_path] = file_sha256
                else:
                    audio_digests[relative_posix_path] = file_sha256
            else:
                pass
    sums_bytes = ("\n".join(sum_lines) + "\n").encode("utf-8")
    sums_digest = hashlib.sha256(sums_bytes).hexdigest()
    (dataset_root / "SHA256SUMS").write_bytes(sums_bytes)
    monkeypatch.setattr("benchmarks.dataset.c3.C3_METADATA_SHA256", metadata_digests)
    monkeypatch.setattr("benchmarks.dataset.c3.C3_SHA256SUMS_DIGEST", sums_digest)
    return dataset_root, audio_digests
