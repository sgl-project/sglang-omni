# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Dataset loading, single-snapshot SHA256SUMS verification, and staged downloader for C3."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile
from huggingface_hub import snapshot_download
from pydantic import JsonValue

from benchmarks.c3.containment import (
    ensure_safe_contained_path,
    read_bytes_nofollow,
    validate_non_symlink_root_directory,
    write_bytes_atomic,
)
from benchmarks.c3.registry import (
    C3_DATASET_ID,
    C3_DATASET_REVISION,
    C3_SUBTASK_SPECS,
    C3JudgeTemplateId,
    C3Language,
    C3Phenomenon,
    C3SubtaskName,
    C3SubtaskSpec,
    C3VerificationScope,
    validate_c3_instance_id,
)
from benchmarks.c3.schemas import C3ReferenceRecord, parse_reference_record_for_spec

logger = logging.getLogger(__name__)

C3_AUTOMATED_SUBTASK_SPECS: tuple[C3SubtaskSpec, ...] = tuple(
    subtask_spec
    for subtask_spec in C3_SUBTASK_SPECS
    if not subtask_spec.requires_human_audio_annotation
)
C3_MINI_INSTANCES_PER_SUBTASK = 2
C3_SHA256SUMS_DIGEST = (
    "e46f1861760bd0c70c0a8d5204493b302e1b4ba33eeddbe9840097a3b5db51f3"
)
C3_METADATA_SHA256: dict[str, str] = {
    "chinese/text/ambiguity/phonological/generation/heteronym.json": "b912c1718fcf3b9ef0cd2b6e98b60d0df4bd1a3e69e8d84b51b201afac637cdb",
    "chinese/text/ambiguity/phonological/generation/pause.json": "30b05f1ab0077dd8eff70cb2aaacb4594a5db832d097ce35f64d66d91262525f",
    "chinese/text/ambiguity/phonological/generation/tone.json": "add83c85b69457df52b61f913b80243873f3d7c9564a222947926e4a289bc39a",
    "chinese/text/ambiguity/phonological/heterograph.json": "fdcebce11b21cd3c11c1da6d3f6f917fc61078cd23ef212c0e06e27c51afc549",
    "chinese/text/ambiguity/phonological/heteronym.json": "c360e0ac8b77a45ecfcc05afcf64bf592470a72adfb5fd2652f55c5c11fc1611",
    "chinese/text/ambiguity/phonological/pause.json": "574da8650b8d99b2d158c86f65f9f27910ddaec64df9cd8c3c8e818f56be17b8",
    "chinese/text/ambiguity/phonological/tone.json": "f56521b320c2584c5d316a9766ed86512fbce53cd92293be010d899039992747",
    "chinese/text/ambiguity/semantic.json": "dd9238ccb224790976a73d2cd516b6f445f28ceb6b8e6bb456a1ad243941c6fa",
    "chinese/text/context-dependency/coreference.json": "2b1a4ba588440b3fb0b027c993dc02967fe7daccccd713af95404eac8418e680",
    "chinese/text/context-dependency/multi-turn.json": "fbdd7591e730a122a4f6b8509acfd7b1cb54e59219e62b2bb2266fc46993e870",
    "chinese/text/context-dependency/omission.json": "5ac19d38b28f1edaba8edfbac92bb070b833d11f1ce3e358750dc32c6b8e07ed",
    "english/text/ambiguity/phonological/generation.json": "c35c26117687936c229a0dcda316a8ce9643dc7dfb0b73da18f6628fb172d0af",
    "english/text/ambiguity/phonological/heterograph.json": "f3680a7e5a4558d020838aa85817eb941395f3c9b89bd80fe1eb15fa38814b70",
    "english/text/ambiguity/phonological/intonation.json": "6b44da7ce979a97e6732e3209c45438de8becb10aad995aaa27274159dd08c8f",
    "english/text/ambiguity/phonological/pause.json": "a5ce9015d5e8866ba52993ace7943c038586798847b203aa7438028667c7a91d",
    "english/text/ambiguity/phonological/stress.json": "6b6f17a9d578c9bf7c56e68cac4325c523b0d22c3920472e1d43be3b954a9b83",
    "english/text/ambiguity/semantic/lexical.json": "e8db2974d62539c89a23f67287b35a89e8c302273c154954a8941a160fa3b26c",
    "english/text/ambiguity/semantic/syntactic.json": "b07f3e93a252c528c458f752b562666de25494679a86a44c66505ae065c833f3",
    "english/text/context-dependency/coreference.json": "0e61847af8ce9d6f02438e96fe7e33f0e1c69a6c7d3aef06f9a9a7d10bd25708",
    "english/text/context-dependency/multi-turn.json": "6b210154ffd12541bf0e2a6421e7ee1ef47a6faf9330cd180aab0b67da0c9008",
    "english/text/context-dependency/omission.json": "d9cb9a2516b2bbc8e151b192e5c7b0151b164c09cde6aab6b31bc10a3f7b603f",
}


@dataclass(frozen=True, kw_only=True)
class C3InstanceRecord:
    instance_id: str
    language: C3Language
    phenomenon: C3Phenomenon
    subtask: C3SubtaskName
    category_path: str
    source_index: int
    audio_relative_paths: tuple[str, ...]
    user_turn_texts: tuple[str, ...]
    reference_record: C3ReferenceRecord
    judge_template_id: C3JudgeTemplateId | None
    is_multiturn: bool
    requires_human_audio_annotation: bool


@dataclass(frozen=True, kw_only=True)
class C3DatasetIdentitySummary:
    expected_huggingface_revision: str
    verification_scope: C3VerificationScope
    metadata_sha256: dict[str, str]
    sha256sums_verified: bool
    selected_audio_sha256_verified: bool
    metadata_matches_expected_revision: bool
    dataset_matches_expected_revision: bool


@dataclass(frozen=True, kw_only=True)
class C3LoadedDatasetSnapshot:
    instances: list[C3InstanceRecord]
    identity: C3DatasetIdentitySummary
    pinned_audio_sha256_by_path: dict[str, str] | None


def load_and_verify_c3_snapshot(
    dataset_root: Path,
    *,
    subtask_specs: list[C3SubtaskSpec] | None = None,
    mini: bool = False,
    max_samples_per_subtask: int | None = None,
    strict_counts: bool = True,
    verify_all_audio_headers: bool = True,
) -> C3LoadedDatasetSnapshot:
    """Read each metadata JSON and audio clip once via O_NOFOLLOW, computing SHA-256 and parsing from the identical bytes."""
    verified_root = validate_non_symlink_root_directory(
        dataset_root, create_if_missing=False
    )
    selected_subtask_specs = (
        subtask_specs if subtask_specs is not None else list(C3_AUTOMATED_SUBTASK_SPECS)
    )
    maximum_instances_per_subtask: int | None
    if mini and max_samples_per_subtask is not None:
        maximum_instances_per_subtask = min(
            C3_MINI_INSTANCES_PER_SUBTASK, max_samples_per_subtask
        )
    elif mini:
        maximum_instances_per_subtask = C3_MINI_INSTANCES_PER_SUBTASK
    else:
        maximum_instances_per_subtask = max_samples_per_subtask

    observed_json_hashes: dict[str, str] = {}
    json_cache: dict[str, list[JsonValue]] = {}
    all_json_matched = True
    for subtask_spec in selected_subtask_specs:
        json_relative_path = subtask_spec.reference_json_relative_path
        if json_relative_path not in json_cache:
            json_bytes = read_bytes_nofollow(verified_root, json_relative_path)
            observed_sha256 = hashlib.sha256(json_bytes).hexdigest()
            observed_json_hashes[json_relative_path] = observed_sha256
            if C3_METADATA_SHA256.get(json_relative_path) != observed_sha256:
                all_json_matched = False
            else:
                pass
            parsed_json = json.loads(json_bytes.decode("utf-8"))
            if not isinstance(parsed_json, list) or not parsed_json:
                raise ValueError(
                    f"Expected non-empty JSON list in {json_relative_path}"
                )
            else:
                json_cache[json_relative_path] = parsed_json
        else:
            pass

    pinned_sums: dict[str, str] | None = None
    try:
        sums_bytes = read_bytes_nofollow(verified_root, "SHA256SUMS")
    except (FileNotFoundError, ValueError):
        sums_bytes = b""
    if sums_bytes and hashlib.sha256(sums_bytes).hexdigest() == C3_SHA256SUMS_DIGEST:
        parsed_sums: dict[str, str] = {}
        sums_well_formed = True
        for raw_line in sums_bytes.decode("utf-8").splitlines():
            if not raw_line.strip():
                continue
            else:
                digest_part, separator, path_part = raw_line.partition("  ")
                if not separator or len(digest_part) != 64 or not path_part:
                    sums_well_formed = False
                    break
                else:
                    parsed_sums[path_part] = digest_part
        if sums_well_formed:
            pinned_sums = parsed_sums
        else:
            pass
    else:
        pass

    instances: list[C3InstanceRecord] = []
    audio_hashes_matched = pinned_sums is not None

    for subtask_spec in selected_subtask_specs:
        json_records = json_cache[subtask_spec.reference_json_relative_path]
        if strict_counts and len(json_records) != subtask_spec.expected_instance_count:
            raise ValueError(
                f"{subtask_spec.reference_json_relative_path} has {len(json_records)} records, "
                f"expected {subtask_spec.expected_instance_count}"
            )
        else:
            pass

        selected_instance_count = (
            min(len(json_records), maximum_instances_per_subtask)
            if maximum_instances_per_subtask is not None
            else len(json_records)
        )
        for source_index in range(selected_instance_count):
            user_turn_texts, reference_record = parse_reference_record_for_spec(
                subtask_spec, json_records[source_index], source_index
            )
            audio_relative_paths = (
                [
                    f"{subtask_spec.speech_directory_relative_path}/{source_index}/{turn_index}.mp3"
                    for turn_index in range(len(user_turn_texts))
                ]
                if subtask_spec.is_multiturn
                else [
                    f"{subtask_spec.speech_directory_relative_path}/{source_index}.mp3"
                ]
            )
            for audio_relative_path in audio_relative_paths:
                if pinned_sums is not None or verify_all_audio_headers:
                    audio_bytes = read_bytes_nofollow(
                        verified_root, audio_relative_path
                    )
                    if verify_all_audio_headers:
                        audio_info = soundfile.info(io.BytesIO(audio_bytes))
                        if audio_info.frames <= 0 or audio_info.samplerate <= 0:
                            raise ValueError(
                                f"Empty or unreadable C3 audio file: {audio_relative_path}"
                            )
                        else:
                            pass
                    else:
                        pass
                    if pinned_sums is not None and audio_hashes_matched:
                        expected_audio_sha256 = pinned_sums.get(audio_relative_path)
                        if (
                            expected_audio_sha256 is None
                            or hashlib.sha256(audio_bytes).hexdigest()
                            != expected_audio_sha256
                        ):
                            audio_hashes_matched = False
                        else:
                            pass
                    else:
                        pass
                else:
                    resolved_clip = ensure_safe_contained_path(
                        verified_root, audio_relative_path, create_parents=False
                    )
                    if not resolved_clip.is_file():
                        raise FileNotFoundError(
                            f"Missing C3 dataset file: {resolved_clip}"
                        )
                    else:
                        pass

            instance_id = validate_c3_instance_id(
                f"{subtask_spec.language}/{subtask_spec.subtask}/{source_index}"
            )
            instances.append(
                C3InstanceRecord(
                    instance_id=instance_id,
                    language=subtask_spec.language,
                    phenomenon=subtask_spec.phenomenon,
                    subtask=subtask_spec.subtask,
                    category_path=subtask_spec.category_path,
                    source_index=source_index,
                    audio_relative_paths=tuple(audio_relative_paths),
                    user_turn_texts=user_turn_texts,
                    reference_record=reference_record,
                    judge_template_id=subtask_spec.judge_template_id,
                    is_multiturn=subtask_spec.is_multiturn,
                    requires_human_audio_annotation=subtask_spec.requires_human_audio_annotation,
                )
            )

    fully_verified = all_json_matched and audio_hashes_matched
    verification_scope: C3VerificationScope = (
        "metadata_and_selected_audio_sha256"
        if audio_hashes_matched
        else (
            "metadata_and_audio_headers"
            if verify_all_audio_headers
            else "metadata_only"
        )
    )
    dataset_identity = C3DatasetIdentitySummary(
        expected_huggingface_revision=C3_DATASET_REVISION,
        verification_scope=verification_scope,
        metadata_sha256=dict(sorted(observed_json_hashes.items())),
        sha256sums_verified=(pinned_sums is not None),
        selected_audio_sha256_verified=audio_hashes_matched,
        metadata_matches_expected_revision=all_json_matched,
        dataset_matches_expected_revision=fully_verified,
    )
    return C3LoadedDatasetSnapshot(
        instances=instances,
        identity=dataset_identity,
        pinned_audio_sha256_by_path=(pinned_sums if fully_verified else None),
    )


def load_c3_instances(
    dataset_root: Path,
    *,
    subtask_specs: list[C3SubtaskSpec] | None = None,
    mini: bool = False,
    max_samples_per_subtask: int | None = None,
    strict_counts: bool = True,
) -> list[C3InstanceRecord]:
    return load_and_verify_c3_snapshot(
        dataset_root,
        subtask_specs=subtask_specs,
        mini=mini,
        max_samples_per_subtask=max_samples_per_subtask,
        strict_counts=strict_counts,
        verify_all_audio_headers=False,
    ).instances


def download_c3_dataset(
    destination_dir: Path,
    *,
    revision: str = C3_DATASET_REVISION,
    mini: bool = False,
    quiet: bool = False,
) -> Path:
    """Download into a fresh private staging directory, verify SHA256SUMS, and install via write_bytes_atomic."""
    if revision != C3_DATASET_REVISION:
        raise ValueError(f"C3 must use pinned revision {C3_DATASET_REVISION}")
    else:
        pass
    target_dir = validate_non_symlink_root_directory(
        destination_dir, create_if_missing=True
    )
    subset_label = "mini (2 per subtask)" if mini else "full (1,079 instances)"
    if not quiet:
        logger.info(
            f"Downloading {C3_DATASET_ID} ({subset_label}) revision={revision} to {target_dir} ..."
        )
    else:
        pass

    if mini:
        allow_patterns: list[str] = [
            "README.md",
            "LICENSE",
            "SHA256SUMS",
            *sorted(C3_METADATA_SHA256.keys()),
        ]
        for subtask_spec in C3_SUBTASK_SPECS:
            for sample_index in range(
                min(C3_MINI_INSTANCES_PER_SUBTASK, subtask_spec.expected_instance_count)
            ):
                allow_patterns.append(
                    f"{subtask_spec.speech_directory_relative_path}/{sample_index}/*.mp3"
                    if subtask_spec.is_multiturn
                    else f"{subtask_spec.speech_directory_relative_path}/{sample_index}.mp3"
                )
    else:
        allow_patterns = [
            "README.md",
            "LICENSE",
            "SHA256SUMS",
            "english/**",
            "chinese/**",
        ]

    with tempfile.TemporaryDirectory(prefix="c3_download_staging_") as staging_name:
        staging_root = Path(staging_name)
        snapshot_download(
            repo_id=C3_DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(staging_root),
            allow_patterns=allow_patterns,
        )
        staged_snapshot = load_and_verify_c3_snapshot(
            staging_root,
            subtask_specs=list(C3_SUBTASK_SPECS),
            mini=mini,
            strict_counts=True,
            verify_all_audio_headers=False,
        )
        if not staged_snapshot.identity.dataset_matches_expected_revision:
            raise RuntimeError(
                f"Downloaded C3 files failed SHA-256 verification against {revision}"
            )
        else:
            pass

        required_relative_files: set[str] = {"SHA256SUMS"}
        for optional_doc in ("README.md", "LICENSE"):
            if (staging_root / optional_doc).is_file():
                required_relative_files.add(optional_doc)
            else:
                pass
        for subtask_spec in C3_SUBTASK_SPECS:
            required_relative_files.add(subtask_spec.reference_json_relative_path)
        for instance in staged_snapshot.instances:
            required_relative_files.update(instance.audio_relative_paths)

        for relative_path in sorted(required_relative_files):
            staged_bytes = read_bytes_nofollow(staging_root, relative_path)
            write_bytes_atomic(target_dir, relative_path, staged_bytes)

    installed_snapshot = load_and_verify_c3_snapshot(
        target_dir,
        subtask_specs=list(C3_SUBTASK_SPECS),
        mini=mini,
        strict_counts=True,
        verify_all_audio_headers=False,
    )
    if not installed_snapshot.identity.dataset_matches_expected_revision:
        raise RuntimeError(
            f"Installed C3 files in {target_dir} failed SHA-256 verification against {revision}"
        )
    else:
        pass
    if not quiet:
        logger.info(f"Verified C3 ({subset_label}) in {target_dir}.")
    else:
        pass
    return target_dir
