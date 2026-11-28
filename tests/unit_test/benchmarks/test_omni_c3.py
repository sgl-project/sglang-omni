# SPDX-License-Identifier: Apache-2.0
"""Protocol, dataset registry, golden prompt digest, and security unit tests for C3."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import traceback
from pathlib import Path

import httpx
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.c3.config_manifest import C3EvalConfig
from benchmarks.c3.containment import write_json_atomic
from benchmarks.c3.http_transport import (
    compute_endpoint_identity_sha256,
    post_chat_completion_with_retries,
    resolve_bearer_headers,
)
from benchmarks.c3.judge_state import (
    C3JudgeEndpointSpec,
    load_c3_judge_configs,
    validate_single_judgment_record,
)
from benchmarks.c3.metrics import score_c3_language_instances
from benchmarks.c3.prompts import (
    C3_EVALUATION_TEMPLATES,
    build_c3_judge_messages,
    compute_c3_judge_prompt_sha256,
    compute_c3_prompt_bundle_sha256,
    parse_c3_judge_verdict,
)
from benchmarks.c3.registry import (
    C3_DATASET_ID,
    C3_DATASET_REVISION,
    select_c3_subtask_specs,
)
from benchmarks.c3.runner import (
    encode_audio_file_to_wav_base64_sync,
    extract_assistant_text_and_audio,
    validate_wav_audio_bytes,
)
from benchmarks.dataset.c3 import (
    C3InstanceRecord,
    download_c3_dataset,
    load_and_verify_c3_snapshot,
    load_c3_instances,
)
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_c3 import run_c3_evaluation
from tests.unit_test.fixtures.c3_fixtures import (
    build_pinned_synthetic_c3_dataset,
    build_synthetic_c3_dataset,
    write_synthetic_audio_clip,
)

EXPECTED_PROMPT_BUNDLE_SHA256 = (
    "f6dc1f4b9b8078861239ade5aff37b9dfe5fa5bb448dc77459645644243d9d03"
)

EXPECTED_SUBTASK_RENDERED_PROMPT_SHA256: dict[str, str] = {
    "en/coreference_detection": "2f817f375e277f0cbe626f7614d2f08b239a9dad6783190677e0ab83ea82eb4d",
    "en/coreference_resolution": "1e4b3c1dd7b9d12f79be91b6310a1f94f9ce5464b61604df7e0575edc1351805",
    "en/multiturn": "9716daef5c2f24b3ed24b2570fa33b883ae01a092e798492e080ed0d5bfe8a80",
    "en/omission_completion": "6ec11a7018447b9aacf140d80eb7ab2519cef80c9f2db7131b07875c0425717a",
    "en/omission_detection": "36c2b6982e427d30e0173949330077fffc591fa7f5ee07368318a123ef179c09",
    "en/phonological_heterograph": "55169a4399fb14dc00c275c50d947a6affd7bd46227d7eaf822ad43136c8340c",
    "en/phonological_intonation": "8630a4bda420f99ad24d3257b27f9518838b002b9e4261e415d72a21eb4274a6",
    "en/phonological_pause": "7d67cd42ca1f8a987afbcc0c7f01d30230a73bbc574be9ac0071a5a95c72a9bd",
    "en/phonological_stress": "2fe3d797b061bc6df60210c442990fa05d53f371d94e61aef9556c6d624270c5",
    "en/semantic_lexical": "7a4a2b9e7965cded84630a0766331c8e0bd67d68230658acf8bf58234b77e8e2",
    "en/semantic_syntactic": "5e067ce16f0d3efa1e8f50504628493016cad2d6242444e471219f95847fafcd",
    "zh/coreference_detection": "f520c3d2600d31a6c505dc2f9a6a05fa9f1e4ab56975d5b5d3a54c62ee4157be",
    "zh/coreference_resolution": "f1d37e444748e882611e8bc52d146f24f7007f40a6a55aed928e57e9a847aa9d",
    "zh/multiturn": "f71fa7bd377d7910c653e3a9da7d7ffc1190df4f516ea39f9520fd0a05aceccf",
    "zh/omission_completion": "2cb50d424468a4821ef988414d6e0139f57aa0b386f5f4074d3119b7528c91b7",
    "zh/omission_detection": "67cc62a9bf7cb88a653d797fcb14aff3dc8930e54714d7f6f32145e250be2438",
    **dict.fromkeys(
        (
            "zh/phonological_heterograph",
            "zh/phonological_heteronym",
            "zh/phonological_pause",
            "zh/phonological_tone",
            "zh/semantic",
        ),
        "8083f8a4e6bb99641a84340932fcf0883579365ae22eb1da54e06c22e4f5fb39",
    ),
}


def test_c3_dataset_registry_pinning_mp3_decode_and_containment_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify prepare.py registration, staged downloader containment, MP3 decoding, and symlink guards."""
    assert DATASETS["c3"] == C3_DATASET_ID
    assert DATASETS["c3-mini"] == f"{C3_DATASET_ID}:mini"
    assert len(select_c3_subtask_specs(include_human_generation=False)) == 21
    assert len(select_c3_subtask_specs(include_human_generation=True)) == 25

    with pytest.raises(ValueError, match="pinned dataset revision"):
        download_dataset("ChengqianMa/C3", revision="deadbeef" * 5)
    with pytest.raises(ValueError, match="Unsupported C3 subset"):
        download_dataset("ChengqianMa/C3:invalid_split")
    with pytest.raises(ValueError, match="pinned revision"):
        download_c3_dataset(tmp_path / "bad_revision", revision="0" * 40)

    mp3_dir = tmp_path / "mp3_holder"
    mp3_dir.mkdir()
    real_mp3_path = mp3_dir / "real_sample.mp3"
    write_synthetic_audio_clip(real_mp3_path, use_real_mp3=True)
    encoded_wav_base64, source_file_sha256 = encode_audio_file_to_wav_base64_sync(
        mp3_dir, "real_sample.mp3"
    )
    assert source_file_sha256 == hashlib.sha256(real_mp3_path.read_bytes()).hexdigest()
    audio_info = soundfile.info(
        io.BytesIO(base64.b64decode(encoded_wav_base64, validate=True))
    )
    assert (
        audio_info.format == "WAV"
        and audio_info.samplerate == 24000
        and audio_info.frames > 0
    )

    dataset_root = build_synthetic_c3_dataset(
        tmp_path / "c3_dataset", instances_with_audio_per_subtask=2
    )

    with pytest.raises(ValueError, match="Unknown C3 subtask"):
        select_c3_subtask_specs(
            subtask_filter=("en/phonological_pause", "en/phonological_generation_typo")
        )
    with pytest.raises(ValueError, match="excluded under automated scope"):
        select_c3_subtask_specs(
            include_human_generation=False,
            subtask_filter=("en/phonological_generation",),
        )

    unverified_full_counts = build_synthetic_c3_dataset(
        tmp_path / "unverified_full_counts"
    )
    unverified_snapshot = load_and_verify_c3_snapshot(
        unverified_full_counts, verify_all_audio_headers=False
    )
    assert unverified_snapshot.identity.dataset_matches_expected_revision is False
    mini_snapshot = load_and_verify_c3_snapshot(
        dataset_root, mini=True, strict_counts=True
    )
    assert len(mini_snapshot.instances) == 42
    with pytest.raises(ValueError, match="do not match the pinned Hugging Face"):
        asyncio.run(
            run_c3_evaluation(
                C3EvalConfig(
                    dataset_root=unverified_full_counts,
                    output_dir=tmp_path / "reject_unverified",
                    phase="generate",
                    scope="auto",
                    base_url="http://127.0.0.1:8000",
                    model="m",
                )
            )
        )

    captured_revisions: list[str] = []

    def fake_unverified_download(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_dir: str,
        allow_patterns: list[str],
    ) -> str:
        assert repo_id == C3_DATASET_ID and repo_type == "dataset" and allow_patterns
        captured_revisions.append(revision)
        build_synthetic_c3_dataset(Path(local_dir), instances_with_audio_per_subtask=2)
        return local_dir

    monkeypatch.setattr(
        "benchmarks.dataset.c3.snapshot_download", fake_unverified_download
    )
    with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
        download_dataset(
            "ChengqianMa/C3:mini", local_dir=str(tmp_path / "bad_download")
        )
    assert captured_revisions == [C3_DATASET_REVISION]

    victim_download_dir = tmp_path / "victim_download_dir"
    victim_download_dir.mkdir()
    trapped_destination = tmp_path / "trapped_destination"
    trapped_destination.mkdir()
    (trapped_destination / "english").symlink_to(victim_download_dir)

    def fake_verified_download(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_dir: str,
        allow_patterns: list[str],
    ) -> str:
        assert (
            repo_id == C3_DATASET_ID
            and repo_type == "dataset"
            and revision == C3_DATASET_REVISION
            and allow_patterns
        )
        build_pinned_synthetic_c3_dataset(
            Path(local_dir), monkeypatch, instances_with_audio_per_subtask=2
        )
        return local_dir

    monkeypatch.setattr(
        "benchmarks.dataset.c3.snapshot_download", fake_verified_download
    )
    with pytest.raises(ValueError, match="Symlink"):
        download_dataset("ChengqianMa/C3:mini", local_dir=str(trapped_destination))
    assert list(victim_download_dir.iterdir()) == []


def test_c3_all_18_official_prompt_templates_and_scoring_invariants(
    tmp_path: Path,
) -> None:
    """Verify upstream prompt bundle SHA-256, golden rendered SHA-256 across all 21 subtasks, and macro math."""
    assert parse_c3_judge_verdict("Analysis...\nThe answer is: yes.", "en") is True
    assert parse_c3_judge_verdict("Incorrect.The answer is: no.", "en") is False
    assert parse_c3_judge_verdict("The answer is: NO!", "en") is False
    assert parse_c3_judge_verdict("分析如下。\n最终答案为：是。", "zh") is True
    assert parse_c3_judge_verdict("未能理解语义。最终答案为：否。", "zh") is False
    assert parse_c3_judge_verdict("最终答案为: 否", "zh") is False
    assert parse_c3_judge_verdict("", "en") is None
    assert parse_c3_judge_verdict("The answer is: yes. Wait, maybe no.", "en") is None
    assert parse_c3_judge_verdict("最终答案为：是不确定。", "zh") is None
    assert parse_c3_judge_verdict("我觉得是对的。", "zh") is None

    assert (
        len(C3_EVALUATION_TEMPLATES["en"]) == 11
        and len(C3_EVALUATION_TEMPLATES["zh"]) == 7
    )
    assert compute_c3_prompt_bundle_sha256() == EXPECTED_PROMPT_BUNDLE_SHA256

    full_dataset = build_synthetic_c3_dataset(tmp_path / "full_c3")
    auto_instances = load_c3_instances(full_dataset)
    assert len(auto_instances) == 1045
    assert sum(1 for instance in auto_instances if instance.language == "en") == 744
    assert sum(1 for instance in auto_instances if instance.language == "zh") == 301

    first_instance_by_subtask: dict[str, C3InstanceRecord] = {}
    observed_template_keys: set[tuple[str, str]] = set()
    for instance in auto_instances:
        qualified_subtask = f"{instance.language}/{instance.subtask}"
        if qualified_subtask not in first_instance_by_subtask:
            first_instance_by_subtask[qualified_subtask] = instance
        else:
            pass
        assert instance.judge_template_id is not None
        observed_template_keys.add((instance.language, instance.judge_template_id))

    assert len(first_instance_by_subtask) == 21 and len(observed_template_keys) == 18

    observed_rendered_sha256: dict[str, str] = {}
    for qualified_subtask, instance in sorted(first_instance_by_subtask.items()):
        assistant_turns = [
            f"Candidate answer round {round_index}"
            for round_index in range(len(instance.user_turn_texts))
        ]
        system_prompt, user_prompt = build_c3_judge_messages(instance, assistant_turns)
        observed_rendered_sha256[qualified_subtask] = compute_c3_judge_prompt_sha256(
            system_prompt, user_prompt
        )
    assert observed_rendered_sha256 == EXPECTED_SUBTASK_RENDERED_PROMPT_SHA256

    verdicts_judge_one = {
        instance.instance_id: (instance.subtask != "omission_completion")
        for instance in auto_instances
    }
    verdicts_judge_two = {
        instance.instance_id: (instance.subtask == "omission_detection")
        for instance in auto_instances
    }

    english_scored = score_c3_language_instances(
        "en",
        auto_instances,
        {"gpt-4o": verdicts_judge_one, "deepseek-r1": verdicts_judge_two},
        ("gpt-4o", "deepseek-r1"),
        scope="auto",
        dataset_matches_pinned_revision=True,
    )
    assert english_scored.phenomena["omission"].mean_accuracy == pytest.approx(0.5)
    assert english_scored.macro_accuracy_by_judge == {
        "gpt-4o": pytest.approx(0.9),
        "deepseek-r1": pytest.approx(0.1),
    }
    assert english_scored.mean_macro_accuracy == pytest.approx(0.5)
    assert english_scored.macro_metric_name == "c3_auto_macro_accuracy"

    partial_one = dict(verdicts_judge_one)
    partial_two = dict(verdicts_judge_two)
    partial_one.pop("en/multiturn/0")
    partial_two.pop("en/multiturn/1")
    partial_scored = score_c3_language_instances(
        "en",
        auto_instances,
        {"gpt-4o": partial_one, "deepseek-r1": partial_two},
        ("gpt-4o", "deepseek-r1"),
        scope="auto",
        dataset_matches_pinned_revision=True,
    )
    assert partial_scored.mean_macro_accuracy is None
    assert partial_scored.subtasks["multiturn"].pending_instances == 2


def test_c3_security_credentials_sanitized_tracebacks_and_checkpoint_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify credential validation, sanitized tracebacks, strict base64/WAV validation, and finish_reason guards."""
    assert compute_endpoint_identity_sha256(
        "https://gateway.internal/deployment-a/v1"
    ) != compute_endpoint_identity_sha256("https://gateway.internal/deployment-b/v1")

    for bad_secret in ("secret value", "secret value", "secret\x7fvalue"):
        monkeypatch.setenv("C3_TEST_API_KEY", bad_secret)
        with pytest.raises(ValueError) as error_info:
            resolve_bearer_headers("C3_TEST_API_KEY")
        assert "secret" not in str(error_info.value)

    sample_judge_entry = {
        "name": "j",
        "model": "m",
        "base_url": "http://127.0.0.1:30000/v1",
    }
    for invalid_config in (
        {"api_key": "TOP_SECRET_ROOT_123", "judges": [sample_judge_entry]},
        {"unexpected_field": True, "judges": [sample_judge_entry]},
    ):
        config_path = tmp_path / "invalid_judge_config.json"
        write_json_atomic(config_path, invalid_config)
        with pytest.raises(ValueError) as config_error:
            load_c3_judge_configs(config_path)
        assert "TOP_SECRET_ROOT_123" not in str(config_error.value)

    with pytest.raises(ValueError, match="strict base64"):
        extract_assistant_text_and_audio(
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello",
                            "audio": {"data": "not!valid!base64@@@"},
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="readable WAV"):
        validate_wav_audio_bytes(b"RIFF" + b"0" * 64)

    dataset_root = build_synthetic_c3_dataset(
        tmp_path / "security_dataset", instances_with_audio_per_subtask=1
    )
    sample_specs = select_c3_subtask_specs(subtask_filter=("en/phonological_pause",))
    sample_instance = load_c3_instances(
        dataset_root, subtask_specs=sample_specs, max_samples_per_subtask=1
    )[0]
    judge_spec = C3JudgeEndpointSpec(
        name="j1", model="m1", base_url="http://127.0.0.1:30001/v1"
    )
    fingerprint_sha256 = judge_spec.compute_fingerprint_sha256()
    base_judgment: dict[str, JsonValue] = {
        "schema_version": 1,
        "instance_id": sample_instance.instance_id,
        "language": sample_instance.language,
        "subtask": sample_instance.subtask,
        "judge_name": "j1",
        "judge_model": "m1",
        "judge_model_revision": None,
        "judge_enable_thinking": None,
        "judge_temperature": 0.0,
        "judge_max_tokens": 4096,
        "judge_endpoint_identity_sha256": compute_endpoint_identity_sha256(
            judge_spec.base_url
        ),
        "judge_fingerprint_sha256": fingerprint_sha256,
        "prediction_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "finish_reason": "stop",
        "raw_verdict_text": "Analysis.\nThe answer is: yes.",
        "correct": True,
    }
    assert (
        validate_single_judgment_record(
            base_judgment,
            instance=sample_instance,
            expected_prediction_sha256="a" * 64,
            expected_prompt_sha256="b" * 64,
            expected_judge_fingerprint_sha256=fingerprint_sha256,
        )
        is True
    )
    for bad_override, expected_match in (
        ({"correct": False}, "disagrees with raw_verdict_text"),
        ({"finish_reason": None}, "Invalid cached judgment"),
        ({"finish_reason": "length"}, "Invalid cached judgment"),
        ({"finish_reason": "content_filter"}, "Invalid cached judgment"),
        ({"judge_fingerprint_sha256": "c" * 64}, "fingerprint mismatch"),
    ):
        with pytest.raises(ValueError, match=expected_match):
            validate_single_judgment_record(
                {**base_judgment, **bad_override},
                instance=sample_instance,
                expected_prediction_sha256="a" * 64,
                expected_prompt_sha256="b" * 64,
                expected_judge_fingerprint_sha256=fingerprint_sha256,
            )

    def failing_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"Failed connecting to {request.url}", request=request)

    async def trigger_failure() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(failing_transport)
        ) as http_client:
            await post_chat_completion_with_retries(
                http_client,
                api_url="https://private.internal/secret-route-xyz/v1/chat/completions",
                request_headers={"Content-Type": "application/json"},
                request_payload={"model": "m", "messages": []},
            )

    monkeypatch.setattr("benchmarks.c3.http_transport.BACKOFF_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="ConnectError") as transport_error_info:
        asyncio.run(trigger_failure())
    formatted_traceback = "".join(
        traceback.format_exception(
            type(transport_error_info.value),
            transport_error_info.value,
            transport_error_info.value.__traceback__,
        )
    )
    assert "secret-route-xyz" not in formatted_traceback
