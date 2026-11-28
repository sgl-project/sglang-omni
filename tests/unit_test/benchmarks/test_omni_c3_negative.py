# SPDX-License-Identifier: Apache-2.0
"""Live HTTP negative-path, prompt-bundle provenance, and export-upgrade unit tests for C3."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from benchmarks.c3.containment import write_json_atomic
from benchmarks.c3.prompts import (
    C3_EVALUATION_TEMPLATES,
    compute_c3_prompt_bundle_sha256,
)
from benchmarks.eval.benchmark_omni_c3 import main as run_c3_cli
from tests.unit_test.fixtures.c3_fixtures import (
    build_synthetic_c3_dataset,
    build_valid_wav_base64,
)

EXPECTED_PROMPT_BUNDLE_SHA256 = (
    "f6dc1f4b9b8078861239ade5aff37b9dfe5fa5bb448dc77459645644243d9d03"
)


def test_c3_prompt_bundle_provenance_live_negative_ledgers_and_export_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify live finish/audio failure ledgers, prompt-bundle evolution, stale rejudging, and auto/single_turn export upgrades."""
    dataset_root = build_synthetic_c3_dataset(
        tmp_path / "c3_negative_dataset", instances_with_audio_per_subtask=1
    )

    valid_wav_base64 = build_valid_wav_base64()
    live_mode: dict[str, JsonValue] = {
        "model_finish": "content_filter",
        "audio_mode": "missing",
        "judge_finish": "stop",
        "fail_judge": False,
    }
    original_async_client = httpx.AsyncClient

    def mock_negative_client(
        *, timeout: float = 180.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        def negative_request_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            if "8000" in str(request.url):
                message_payload: dict[str, JsonValue] = {
                    "role": "assistant",
                    "content": "Model answer",
                }
                if "audio" in body.get("modalities", []):
                    if live_mode["audio_mode"] == "valid":
                        message_payload["audio"] = {
                            "data": valid_wav_base64,
                            "transcript": "spoken",
                        }
                    elif live_mode["audio_mode"] == "corrupt_wav":
                        message_payload["audio"] = {
                            "data": base64.b64encode(b"NOT_A_WAV_HEADER").decode(
                                "ascii"
                            ),
                            "transcript": "spoken",
                        }
                    else:
                        pass
                else:
                    pass
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": message_payload,
                                "finish_reason": live_mode["model_finish"],
                            }
                        ]
                    },
                )
            elif live_mode["fail_judge"]:
                return httpx.Response(502, json={"error": "judge error"})
            else:
                judge_messages = body["messages"]
                user_text = str(judge_messages[1]["content"])
                verdict_text = (
                    "分析完成。\n最终答案为：是。"
                    if "最终答案" in user_text
                    else "Analysis.\nThe answer is: yes."
                )
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": verdict_text,
                                },
                                "finish_reason": live_mode["judge_finish"],
                            }
                        ]
                    },
                )

        return original_async_client(
            transport=httpx.MockTransport(negative_request_handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr("benchmarks.c3.http_transport.BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(httpx, "AsyncClient", mock_negative_client)

    def invoke_subtask_cli(output_dir: Path, subtasks: str, *extra_flags: str) -> int:
        return run_c3_cli(
            [
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--allow-unverified-dataset",
                "--subtasks",
                subtasks,
                "--max-samples-per-subtask",
                "1",
                *extra_flags,
            ]
        )

    judge_config = tmp_path / "judge_config.json"
    write_json_atomic(
        judge_config,
        {
            "judges": [
                {
                    "name": "primary_judge",
                    "model": "qwen3.8-27b",
                    "base_url": "http://127.0.0.1:30001/v1",
                    "max_concurrency": 3,
                    "enable_thinking": False,
                }
            ]
        },
    )

    negative_output_dir = tmp_path / "negative_output"
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase",
            "generate",
            "--model",
            "m1",
            "--base-url",
            "http://127.0.0.1:8000",
            "--warmup",
            "0",
        )
        == 1
    )
    inference_failures = json.loads(
        (negative_output_dir / "inference_failures.json").read_text(encoding="utf-8")
    )
    assert "content_filter" in inference_failures["en/phonological_pause/0"]

    live_mode["model_finish"] = "stop"
    human_negative_dir = tmp_path / "human_negative_dir"
    for bad_audio_mode, expected_substring in (
        ("missing", "Missing generated WAV audio"),
        ("corrupt_wav", "readable WAV container"),
    ):
        live_mode["audio_mode"] = bad_audio_mode
        assert (
            invoke_subtask_cli(
                human_negative_dir,
                "en/phonological_generation",
                "--phase",
                "generate",
                "--scope",
                "subset",
                "--modalities",
                "text,audio",
                "--model",
                "m1",
                "--base-url",
                "http://127.0.0.1:8000",
                "--warmup",
                "0",
            )
            == 1
        )
        human_failures = json.loads(
            (human_negative_dir / "inference_failures.json").read_text(encoding="utf-8")
        )
        assert expected_substring in human_failures["en/phonological_generation/0"]

    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase=generate",
            "--model=m1",
            "--base-url=http://127.0.0.1:8000",
            "--warmup=0",
        )
        == 0
    )
    assert not (negative_output_dir / "inference_failures.json").exists()

    length_model_dir = tmp_path / "length_model_dir"
    live_mode["model_finish"] = "length"
    live_mode["judge_finish"] = "length"
    assert (
        invoke_subtask_cli(
            length_model_dir,
            "en/phonological_pause",
            "--model=m1",
            "--base-url=http://127.0.0.1:8000",
            "--warmup=0",
            f"--judge-config={judge_config}",
        )
        == 1
    )
    assert not (length_model_dir / "inference_failures.json").exists()
    assert (
        "truncated"
        in json.loads(
            (length_model_dir / "judge_failures.json").read_text(encoding="utf-8")
        )["primary_judge"]["en/phonological_pause/0"]
    )

    live_mode["judge_finish"] = "stop"
    assert (
        invoke_subtask_cli(
            length_model_dir,
            "en/phonological_pause",
            "--phase=judge",
            f"--judge-config={judge_config}",
        )
        == 0
    )
    assert (
        invoke_subtask_cli(length_model_dir, "en/phonological_pause", "--phase=score")
        == 0
    )
    length_model_results = json.loads(
        (length_model_dir / "c3_results.json").read_text(encoding="utf-8")
    )
    assert length_model_results["completed_instances"] == 1
    assert length_model_results["truncated_instances"] == 1
    live_mode["model_finish"] = "stop"

    live_mode["judge_finish"] = None
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase=judge",
            f"--judge-config={judge_config}",
        )
        == 1
    )
    judge_failures = json.loads(
        (negative_output_dir / "judge_failures.json").read_text(encoding="utf-8")
    )
    assert "finish_reason" in judge_failures["primary_judge"]["en/phonological_pause/0"]

    live_mode["judge_finish"] = "stop"
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase=judge",
            f"--judge-config={judge_config}",
        )
        == 0
    )
    assert not (negative_output_dir / "judge_failures.json").exists()

    with pytest.raises(ValueError, match="endpoint_identity_sha256"):
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase=score",
            "--base-url=http://127.0.0.1:8000/other-deployment/v1",
        )

    alternate_execution_judge_config = tmp_path / "alternate_execution_judge.json"
    write_json_atomic(
        alternate_execution_judge_config,
        {
            "judges": [
                {
                    "name": "primary_judge",
                    "model": "qwen3.8-27b",
                    "base_url": "http://127.0.0.1:30001/v1",
                    "api_key_env": "OPTIONAL_UNUSED_ENV",
                    "max_concurrency": 64,
                    "enable_thinking": False,
                }
            ]
        },
    )
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase",
            "score",
            "--judge-config",
            str(alternate_execution_judge_config),
        )
        == 0
    )
    scored_provenance = json.loads(
        (negative_output_dir / "c3_results.json").read_text(encoding="utf-8")
    )["provenance"]["declared_server_config"]
    assert scored_provenance["judges"][0]["max_concurrency"] == 3
    assert scored_provenance["judges"][0]["api_key_env"] is None

    original_pause_template = C3_EVALUATION_TEMPLATES["en"][
        "ambiguity_phonological_pause"
    ]
    monkeypatch.setitem(
        C3_EVALUATION_TEMPLATES["en"],
        "ambiguity_phonological_pause",
        original_pause_template + "\nExtra rubric note.",
    )
    new_bundle_sha256 = compute_c3_prompt_bundle_sha256()
    assert new_bundle_sha256 != EXPECTED_PROMPT_BUNDLE_SHA256
    with pytest.raises(ValueError, match="prompt bundle differs"):
        invoke_subtask_cli(
            negative_output_dir, "en/phonological_pause", "--phase", "score"
        )

    live_mode["fail_judge"] = True
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase",
            "judge",
            "--judge-config",
            str(judge_config),
        )
        == 1
    )
    assert (
        json.loads(
            (negative_output_dir / "judgments.json").read_text(encoding="utf-8")
        )["primary_judge"]
        == {}
    )
    assert (
        invoke_subtask_cli(
            negative_output_dir, "en/phonological_pause", "--phase", "score"
        )
        == 1
    )

    live_mode["fail_judge"] = False
    assert (
        invoke_subtask_cli(
            negative_output_dir,
            "en/phonological_pause",
            "--phase",
            "judge",
            "--judge-config",
            str(judge_config),
        )
        == 0
    )
    assert (
        invoke_subtask_cli(
            negative_output_dir, "en/phonological_pause", "--phase", "score"
        )
        == 0
    )
    updated_provenance = json.loads(
        (negative_output_dir / "c3_results.json").read_text(encoding="utf-8")
    )["provenance"]["declared_server_config"]
    assert updated_provenance["prompt_bundle_sha256"] == new_bundle_sha256
    assert (
        updated_provenance["generation_prompt_bundle_sha256"]
        == EXPECTED_PROMPT_BUNDLE_SHA256
    )

    monkeypatch.setitem(
        C3_EVALUATION_TEMPLATES["en"],
        "ambiguity_phonological_pause",
        original_pause_template,
    )
    live_mode["audio_mode"] = "valid"
    full_counts_dataset = build_synthetic_c3_dataset(tmp_path / "full_counts_dataset")

    for requested_scope, expected_scope, expected_selected, expected_judged in (
        ("auto", "full", 323, 301),
        ("single_turn", "subset", 285, 263),
    ):
        export_dir = tmp_path / f"{requested_scope}_export_out"
        assert (
            run_c3_cli(
                [
                    "--dataset-root",
                    str(full_counts_dataset),
                    "--output-dir",
                    str(export_dir),
                    "--allow-unverified-dataset",
                    "--scope",
                    requested_scope,
                    "--language",
                    "zh",
                    "--export-generation-audio",
                    "--modalities",
                    "text,audio",
                    "--max-concurrency",
                    "32",
                    "--model",
                    "m1",
                    "--base-url",
                    "http://127.0.0.1:8000",
                    "--warmup",
                    "0",
                    "--judge-config",
                    str(judge_config),
                ]
            )
            == 1
        )
        export_summary = json.loads(
            (export_dir / "c3_results.json").read_text(encoding="utf-8")
        )
        assert export_summary["scope"] == expected_scope
        assert export_summary["selected_instances"] == expected_selected
        assert export_summary["completed_instances"] == expected_selected
        assert export_summary["resolved_judgments"] == expected_judged
        assert export_summary["failed_instances"] == {}
        assert export_summary["failed_judgments"] == {}
        assert export_summary["required_human_annotations"] == 22
        assert export_summary["resolved_human_annotations"] == 0
