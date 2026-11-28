# SPDX-License-Identifier: Apache-2.0
"""End-to-end recovery, roster persistence, and human-generation unit tests for C3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from benchmarks.c3.containment import ensure_safe_contained_path, write_json_atomic
from benchmarks.c3.prompts import load_manual_generation_annotations
from benchmarks.c3.registry import select_c3_subtask_specs
from benchmarks.c3.runner import validate_cached_predictions
from benchmarks.dataset.c3 import load_c3_instances
from benchmarks.eval.benchmark_omni_c3 import main as run_c3_cli
from tests.unit_test.fixtures.c3_fixtures import (
    build_synthetic_c3_dataset,
    build_valid_wav_base64,
)


def test_c3_end_to_end_multiturn_resume_roster_persistence_and_human_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify input isolation, multi-turn history, roster persistence, and human-generation bindings."""
    dataset_root = build_synthetic_c3_dataset(
        tmp_path / "c3_end_to_end", instances_with_audio_per_subtask=1
    )

    valid_wav_base64 = build_valid_wav_base64()
    observed_model_payloads: list[dict[str, JsonValue]] = []
    fail_state = {
        "fail_all_model": False,
        "fail_zh_turn3": True,
        "fail_judge_b": True,
    }
    original_async_client = httpx.AsyncClient

    def mock_http_client(
        *, timeout: float = 180.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        def request_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            url_text = str(request.url)
            if "8000" in url_text:
                if fail_state["fail_all_model"]:
                    return httpx.Response(503, json={"error": "model down"})
                else:
                    pass
                observed_model_payloads.append(body)
                assert "REF_SENTINEL" not in json.dumps(body, ensure_ascii=False)
                request_messages = body["messages"]
                assert isinstance(request_messages, list)
                turn_number = (len(request_messages) + 1) // 2
                if len(request_messages) == 5 and fail_state["fail_zh_turn3"]:
                    fail_state["fail_zh_turn3"] = False
                    return httpx.Response(400, json={"error": "simulated failure"})
                else:
                    message_payload: dict[str, JsonValue] = {
                        "role": "assistant",
                        "content": f"Generated reply for round {turn_number}",
                    }
                    if "audio" in body.get("modalities", []):
                        message_payload["audio"] = {
                            "data": valid_wav_base64,
                            "transcript": f"Generated reply for round {turn_number}",
                        }
                    else:
                        pass
                    return httpx.Response(
                        200,
                        json={
                            "choices": [
                                {
                                    "message": message_payload,
                                    "finish_reason": "stop",
                                }
                            ]
                        },
                    )
            elif "30002" in url_text and fail_state["fail_judge_b"]:
                return httpx.Response(500, json={"error": "judge B down"})
            else:
                judge_messages = body["messages"]
                user_text = judge_messages[1]["content"]
                assert "REF_SENTINEL" in user_text
                verdict_line = (
                    "分析完成。\n最终答案为：是。"
                    if "最终答案" in user_text
                    else "Analysis complete.\nThe answer is: yes."
                )
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": verdict_line,
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )

        return original_async_client(
            transport=httpx.MockTransport(request_handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr("benchmarks.c3.http_transport.BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(httpx, "AsyncClient", mock_http_client)

    def invoke_cli(output_dir: Path, subtasks: str, *extra_flags: str) -> int:
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

    judge_a_entry: dict[str, JsonValue] = {
        "name": "team:judge_a",
        "model": "qwen3.8-27b",
        "base_url": "http://127.0.0.1:30001/v1",
        "max_concurrency": 2,
        "enable_thinking": False,
    }
    judge_b_entry: dict[str, JsonValue] = {
        "name": "team:judge_b",
        "model": "deepseek-v3",
        "base_url": "http://127.0.0.1:30002/v1",
        "max_concurrency": 2,
        "enable_thinking": False,
    }
    judge_one_config = tmp_path / "judge_one.json"
    write_json_atomic(judge_one_config, {"judges": [judge_a_entry]})
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    launch_command = (
        "CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY=sk-secret123456 "
        f"python -m sglang_omni.cli serve --model-path {model_name} --port 8000"
    )
    model_flags = (
        "--model",
        model_name,
        "--base-url",
        "http://127.0.0.1:8000",
        "--launch-command",
        launch_command,
    )

    zero_dir = tmp_path / "zero_success_out"
    fail_state["fail_all_model"] = True
    assert (
        invoke_cli(
            zero_dir,
            "en/phonological_pause",
            "--phase",
            "generate",
            *model_flags,
            "--warmup",
            "0",
        )
        == 1
    )
    assert (zero_dir / "predictions.json").is_file()
    assert (
        invoke_cli(
            zero_dir,
            "en/phonological_pause",
            "--phase",
            "judge",
            "--judge-config",
            str(judge_one_config),
        )
        == 1
    )
    assert invoke_cli(zero_dir, "en/phonological_pause", "--phase", "score") == 1
    fail_state["fail_all_model"] = False

    output_dir = tmp_path / "run_out"
    paired_subtasks = "en/phonological_pause,zh/multiturn"
    assert (
        invoke_cli(
            output_dir,
            paired_subtasks,
            *model_flags,
            "--max-concurrency",
            "4",
            "--warmup",
            "0",
            "--judge-config",
            str(judge_one_config),
        )
        == 1
    )
    assert (output_dir / "inference_failures.json").is_file()

    assert (
        invoke_cli(
            output_dir,
            paired_subtasks,
            *model_flags,
            "--max-concurrency",
            "1",
            "--warmup",
            "0",
            "--judge-config",
            str(judge_one_config),
        )
        == 0
    )
    assert not (output_dir / "inference_failures.json").exists()
    saved_run_manifest = json.loads(
        (output_dir / "run.json").read_text(encoding="utf-8")
    )
    assert saved_run_manifest["concurrency_history"] == [4, 1]
    assert "sk-secret123456" not in json.dumps(saved_run_manifest)

    assert (
        invoke_cli(
            output_dir,
            paired_subtasks,
            "--phase",
            "generate",
            *model_flags,
            "--max-concurrency",
            "16",
            "--warmup",
            "25",
        )
        == 0
    )
    preserved_manifest = json.loads(
        (output_dir / "run.json").read_text(encoding="utf-8")
    )
    assert preserved_manifest["max_concurrency"] == 1
    assert preserved_manifest["warmup_requests"] == 0
    assert preserved_manifest["concurrency_history"] == [4, 1]

    five_message_calls = [
        payload
        for payload in observed_model_payloads
        if isinstance(payload.get("messages"), list) and len(payload["messages"]) == 5
    ]
    assert five_message_calls
    final_messages = five_message_calls[-1]["messages"]
    assert isinstance(final_messages, list)
    assert [entry["role"] for entry in final_messages if isinstance(entry, dict)] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    two_judges_config = tmp_path / "two_judges.json"
    write_json_atomic(two_judges_config, {"judges": [judge_a_entry, judge_b_entry]})
    with pytest.raises(ValueError, match="differs from authoritative"):
        invoke_cli(
            output_dir,
            paired_subtasks,
            "--phase",
            "score",
            "--judge-config",
            str(two_judges_config),
        )

    assert (
        invoke_cli(
            output_dir,
            paired_subtasks,
            "--phase",
            "judge",
            "--judge-config",
            str(two_judges_config),
        )
        == 1
    )
    assert (output_dir / "judge_failures.json").is_file()
    with pytest.raises(ValueError, match="omits previously required judge"):
        invoke_cli(
            output_dir,
            paired_subtasks,
            "--phase",
            "judge",
            "--judge-config",
            str(judge_one_config),
        )

    assert invoke_cli(output_dir, paired_subtasks, "--phase", "score") == 1

    fail_state["fail_judge_b"] = False
    assert (
        invoke_cli(
            output_dir,
            paired_subtasks,
            "--phase",
            "judge",
            "--judge-config",
            str(two_judges_config),
        )
        == 0
    )
    assert not (output_dir / "judge_failures.json").exists()
    assert invoke_cli(output_dir, paired_subtasks, "--phase", "score") == 0
    scored_json = json.loads(
        (output_dir / "c3_results.json").read_text(encoding="utf-8")
    )
    scored_markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")
    assert scored_json["scoring_attempt_id"] in scored_markdown
    assert (
        hashlib.sha256(scored_markdown.encode("utf-8")).hexdigest()
        == scored_json["results_markdown_sha256"]
    )

    with pytest.raises(ValueError, match="modalities text,audio"):
        invoke_cli(
            tmp_path / "human_text_only_rejected",
            "en/phonological_generation",
            "--phase=generate",
            "--scope=subset",
            "--modalities=text",
            *model_flags[:4],
        )

    human_output_dir = tmp_path / "human_generation_run"
    assert (
        invoke_cli(
            human_output_dir,
            "en/phonological_generation",
            "--scope=subset",
            "--modalities=text,audio",
            *model_flags[:4],
            "--warmup=0",
        )
        == 1
    )
    human_predictions = json.loads(
        (human_output_dir / "predictions.json").read_text(encoding="utf-8")
    )
    human_prediction_entry = human_predictions["en/phonological_generation/0"]
    exported_wav_path = ensure_safe_contained_path(
        human_output_dir,
        human_prediction_entry["saved_generation_audio"],
        create_parents=False,
    )
    assert exported_wav_path.is_file()

    human_specs = select_c3_subtask_specs(
        include_human_generation=True,
        subtask_filter=("en/phonological_generation",),
    )
    human_instances = load_c3_instances(
        dataset_root, subtask_specs=human_specs, max_samples_per_subtask=1
    )

    bad_manual_path = tmp_path / "bad_manual.json"
    write_json_atomic(
        bad_manual_path,
        {
            "en/phonological_generation/0": {
                "correct": True,
                "prediction_sha256": human_prediction_entry["prediction_sha256"],
                "generated_audio_sha256": "0" * 64,
            }
        },
    )
    with pytest.raises(ValueError, match="generated_audio_sha256 mismatch"):
        load_manual_generation_annotations(
            bad_manual_path,
            allowed_human_instances={
                instance.instance_id: instance for instance in human_instances
            },
            expected_hashes_by_instance={
                "en/phonological_generation/0": (
                    human_prediction_entry["prediction_sha256"],
                    human_prediction_entry["generated_audio_sha256"],
                )
            },
        )

    valid_manual_path = tmp_path / "valid_manual.json"
    write_json_atomic(
        valid_manual_path,
        {
            "en/phonological_generation/0": {
                "correct": "The answer is: yes.",
                "prediction_sha256": human_prediction_entry["prediction_sha256"],
                "generated_audio_sha256": human_prediction_entry[
                    "generated_audio_sha256"
                ],
                "annotator_id": "expert_1",
            }
        },
    )
    assert (
        invoke_cli(
            human_output_dir,
            "en/phonological_generation",
            "--phase=score",
            "--scope=subset",
            f"--manual-generation-annotations={valid_manual_path}",
        )
        == 0
    )

    exported_wav_path.unlink()
    with pytest.raises(ValueError, match="Missing exported generation WAV"):
        validate_cached_predictions(
            human_predictions,
            {instance.instance_id: instance for instance in human_instances},
            dataset_root,
            human_output_dir,
            require_generation_audio_export=True,
        )

    tampered_predictions = json.loads(json.dumps(human_predictions))
    tampered_predictions["en/phonological_generation/0"]["finish_reasons"] = [
        "content_filter"
    ]
    with pytest.raises(ValueError, match="Invalid cached prediction"):
        validate_cached_predictions(
            tampered_predictions,
            {instance.instance_id: instance for instance in human_instances},
            dataset_root,
            human_output_dir,
            require_generation_audio_export=False,
        )
