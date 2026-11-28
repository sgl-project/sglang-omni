# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Asynchronous OpenAI-compatible HTTP judging runner for C3."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from pydantic import JsonValue

from benchmarks.c3.containment import read_bytes_nofollow, write_contained_json_atomic
from benchmarks.c3.http_transport import (
    build_chat_completions_url,
    compute_endpoint_identity_sha256,
    post_chat_completion_with_retries,
    resolve_bearer_headers,
    sanitize_endpoint_origin,
)
from benchmarks.c3.judge_state import (
    C3JudgeEndpointSpec,
    C3JudgePhaseResult,
    C3PendingJudgeWorkItem,
    load_judgments_json_map,
    load_structured_judge_failures,
    persist_structured_judge_failures,
    validate_single_judgment_record,
)
from benchmarks.c3.prompts import (
    build_c3_judge_messages,
    compute_c3_judge_prompt_sha256,
    compute_c3_prompt_bundle_sha256,
    parse_c3_judge_verdict,
)
from benchmarks.c3.runner import extract_assistant_text_and_audio
from benchmarks.c3.schemas import (
    JUDGMENT_CHECKPOINT_SCHEMA_VERSION,
    C3JudgeManifestRecord,
    C3JudgmentCheckpointRecord,
    C3PredictionCheckpointRecord,
    C3PublicJudgeManifestEntry,
    SuccessfulFinishReason,
)
from benchmarks.dataset.c3 import C3InstanceRecord


async def judge_c3_predictions_via_http(
    instances: list[C3InstanceRecord],
    predictions_by_id: dict[str, C3PredictionCheckpointRecord],
    judge_specs: list[C3JudgeEndpointSpec],
    *,
    output_dir: Path,
    timeout_seconds: float,
    trust_env: bool,
) -> C3JudgePhaseResult:
    request_headers_by_judge: dict[str, dict[str, str]] = {
        judge_spec.name: resolve_bearer_headers(judge_spec.api_key_env)
        for judge_spec in judge_specs
    }

    current_prompt_bundle_sha256 = compute_c3_prompt_bundle_sha256()
    supplied_by_name = {judge_spec.name: judge_spec for judge_spec in judge_specs}
    saved_entries_by_name: dict[str, C3PublicJudgeManifestEntry] = {}
    saved_order: list[str] = []

    try:
        manifest_bytes = read_bytes_nofollow(output_dir, "judge_manifest.json")
    except FileNotFoundError:
        manifest_bytes = b""

    if manifest_bytes:
        parsed_manifest = C3JudgeManifestRecord.model_validate_json(
            manifest_bytes.decode("utf-8")
        )
        for saved_entry in parsed_manifest.judges:
            saved_entries_by_name[saved_entry.name] = saved_entry
            saved_order.append(saved_entry.name)

        missing_required_judges = [
            saved_name
            for saved_name in saved_order
            if saved_name not in supplied_by_name
        ]
        if missing_required_judges:
            raise ValueError(
                f"Judge configuration omits previously required judge(s) {missing_required_judges} "
                "from judge_manifest.json; include all required judges or choose a new --output-dir"
            )
        else:
            for judge_spec in judge_specs:
                if judge_spec.name in saved_entries_by_name and (
                    saved_entries_by_name[judge_spec.name].judge_fingerprint_sha256
                    != judge_spec.compute_fingerprint_sha256()
                ):
                    raise ValueError(
                        f"Judge {judge_spec.name!r} configuration changed relative to judge_manifest.json; "
                        "choose a new --output-dir or a distinct judge name"
                    )
                else:
                    pass
    else:
        pass

    serialized_records_by_judge = load_judgments_json_map(output_dir)
    for judge_spec in judge_specs:
        serialized_records_by_judge.setdefault(judge_spec.name, {})

    verdicts_by_judge: dict[str, dict[str, bool]] = {
        judge_spec.name: {} for judge_spec in judge_specs
    }
    automated_selected_ids = {
        instance.instance_id
        for instance in instances
        if not instance.requires_human_audio_annotation
    }
    prior_failures = load_structured_judge_failures(output_dir)
    failed_judgments_by_judge: dict[str, dict[str, str]] = {
        judge_spec.name: {
            instance_id: reason
            for instance_id, reason in prior_failures.get(judge_spec.name, {}).items()
            if instance_id in automated_selected_ids
        }
        for judge_spec in judge_specs
    }

    automated_instances_with_predictions = [
        instance
        for instance in instances
        if not instance.requires_human_audio_annotation
        and instance.instance_id in predictions_by_id
    ]
    pending_by_judge: dict[str, list[C3PendingJudgeWorkItem]] = {
        judge_spec.name: [] for judge_spec in judge_specs
    }

    for judge_spec in judge_specs:
        fingerprint_sha256 = judge_spec.compute_fingerprint_sha256()
        for instance in automated_instances_with_predictions:
            prediction_record = predictions_by_id[instance.instance_id]
            system_prompt, user_prompt = build_c3_judge_messages(
                instance, prediction_record.assistant_turns
            )
            prompt_sha256 = compute_c3_judge_prompt_sha256(system_prompt, user_prompt)

            cached_payload = serialized_records_by_judge[judge_spec.name].get(
                instance.instance_id
            )
            if isinstance(cached_payload, dict):
                validated_verdict = validate_single_judgment_record(
                    cached_payload,
                    instance=instance,
                    expected_prediction_sha256=prediction_record.prediction_sha256,
                    expected_prompt_sha256=prompt_sha256,
                    expected_judge_fingerprint_sha256=fingerprint_sha256,
                )
                if validated_verdict is not None:
                    verdicts_by_judge[judge_spec.name][
                        instance.instance_id
                    ] = validated_verdict
                    failed_judgments_by_judge[judge_spec.name].pop(
                        instance.instance_id, None
                    )
                    continue
                else:
                    serialized_records_by_judge[judge_spec.name].pop(
                        instance.instance_id, None
                    )
            else:
                pass
            pending_by_judge[judge_spec.name].append(
                C3PendingJudgeWorkItem(
                    instance=instance,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    prompt_sha256=prompt_sha256,
                    prediction_sha256=prediction_record.prediction_sha256,
                )
            )

    authoritative_order: list[str] = list(saved_order)
    for judge_spec in judge_specs:
        if judge_spec.name not in authoritative_order:
            authoritative_order.append(judge_spec.name)
        else:
            pass

    authoritative_public_judges: list[C3PublicJudgeManifestEntry] = []
    for judge_name in authoritative_order:
        judge_spec = supplied_by_name[judge_name]
        has_new_work = bool(pending_by_judge[judge_name])
        saved_entry = saved_entries_by_name.get(judge_name)
        prior_concurrency_history = (
            list(saved_entry.concurrency_history)
            if saved_entry is not None and saved_entry.concurrency_history
            else ([saved_entry.max_concurrency] if saved_entry is not None else [])
        )
        if saved_entry is not None and not has_new_work:
            authoritative_public_judges.append(
                saved_entry.model_copy(
                    update={"concurrency_history": prior_concurrency_history}
                )
            )
        else:
            if judge_spec.max_concurrency not in prior_concurrency_history:
                prior_concurrency_history.append(judge_spec.max_concurrency)
            else:
                pass
            authoritative_public_judges.append(
                C3PublicJudgeManifestEntry(
                    name=judge_spec.name,
                    model=judge_spec.model,
                    model_revision=judge_spec.model_revision,
                    base_url=sanitize_endpoint_origin(judge_spec.base_url),
                    endpoint_identity_sha256=compute_endpoint_identity_sha256(
                        judge_spec.base_url
                    ),
                    judge_fingerprint_sha256=judge_spec.compute_fingerprint_sha256(),
                    api_key_env=judge_spec.api_key_env,
                    max_concurrency=judge_spec.max_concurrency,
                    concurrency_history=list(prior_concurrency_history),
                    enable_thinking=judge_spec.enable_thinking,
                    temperature=judge_spec.temperature,
                    max_tokens=judge_spec.max_tokens,
                )
            )

    judge_manifest_record = C3JudgeManifestRecord(
        schema_version=JUDGMENT_CHECKPOINT_SCHEMA_VERSION,
        prompt_bundle_sha256=current_prompt_bundle_sha256,
        judges=authoritative_public_judges,
    )
    write_contained_json_atomic(
        output_dir,
        "judge_manifest.json",
        judge_manifest_record.model_dump(mode="json"),
    )
    write_contained_json_atomic(
        output_dir, "judgments.json", dict(serialized_records_by_judge)
    )
    persist_structured_judge_failures(output_dir, failed_judgments_by_judge)

    save_lock = asyncio.Lock()

    async def execute_judge_batch(judge_spec: C3JudgeEndpointSpec) -> None:
        pending_for_judge = pending_by_judge[judge_spec.name]
        if not pending_for_judge:
            return
        else:
            pass
        request_headers = request_headers_by_judge[judge_spec.name]
        api_url = build_chat_completions_url(judge_spec.base_url)
        fingerprint_sha256 = judge_spec.compute_fingerprint_sha256()
        endpoint_identity_sha256 = compute_endpoint_identity_sha256(judge_spec.base_url)
        semaphore = asyncio.Semaphore(judge_spec.max_concurrency)

        async with httpx.AsyncClient(
            timeout=timeout_seconds, trust_env=trust_env
        ) as http_client:

            async def evaluate_and_record_judgment(
                pending_judgment: C3PendingJudgeWorkItem,
            ) -> None:
                async with semaphore:
                    request_payload: dict[str, JsonValue] = {
                        "model": judge_spec.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": pending_judgment.system_prompt,
                            },
                            {"role": "user", "content": pending_judgment.user_prompt},
                        ],
                        "max_tokens": judge_spec.max_tokens,
                        "stream": False,
                    }
                    if judge_spec.temperature is not None:
                        request_payload["temperature"] = judge_spec.temperature
                    else:
                        pass
                    if judge_spec.enable_thinking is not None:
                        request_payload["chat_template_kwargs"] = {
                            "enable_thinking": judge_spec.enable_thinking
                        }
                    else:
                        pass

                    try:
                        response_json = await post_chat_completion_with_retries(
                            http_client,
                            api_url=api_url,
                            request_headers=request_headers,
                            request_payload=request_payload,
                        )
                        raw_verdict_text, _, finish_reason = (
                            extract_assistant_text_and_audio(response_json)
                        )
                        judge_label = (
                            f"judge {judge_spec.name} on {pending_judgment.instance.instance_id} "
                            f"(max_tokens={judge_spec.max_tokens})"
                        )
                        if (
                            not isinstance(finish_reason, str)
                            or not finish_reason.strip()
                        ):
                            raise RuntimeError(
                                f"Missing or non-string finish_reason on {judge_label}"
                            )
                        elif finish_reason == "length":
                            raise RuntimeError(
                                f"{judge_label} truncated (finish_reason='length')"
                            )
                        elif finish_reason == "stop":
                            verified_finish_reason: SuccessfulFinishReason = "stop"
                        elif finish_reason == "eos":
                            verified_finish_reason = "eos"
                        else:
                            raise RuntimeError(
                                f"Unsupported non-stop finish_reason={finish_reason!r} on {judge_label}"
                            )
                        parsed_verdict = parse_c3_judge_verdict(
                            raw_verdict_text, pending_judgment.instance.language
                        )
                        if parsed_verdict is None:
                            raise RuntimeError(
                                "Malformed or missing final judge verdict"
                            )
                        else:
                            pass
                    except (RuntimeError, ValueError) as judgment_error:
                        failure_reason = (
                            str(judgment_error)
                            if str(judgment_error)
                            else type(judgment_error).__name__
                        )
                        async with save_lock:
                            failed_judgments_by_judge[judge_spec.name][
                                pending_judgment.instance.instance_id
                            ] = failure_reason
                            await asyncio.to_thread(
                                persist_structured_judge_failures,
                                output_dir,
                                failed_judgments_by_judge,
                            )
                        return

                    judgment_record = C3JudgmentCheckpointRecord(
                        schema_version=JUDGMENT_CHECKPOINT_SCHEMA_VERSION,
                        instance_id=pending_judgment.instance.instance_id,
                        language=pending_judgment.instance.language,
                        subtask=pending_judgment.instance.subtask,
                        judge_name=judge_spec.name,
                        judge_model=judge_spec.model,
                        judge_model_revision=judge_spec.model_revision,
                        judge_enable_thinking=judge_spec.enable_thinking,
                        judge_temperature=judge_spec.temperature,
                        judge_max_tokens=judge_spec.max_tokens,
                        judge_endpoint_identity_sha256=endpoint_identity_sha256,
                        judge_fingerprint_sha256=fingerprint_sha256,
                        prediction_sha256=pending_judgment.prediction_sha256,
                        prompt_sha256=pending_judgment.prompt_sha256,
                        finish_reason=verified_finish_reason,
                        raw_verdict_text=raw_verdict_text,
                        correct=parsed_verdict,
                    )
                    async with save_lock:
                        serialized_records_by_judge[judge_spec.name][
                            pending_judgment.instance.instance_id
                        ] = judgment_record.model_dump(mode="json")
                        verdicts_by_judge[judge_spec.name][
                            pending_judgment.instance.instance_id
                        ] = parsed_verdict
                        failed_judgments_by_judge[judge_spec.name].pop(
                            pending_judgment.instance.instance_id, None
                        )
                        await asyncio.to_thread(
                            write_contained_json_atomic,
                            output_dir,
                            "judgments.json",
                            dict(serialized_records_by_judge),
                        )
                        await asyncio.to_thread(
                            persist_structured_judge_failures,
                            output_dir,
                            failed_judgments_by_judge,
                        )

            await asyncio.gather(
                *(
                    evaluate_and_record_judgment(pending_judgment)
                    for pending_judgment in pending_for_judge
                )
            )

    await asyncio.gather(
        *(execute_judge_batch(judge_spec) for judge_spec in judge_specs)
    )

    non_empty_failures_by_judge = {
        judge_name: dict(failure_reasons_by_instance)
        for judge_name, failure_reasons_by_instance in failed_judgments_by_judge.items()
        if failure_reasons_by_instance
    }
    return C3JudgePhaseResult(
        authoritative_judge_names=tuple(authoritative_order),
        authoritative_public_judges=authoritative_public_judges,
        prompt_bundle_sha256=current_prompt_bundle_sha256,
        verdicts_by_judge=verdicts_by_judge,
        failed_judgments_by_judge=non_empty_failures_by_judge,
    )
