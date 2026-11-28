# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Asynchronous OpenAI-compatible generation phase runner for C3."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import httpx

from benchmarks.c3.config_manifest import (
    C3EvalConfig,
    load_inference_failure_ledger,
    persist_inference_failure_ledger,
    resolve_effective_scope,
    serialize_predictions_map,
)
from benchmarks.c3.containment import (
    read_bytes_nofollow,
    validate_non_symlink_root_directory,
    write_contained_json_atomic,
)
from benchmarks.c3.http_transport import (
    build_chat_completions_url,
    compute_endpoint_identity_sha256,
    resolve_bearer_headers,
    sanitize_endpoint_origin,
)
from benchmarks.c3.prompts import compute_c3_prompt_bundle_sha256
from benchmarks.c3.runner import (
    C3InferencePhaseResult,
    run_c3_inference_for_instance,
    validate_cached_predictions,
)
from benchmarks.c3.schemas import C3PredictionCheckpointRecord, C3RunManifestRecord
from benchmarks.dataset.c3 import C3InstanceRecord
from benchmarks.eval.benchmark_duplex_mpe import sanitize_launch_command

logger = logging.getLogger(__name__)


async def run_c3_generate_phase(
    config: C3EvalConfig,
    instances: list[C3InstanceRecord],
    *,
    pinned_audio_sha256_by_path: dict[str, str] | None = None,
) -> C3InferencePhaseResult:
    assert config.base_url is not None and config.model is not None
    dataset_root = validate_non_symlink_root_directory(
        config.dataset_root, create_if_missing=False
    )
    output_dir = validate_non_symlink_root_directory(
        config.output_dir, create_if_missing=True
    )

    has_human_instances = any(
        instance.requires_human_audio_annotation for instance in instances
    )
    if has_human_instances and "audio" not in config.modalities:
        raise ValueError(
            "Selected C3 subtasks include human-scored phonological generation items; "
            "pass --modalities text,audio so generated WAV audio can be exported and verified"
        )
    else:
        pass
    require_audio_export = has_human_instances

    try:
        existing_predictions_bytes = read_bytes_nofollow(output_dir, "predictions.json")
        has_predictions_file = True
    except FileNotFoundError:
        existing_predictions_bytes = b""
        has_predictions_file = False

    try:
        existing_manifest_bytes = read_bytes_nofollow(output_dir, "run.json")
        has_manifest_file = True
    except FileNotFoundError:
        existing_manifest_bytes = b""
        has_manifest_file = False

    if has_predictions_file and not has_manifest_file:
        raise ValueError(
            "Existing predictions.json has no run.json; choose a new --output-dir"
        )
    else:
        pass

    request_headers = resolve_bearer_headers(config.api_key_env)
    api_url = build_chat_completions_url(config.base_url)
    sanitized_launch = sanitize_launch_command(config.launch_command)
    effective_scope = resolve_effective_scope(config, instances)
    selected_instances_sha256 = hashlib.sha256(
        json.dumps([instance.instance_id for instance in instances]).encode("utf-8")
    ).hexdigest()

    candidate_manifest = C3RunManifestRecord(
        model=config.model,
        model_revision=config.model_revision,
        launch_command=sanitized_launch,
        base_url=sanitize_endpoint_origin(config.base_url),
        endpoint_identity_sha256=compute_endpoint_identity_sha256(config.base_url),
        modalities=list(config.modalities),
        export_generation_audio=require_audio_export,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
        max_concurrency=config.max_concurrency,
        warmup_requests=config.warmup_requests,
        request_timeout_seconds=config.request_timeout_seconds,
        requested_scope=config.scope,
        effective_scope=effective_scope,
        language=config.language,
        subtasks=list(config.subtasks) if config.subtasks is not None else None,
        mini=config.mini,
        max_samples_per_subtask=config.max_samples_per_subtask,
        selected_instances_sha256=selected_instances_sha256,
        prompt_bundle_sha256=compute_c3_prompt_bundle_sha256(),
    )

    existing_manifest: C3RunManifestRecord | None = None
    if has_manifest_file:
        loaded_manifest = C3RunManifestRecord.model_validate_json(
            existing_manifest_bytes.decode("utf-8")
        )
        mutable_manifest_fields = {
            "concurrency_history",
            "warmup_requests",
            "warmup_history",
            "max_concurrency",
            "request_timeout_seconds",
        }
        is_compatible = (
            loaded_manifest.model_dump(exclude=mutable_manifest_fields)
            == candidate_manifest.model_dump(exclude=mutable_manifest_fields)
            and candidate_manifest.request_timeout_seconds
            >= loaded_manifest.request_timeout_seconds
        )
        if not is_compatible:
            raise ValueError(
                "Run configuration in run.json changed; choose a new --output-dir"
            )
        else:
            existing_manifest = loaded_manifest
    else:
        pass

    instances_by_id = {instance.instance_id: instance for instance in instances}
    saved_predictions: dict[str, C3PredictionCheckpointRecord] = {}
    if has_predictions_file:
        parsed_predictions_json = json.loads(existing_predictions_bytes.decode("utf-8"))
        if not isinstance(parsed_predictions_json, dict):
            raise ValueError(
                "predictions.json must be a JSON object keyed by instance_id"
            )
        else:
            saved_predictions = validate_cached_predictions(
                parsed_predictions_json,
                instances_by_id,
                dataset_root,
                output_dir,
                require_generation_audio_export=require_audio_export,
                pinned_audio_sha256_by_path=pinned_audio_sha256_by_path,
            )
    else:
        pass

    failed_instances: dict[str, str] = {
        instance_id: reason
        for instance_id, reason in load_inference_failure_ledger(output_dir).items()
        if instance_id in instances_by_id and instance_id not in saved_predictions
    }
    pending_instances = [
        instance
        for instance in instances
        if instance.instance_id not in saved_predictions
    ]

    if not pending_instances and existing_manifest is not None:
        write_contained_json_atomic(
            output_dir,
            "predictions.json",
            serialize_predictions_map(saved_predictions),
        )
        persist_inference_failure_ledger(output_dir, failed_instances)
        return C3InferencePhaseResult(
            predictions_by_id=saved_predictions, failed_instances=failed_instances
        )
    else:
        pass

    prior_concurrency_history = (
        list(existing_manifest.concurrency_history)
        if existing_manifest is not None
        else []
    )
    prior_warmup_history = (
        list(existing_manifest.warmup_history) if existing_manifest is not None else []
    )
    if config.max_concurrency not in prior_concurrency_history:
        prior_concurrency_history.append(config.max_concurrency)
    else:
        pass
    if config.warmup_requests not in prior_warmup_history:
        prior_warmup_history.append(config.warmup_requests)
    else:
        pass

    updated_manifest = candidate_manifest.model_copy(
        update={
            "concurrency_history": prior_concurrency_history,
            "warmup_history": prior_warmup_history,
        }
    )
    write_contained_json_atomic(
        output_dir, "run.json", updated_manifest.model_dump(mode="json")
    )
    write_contained_json_atomic(
        output_dir, "predictions.json", serialize_predictions_map(saved_predictions)
    )
    persist_inference_failure_ledger(output_dir, failed_instances)

    if not pending_instances:
        return C3InferencePhaseResult(
            predictions_by_id=saved_predictions, failed_instances=failed_instances
        )
    else:
        pass

    semaphore = asyncio.Semaphore(config.max_concurrency)
    save_lock = asyncio.Lock()

    async with httpx.AsyncClient(
        timeout=config.request_timeout_seconds, trust_env=config.trust_env
    ) as http_client:
        if config.warmup_requests > 0:
            warmup_instance = instances[0]
            for warmup_index in range(config.warmup_requests):
                try:
                    await run_c3_inference_for_instance(
                        warmup_instance,
                        dataset_root=dataset_root,
                        output_dir=output_dir,
                        http_client=http_client,
                        api_url=api_url,
                        request_headers=request_headers,
                        model_name=config.model,
                        modalities=config.modalities,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        seed=config.seed,
                        require_generation_audio_export=False,
                        pinned_audio_sha256_by_path=pinned_audio_sha256_by_path,
                    )
                except (RuntimeError, ValueError, OSError) as warmup_error:
                    failure_reason = (
                        str(warmup_error)
                        if str(warmup_error)
                        else type(warmup_error).__name__
                    )
                    logger.warning(
                        f"C3 warmup request {warmup_index} failed ({failure_reason}); continuing"
                    )
        else:
            pass

        async def generate_and_record_instance_prediction(
            instance: C3InstanceRecord,
        ) -> None:
            async with semaphore:
                try:
                    prediction_record = await run_c3_inference_for_instance(
                        instance,
                        dataset_root=dataset_root,
                        output_dir=output_dir,
                        http_client=http_client,
                        api_url=api_url,
                        request_headers=request_headers,
                        model_name=config.model,
                        modalities=config.modalities,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        seed=config.seed,
                        require_generation_audio_export=require_audio_export,
                        pinned_audio_sha256_by_path=pinned_audio_sha256_by_path,
                    )
                except (RuntimeError, ValueError, OSError) as inference_error:
                    failure_reason = (
                        str(inference_error)
                        if str(inference_error)
                        else type(inference_error).__name__
                    )
                    async with save_lock:
                        failed_instances[instance.instance_id] = failure_reason
                        await asyncio.to_thread(
                            persist_inference_failure_ledger,
                            output_dir,
                            failed_instances,
                        )
                    return

                async with save_lock:
                    saved_predictions[instance.instance_id] = (
                        prediction_record.model_copy(
                            update={"recorded_concurrency": config.max_concurrency}
                        )
                    )
                    failed_instances.pop(instance.instance_id, None)
                    await asyncio.to_thread(
                        write_contained_json_atomic,
                        output_dir,
                        "predictions.json",
                        serialize_predictions_map(saved_predictions),
                    )
                    await asyncio.to_thread(
                        persist_inference_failure_ledger,
                        output_dir,
                        failed_instances,
                    )

        await asyncio.gather(
            *(
                generate_and_record_instance_prediction(instance)
                for instance in pending_instances
            )
        )

    return C3InferencePhaseResult(
        predictions_by_id=saved_predictions, failed_instances=failed_instances
    )
