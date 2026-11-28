# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Audio encoding, single-instance dialogue inference, and cached-prediction validation for C3."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import soundfile
from pydantic import JsonValue

from benchmarks.c3.containment import read_bytes_nofollow, write_bytes_atomic
from benchmarks.c3.http_transport import post_chat_completion_with_retries
from benchmarks.c3.registry import C3OutputModality
from benchmarks.c3.schemas import C3PredictionCheckpointRecord, ModelFinishReason
from benchmarks.dataset.c3 import C3InstanceRecord

PREDICTION_CHECKPOINT_SCHEMA_VERSION = 1


def encode_audio_file_to_wav_base64_sync(
    dataset_root: Path, audio_relative_path: str
) -> tuple[str, str]:
    """Read dataset clip once via O_NOFOLLOW and encode that exact byte snapshot to PCM16 WAV base64."""
    source_audio_bytes = read_bytes_nofollow(dataset_root, audio_relative_path)
    source_file_sha256 = hashlib.sha256(source_audio_bytes).hexdigest()
    waveform, sample_rate_hz = soundfile.read(
        io.BytesIO(source_audio_bytes), dtype="float32"
    )
    if waveform.size == 0 or sample_rate_hz <= 0:
        raise ValueError("Decoded empty waveform from C3 audio bytes")
    else:
        wav_bytes_buffer = io.BytesIO()
        soundfile.write(
            wav_bytes_buffer, waveform, sample_rate_hz, format="WAV", subtype="PCM_16"
        )
        encoded_wav_base64 = base64.b64encode(wav_bytes_buffer.getvalue()).decode(
            "ascii"
        )
        return encoded_wav_base64, source_file_sha256


def validate_wav_audio_bytes(audio_bytes: bytes) -> tuple[int, int]:
    """Verify decoded bytes form a readable non-empty WAV container and return (frames, sample_rate_hz)."""
    if not audio_bytes:
        raise ValueError("Generated audio payload is empty")
    else:
        pass
    try:
        audio_info = soundfile.info(io.BytesIO(audio_bytes))
    except RuntimeError as error:
        raise ValueError(
            f"Generated audio is not a readable WAV container: {error}"
        ) from None
    if (
        audio_info.format != "WAV"
        or audio_info.frames <= 0
        or audio_info.samplerate <= 0
    ):
        raise ValueError("Generated audio must be a non-empty WAV container")
    else:
        return audio_info.frames, audio_info.samplerate


def extract_assistant_text_and_audio(
    response_json: dict[str, JsonValue],
) -> tuple[str, bytes | None, str | None]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat completion response missing choices[0]")
    else:
        first_choice = choices[0]
        finish_reason_raw = first_choice.get("finish_reason")
        finish_reason = (
            finish_reason_raw if isinstance(finish_reason_raw, str) else None
        )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat completion response missing message object")
        else:
            pass

    text_value = ""
    raw_content = message.get("content")
    if isinstance(raw_content, str) and raw_content.strip():
        text_value = raw_content.strip()
    else:
        audio_block_for_text = message.get("audio")
        if isinstance(audio_block_for_text, dict):
            transcript_raw = audio_block_for_text.get("transcript")
            if isinstance(transcript_raw, str) and transcript_raw.strip():
                text_value = transcript_raw.strip()
            else:
                pass
        else:
            pass

    audio_bytes: bytes | None = None
    audio_block = message.get("audio")
    if isinstance(audio_block, dict) and audio_block.get("data") is not None:
        raw_audio_base64 = audio_block.get("data")
        if not isinstance(raw_audio_base64, str) or not raw_audio_base64.strip():
            raise ValueError(
                "Chat completion audio.data must be a non-empty base64 string"
            )
        else:
            try:
                audio_bytes = base64.b64decode(raw_audio_base64, validate=True)
            except (binascii.Error, ValueError):
                raise ValueError(
                    "Chat completion audio.data failed strict base64 decoding"
                ) from None
    else:
        pass
    return text_value, audio_bytes, finish_reason


def compute_prediction_sha256(assistant_turns: list[str]) -> str:
    serialized_turns_bytes = json.dumps(assistant_turns, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized_turns_bytes).hexdigest()


@dataclass(frozen=True, kw_only=True)
class C3InferencePhaseResult:
    predictions_by_id: dict[str, C3PredictionCheckpointRecord]
    failed_instances: dict[str, str]


async def run_c3_inference_for_instance(
    instance: C3InstanceRecord,
    *,
    dataset_root: Path,
    output_dir: Path,
    http_client: httpx.AsyncClient,
    api_url: str,
    request_headers: dict[str, str],
    model_name: str,
    modalities: tuple[C3OutputModality, ...],
    temperature: float,
    max_tokens: int,
    seed: int | None,
    require_generation_audio_export: bool,
    pinned_audio_sha256_by_path: dict[str, str] | None,
) -> C3PredictionCheckpointRecord:
    messages: list[dict[str, JsonValue]] = []
    assistant_turns: list[str] = []
    finish_reasons: list[ModelFinishReason] = []
    source_audio_sha256: list[str] = []
    saved_audio_relative_path: str | None = None
    generated_audio_sha256: str | None = None

    for turn_index, audio_relative_path in enumerate(instance.audio_relative_paths):
        encoded_wav_base64, source_file_sha256 = await asyncio.to_thread(
            encode_audio_file_to_wav_base64_sync, dataset_root, audio_relative_path
        )
        if pinned_audio_sha256_by_path is not None:
            expected_pinned_sha256 = pinned_audio_sha256_by_path.get(
                audio_relative_path
            )
            if expected_pinned_sha256 != source_file_sha256:
                raise ValueError(
                    f"Source audio {audio_relative_path} changed or failed pinned SHA-256 check during inference"
                )
            else:
                pass
        else:
            pass
        source_audio_sha256.append(source_file_sha256)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_wav_base64, "format": "wav"},
                    }
                ],
            }
        )

        request_payload: dict[str, JsonValue] = {
            "model": model_name,
            "messages": list(messages),
            "modalities": list(modalities),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            request_payload["seed"] = seed
        else:
            pass
        if "audio" in modalities:
            request_payload["audio"] = {"format": "wav"}
        else:
            pass

        response_json = await post_chat_completion_with_retries(
            http_client,
            api_url=api_url,
            request_headers=request_headers,
            request_payload=request_payload,
        )
        assistant_text, audio_bytes, finish_reason = extract_assistant_text_and_audio(
            response_json
        )
        turn_label = (
            f"{instance.instance_id} turn {turn_index} (max_tokens={max_tokens})"
        )
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise RuntimeError(f"Missing or non-string finish_reason on {turn_label}")
        elif finish_reason == "stop":
            verified_finish_reason: ModelFinishReason = "stop"
        elif finish_reason == "eos":
            verified_finish_reason = "eos"
        elif finish_reason == "length":
            verified_finish_reason = "length"
        else:
            raise RuntimeError(
                f"Unsupported finish_reason={finish_reason!r} on {turn_label}"
            )
        if not assistant_text:
            raise RuntimeError(f"Empty assistant text on turn {turn_index}")
        else:
            pass

        assistant_turns.append(assistant_text)
        finish_reasons.append(verified_finish_reason)
        messages.append({"role": "assistant", "content": assistant_text})

        if instance.requires_human_audio_annotation and require_generation_audio_export:
            if audio_bytes is None:
                raise RuntimeError(
                    f"Missing generated WAV audio for human-scored instance {instance.instance_id}"
                )
            else:
                await asyncio.to_thread(validate_wav_audio_bytes, audio_bytes)
                relative_wav_path = (
                    f"generation_audio/{instance.language}/"
                    f"{instance.subtask}/{instance.source_index}.wav"
                )
                await asyncio.to_thread(
                    write_bytes_atomic, output_dir, relative_wav_path, audio_bytes
                )
                saved_audio_relative_path = relative_wav_path
                generated_audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        else:
            pass

    return C3PredictionCheckpointRecord(
        schema_version=PREDICTION_CHECKPOINT_SCHEMA_VERSION,
        instance_id=instance.instance_id,
        language=instance.language,
        phenomenon=instance.phenomenon,
        subtask=instance.subtask,
        source_index=instance.source_index,
        source_audio_sha256=source_audio_sha256,
        assistant_turns=assistant_turns,
        finish_reasons=finish_reasons,
        final_answer=assistant_turns[-1],
        prediction_sha256=compute_prediction_sha256(assistant_turns),
        saved_generation_audio=saved_audio_relative_path,
        generated_audio_sha256=generated_audio_sha256,
    )


def validate_cached_predictions(
    raw_cached_rows: dict[str, JsonValue],
    instances_by_id: dict[str, C3InstanceRecord],
    dataset_root: Path,
    output_dir: Path,
    *,
    require_generation_audio_export: bool = False,
    pinned_audio_sha256_by_path: dict[str, str] | None = None,
) -> dict[str, C3PredictionCheckpointRecord]:
    validated_predictions_by_id: dict[str, C3PredictionCheckpointRecord] = {}
    for instance_id, raw_row in raw_cached_rows.items():
        try:
            prediction_record = C3PredictionCheckpointRecord.model_validate(raw_row)
        except ValueError as validation_error:
            raise ValueError(
                f"Invalid cached prediction for {instance_id}: {validation_error}"
            ) from None
        if prediction_record.schema_version != PREDICTION_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported prediction schema_version for {instance_id}: {prediction_record.schema_version}"
            )
        elif instance_id not in instances_by_id:
            validated_predictions_by_id[instance_id] = prediction_record
            continue
        else:
            pass

        instance = instances_by_id[instance_id]
        if (
            len(prediction_record.assistant_turns) != len(instance.audio_relative_paths)
            or len(prediction_record.finish_reasons)
            != len(prediction_record.assistant_turns)
            or any(
                not turn_text.strip() for turn_text in prediction_record.assistant_turns
            )
        ):
            raise ValueError(
                f"Malformed assistant_turns or finish_reasons in cached prediction for {instance_id}"
            )
        elif prediction_record.final_answer != prediction_record.assistant_turns[-1]:
            raise ValueError(f"Cached final_answer mismatch for {instance_id}")
        elif prediction_record.prediction_sha256 != compute_prediction_sha256(
            prediction_record.assistant_turns
        ):
            raise ValueError(f"Cached prediction_sha256 mismatch for {instance_id}")
        else:
            pass

        expected_audio_sha256_list = [
            hashlib.sha256(
                read_bytes_nofollow(dataset_root, audio_relative_path)
            ).hexdigest()
            for audio_relative_path in instance.audio_relative_paths
        ]
        if prediction_record.source_audio_sha256 != expected_audio_sha256_list:
            raise ValueError(f"Source audio SHA-256 changed on disk for {instance_id}")
        elif pinned_audio_sha256_by_path is not None:
            pinned_expected_sha256_list = [
                pinned_audio_sha256_by_path.get(audio_relative_path)
                for audio_relative_path in instance.audio_relative_paths
            ]
            if prediction_record.source_audio_sha256 != pinned_expected_sha256_list:
                raise ValueError(
                    f"Cached prediction for {instance_id} does not match pinned SHA256SUMS audio digests"
                )
            else:
                pass
        else:
            pass

        must_verify_saved_audio = instance.requires_human_audio_annotation and (
            require_generation_audio_export
            or prediction_record.saved_generation_audio is not None
            or prediction_record.generated_audio_sha256 is not None
        )
        if must_verify_saved_audio:
            expected_relative_wav = (
                f"generation_audio/{instance.language}/"
                f"{instance.subtask}/{instance.source_index}.wav"
            )
            if (
                prediction_record.saved_generation_audio != expected_relative_wav
                or prediction_record.generated_audio_sha256 is None
            ):
                raise ValueError(
                    f"Cached human-generation prediction for {instance_id} is missing exported WAV metadata"
                )
            else:
                try:
                    wav_bytes = read_bytes_nofollow(output_dir, expected_relative_wav)
                except FileNotFoundError:
                    raise ValueError(
                        f"Missing exported generation WAV on disk for {instance_id}: {expected_relative_wav}"
                    ) from None
                validate_wav_audio_bytes(wav_bytes)
                if (
                    hashlib.sha256(wav_bytes).hexdigest()
                    != prediction_record.generated_audio_sha256
                ):
                    raise ValueError(
                        f"Exported generation WAV SHA-256 mismatch on disk for {instance_id}"
                    )
                else:
                    pass
        else:
            pass

        validated_predictions_by_id[instance_id] = prediction_record
    return validated_predictions_by_id
