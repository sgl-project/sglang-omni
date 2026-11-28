# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Pydantic dataset-metadata, checkpoint, and manifest schemas for C3."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
)

from benchmarks.c3.registry import (
    C3EvaluationScope,
    C3Language,
    C3LanguageSelection,
    C3OutputModality,
    C3Phenomenon,
    C3SubtaskName,
    C3SubtaskSpec,
)

JUDGMENT_CHECKPOINT_SCHEMA_VERSION = 1
SuccessfulFinishReason = Literal["stop", "eos"]
ModelFinishReason = Literal["stop", "eos", "length"]
NonBlankReferenceText = Annotated[str, StringConstraints(pattern=r"(?s).*\S.*")]


class C3EnglishExplanationReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: NonBlankReferenceText
    notation: NonBlankReferenceText


class C3EnglishHeterographReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: NonBlankReferenceText
    problem: NonBlankReferenceText
    notation: NonBlankReferenceText


class C3SyntacticNotationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ambiguity: NonBlankReferenceText
    interpretations: tuple[NonBlankReferenceText, ...] = Field(min_length=1)


class C3EnglishSyntacticReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: NonBlankReferenceText
    notation: C3SyntacticNotationBlock


class C3EnglishCoreferenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: NonBlankReferenceText
    problem: NonBlankReferenceText
    answer: NonBlankReferenceText


class C3EnglishMultiturnReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    turns: tuple[NonBlankReferenceText, ...] = Field(min_length=2)


class C3ChineseAmbiguityReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: NonBlankReferenceText
    annotation: NonBlankReferenceText
    answer: str | None = None


class C3ChineseContextReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    history: tuple[NonBlankReferenceText, ...] = Field(min_length=1)
    omission_complete: str | None = None
    coreference_complete: str | None = None


class C3ChineseMultiturnReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    history: tuple[NonBlankReferenceText, ...] = Field(min_length=1)
    question: NonBlankReferenceText


C3ReferenceRecord = (
    C3EnglishExplanationReference
    | C3EnglishHeterographReference
    | C3EnglishSyntacticReference
    | C3EnglishCoreferenceReference
    | C3EnglishMultiturnReference
    | C3ChineseAmbiguityReference
    | C3ChineseContextReference
    | C3ChineseMultiturnReference
)


def parse_reference_record_for_spec(
    subtask_spec: C3SubtaskSpec, raw_record: JsonValue, source_index: int
) -> tuple[tuple[str, ...], C3ReferenceRecord]:
    record_location = f"{subtask_spec.reference_json_relative_path}[{source_index}]"
    try:
        if subtask_spec.is_multiturn and subtask_spec.language == "en":
            if not isinstance(raw_record, list):
                raise ValueError(f"{record_location} must be a JSON list of user turns")
            else:
                en_multiturn = C3EnglishMultiturnReference.model_validate(
                    {"turns": tuple(raw_record)}
                )
                return en_multiturn.turns, en_multiturn
        elif not isinstance(raw_record, dict):
            raise ValueError(f"{record_location} must be a JSON object")
        elif subtask_spec.is_multiturn and subtask_spec.language == "zh":
            zh_multiturn = C3ChineseMultiturnReference.model_validate(
                {
                    "history": (
                        tuple(raw_record["history"])
                        if isinstance(raw_record.get("history"), list)
                        else raw_record.get("history")
                    ),
                    "question": raw_record.get("question"),
                }
            )
            return (*zh_multiturn.history, zh_multiturn.question), zh_multiturn
        elif subtask_spec.language == "en":
            if subtask_spec.subtask == "phonological_heterograph":
                en_heterograph = C3EnglishHeterographReference.model_validate(
                    raw_record
                )
                return (en_heterograph.content,), en_heterograph
            elif subtask_spec.subtask == "semantic_syntactic":
                raw_notation = raw_record.get("notation")
                normalized_syntactic = dict(raw_record)
                if isinstance(raw_notation, dict) and isinstance(
                    raw_notation.get("interpretations"), list
                ):
                    normalized_syntactic["notation"] = {
                        **raw_notation,
                        "interpretations": tuple(raw_notation["interpretations"]),
                    }
                else:
                    pass
                en_syntactic = C3EnglishSyntacticReference.model_validate(
                    normalized_syntactic
                )
                return (en_syntactic.content,), en_syntactic
            elif subtask_spec.phenomenon == "coreference":
                en_coreference = C3EnglishCoreferenceReference.model_validate(
                    raw_record
                )
                return (en_coreference.content,), en_coreference
            else:
                en_explanation = C3EnglishExplanationReference.model_validate(
                    raw_record
                )
                return (en_explanation.content,), en_explanation
        elif subtask_spec.phenomenon in ("omission", "coreference"):
            normalized_context = dict(raw_record)
            if isinstance(raw_record.get("history"), list):
                normalized_context["history"] = tuple(raw_record["history"])
            else:
                pass
            zh_context = C3ChineseContextReference.model_validate(normalized_context)
            if subtask_spec.subtask == "omission_completion" and (
                zh_context.omission_complete is None
                or not zh_context.omission_complete.strip()
            ):
                raise ValueError(
                    f"{record_location} missing non-empty omission_complete"
                )
            elif subtask_spec.subtask == "coreference_resolution" and (
                zh_context.coreference_complete is None
                or not zh_context.coreference_complete.strip()
            ):
                raise ValueError(
                    f"{record_location} missing non-empty coreference_complete"
                )
            else:
                return ("\n".join(zh_context.history),), zh_context
        else:
            zh_ambiguity = C3ChineseAmbiguityReference.model_validate(raw_record)
            return (zh_ambiguity.content,), zh_ambiguity
    except ValidationError as validation_error:
        raise ValueError(
            f"Invalid record at {record_location}: {validation_error}"
        ) from None


class C3PredictionCheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int
    instance_id: str
    language: C3Language
    phenomenon: C3Phenomenon
    subtask: C3SubtaskName
    source_index: int
    source_audio_sha256: list[str]
    assistant_turns: list[str]
    finish_reasons: list[ModelFinishReason]
    final_answer: str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    saved_generation_audio: str | None = None
    generated_audio_sha256: str | None = None
    recorded_concurrency: int | None = None


class C3RunManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = 1
    benchmark: Literal["c3_spoken"] = "c3_spoken"
    model: str
    model_revision: str | None = None
    launch_command: str | None = None
    base_url: str
    endpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    modalities: list[C3OutputModality]
    export_generation_audio: bool
    temperature: float
    max_tokens: int
    seed: int | None = None
    max_concurrency: int
    warmup_requests: int
    request_timeout_seconds: float
    requested_scope: C3EvaluationScope
    effective_scope: C3EvaluationScope
    language: C3LanguageSelection
    subtasks: list[str] | None = None
    mini: bool
    max_samples_per_subtask: int | None = None
    selected_instances_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    concurrency_history: list[int] = Field(default_factory=list)
    warmup_history: list[int] = Field(default_factory=list)


class C3PublicJudgeManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    model: str
    model_revision: str | None = None
    base_url: str
    endpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    api_key_env: str | None = None
    max_concurrency: int
    concurrency_history: list[int] = Field(default_factory=list)
    enable_thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int


class C3JudgeManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = JUDGMENT_CHECKPOINT_SCHEMA_VERSION
    prompt_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judges: list[C3PublicJudgeManifestEntry]


class C3JudgmentCheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int
    instance_id: str
    language: C3Language
    subtask: C3SubtaskName
    judge_name: str
    judge_model: str
    judge_model_revision: str | None = None
    judge_enable_thinking: bool | None = None
    judge_temperature: float | None = None
    judge_max_tokens: int
    judge_endpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason: SuccessfulFinishReason
    raw_verdict_text: str
    correct: bool
