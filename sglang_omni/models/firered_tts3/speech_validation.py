# SPDX-License-Identifier: Apache-2.0
"""Validate FireRedTTS3 generation parameters at request boundaries."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StrictBool,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from sglang_omni.models.firered_tts3.task_prompts import FireRedTTS3AcousticParameters
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import bad_request
from sglang_omni.serve.speech_service import (
    PreparedSpeechReferences,
    SpeechRequestUpdates,
    SpeechRequestValidator,
)
from sglang_omni.utils.json import JsonValue

FireRedTTS3SpeechTask = Literal["Base", "VoiceDesign", "SemanticEdit", "AcousticEdit"]


class FireRedTTS3GenerationOverrides(BaseModel):
    """Validate generation parameters supplied at request or stage boundaries."""

    model_config = ConfigDict(allow_inf_nan=False)

    n_timesteps: PositiveInt | None = None
    inference_cfg: float | None = None
    stop_threshold: float | None = None
    min_patches: NonNegativeInt | None = None
    max_patches: PositiveInt | None = None
    seed: int | None = None

    @field_validator("*", mode="before")
    @classmethod
    def validate_numeric_value(
        cls, value: JsonValue, validation_info: ValidationInfo
    ) -> JsonValue:
        if isinstance(value, bool) or (
            value is None and validation_info.field_name != "seed"
        ):
            raise ValueError("generation parameters must be numeric values")
        else:
            return value


class FireRedTTS3PreprocessingOverrides(BaseModel):
    """Validate speaker prompt options before dispatch."""

    dual_stream: StrictBool | None = None
    style: StrictStr | None = None


class FireRedTTS3SpeechRequestValidator(SpeechRequestValidator):
    """Validate native speed, generation settings and Base task admission."""

    supports_dual_stream: ClassVar[bool] = False
    requires_explicit_language: ClassVar[bool] = True
    supported_task_types: ClassVar[frozenset[FireRedTTS3SpeechTask]] = frozenset(
        {"Base"}
    )

    @classmethod
    def normalize_task_type(cls, value: str) -> FireRedTTS3SpeechTask:
        normalized = value.strip().replace("_", "").replace("-", "").lower()
        for task_type in cls.supported_task_types:
            if normalized == task_type.lower():
                return task_type
            else:
                pass
        raise bad_request(
            f"task_type must be one of: {', '.join(sorted(cls.supported_task_types))}",
            param="task_type",
        )

    def prepare_generation_updates(
        self, request: CreateSpeechRequest
    ) -> SpeechRequestUpdates:
        updates = super().prepare_generation_updates(request)
        stage_parameters = request.stage_params or {}
        talker_parameters = {
            parameter_name: value
            for parameter_name, value in stage_parameters.get("talker", {}).items()
            if value is not None
        }
        if request.seed is not None:
            talker_parameters["seed"] = request.seed
        else:
            pass
        preprocessing_parameters = stage_parameters.get("preprocessing", {})
        for stage_name, parameter_schema, stage_values in (
            ("talker", FireRedTTS3GenerationOverrides, talker_parameters),
            (
                "preprocessing",
                FireRedTTS3PreprocessingOverrides,
                preprocessing_parameters,
            ),
        ):
            try:
                parameter_schema.model_validate(stage_values)
            except ValidationError as error:
                detail = error.errors()[0]
                parameter_name = str(detail["loc"][0])
                parameter_path = (
                    "seed"
                    if stage_name == "talker"
                    and parameter_name == "seed"
                    and request.seed is not None
                    else f"stage_params.{stage_name}.{parameter_name}"
                )
                raise bad_request(
                    f"{parameter_path}: {detail['msg']}", param=parameter_path
                ) from error
        if (
            preprocessing_parameters.get("dual_stream")
            and not self.supports_dual_stream
        ):
            raise bad_request(
                "dual_stream requires a dual-stream-capable checkpoint",
                param="stage_params.preprocessing.dual_stream",
            )
        else:
            pass
        return updates

    def resolve_encode_speed(self, request: CreateSpeechRequest) -> float:
        # note (kermit): FireRed consumes speed natively; encode-time resampling would apply it twice.
        return 1.0

    def normalize_language(self, value: str) -> str:
        language = super().normalize_language(value)
        if language == "Auto" and self.requires_explicit_language:
            raise bad_request(
                "FireRedTTS3-Base requires an explicit language", param="language"
            )
        else:
            return language

    def validate_speech_references(
        self,
        request: CreateSpeechRequest,
        prepared_references: PreparedSpeechReferences,
    ) -> None:
        super().validate_speech_references(request, prepared_references)
        if any(
            not (reference.get("audio_path") or reference.get("data"))
            for reference in prepared_references.reference_descriptors
        ):
            raise bad_request("reference audio is required", param="references")
        else:
            pass


class FireRedTTS3InstructSpeechRequestValidator(FireRedTTS3SpeechRequestValidator):
    """Validate Instruct task requirements before backend dispatch."""

    requires_explicit_language: ClassVar[bool] = False
    supported_task_types: ClassVar[frozenset[FireRedTTS3SpeechTask]] = frozenset(
        {"Base", "VoiceDesign", "SemanticEdit", "AcousticEdit"}
    )

    def validate_input_text(
        self, input_text: str, *, task_type: str | None = None
    ) -> None:
        normalized_task = (
            self.normalize_task_type(task_type) if task_type is not None else "Base"
        )
        if (
            normalized_task in {"SemanticEdit", "AcousticEdit"}
            and not input_text.strip()
        ):
            return
        else:
            super().validate_input_text(input_text, task_type=normalized_task)

    def prepare_generation_updates(
        self, request: CreateSpeechRequest
    ) -> SpeechRequestUpdates:
        updates = super().prepare_generation_updates(request)
        if updates.get("task_type") in {"VoiceDesign", "SemanticEdit"} and not (
            request.instructions and request.instructions.strip()
        ):
            raise bad_request(
                "instructions are required for this task", param="instructions"
            )
        elif updates.get("task_type") == "AcousticEdit" and not (
            request.instructions and request.instructions.strip()
        ):
            preprocessing_parameters = (request.stage_params or {}).get(
                "preprocessing", {}
            )
            acoustic_values = {
                parameter_name: value
                for parameter_name, value in preprocessing_parameters.items()
                if value is not None
            }
            acoustic_values.setdefault("acoustic_value", request.speed)
            try:
                FireRedTTS3AcousticParameters.model_validate(acoustic_values)
            except ValidationError as error:
                detail = error.errors()[0]
                parameter_path = f"stage_params.preprocessing.{detail['loc'][0]}"
                raise bad_request(
                    f"{parameter_path}: {detail['msg']}", param=parameter_path
                ) from error
        else:
            pass
        return updates

    def validate_speech_references(
        self,
        request: CreateSpeechRequest,
        prepared_references: PreparedSpeechReferences,
    ) -> None:
        super().validate_speech_references(request, prepared_references)
        task_type = (
            self.normalize_task_type(request.task_type)
            if request.task_type is not None
            else "Base"
        )
        references = prepared_references.reference_descriptors
        if task_type == "VoiceDesign":
            if references:
                raise bad_request(
                    "VoiceDesign does not accept speech references", param="references"
                )
            else:
                pass
        elif len(references) != 1:
            raise bad_request(
                "exactly one speech reference is required for this task",
                param="ref_audio" if not references else "references",
            )
        elif task_type == "Base" and not (
            isinstance(reference_text := references[0].get("text"), str)
            and reference_text.strip()
        ):
            raise bad_request("reference transcript is required", param="ref_text")
        else:
            pass


class FireRedTTS3SpeakerSpeechRequestValidator(FireRedTTS3SpeechRequestValidator):
    """Speaker checkpoints use speaker tokens and cannot consume reference audio."""

    supports_dual_stream: ClassVar[bool] = True
    requires_explicit_language: ClassVar[bool] = False

    def validate_speech_references(
        self,
        request: CreateSpeechRequest,
        prepared_references: PreparedSpeechReferences,
    ) -> None:
        if prepared_references.reference_descriptors:
            raise bad_request(
                "this model does not accept speech references",
                param="ref_audio" if request.ref_audio is not None else "references",
            )
        else:
            super().validate_speech_references(request, prepared_references)
