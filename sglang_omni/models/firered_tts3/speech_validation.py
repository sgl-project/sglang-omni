# SPDX-License-Identifier: Apache-2.0
"""Validate FireRedTTS3 generation parameters at request boundaries."""

from __future__ import annotations

from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from sglang_omni.serve.protocol import SUPPORTED_TTS_TASK_TYPES, CreateSpeechRequest
from sglang_omni.serve.speech_errors import bad_request
from sglang_omni.serve.speech_service import (
    PreparedSpeechReferences,
    SpeechRequestValidator,
)
from sglang_omni.utils.json import JsonValue


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


class FireRedTTS3SpeechRequestValidator(SpeechRequestValidator):
    """Apply native editing, speed and checkpoint parameter rules at the API boundary."""

    supports_dual_stream: ClassVar[bool] = False

    @staticmethod
    def normalize_task_type(value: str) -> str:
        supported = SUPPORTED_TTS_TASK_TYPES | {"SemanticEdit", "AcousticEdit"}
        normalized = value.strip().replace("_", "").replace("-", "").lower()
        for task_type in supported:
            if normalized == task_type.lower():
                return task_type
            else:
                pass
        raise bad_request(
            f"task_type must be one of: {', '.join(sorted(supported))}",
            param="task_type",
        )

    def validate_input_text(
        self, input_text: str, *, task_type: str | None = None
    ) -> None:
        task_type = (
            self.normalize_task_type(task_type) if task_type is not None else None
        )
        if (
            task_type in {"SemanticEdit", "AcousticEdit"}
            and isinstance(input_text, str)
            and not input_text.strip()
        ):
            return
        else:
            super().validate_input_text(input_text, task_type=task_type)

    def validate_extra_params(self, request: CreateSpeechRequest) -> None:
        extra_params = request.extra_params or {}
        stage_parameters = extra_params.get("stage_params", request.stage_params or {})
        if isinstance(stage_parameters, dict):
            talker_parameters = stage_parameters.get("talker")
        else:
            talker_parameters = None
        if not isinstance(talker_parameters, dict):
            talker_parameters = {}
        else:
            pass
        generation_values = {}
        for parameter_name in FireRedTTS3GenerationOverrides.model_fields:
            if parameter_name == "seed" and request.seed is not None:
                generation_values[parameter_name] = request.seed
            elif talker_parameters.get(parameter_name) is not None:
                generation_values[parameter_name] = talker_parameters[parameter_name]
            elif extra_params.get(parameter_name) is not None:
                generation_values[parameter_name] = extra_params[parameter_name]
            else:
                pass
        try:
            FireRedTTS3GenerationOverrides.model_validate(generation_values)
        except ValidationError as error:
            detail = error.errors()[0]
            parameter_name = str(detail["loc"][0])
            if parameter_name == "seed" and request.seed is not None:
                parameter_path = "seed"
            elif talker_parameters.get(parameter_name) is not None:
                stage_path = (
                    "extra_params.stage_params"
                    if "stage_params" in extra_params
                    else "stage_params"
                )
                parameter_path = f"{stage_path}.talker.{parameter_name}"
            else:
                parameter_path = f"extra_params.{parameter_name}"
            raise bad_request(
                f"{parameter_path}: {detail['msg']}", param=parameter_path
            ) from error
        if "dual_stream" in extra_params:
            if not isinstance(extra_params["dual_stream"], bool):
                raise bad_request(
                    "extra_params.dual_stream must be a boolean", param="extra_params"
                )
            elif extra_params["dual_stream"] and not self.supports_dual_stream:
                raise bad_request(
                    "extra_params.dual_stream requires a dual-stream-capable "
                    "checkpoint (this deployment does not support it)",
                    param="extra_params",
                )
            else:
                pass
        else:
            pass
        if extra_params.get("style") is not None and not isinstance(
            extra_params["style"], str
        ):
            raise bad_request(
                "extra_params.style must be a string", param="extra_params"
            )
        else:
            pass

    def resolve_encode_speed(self, request: CreateSpeechRequest) -> float:
        # note (kermit): FireRed consumes speed natively; encode-time resampling would apply it twice.
        return 1.0


class FireRedTTS3SpeakerSpeechRequestValidator(FireRedTTS3SpeechRequestValidator):
    """Speaker checkpoints use speaker tokens and cannot consume reference audio."""

    supports_dual_stream: ClassVar[bool] = True

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
