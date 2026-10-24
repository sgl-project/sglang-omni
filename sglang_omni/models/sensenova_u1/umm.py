# SPDX-License-Identifier: Apache-2.0
"""Single-phase SenseNova generation on the shared UMM controller."""

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sglang_omni.client.image_media import InlineImage
from sglang_omni.models.sensenova_u1.admission import (
    ADMISSION_METADATA_KEY,
    DEFAULT_UMM_ADMISSION_LIMITS,
    SenseNovaAdmissionLimits,
    validate_admission,
)
from sglang_omni.models.sensenova_u1.cancellation import SENSENOVA_DEADLINE_METADATA_KEY
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)


class SenseNovaImageResult(BaseModel):
    model_config = ConfigDict(strict=True)

    modality: Literal["image"]
    image_b64: str | None = None
    images_b64: list[str] | None = None


@dataclass(kw_only=True)
class SenseNovaExecution:
    expected_images: int
    retained_bytes: int
    reserved_output_bytes: int


class SenseNovaUMMAdapter:
    def __init__(self, model_path: str) -> None:
        """The shared factory supplies model_path; the generation stage loads it."""
        self.limits: SenseNovaAdmissionLimits = DEFAULT_UMM_ADMISSION_LIMITS.model_copy(
            deep=True
        )
        self.executions: dict[str, SenseNovaExecution] = {}

    def start(self, context: UMMContext) -> UMMStep:
        try:
            admitted = validate_admission(context.request, self.limits)
        except (ValueError, OSError) as error:
            raise UMMRequestError(str(error)) from error
        self.executions[context.execution_id] = SenseNovaExecution(
            expected_images=admitted.expected_images,
            retained_bytes=admitted.input_bytes,
            reserved_output_bytes=admitted.reserved_output_bytes,
        )
        request = replace(
            context.request,
            metadata={
                **context.request.metadata,
                ADMISSION_METADATA_KEY: self.limits.model_dump(),
                SENSENOVA_DEADLINE_METADATA_KEY: context.deadline,
            },
        )
        return UMMStep(phase="generate", request=request)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        """Complete the request; start dispatches only the generate phase."""
        generation = SenseNovaImageResult.model_validate(result)
        images = generation.images_b64
        if images is None:
            images = [generation.image_b64] if generation.image_b64 is not None else []
        else:
            pass
        execution = self.executions[context.execution_id]
        if len(images) != execution.expected_images:
            raise ValueError("SenseNova output count does not match the request")
        else:
            pass
        execution.reserved_output_bytes = 0
        return UMMStep(
            outputs=tuple(
                UMMOutput("image", InlineImage.from_png(encoded_image).model_dump())
                for encoded_image in images
            )
        )

    def usage(self, context: UMMContext) -> int:
        execution = self.executions[context.execution_id]
        return execution.retained_bytes + execution.reserved_output_bytes

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.executions.pop(context.execution_id, None)
