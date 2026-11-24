# SPDX-License-Identifier: Apache-2.0
"""Validate Stable Audio 3 speech requests at the HTTP boundary."""

from pydantic import ValidationError

from sglang_omni.models.stable_audio3.request_builders import (
    GENERATION_STAGE,
    SEED_LIMIT,
    GenerationOptions,
)
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import bad_request
from sglang_omni.serve.speech_service import (
    SpeechRequestUpdates,
    SpeechRequestValidator,
)


class StableAudio3SpeechRequestValidator(SpeechRequestValidator):
    """A whole clip from a prompt: no voices, no streaming, and generation options under stage_params.generation."""

    def prepare_generation_updates(
        self, request: CreateSpeechRequest
    ) -> SpeechRequestUpdates:
        updates = super().prepare_generation_updates(request)
        stage_params = request.stage_params or {}
        if request.voice not in ("", "default"):
            raise bad_request("Stable Audio 3 has no voices", param="voice")
        elif updates["stream"]:
            raise bad_request("Stable Audio 3 does not stream", param="stream")
        elif request.seed is not None and not 0 <= request.seed < SEED_LIMIT:
            raise bad_request(f"seed must be in [0, {SEED_LIMIT})", param="seed")
        elif set(stage_params) - {GENERATION_STAGE}:
            raise bad_request(
                f"stage_params must target {GENERATION_STAGE}", param="stage_params"
            )
        else:
            pass
        try:
            GenerationOptions.model_validate(stage_params.get(GENERATION_STAGE, {}))
        except ValidationError as error:
            detail = error.errors()[0]
            path = ".".join(
                ["stage_params", GENERATION_STAGE, *map(str, detail["loc"])]
            )
            raise bad_request(f"{path}: {detail['msg']}", param=path) from error
        return updates
