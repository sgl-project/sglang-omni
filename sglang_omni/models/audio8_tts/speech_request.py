# SPDX-License-Identifier: Apache-2.0
"""Validate the buffered Audio8 speech API request."""

from pydantic import ValidationError

from sglang_omni.models.audio8_tts.payload_types import Audio8SamplingOptions
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import bad_request
from sglang_omni.serve.speech_service import (
    SpeechRequestUpdates,
    SpeechRequestValidator,
)


class Audio8SpeechRequestValidator(SpeechRequestValidator):
    def prepare_generation_updates(
        self, request: CreateSpeechRequest
    ) -> SpeechRequestUpdates:
        references = request.references or []
        if request.stream or request.stream_format == "sse":
            raise bad_request("Audio8 TTS requires stream=false", param="stream")
        elif request.voice.casefold() != "default":
            raise bad_request("Audio8 TTS uses voice=default", param="voice")
        elif len(references) + int(request.ref_audio is not None) > 1:
            raise bad_request(
                "Audio8 TTS accepts one speech reference", param="references"
            )
        elif request.ref_text is not None and request.ref_audio is None:
            raise bad_request("ref_text requires ref_audio", param="ref_audio")
        else:
            pass

        if references:
            reference = references[0]
            if reference.vq_codes is not None:
                raise bad_request(
                    "Audio8 TTS accepts reference waveforms",
                    param="references.0.vq_codes",
                )
            elif not any(
                (
                    reference.audio_path,
                    reference.ref_audio,
                    reference.audio,
                    reference.data,
                )
            ):
                raise bad_request(
                    "Audio8 TTS requires reference audio", param="references.0"
                )
            else:
                pass
        else:
            pass

        try:
            Audio8SamplingOptions.model_validate(
                request.model_dump(
                    include=set(Audio8SamplingOptions.model_fields),
                    exclude_none=True,
                )
            )
        except ValidationError as error:
            invalid_option = error.errors(include_url=False)[0]
            parameter = str(invalid_option["loc"][0])
            raise bad_request(invalid_option["msg"], param=parameter) from error

        return super().prepare_generation_updates(request)
