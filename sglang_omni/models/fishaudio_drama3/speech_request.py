# SPDX-License-Identifier: Apache-2.0
"""Lower a /v1/audio/speech request into the Drama 3 engine's generation input."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from tokenizers import Tokenizer

from sglang_omni.client.types import SamplingParams
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.prompt import (
    SEMANTIC_BEGIN_TOKEN,
    SPEAKER_IDS,
    SPEAKER_TAG_PATTERN,
    ReferencePart,
    build_prompt_rows,
    with_speaker_tag,
)
from sglang_omni.models.fishaudio_drama3.request_builders import (
    MINIMUM_PROMPT_ROWS,
    REQUEST_ERROR_PREFIX,
    Drama3GenerationInput,
)
from sglang_omni.models.fishaudio_drama3.text_config import checked_int
from sglang_omni.models.fishaudio_s2_pro.stages import (
    FishReferenceInput,
    fish_reference_payload_is_supported,
)
from sglang_omni.proto.request import (
    EXPLICIT_GENERATION_PARAMS_KEY,
    OmniRequest,
    StagePayload,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 30
# Lowered speech fields the model cannot honor; a value other than None was given.
UNSUPPORTED_TTS_PARAMS = (
    "language",
    "instructions",
    "x_vector_only_mode",
    "stream_codec_output",
    "suppress_bootstrap_silence",
    "token_count",
    "duration_tokens",
    "initial_codec_chunk_frames",
    "seed",
)
UNSUPPORTED_PARAMS = ("seed", "stage_params", "stage_sampling")
# A direct Client fills the sampling fields its caller left alone with these values.
CLIENT_SAMPLING_PLACEHOLDERS = SamplingParams().to_dict()
PLACEHOLDER_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty")
INT64_LIMITS = torch.iinfo(torch.int64)


def explicit_generation_fields(
    request: OmniRequest, tts_params: Mapping[str, object]
) -> frozenset[str]:
    """Sampling fields the caller set, as recorded in tts_params for speech requests or
    in metadata[EXPLICIT_GENERATION_PARAMS_KEY] otherwise. A direct Client caller lists
    there every sampling field it sets, placeholder values included."""
    is_recorded = isinstance(request.metadata.get("tts_params"), Mapping) or isinstance(
        request.metadata.get(EXPLICIT_GENERATION_PARAMS_KEY), list | tuple
    )
    unrecorded_fields = (
        []
        if is_recorded
        else [
            field_name
            for field_name in PLACEHOLDER_SAMPLING_FIELDS
            if request.params.get(field_name, CLIENT_SAMPLING_PLACEHOLDERS[field_name])
            != CLIENT_SAMPLING_PLACEHOLDERS[field_name]
        ]
    )
    if unrecorded_fields:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {', '.join(unrecorded_fields)} set without "
            f"{EXPLICIT_GENERATION_PARAMS_KEY}; list every sampling field you set in "
            f"the request metadata"
        )
    else:
        pass
    # Serving never fills in max_new_tokens, so a value is always the caller's.
    field_names = (
        {"max_new_tokens"}
        if request.params.get("max_new_tokens") is not None
        else set()
    )
    for source in (tts_params, request.metadata):
        recorded = source.get(EXPLICIT_GENERATION_PARAMS_KEY)
        if isinstance(recorded, list | tuple):
            field_names.update(str(field_name) for field_name in recorded)
        else:
            pass
    return frozenset(field_names)


def given_number(
    params: Mapping[str, object],
    explicit_fields: frozenset[str],
    field_name: str,
    default: float,
) -> float:
    """The caller's value of a sampling field it set, else the model's default."""
    value = params.get(field_name) if field_name in explicit_fields else default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {field_name} must be a number; got {value!r}"
        )
    else:
        return float(value)


def given_integer(
    params: Mapping[str, object],
    explicit_fields: frozenset[str],
    field_name: str,
    default: int | None,
) -> int | None:
    """The caller's value of an integer field it set, else the model's default."""
    if field_name not in explicit_fields:
        return default
    else:
        pass
    value = params.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {field_name} must be an integer; got {value!r}"
        )
    else:
        return value


def speech_inputs(inputs: object) -> tuple[str, list[Mapping[str, object]]]:
    """The text to speak and the reference descriptors of a lowered speech prompt."""
    if isinstance(inputs, str):
        return inputs, []
    elif not isinstance(inputs, Mapping) or not isinstance(inputs.get("text"), str):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} the speech input must be text or an object with "
            f"text and references"
        )
    else:
        pass
    references = inputs.get("references") or []
    if not isinstance(references, list) or not all(
        isinstance(reference, Mapping) for reference in references
    ):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} references must be a list of objects")
    else:
        return inputs["text"], references


def reference_codes(vq_codes: object, index: int) -> torch.Tensor:
    """A reference's VQ codes as a CPU int64 tensor; the prompt builder checks shape
    and codebook range."""
    is_integer_grid = (
        isinstance(vq_codes, list)
        and bool(vq_codes)
        and all(
            isinstance(row, list)
            and bool(row)
            and all(
                isinstance(code, int)
                and not isinstance(code, bool)
                and INT64_LIMITS.min <= code <= INT64_LIMITS.max
                for code in row
            )
            for row in vq_codes
        )
        and len({len(row) for row in vq_codes}) == 1
    )
    if not is_integer_grid:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} references.{index}.vq_codes must be a "
            f"[codebooks, frames] grid of int64 integers"
        )
    else:
        return torch.tensor(vq_codes, dtype=torch.int64, device="cpu")


class Drama3SpeechPreprocessor:
    """Turns a /v1/audio/speech stage payload into one carrying Drama3GenerationInput."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        config: FishOmniConfig,
        max_prompt_rows: int,
        reference_encoder: ReferenceEncodeService[
            FishReferenceInput, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        semantic_begin_id = tokenizer.token_to_id(SEMANTIC_BEGIN_TOKEN)
        if semantic_begin_id != config.semantic_start_token_id:
            raise ValueError(
                f"the tokenizer's {SEMANTIC_BEGIN_TOKEN} id {semantic_begin_id} is not "
                f"the config's semantic_start_token_id {config.semantic_start_token_id}"
            )
        else:
            pass
        self.tokenizer = tokenizer
        self.num_codebooks = config.audio_decoder_config.num_codebooks
        self.codebook_size = config.audio_decoder_config.vocab_size
        self.max_prompt_rows = checked_int(
            max_prompt_rows, "max_prompt_rows", MINIMUM_PROMPT_ROWS
        )
        self.reference_encoder = reference_encoder

    def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        tts_params = request.metadata.get("tts_params") or {}
        if not isinstance(tts_params, Mapping):
            raise TypeError(
                f"tts_params must be a mapping; got {type(tts_params).__name__}"
            )
        else:
            pass
        refused_fields = sorted(
            {
                field_name
                for field_name in UNSUPPORTED_TTS_PARAMS
                if tts_params.get(field_name) is not None
            }
            | {
                field_name
                for field_name in UNSUPPORTED_PARAMS
                if request.params.get(field_name) is not None
            }
        )
        voice = tts_params.get("voice")
        is_default_voice = voice is None or (
            isinstance(voice, str) and voice.lower() in ("", "default")
        )
        if refused_fields:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} {', '.join(refused_fields)} not supported"
            )
        elif tts_params.get("task_type") not in (None, "Base"):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} task_type {tts_params['task_type']!r} is not "
                f"supported; only Base is"
            )
        elif not is_default_voice and tts_params.get("uploaded_voice_name") is None:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} unknown voice {voice!r}; use default or an "
                f"uploaded voice"
            )
        else:
            pass

        explicit_fields = explicit_generation_fields(request, tts_params)
        temperature = given_number(
            request.params, explicit_fields, "temperature", DEFAULT_TEMPERATURE
        )
        top_p = given_number(request.params, explicit_fields, "top_p", DEFAULT_TOP_P)
        top_k = given_integer(request.params, explicit_fields, "top_k", DEFAULT_TOP_K)
        assert top_k is not None
        max_new_tokens = given_integer(
            request.params, explicit_fields, "max_new_tokens", None
        )
        repetition_penalty = given_number(
            request.params, explicit_fields, "repetition_penalty", 1.0
        )
        if repetition_penalty != 1.0:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} repetition_penalty must be 1.0; got "
                f"{repetition_penalty}"
            )
        else:
            pass

        text, reference_descriptors = speech_inputs(request.inputs)
        transcripts: list[str] = []
        for index, descriptor in enumerate(reference_descriptors):
            transcript = descriptor.get("text") or ""
            if not isinstance(transcript, str):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} references.{index}.text must be a string"
                )
            else:
                transcripts.append(transcript)
        # Malformed tags, an unbounded run of digits included, are left to the builder.
        other_speaker_ids = {
            int(speaker_tag.group(1))
            for speech_text in [text, *transcripts]
            for speaker_tag in SPEAKER_TAG_PATTERN.finditer(speech_text)
            if speaker_tag.group(1) in SPEAKER_IDS - {"0"}
        }
        if other_speaker_ids:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} only speaker 0 is supported; got speaker "
                f"tags {sorted(other_speaker_ids)}"
            )
        else:
            pass
        references = [
            ReferencePart(
                transcript=with_speaker_tag(transcript, 0),
                audio_codes=self.reference_audio_codes(descriptor, index),
            )
            for index, (descriptor, transcript) in enumerate(
                zip(reference_descriptors, transcripts, strict=True)
            )
        ]
        try:
            prompt_rows = build_prompt_rows(
                text,
                references,
                self.tokenizer,
                num_codebooks=self.num_codebooks,
                codebook_size=self.codebook_size,
            )
        except ValueError as error:
            raise ValueError(f"{REQUEST_ERROR_PREFIX} {error}") from error
        if prompt_rows.shape[0] > self.max_prompt_rows:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a prompt of {prompt_rows.shape[0]} rows exceeds "
                f"the maximum allowed length of {self.max_prompt_rows} rows"
            )
        else:
            pass
        return StagePayload(
            request_id=payload.request_id,
            request=request,
            data=Drama3GenerationInput(
                prompt_rows=prompt_rows,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )

    def reference_audio_codes(
        self, descriptor: Mapping[str, object], index: int
    ) -> torch.Tensor:
        """A reference's given VQ codes, or its audio encoded into codes."""
        vq_codes = descriptor.get("vq_codes")
        if vq_codes is not None:
            return reference_codes(vq_codes, index)
        elif fish_reference_payload_is_supported(descriptor):
            try:
                return self.reference_encoder.get_or_encode(
                    dict(descriptor), desc=f"Drama 3 reference {index}"
                )
            except ValueError as error:
                # note (SunskyXH): audio that does not decode raises ValueError (PyAV's
                # InvalidDataError is one); codec or device failures stay 500s.
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} references.{index}: {error}"
                ) from error
        else:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} references.{index} needs audio or vq_codes"
            )
