# SPDX-License-Identifier: Apache-2.0
"""Lower a /v1/audio/speech request into the Drama 3 engine's generation input."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.prompt import (
    SEMANTIC_BEGIN_TOKEN,
    SPEAKER_IDS,
    SPEAKER_TAG_PATTERN,
    InfillSpan,
    ReferencePart,
    SpokenTake,
    audio_frame_rows,
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
from sglang_omni.proto.request import StagePayload
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
INT64_LIMITS = torch.iinfo(torch.int64)


def given_number(
    params: Mapping[str, object], field_name: str, default: float
) -> float:
    """The caller's value of a sampling field it set, else the model's default."""
    value = params.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {field_name} must be a number; got {value!r}"
        )
    else:
        return float(value)


def given_integer(
    params: Mapping[str, object], field_name: str, default: int | None
) -> int | None:
    """The caller's value of an integer field it set, else the model's default."""
    if field_name not in params:
        return default
    else:
        pass
    value = params[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {field_name} must be an integer; got {value!r}"
        )
    else:
        return value


def mapping_list(value: object, location: str) -> list[Mapping[str, object]]:
    """A JSON list of objects, or an empty list when the field was omitted."""
    if value is None:
        return []
    elif not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} {location} must be a list of objects")
    else:
        return value


def speech_inputs(
    inputs: object,
) -> tuple[
    str,
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    Mapping[str, object] | None,
]:
    """Text, references, previous takes and the infill span of a lowered prompt."""
    if isinstance(inputs, str):
        return inputs, [], [], None
    elif not isinstance(inputs, Mapping) or not isinstance(inputs.get("text"), str):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} the speech input must be text or an object with "
            f"text and references"
        )
    else:
        pass
    infill = inputs.get("infill")
    if infill is not None and not isinstance(infill, Mapping):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} infill must be an object")
    else:
        return (
            inputs["text"],
            mapping_list(inputs.get("references"), "references"),
            mapping_list(inputs.get("previous_takes"), "previous_takes"),
            infill,
        )


def reference_codes(
    vq_codes: object, location: str, *, allow_empty: bool = False
) -> torch.Tensor:
    """A reference's VQ codes as a CPU int64 tensor; the prompt builder checks shape
    and codebook range."""
    rows_are_lists = (
        isinstance(vq_codes, list)
        and bool(vq_codes)
        and all(isinstance(row, list) for row in vq_codes)
    )
    is_empty_grid = (
        allow_empty and rows_are_lists and all(row == [] for row in vq_codes)
    )
    is_integer_grid = (
        rows_are_lists
        and all(
            bool(row)
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
    if is_empty_grid:
        return torch.zeros((len(vq_codes), 0), dtype=torch.int64, device="cpu")
    elif not is_integer_grid:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {location}.vq_codes must be a "
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
        if (
            semantic_begin_id is None
            or semantic_begin_id != config.semantic_start_token_id
        ):
            raise ValueError(
                f"the tokenizer's {SEMANTIC_BEGIN_TOKEN} id {semantic_begin_id} is not "
                f"the config's semantic_start_token_id {config.semantic_start_token_id}"
            )
        else:
            pass
        self.tokenizer = tokenizer
        self.semantic_begin_id = semantic_begin_id
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

        temperature = given_number(request.params, "temperature", DEFAULT_TEMPERATURE)
        top_p = given_number(request.params, "top_p", DEFAULT_TOP_P)
        top_k = given_integer(request.params, "top_k", DEFAULT_TOP_K)
        assert top_k is not None
        max_new_tokens = given_integer(request.params, "max_new_tokens", None)
        repetition_penalty = given_number(request.params, "repetition_penalty", 1.0)
        if repetition_penalty != 1.0:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} repetition_penalty must be 1.0; got "
                f"{repetition_penalty}"
            )
        else:
            pass

        text, reference_descriptors, previous_take_descriptors, infill = speech_inputs(
            request.inputs
        )
        if infill is not None and request.params.get("stream"):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} infill does not stream; the kept audio is "
                f"spliced around the generated span"
            )
        elif infill is not None and (
            reference_descriptors or previous_take_descriptors
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} infill does not take references or "
                f"previous takes"
            )
        else:
            pass
        transcripts: list[str] = []
        for index, descriptor in enumerate(reference_descriptors):
            transcript = descriptor.get("text") or ""
            if not isinstance(transcript, str):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} references.{index}.text must be a string"
                )
            else:
                transcripts.append(transcript)
        previous_take_texts: list[str] = []
        for index, descriptor in enumerate(previous_take_descriptors):
            take_text = descriptor.get("text")
            if not isinstance(take_text, str):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} previous_takes.{index}.text must be a "
                    f"string"
                )
            else:
                previous_take_texts.append(take_text)
        # Malformed tags, an unbounded run of digits included, are left to the builder.
        other_speaker_ids = {
            int(speaker_tag.group(1))
            for speech_text in [text, *transcripts, *previous_take_texts]
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
                audio_codes=self.reference_audio_codes(
                    descriptor, f"references.{index}"
                ),
            )
            for index, (descriptor, transcript) in enumerate(
                zip(reference_descriptors, transcripts, strict=True)
            )
        ]
        previous_takes = [
            SpokenTake(
                text=take_text,
                audio_codes=self.reference_audio_codes(
                    descriptor, f"previous_takes.{index}"
                ),
            )
            for index, (descriptor, take_text) in enumerate(
                zip(previous_take_descriptors, previous_take_texts, strict=True)
            )
        ]
        infill_span = None if infill is None else self.infill_span(infill)
        try:
            prompt_rows = build_prompt_rows(
                text,
                references,
                self.tokenizer,
                num_codebooks=self.num_codebooks,
                codebook_size=self.codebook_size,
                previous_takes=previous_takes,
                infill_span=infill_span,
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
                kept_prefix_rows=(
                    None
                    if infill_span is None
                    else audio_frame_rows(
                        infill_span.prefix_codes, self.semantic_begin_id
                    )
                ),
                kept_suffix_rows=(
                    None
                    if infill_span is None
                    else audio_frame_rows(
                        infill_span.suffix_codes, self.semantic_begin_id
                    )
                ),
            ),
        )

    def infill_span(self, infill: Mapping[str, object]) -> InfillSpan:
        prefix = infill.get("prefix")
        suffix = infill.get("suffix")
        if not isinstance(prefix, Mapping) or not isinstance(suffix, Mapping):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} infill.prefix and infill.suffix must be objects"
            )
        else:
            pass
        raw_audio_token_count = infill.get("audio_token_count")
        if raw_audio_token_count is None:
            audio_token_count = None
        elif (
            isinstance(raw_audio_token_count, int)
            and not isinstance(raw_audio_token_count, bool)
            and raw_audio_token_count >= 1
        ):
            audio_token_count = raw_audio_token_count
        else:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} infill.audio_token_count must be an integer "
                f">= 1; got {raw_audio_token_count!r}"
            )
        return InfillSpan(
            prefix_codes=self.span_codes(prefix, "infill.prefix"),
            suffix_codes=self.span_codes(suffix, "infill.suffix"),
            audio_token_count=audio_token_count,
        )

    def span_codes(
        self, descriptor: Mapping[str, object], location: str
    ) -> torch.Tensor:
        """An empty object is that side of the recording."""
        if descriptor:
            return self.reference_audio_codes(descriptor, location, allow_empty=True)
        else:
            return torch.zeros((self.num_codebooks, 0), dtype=torch.int64, device="cpu")

    def reference_audio_codes(
        self,
        descriptor: Mapping[str, object],
        location: str,
        *,
        allow_empty: bool = False,
    ) -> torch.Tensor:
        """A clip's given VQ codes, or its audio encoded into codes."""
        vq_codes = descriptor.get("vq_codes")
        if vq_codes is not None:
            return reference_codes(vq_codes, location, allow_empty=allow_empty)
        elif fish_reference_payload_is_supported(descriptor):
            try:
                return self.reference_encoder.get_or_encode(
                    dict(descriptor), desc=f"Drama 3 {location}"
                )
            except ValueError as error:
                # note (SunskyXH): audio that does not decode raises ValueError (PyAV's
                # InvalidDataError is one); codec or device failures stay 500s.
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} {location}: {error}"
                ) from error
        else:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} {location} needs audio or vq_codes"
            )
