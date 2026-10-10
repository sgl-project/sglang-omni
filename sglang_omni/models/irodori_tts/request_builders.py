# SPDX-License-Identifier: Apache-2.0
"""Request parsing for Irodori-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from sglang_omni.proto import StagePayload


@dataclass(frozen=True, kw_only=True)
class IrodoriReference:
    audio_path: str | None = None
    audio_base64: str | None = None
    media_type: str = "audio/wav"


@dataclass(frozen=True, kw_only=True)
class IrodoriSynthesisRequest:
    text: str
    caption: str | None
    references: tuple[IrodoriReference, ...]
    seed: int | None
    num_steps: int | None
    seconds: float | None
    duration_scale: float | None
    cfg_scale_text: float | None
    cfg_scale_caption: float | None
    cfg_scale_speaker: float | None


def request_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    else:
        pass
    if isinstance(value, Mapping):
        return value
    else:
        pass
    raise TypeError(f"Irodori {field_name} must be an object")


def reference_from_value(value: object) -> IrodoriReference:
    if isinstance(value, str):
        if value.startswith("data:"):
            data_uri_header, data_uri_separator, encoded_audio = (
                value[5:].partition(",")
            )
            if not data_uri_separator or not data_uri_header.endswith(";base64"):
                raise ValueError("Irodori reference data URI must be base64 encoded")
            else:
                pass
            media_type = data_uri_header.removesuffix(";base64") or "audio/wav"
            return IrodoriReference(audio_base64=encoded_audio, media_type=media_type)
        else:
            pass
        parsed_path = urlparse(value)
        if parsed_path.scheme == "file":
            return IrodoriReference(audio_path=unquote(parsed_path.path))
        else:
            pass
        if parsed_path.scheme:
            raise ValueError("Irodori reference audio must be a local path or data URI")
        else:
            pass
        return IrodoriReference(audio_path=value)
    else:
        pass
    reference = request_mapping(value, "reference audio")
    for field_name in ("audio_path", "path", "ref_audio", "audio"):
        audio_path = reference.get(field_name)
        if audio_path is not None:
            if not isinstance(audio_path, str):
                raise TypeError("Irodori reference audio path must be a string")
            else:
                pass
            return reference_from_value(audio_path)
        else:
            pass
    encoded_audio = reference.get("data")
    if encoded_audio is None:
        encoded_audio = reference.get("base64")
    else:
        pass
    if encoded_audio is None:
        raise ValueError("Irodori reference is missing audio")
    else:
        pass
    if not isinstance(encoded_audio, str):
        raise TypeError("Irodori reference audio data must be a base64 string")
    else:
        pass
    media_type = reference.get("media_type") or "audio/wav"
    if not isinstance(media_type, str):
        raise TypeError("Irodori reference media_type must be a string")
    else:
        pass
    return reference_from_value(
        f"data:{media_type};base64,{encoded_audio}"
        if not encoded_audio.startswith("data:")
        else encoded_audio
    )


def optional_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    else:
        pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Irodori {field_name} must be an integer")
    else:
        pass
    return value


def optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    else:
        pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Irodori {field_name} must be a number")
    else:
        pass
    return float(value)


def build_irodori_request(payload: StagePayload) -> IrodoriSynthesisRequest:
    raw_inputs = payload.request.inputs
    input_mapping = (
        {} if isinstance(raw_inputs, str) else request_mapping(raw_inputs, "inputs")
    )
    if isinstance(raw_inputs, str):
        text = raw_inputs
    else:
        text_value = input_mapping.get("text", input_mapping.get("input"))
        if not isinstance(text_value, str):
            raise ValueError("Irodori requires a text input")
        else:
            pass
        text = text_value
    if not text.strip():
        raise ValueError("Irodori requires non-empty Japanese text")
    else:
        pass

    parameters = request_mapping(payload.request.params, "parameters")
    metadata = request_mapping(payload.request.metadata, "metadata")
    tts_parameters = request_mapping(metadata.get("tts_params"), "TTS parameters")

    caption_value = input_mapping.get("caption")
    if caption_value is None:
        caption_value = input_mapping.get("instructions")
    else:
        pass
    if caption_value is None:
        caption_value = tts_parameters.get("instructions")
    else:
        pass
    if caption_value is not None and not isinstance(caption_value, str):
        raise TypeError("Irodori caption must be a string")
    else:
        pass
    caption = caption_value.strip() if isinstance(caption_value, str) else None
    if caption == "":
        caption = None
    else:
        pass

    raw_references = input_mapping.get("references")
    if raw_references is None:
        raw_reference = input_mapping.get("ref_audio")
        if raw_reference is None:
            raw_reference = tts_parameters.get("ref_audio")
        else:
            pass
        raw_references = [] if raw_reference is None else [raw_reference]
    else:
        pass
    if not isinstance(raw_references, (list, tuple)):
        raise TypeError("Irodori references must be a list")
    else:
        pass
    references = tuple(reference_from_value(reference) for reference in raw_references)

    def parameter_value(parameter_name: str) -> object:
        value = parameters.get(parameter_name)
        if value is None:
            value = tts_parameters.get(parameter_name)
        else:
            pass
        return value

    return IrodoriSynthesisRequest(
        text=text,
        caption=caption,
        references=references,
        seed=optional_integer(parameter_value("seed"), "seed"),
        num_steps=optional_integer(parameter_value("num_steps"), "num_steps"),
        seconds=optional_float(parameter_value("seconds"), "seconds"),
        duration_scale=optional_float(
            parameter_value("duration_scale"), "duration_scale"
        ),
        cfg_scale_text=optional_float(
            parameter_value("cfg_scale_text"), "cfg_scale_text"
        ),
        cfg_scale_caption=optional_float(
            parameter_value("cfg_scale_caption"), "cfg_scale_caption"
        ),
        cfg_scale_speaker=optional_float(
            parameter_value("cfg_scale_speaker"), "cfg_scale_speaker"
        ),
    )
