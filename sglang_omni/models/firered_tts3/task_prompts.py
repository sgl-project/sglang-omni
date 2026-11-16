# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare Base and Instruct reference-audio prompts for FireRedTTS3."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    FireRedTTS3TaskType,
    first_not_none,
    require_edit_source,
    resolve_request_reference,
    resolve_text_sampling,
)
from sglang_omni.utils.json import JsonValue

CHATML_LATENT_IN_PAD_SYM = "<|image_pad|>"
CHATML_LATENT_OUT_PAD_SYM = "<|video_pad|>"


MULTI_LANG_TAGS = (
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Russian",
    "Arabic",
    "Turkish",
    "Indonesian",
    "Portuguese",
    "Italian",
    "Dutch",
    "Vietnamese",
    "German",
    "Ukrainian",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
)

MULTI_DIALECT_TAGS = tuple(
    f"ZH_{name}"
    for name in (
        "Anhui",
        "Fujian",
        "Gansu",
        "Guizhou",
        "Hebei",
        "Henan",
        "Hubei",
        "Hunan",
        "Jiangxi",
        "Liaoning",
        "Minnan",
        "Ningxia",
        "Shaanxi",
        "Shandong",
        "Shanghai",
        "Shanxi",
        "Sichuan",
        "Tianjin",
        "Wenzhou",
        "Wu",
        "Yunnan",
    )
)


class FireRedTTS3AcousticParameters(BaseModel):
    """Validate native acoustic-edit composition for serving and SDK requests."""

    model_config = ConfigDict(allow_inf_nan=False)

    acoustic_attribute: Literal["Speed", "Pitch", "Volume"] = "Speed"
    acoustic_value: float = 1.0

    @field_validator("acoustic_attribute", mode="before")
    @classmethod
    def normalize_acoustic_attribute(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return value.strip().capitalize()
        else:
            return value

    @field_validator("acoustic_value", mode="before")
    @classmethod
    def validate_acoustic_value(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, bool):
            raise ValueError("acoustic_value must be numeric")
        else:
            return value


def convert_to_chatml(
    input_text: str,
    audio_input_patch_count: int = 0,
    output_text: str = "",
    audio_output_patch_count: int = 0,
) -> str:
    audio_input = (
        f"<|sosp|>{CHATML_LATENT_IN_PAD_SYM * audio_input_patch_count}<|eosp|>\n"
        if audio_input_patch_count > 0
        else ""
    )
    audio_output = (
        f"<|sosp|>{CHATML_LATENT_OUT_PAD_SYM * audio_output_patch_count}<|eosp|>\n"
        if audio_output_patch_count > 0
        else ""
    )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{audio_input}{input_text} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{output_text}{audio_output}<|im_end|>\n"
    )


def resolve_acoustic_instruction(
    instruction_text: str | None,
    request_inputs: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
) -> str:
    """Use an explicit instruction or a trained acoustic-edit template."""
    if instruction_text is not None and instruction_text.strip():
        return instruction_text.strip()
    else:
        pass
    acoustic_parameters = FireRedTTS3AcousticParameters.model_validate(
        {
            parameter_name: value
            for parameter_name, value in {
                "acoustic_attribute": first_not_none(
                    preprocessing_parameters.get("acoustic_attribute"),
                    tts_parameters.get("acoustic_attribute"),
                ),
                "acoustic_value": first_not_none(
                    preprocessing_parameters.get("acoustic_value"),
                    tts_parameters.get("acoustic_value"),
                    request_inputs.get("speed"),
                    tts_parameters.get("speed"),
                ),
            }.items()
            if value is not None
        }
    )
    acoustic_value = acoustic_parameters.acoustic_value
    if acoustic_parameters.acoustic_attribute == "Speed":
        return f"adjust the speed to {acoustic_value:.1f}x"
    elif acoustic_parameters.acoustic_attribute == "Volume":
        return f"adjust the volume to {acoustic_value:.1f}"
    else:
        steps = int(round(acoustic_value))
        return f"shift the pitch by {steps} step{'' if abs(steps) == 1 else 's'}"


def compose_base_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    request_inputs: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    if not text.strip():
        raise ValueError("FireRedTTS3 requires non-empty input text")
    else:
        pass
    text = text.strip()
    prompt_audio_path, prompt_text_value = resolve_request_reference(
        request_inputs, tts_parameters
    )
    if prompt_audio_path is None:
        raise ValueError("FireRedTTS3-Base requires a reference audio (voice cloning)")
    else:
        pass
    if prompt_text_value is None or not str(prompt_text_value).strip():
        raise ValueError(
            "FireRedTTS3-Base requires the reference audio transcript (ref_text / references[0].text)"
        )
    else:
        pass
    prompt_text = str(prompt_text_value).strip()
    language = str(
        first_not_none(
            request_inputs.get("language"),
            tts_parameters.get("language"),
            default="Chinese",
        )
    )
    if language not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(
            f"unsupported FireRedTTS3 language {language!r}; expected one of MULTI_LANG_TAGS/MULTI_DIALECT_TAGS"
        )
    else:
        pass
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="base",
        task_type="tts",
        text=text,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        language=language,
        input_ids=tokenizer(
            f"<|{language}|><|sot|>{prompt_text}{text}<|eot|>",
            add_special_tokens=False,
        )["input_ids"],
    )


def compose_instruct_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    task_type: FireRedTTS3TaskType,
    request_inputs: dict[str, JsonValue],
    request_parameters: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    instruction_text = first_not_none(
        request_inputs.get("instructions"), tts_parameters.get("instructions")
    )
    if instruction_text is not None:
        instruction_text = str(instruction_text).strip() or None
    else:
        pass
    reference_audio, prompt_text_value = resolve_request_reference(
        request_inputs, tts_parameters
    )
    prompt_audio_path: str | None = None
    audio_input_path: str | None = None
    prompt_text: str | None = None
    reference_pad_token_id: int | None = None
    if task_type == "tts":
        if not text.strip():
            raise ValueError("FireRedTTS3 tts requires non-empty input text")
        else:
            pass
        if reference_audio is None:
            raise ValueError(
                "FireRedTTS3-Instruct tts requires a reference audio (ref_audio / references[0])"
            )
        else:
            pass
        if prompt_text_value is None or not str(prompt_text_value).strip():
            raise ValueError(
                "FireRedTTS3-Instruct tts requires the reference audio transcript (ref_text / references[0].text)"
            )
        else:
            pass
        prompt_audio_path = reference_audio
        prompt_text = str(prompt_text_value).strip()
        reference_pad_token_id = tokenizer.convert_tokens_to_ids(
            CHATML_LATENT_OUT_PAD_SYM
        )
        prompt_template = convert_to_chatml(
            input_text=f"Convert text to speech.\n{prompt_text}{text.strip()}",
            audio_output_patch_count=1,
        ).removesuffix("<|eosp|>\n<|im_end|>\n")
    elif task_type == "voice_design":
        if not text.strip():
            raise ValueError("FireRedTTS3 voice_design requires non-empty input text")
        else:
            pass
        if not instruction_text:
            raise ValueError("FireRedTTS3 voice_design requires instructions")
        else:
            pass
        prompt_template = convert_to_chatml(
            input_text=f"{instruction_text}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{text.strip()}",
            output_text="<|sot|>",
        ).removesuffix("<|im_end|>\n")
    elif task_type == "semantic_edit":
        if not instruction_text:
            raise ValueError("FireRedTTS3 semantic_edit requires instructions")
        else:
            pass
        audio_input_path = require_edit_source(reference_audio, task_type)
        reference_pad_token_id = tokenizer.convert_tokens_to_ids(
            CHATML_LATENT_IN_PAD_SYM
        )
        prompt_template = convert_to_chatml(
            input_text=f"Identify the content of the audio. {instruction_text.strip()}",
            audio_input_patch_count=1,
            output_text="<|sot|>",
        ).removesuffix("<|im_end|>\n")
    elif task_type == "acoustic_edit":
        audio_input_path = require_edit_source(reference_audio, task_type)
        instruction = resolve_acoustic_instruction(
            instruction_text, request_inputs, tts_parameters, preprocessing_parameters
        )
        reference_pad_token_id = tokenizer.convert_tokens_to_ids(
            CHATML_LATENT_IN_PAD_SYM
        )
        prompt_template = convert_to_chatml(
            input_text=instruction,
            audio_input_patch_count=1,
            audio_output_patch_count=1,
        ).removesuffix("<|video_pad|><|eosp|>\n<|im_end|>\n")
    else:
        raise AssertionError(f"unhandled FireRedTTS3-Instruct task: {task_type}")
    text_temperature, text_top_p, text_top_k = resolve_text_sampling(
        task_type, request_parameters
    )
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="instruct",
        task_type=task_type,
        text=text.strip(),
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        instructions=instruction_text,
        audio_in_path=audio_input_path,
        input_ids=tokenizer(prompt_template, add_special_tokens=False)["input_ids"],
        reference_pad_token_id=reference_pad_token_id,
        text_temperature=text_temperature,
        text_top_p=text_top_p,
        text_top_k=text_top_k,
    )
