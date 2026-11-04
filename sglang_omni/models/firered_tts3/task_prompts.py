# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare Base and Instruct reference-audio prompts for FireRedTTS3."""

from __future__ import annotations

import base64
import io
import math
from urllib.request import urlopen

import soundfile as sf
import torch
import torchaudio
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import (
    MAX_EDIT_SECONDS,
    FireRedTTS3State,
    FireRedTTS3TaskType,
    first_not_none,
    require_edit_source,
    resolve_request_reference,
    resolve_text_sampling,
)
from sglang_omni.utils.json import JsonValue

AUDIO_SAMPLE_RATE = 24000
AUDIO_URL_TIMEOUT_SECONDS = 60
SAMPLES_PER_PATCH = 3840
ACOUSTIC_ATTRIBUTES = ("Speed", "Pitch", "Volume")
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


def convert_to_chatml(
    text_in: str,
    latent_in_len: int = 0,
    text_out: str = "",
    latent_out_len: int = 0,
) -> str:
    audio_input = (
        f"<|sosp|>{CHATML_LATENT_IN_PAD_SYM * latent_in_len}<|eosp|>\n"
        if latent_in_len > 0
        else ""
    )
    audio_output = (
        f"<|sosp|>{CHATML_LATENT_OUT_PAD_SYM * latent_out_len}<|eosp|>\n"
        if latent_out_len > 0
        else ""
    )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{audio_input}{text_in} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{text_out}{audio_output}<|im_end|>\n"
    )


def count_audio_patches(audio_path: str, *, cap_seconds: float | None) -> int:
    """Count prompt slots after clipping and resampling the reference audio."""
    # note (kermit): Prompt counting preserves its URL timeout and SoundFile error contract.
    audio_source: str | io.BytesIO
    if audio_path.startswith(("http://", "https://")):
        with urlopen(audio_path, timeout=AUDIO_URL_TIMEOUT_SECONDS) as response:
            audio_source = io.BytesIO(response.read())
    elif audio_path.startswith("data:"):
        _, _, encoded_audio = audio_path.partition(",")
        audio_source = io.BytesIO(base64.b64decode(encoded_audio))
    else:
        audio_source = audio_path
    waveform_samples, sample_rate = sf.read(audio_source, dtype="float32")
    if waveform_samples.ndim > 1:
        waveform_samples = waveform_samples.mean(axis=1)
    else:
        pass
    waveform = torch.from_numpy(waveform_samples)
    if cap_seconds is not None:
        waveform = waveform[..., : int(sample_rate * cap_seconds)]
    else:
        pass
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        pass
    waveform = waveform[:1].float()
    if sample_rate != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, AUDIO_SAMPLE_RATE
        )
    else:
        pass
    return math.ceil(waveform.shape[1] / SAMPLES_PER_PATCH)


def resolve_acoustic_instruction(
    instruction_text: str | None,
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
) -> str:
    """Use an explicit instruction or a trained acoustic-edit template."""
    if instruction_text is not None and instruction_text.strip():
        return instruction_text.strip()
    else:
        pass
    attribute = str(
        first_not_none(
            preprocessing_parameters.get("acoustic_attribute"),
            tts_parameters.get("acoustic_attribute"),
            default="Speed",
        )
    )
    attribute = attribute.strip().capitalize()
    if attribute not in ACOUSTIC_ATTRIBUTES:
        raise ValueError(
            f"acoustic_attribute must be one of {ACOUSTIC_ATTRIBUTES}, got {attribute!r}"
        )
    else:
        pass
    value = first_not_none(
        preprocessing_parameters.get("acoustic_value"),
        tts_parameters.get("acoustic_value"),
        tts_parameters.get("speed"),
        default=1.0,
    )
    acoustic_value = float(value)
    if attribute == "Speed":
        return f"adjust the speed to {acoustic_value:.1f}x"
    elif attribute == "Volume":
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
        prompt_patch_count = count_audio_patches(prompt_audio_path, cap_seconds=None)
        prompt_template = convert_to_chatml(
            text_in=f"Convert text to speech.\n{prompt_text}{text.strip()}",
            latent_out_len=prompt_patch_count,
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
            text_in=f"{instruction_text}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{text.strip()}",
            text_out="<|sot|>",
        ).removesuffix("<|im_end|>\n")
    elif task_type == "semantic_edit":
        if not instruction_text:
            raise ValueError("FireRedTTS3 semantic_edit requires instructions")
        else:
            pass
        audio_input_path = require_edit_source(reference_audio, task_type)
        audio_input_patch_count = count_audio_patches(
            audio_input_path, cap_seconds=MAX_EDIT_SECONDS
        )
        prompt_template = convert_to_chatml(
            text_in=f"Identify the content of the audio. {instruction_text.strip()}",
            latent_in_len=audio_input_patch_count,
            text_out="<|sot|>",
        ).removesuffix("<|im_end|>\n")
    elif task_type == "acoustic_edit":
        audio_input_path = require_edit_source(reference_audio, task_type)
        instruction = resolve_acoustic_instruction(
            instruction_text, tts_parameters, preprocessing_parameters
        )
        audio_input_patch_count = count_audio_patches(
            audio_input_path, cap_seconds=MAX_EDIT_SECONDS
        )
        prompt_template = convert_to_chatml(
            text_in=instruction,
            latent_in_len=audio_input_patch_count,
            latent_out_len=1,
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
        text_temperature=text_temperature,
        text_top_p=text_top_p,
        text_top_k=text_top_k,
    )
