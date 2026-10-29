# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare FireRedTTS3 requests and encode both Speaker prompt forms."""

from __future__ import annotations

import regex
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    detect_fireredtts3_variant,
)
from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    as_dict,
    engine_token_budget,
    first_not_none,
    normalize_instruct_task,
    resolve_request_reference,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.task_prompts import (
    compose_base_state,
    compose_instruct_state,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.json import JsonValue

MAX_TEXT_CHARS = 400
SPEAKER_TEXT_HOLDER = "<|text_holder|>"
SPEAKER_TEXT_PAD = "<|text_pad|>"
SPEAKER_AUDIO_START = "<|audio_start|>"
SPACE_BEFORE_HAN_PATTERN = regex.compile(r"\s+(?=\p{Han})")


TASK_GUIDANCE_SCALES: dict[str, float] = {
    "tts": 2.0,
    "voice_design": 1.2,
    "semantic_edit": 1.2,
    "acoustic_edit": 1.2,
}


def preprocess_firered_tts3_payload(
    payload: StagePayload,
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: FireRedTTS3Config,
    n_timesteps: int,
    max_patches: int,
) -> StagePayload:
    input_values = payload.request.inputs
    request_inputs = (
        {"text": input_values}
        if isinstance(input_values, str)
        else as_dict(input_values)
    )
    request_parameters = as_dict(payload.request.params)
    tts_parameters = as_dict(as_dict(payload.request.metadata).get("tts_params"))
    stage_parameters = as_dict(request_parameters.get("stage_params"))
    talker_parameters = as_dict(stage_parameters.get("talker"))
    preprocessing_parameters = as_dict(stage_parameters.get("preprocessing"))
    should_stream = bool(request_parameters.get("stream", False))
    text = str(
        first_not_none(
            request_inputs.get("input"), request_inputs.get("text"), default=""
        )
    )
    if len(text.strip()) > MAX_TEXT_CHARS:
        raise ValueError(
            f"FireRedTTS3 input text must be at most {MAX_TEXT_CHARS} characters"
        )
    else:
        pass
    generation_defaults = dict(config.generation_defaults or {})

    def resolve_number(parameter_name: str, fallback: int | float | None) -> JsonValue:
        return first_not_none(
            request_inputs.get(parameter_name),
            tts_parameters.get(parameter_name),
            talker_parameters.get(parameter_name),
            request_parameters.get(parameter_name),
            default=fallback,
        )

    sampling_seed = resolve_number("seed", None)
    variant = detect_fireredtts3_variant(config)
    interleave_ratio = (
        (
            int(config.text_latent_interleave_ratio[0]),
            int(config.text_latent_interleave_ratio[1]),
        )
        if config.text_latent_interleave_ratio is not None
        else None
    )
    if variant == "instruct":
        task_type = normalize_instruct_task(
            first_not_none(
                request_inputs.get("task_type"),
                tts_parameters.get("task_type"),
                request_parameters.get("task_type"),
            )
        )
        request_state = compose_instruct_state(
            config,
            tokenizer=tokenizer,
            task_type=task_type,
            request_inputs=request_inputs,
            request_parameters=request_parameters,
            tts_parameters=tts_parameters,
            preprocessing_parameters=preprocessing_parameters,
            text=text,
        )
    elif variant == "base":
        request_state = compose_base_state(
            config,
            tokenizer=tokenizer,
            request_inputs=request_inputs,
            tts_parameters=tts_parameters,
            text=text,
        )
    elif variant == "speaker":
        request_state = compose_speaker_state(
            config,
            tokenizer=tokenizer,
            request_inputs=request_inputs,
            request_parameters=request_parameters,
            tts_parameters=tts_parameters,
            preprocessing_parameters=preprocessing_parameters,
            text=text,
        )
    else:
        raise AssertionError(f"unhandled FireRedTTS3 variant: {variant}")
    request_state.n_timesteps = int(
        resolve_number(
            "n_timesteps", generation_defaults.get("n_timesteps", n_timesteps)
        )
    )
    guidance_default = generation_defaults.get("inference_cfg", 2.0)
    if variant == "instruct":
        guidance_default = TASK_GUIDANCE_SCALES.get(
            request_state.task_type, guidance_default
        )
    else:
        pass
    request_state.inference_cfg = float(
        resolve_number("inference_cfg", guidance_default)
    )
    request_state.stop_threshold = float(
        resolve_number("stop_threshold", generation_defaults.get("stop_threshold", 0.5))
    )
    request_state.min_patches = int(
        resolve_number("min_patches", generation_defaults.get("min_patches", 6))
    )
    request_state.max_patches = int(resolve_number("max_patches", max_patches))
    request_state.seed = None if sampling_seed is None else int(sampling_seed)
    request_state.stream = should_stream
    request_state.max_engine_tokens = int(
        request_parameters.get("max_new_tokens")
        or engine_token_budget(request_state, interleave_ratio=interleave_ratio)
    )
    return store_firered_tts3_state(payload, request_state)


def compose_speaker_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    request_inputs: dict[str, JsonValue],
    request_parameters: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    """Build the prefill or interleaved speaker prompt."""
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty input text")
    else:
        pass
    reference_audio, prompt_text = resolve_request_reference(
        request_inputs, tts_parameters
    )
    if reference_audio is not None or (
        prompt_text is not None and str(prompt_text).strip()
    ):
        raise ValueError(
            "FireRedTTS3 speaker variant takes text-only tts; reference audio is not used (deploy a Base/Instruct checkpoint for voice cloning)"
        )
    else:
        pass
    task_type = first_not_none(
        request_inputs.get("task_type"),
        tts_parameters.get("task_type"),
        request_parameters.get("task_type"),
    )
    if task_type is not None and str(task_type).strip().lower() not in ("tts", "base"):
        raise ValueError(
            f"FireRedTTS3 speaker variant supports only text-only tts; got task_type {task_type!r}"
        )
    else:
        pass
    dual_stream = bool(
        first_not_none(
            request_parameters.get("dual_stream"),
            tts_parameters.get("dual_stream"),
            preprocessing_parameters.get("dual_stream"),
            default=False,
        )
    )
    interleave_ratio = config.text_latent_interleave_ratio
    if dual_stream and interleave_ratio is None:
        raise ValueError(
            "FireRedTTS3 dual_stream requires a checkpoint with text_latent_interleave_ratio in its config"
        )
    else:
        pass
    speaker_token = str(config.speaker_token or "<|rand_speaker|>")
    if dual_stream:
        style = first_not_none(
            request_parameters.get("style"),
            tts_parameters.get("style"),
            preprocessing_parameters.get("style"),
        )
        prompt_token_ids, text_token_ids, text_padding_token_id = (
            encode_speaker_dual_stream_prompt(
                tokenizer,
                text=text.strip(),
                speaker_token=speaker_token,
                style=None if style is None else str(style),
                interleave_ratio=(int(interleave_ratio[0]), int(interleave_ratio[1])),
            )
        )
        return FireRedTTS3State(
            sample_rate=int(config.audio_sample_rate),
            variant="speaker",
            task_type="tts",
            text=text.strip(),
            input_ids=prompt_token_ids,
            dual_stream=True,
            text_stream_ids=text_token_ids,
            text_pad_id=text_padding_token_id,
        )
    else:
        pass
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="speaker",
        task_type="tts",
        text=text.strip(),
        input_ids=encode_speaker_prefill_ids(
            tokenizer, text=text.strip(), speaker_token=speaker_token
        ),
    )


def compose_speaker_chatml(
    speaker_token: str,
    *,
    style: str = "",
    text_len: int = 0,
    dual_stream: bool = False,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> str:
    """Build the speaker prefix through its audio-start token."""
    if dual_stream:
        user_prompt = f"{style} Align the interleaved text and speech chunks.".strip()
        text_steps, audio_steps = interleave_ratio
        if text_steps < 1 or audio_steps < 1:
            raise ValueError(
                f"FireRedTTS3 interleave_ratio {interleave_ratio!r} invalid; "
                "expected [text_steps, audio_steps] with both >= 1"
            )
        else:
            pass
    else:
        user_prompt = f"{style} Convert text to speech.".strip()
        user_prompt = f"{user_prompt}\n{SPEAKER_TEXT_HOLDER * text_len}"
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{speaker_token}{SPEAKER_AUDIO_START}"
    )


def is_cjk_char(character: str) -> bool:
    return (
        "一" <= character <= "鿿"
        or "㐀" <= character <= "䶿"
        or "豈" <= character <= "﫿"
        or "\U00020000" <= character <= "\U0002a6df"
    )


def encode_speaker_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Expand multi-character Han tokens while preserving unencodable words."""
    text = SPACE_BEFORE_HAN_PATTERN.sub("", text)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    unknown_token_id = tokenizer.unk_token_id
    out: list[int] = []
    for token_id in tokens:
        decoded = tokenizer.decode([int(token_id)]).strip()
        if len(decoded) < 2 or not all(is_cjk_char(c) for c in decoded):
            out.append(int(token_id))
            continue
        else:
            pass
        char_ids: list[int] = []
        for character in decoded:
            ids = tokenizer(character, add_special_tokens=False)["input_ids"]
            if unknown_token_id is not None and unknown_token_id in ids:
                char_ids = []
                break
            else:
                pass
            char_ids.extend(int(i) for i in ids)
        if char_ids:
            out.extend(char_ids)
        else:
            out.append(int(token_id))
    return out


def resolve_speaker_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """Reject a tokenizer missing a required speaker token."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(
            f"FireRedTTS3 speaker requests require a tokenizer with the "
            f"{token} token — this model's tokenizer has none (the speaker "
            "variant is a separate checkpoint)"
        )
    else:
        pass
    return int(token_id)


def encode_speaker_prefill_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
) -> list[int]:
    """Replace speaker text holders with the character-level token stream."""
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
    chatml = compose_speaker_chatml(
        speaker_token,
        style="" if style is None else style,
        text_len=len(text_ids),
    )
    ids = encode_speaker_text(tokenizer, chatml)
    holder_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_HOLDER)
    prompt_ids: list[int] = []
    fed = 0
    for tid in ids:
        if int(tid) == holder_id:
            prompt_ids.append(text_ids[fed])
            fed += 1
        else:
            prompt_ids.append(int(tid))
    if fed != len(text_ids):
        raise ValueError(
            f"FireRedTTS3 speaker template carries {fed} text slots "
            f"for {len(text_ids)} text tokens — composition bug"
        )
    else:
        pass
    return prompt_ids


def encode_speaker_dual_stream_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> tuple[list[int], list[int], int]:
    """Return the speaker prefix, interleaved text tokens, and text-padding id."""
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty text")
    else:
        pass
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
    chatml = compose_speaker_chatml(
        speaker_token,
        style="Use a 中性 tone." if style is None else style,
        dual_stream=True,
        interleave_ratio=interleave_ratio,
    )
    prompt_ids = encode_speaker_text(tokenizer, chatml)
    text_pad_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_PAD)
    return prompt_ids, text_ids, text_pad_id
