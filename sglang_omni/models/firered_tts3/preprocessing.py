# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare FireRedTTS3 requests and encode both Speaker prompt forms."""

from __future__ import annotations

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
from sglang_omni.models.firered_tts3.speaker_prompts import (
    SPACE_BEFORE_HAN_PATTERN,
    SPEAKER_AUDIO_START,
    SPEAKER_TEXT_HOLDER,
    SPEAKER_TEXT_PAD,
    compose_speaker_chatml,
    encode_speaker_dual_stream_prompt,
    encode_speaker_prefill_ids,
    encode_speaker_text,
)
from sglang_omni.models.firered_tts3.speaker_prompts import is_han_character
from sglang_omni.models.firered_tts3.speaker_prompts import (
    is_han_character as is_cjk_char,
)
from sglang_omni.models.firered_tts3.speaker_prompts import resolve_speaker_token_id
from sglang_omni.models.firered_tts3.speech_validation import (
    FireRedTTS3GenerationOverrides,
)
from sglang_omni.models.firered_tts3.task_prompts import (
    compose_base_state,
    compose_instruct_state,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.json import JsonValue

__all__ = [
    "MAX_TEXT_CHARS",
    "SPACE_BEFORE_HAN_PATTERN",
    "SPEAKER_AUDIO_START",
    "SPEAKER_TEXT_HOLDER",
    "SPEAKER_TEXT_PAD",
    "TASK_GUIDANCE_SCALES",
    "compose_speaker_chatml",
    "compose_speaker_state",
    "encode_speaker_dual_stream_prompt",
    "encode_speaker_prefill_ids",
    "encode_speaker_text",
    "is_cjk_char",
    "is_han_character",
    "preprocess_firered_tts3_payload",
    "resolve_speaker_token_id",
]

MAX_TEXT_CHARS = 400


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
    guidance_default = generation_defaults.get("inference_cfg", 2.0)
    if variant == "instruct":
        guidance_default = TASK_GUIDANCE_SCALES.get(
            request_state.task_type, guidance_default
        )
    else:
        pass
    generation_parameters = FireRedTTS3GenerationOverrides.model_validate(
        {
            "n_timesteps": resolve_number(
                "n_timesteps", generation_defaults.get("n_timesteps", n_timesteps)
            ),
            "inference_cfg": resolve_number("inference_cfg", guidance_default),
            "stop_threshold": resolve_number(
                "stop_threshold", generation_defaults.get("stop_threshold", 0.5)
            ),
            "min_patches": resolve_number(
                "min_patches", generation_defaults.get("min_patches", 6)
            ),
            "max_patches": resolve_number("max_patches", max_patches),
            "seed": sampling_seed,
        }
    )
    assert generation_parameters.n_timesteps is not None
    assert generation_parameters.inference_cfg is not None
    assert generation_parameters.stop_threshold is not None
    assert generation_parameters.min_patches is not None
    assert generation_parameters.max_patches is not None
    request_state.n_timesteps = generation_parameters.n_timesteps
    request_state.inference_cfg = generation_parameters.inference_cfg
    request_state.stop_threshold = generation_parameters.stop_threshold
    request_state.min_patches = generation_parameters.min_patches
    request_state.max_patches = generation_parameters.max_patches
    request_state.seed = generation_parameters.seed
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
