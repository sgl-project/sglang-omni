# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for Qwen3-TTS."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.pending_text_queue import PendingTextTensorQueue
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS = 2048
QWEN3_TTS_TASK_BASE = "Base"
QWEN3_TTS_TASK_CUSTOM_VOICE = "CustomVoice"
QWEN3_TTS_TASK_VOICE_DESIGN = "VoiceDesign"
QWEN3_TTS_DEFAULT_CUSTOM_VOICE = "Vivian"
_QWEN3_TTS_PREPARED_MARKER = "_qwen3_tts_prepared_request"

_GENERATION_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
    "max_new_tokens",
)

_IMPLICIT_SAMPLING_DEFAULTS = {
    "temperature": {1.0, 0.8},
    "top_p": {1.0, 0.8},
    "top_k": {-1, 30},
    "repetition_penalty": {1.0, 1.1},
}

_QWEN3_TTS_SAMPLING_SEED_MASK = 0x7FFFFFFF


def _new_qwen3_tts_sampling_seed() -> int:
    return int.from_bytes(os.urandom(4), "little") & _QWEN3_TTS_SAMPLING_SEED_MASK


def _normalize_qwen3_tts_seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise ValueError("Qwen3-TTS seed must be an integer")
    if isinstance(seed, float) and not seed.is_integer():
        raise ValueError("Qwen3-TTS seed must be an integer")
    try:
        normalized = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Qwen3-TTS seed must be an integer") from exc
    return normalized & _QWEN3_TTS_SAMPLING_SEED_MASK


def _derive_qwen3_tts_child_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(
        f"qwen3-tts:{seed}:{label}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & _QWEN3_TTS_SAMPLING_SEED_MASK


def derive_qwen3_tts_sampling_seeds(seed: int) -> tuple[int, int]:
    """Split a public request seed into semantic and subtalker sampling seeds."""

    normalized = _normalize_qwen3_tts_seed(seed)
    return (
        _derive_qwen3_tts_child_seed(normalized, "semantic"),
        _derive_qwen3_tts_child_seed(normalized, "subtalker"),
    )


@dataclass
class Qwen3TTSSGLangRequestData(SGLangARRequestData):
    """Qwen3-TTS scheduler-owned request state."""

    enforce_request_limits: bool = True
    output_codes: list[torch.Tensor] = field(default_factory=list)
    ref_code: torch.Tensor | None = None
    ref_code_len: int = 0
    prompt_input_embeds: torch.Tensor | None = None
    semantic_sampling_seed: int = field(default_factory=_new_qwen3_tts_sampling_seed)
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50
    subtalker_sampling_seed: int = field(default_factory=_new_qwen3_tts_sampling_seed)
    engine_start_s: float = 0.0


@dataclass
class Qwen3TTSPreparedRequest:
    """Heavy Qwen3-TTS preprocessing output consumed by the AR scheduler."""

    state: Qwen3TTSState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trailing_text_hidden: torch.Tensor
    ref_code: torch.Tensor | None
    prompt_input_embeds: torch.Tensor
    tts_pad_embed: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class Qwen3TTSPreprocessingContext:
    model: Any
    wrapper: Any


_PREPROCESSING_CONTEXT: Qwen3TTSPreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, Qwen3TTSPreparedRequest] = {}
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_qwen3_tts_preprocessing_context(*, model: Any, wrapper: Any) -> None:
    """Register model objects used by the preprocessing stage."""

    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = Qwen3TTSPreprocessingContext(
            model=model,
            wrapper=wrapper,
        )
        _PREPARED_REQUESTS.clear()


def clear_qwen3_tts_preprocessing_context() -> None:
    """Clear Qwen3-TTS preprocessing globals, mainly for tests and reloads."""

    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = None
        _PREPARED_REQUESTS.clear()


def _prepared_request_id(payload: StagePayload) -> str | None:
    data = payload.data
    if not isinstance(data, dict):
        return None
    marker = data.get(_QWEN3_TTS_PREPARED_MARKER)
    return str(marker) if marker is not None else None


def pop_prepared_qwen3_tts_request(
    payload: StagePayload,
) -> Qwen3TTSPreparedRequest | None:
    """Consume the prepared request referenced by a preprocessed payload."""

    prepared_request_id = _prepared_request_id(payload)
    if prepared_request_id is None:
        return None
    with _PREPARED_REQUESTS_LOCK:
        prepared = _PREPARED_REQUESTS.pop(prepared_request_id, None)
    if prepared is None:
        raise RuntimeError(
            "Qwen3-TTS preprocessing state is missing for prepared payload "
            f"{prepared_request_id!r}; the AR scheduler must not rebuild it"
        )
    return prepared


def cleanup_prepared_qwen3_tts_request(request_id: str) -> None:
    """Drop any prepared Qwen3-TTS handoff state for an aborted request."""

    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS.pop(str(request_id), None)


def build_qwen3_tts_state(payload: StagePayload) -> Qwen3TTSState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_qwen3_tts_inputs(inputs)
    has_reference = has_voice_clone_reference(references, tts_params)
    task_type_raw = tts_params.get("task_type") or params.get("task_type")
    task_type_explicit = task_type_raw is not None and str(task_type_raw).strip() != ""
    task_type = normalize_qwen3_tts_task_type(
        task_type_raw, has_reference=has_reference
    )
    language = normalize_language(tts_params.get("language") or params.get("language"))
    instructions = resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    ref_audio = None
    ref_text = None
    x_vector_only_mode = False
    non_streaming_mode = resolve_non_streaming_mode(
        task_type=task_type,
        params=params,
        tts_params=tts_params,
    )
    voice = normalize_qwen3_tts_voice(
        tts_params.get("voice") or tts_params.get("speaker") or params.get("voice")
    )

    if task_type == QWEN3_TTS_TASK_BASE:
        ref_audio, ref_text = resolve_voice_clone_reference(references, tts_params)
        x_vector_only_mode = resolve_x_vector_only_mode(
            params=params,
            tts_params=tts_params,
            ref_text=ref_text,
        )
        if not x_vector_only_mode and not ref_text:
            raise ValueError(
                "Qwen3-TTS Base requires non-empty ref_text unless "
                "x_vector_only_mode is enabled"
            )
    elif task_type == QWEN3_TTS_TASK_CUSTOM_VOICE:
        if has_param(tts_params, params, "ref_audio") or references_contain_audio(
            references
        ):
            raise ValueError("Qwen3-TTS CustomVoice does not accept ref_audio")
        if has_param(tts_params, params, "ref_text") or references_contain_text(
            references
        ):
            raise ValueError("Qwen3-TTS CustomVoice does not accept ref_text")
        if has_param(tts_params, params, "x_vector_only_mode"):
            raise ValueError("Qwen3-TTS CustomVoice does not accept x_vector_only_mode")
        voice = voice or QWEN3_TTS_DEFAULT_CUSTOM_VOICE
        non_streaming_mode = True
    elif task_type == QWEN3_TTS_TASK_VOICE_DESIGN:
        if has_param(tts_params, params, "ref_audio") or references_contain_audio(
            references
        ):
            raise ValueError("Qwen3-TTS VoiceDesign does not accept ref_audio")
        if has_param(tts_params, params, "ref_text") or references_contain_text(
            references
        ):
            raise ValueError("Qwen3-TTS VoiceDesign does not accept ref_text")
        if has_param(tts_params, params, "x_vector_only_mode"):
            raise ValueError("Qwen3-TTS VoiceDesign does not accept x_vector_only_mode")
        if not instructions:
            raise ValueError("Qwen3-TTS VoiceDesign requires instructions")
        voice = None
        non_streaming_mode = True
    else:
        raise AssertionError(f"unhandled Qwen3-TTS task type: {task_type}")

    seed = tts_params["seed"] if "seed" in tts_params else params.get("seed")
    normalized_seed = _normalize_qwen3_tts_seed(seed) if seed is not None else None

    return Qwen3TTSState(
        text=text,
        task_type=task_type,
        task_type_explicit=task_type_explicit,
        language=language,
        voice=voice,
        instructions=instructions,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        non_streaming_mode=non_streaming_mode,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
        seed=normalized_seed,
    )


def normalize_qwen3_tts_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs, []
    if isinstance(inputs, dict):
        text = inputs.get("text", inputs.get("input", ""))
        references = inputs.get("references") or []
        if not isinstance(references, list):
            raise ValueError("Qwen3-TTS references must be a list")
        normalized_references = [
            dict(reference) for reference in references if isinstance(reference, dict)
        ]
        return str(text), normalized_references
    return str(inputs) if inputs is not None else "", []


def resolve_voice_clone_reference(
    references: list[dict[str, Any]],
    tts_params: dict[str, Any],
) -> tuple[Any, str | None]:
    reference = references[0] if references else {}
    ref_audio = (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or tts_params.get("ref_audio")
    )
    ref_text = reference.get("text") or tts_params.get("ref_text")
    if ref_audio is None:
        raise ValueError(
            "Qwen3-TTS Base requires reference audio via ref_audio or references[0].audio_path"
        )
    return ref_audio, str(ref_text) if ref_text is not None else None


def has_voice_clone_reference(
    references: list[dict[str, Any]],
    tts_params: dict[str, Any],
) -> bool:
    if references_contain_audio(references) or references_contain_text(references):
        return True
    return (
        tts_params.get("ref_audio") is not None
        or tts_params.get("ref_text") is not None
    )


def references_contain_audio(references: list[dict[str, Any]]) -> bool:
    return any(
        reference.get(key) is not None
        for reference in references
        for key in ("audio_path", "ref_audio", "audio")
    )


def references_contain_text(references: list[dict[str, Any]]) -> bool:
    return any(reference.get("text") is not None for reference in references)


def normalize_qwen3_tts_task_type(
    task_type: Any,
    *,
    has_reference: bool,
) -> str:
    if task_type is None or str(task_type).strip() == "":
        return QWEN3_TTS_TASK_BASE if has_reference else QWEN3_TTS_TASK_CUSTOM_VOICE
    normalized = str(task_type).replace("_", "").replace("-", "").lower()
    if normalized == "base":
        return QWEN3_TTS_TASK_BASE
    if normalized == "customvoice":
        return QWEN3_TTS_TASK_CUSTOM_VOICE
    if normalized == "voicedesign":
        return QWEN3_TTS_TASK_VOICE_DESIGN
    raise ValueError(
        "Qwen3-TTS task_type must be one of Base, CustomVoice, or VoiceDesign"
    )


def resolve_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_qwen3_tts_voice(value: Any) -> str | None:
    voice = resolve_optional_text(value)
    if voice is None or voice.lower() == "default":
        return None
    return voice


def has_param(
    tts_params: dict[str, Any],
    params: dict[str, Any],
    name: str,
) -> bool:
    return name in tts_params or name in params


def resolve_non_streaming_mode(
    *,
    task_type: str,
    params: dict[str, Any],
    tts_params: dict[str, Any],
) -> bool:
    for source in (params, tts_params):
        if "non_streaming_mode" in source:
            return bool(source["non_streaming_mode"])
    return task_type in (QWEN3_TTS_TASK_CUSTOM_VOICE, QWEN3_TTS_TASK_VOICE_DESIGN)


def normalize_language(language: Any) -> str:
    if language is None or language == "":
        return "auto"
    return str(language)


def resolve_x_vector_only_mode(
    *,
    params: dict[str, Any],
    tts_params: dict[str, Any],
    ref_text: str | None,
) -> bool:
    for source in (params, tts_params):
        if "x_vector_only_mode" in source:
            return bool(source["x_vector_only_mode"])
    return not bool(ref_text)


def build_generation_kwargs(
    params: dict[str, Any],
    *,
    tts_params: dict[str, Any],
) -> dict[str, Any]:
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()

    selected_fields = set()
    for field in _GENERATION_FIELDS:
        value = params.get(field)
        if value is None:
            continue
        if field in _IMPLICIT_SAMPLING_DEFAULTS and field not in explicit_fields:
            if value in _IMPLICIT_SAMPLING_DEFAULTS[field]:
                continue
        selected_fields.add(field)

    max_new_tokens = params.get("max_new_tokens")
    if max_new_tokens is None:
        max_new_tokens = QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS
    generation_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    for field in _GENERATION_FIELDS:
        if field == "max_new_tokens":
            continue
        if field in selected_fields and params.get(field) is not None:
            generation_kwargs[field] = params[field]
    return generation_kwargs


def build_embedding_cache_key_ids(input_embeds: torch.Tensor) -> list[int]:
    """Build stable radix-cache token ids for a precomputed embedding prefix."""
    rows = input_embeds.detach().to(dtype=torch.float32, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def _build_qwen3_tts_pad_embed(model: Any) -> torch.Tensor:
    feedback_buffer = model.model._feedback_buffer
    with torch.no_grad():
        return (
            model.text_projection(
                model.get_text_embeddings()(
                    torch.tensor(
                        [[model.root_config.tts_pad_token_id]],
                        device=model.device,
                        dtype=torch.long,
                    )
                )
            )
            .squeeze(0)
            .squeeze(0)
            .detach()
            .to(device=feedback_buffer.device, dtype=feedback_buffer.dtype)
        )


def _build_instruct_id(wrapper: Any, instructions: str | None) -> torch.Tensor | None:
    if not instructions:
        return None
    if hasattr(wrapper, "_build_instruct_text"):
        instruct_text = wrapper._build_instruct_text(instructions)
    else:
        instruct_text = f"<|im_start|>user\n{instructions}<|im_end|>\n"
    return wrapper._tokenize_texts([instruct_text])[0]


def _normalized_model_type(model: Any) -> str:
    model_type = getattr(model, "tts_model_type", None)
    if model_type is None:
        model_type = getattr(
            getattr(model, "root_config", None), "tts_model_type", None
        )
    normalized = str(model_type or "base").replace("-", "_").strip().lower()
    if normalized == "customvoice":
        return "custom_voice"
    if normalized == "voicedesign":
        return "voice_design"
    return normalized


def _validate_qwen3_tts_model_task(model: Any, state: Qwen3TTSState) -> None:
    model_type = _normalized_model_type(model)
    if model_type == "base" and state.task_type != QWEN3_TTS_TASK_BASE:
        if not state.task_type_explicit:
            raise ValueError(
                "Qwen3-TTS Base requires ref_audio or speaker_embedding; "
                "text-only requests require a CustomVoice or VoiceDesign checkpoint"
            )
        raise ValueError(
            f"Qwen3-TTS Base checkpoint does not support {state.task_type}"
        )
    if model_type == "custom_voice" and state.task_type != QWEN3_TTS_TASK_CUSTOM_VOICE:
        raise ValueError(
            f"Qwen3-TTS CustomVoice checkpoint does not support {state.task_type}"
        )
    if model_type == "voice_design" and state.task_type != QWEN3_TTS_TASK_VOICE_DESIGN:
        raise ValueError(
            f"Qwen3-TTS VoiceDesign checkpoint does not support {state.task_type}"
        )


def _prepare_qwen3_tts_base_request(
    *,
    state: Qwen3TTSState,
    model: Any,
    wrapper: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    with torch.no_grad():
        prompt_items = wrapper.create_voice_clone_prompt(
            ref_audio=state.ref_audio,
            ref_text=state.ref_text,
            x_vector_only_mode=state.x_vector_only_mode,
        )
    if len(prompt_items) != 1:
        raise ValueError("Qwen3-TTS expects exactly one voice-clone prompt")
    voice_clone_prompt = wrapper._prompt_items_to_voice_clone_prompt(prompt_items)

    input_id = wrapper._tokenize_texts([wrapper._build_assistant_text(state.text)])[0]
    ref_text = prompt_items[0].ref_text
    ref_id = (
        wrapper._tokenize_texts([wrapper._build_ref_text(ref_text)])[0]
        if ref_text
        else None
    )
    instruct_id = _build_instruct_id(wrapper, state.instructions)
    with torch.no_grad():
        return model.build_voice_clone_inputs(
            input_id=input_id,
            ref_id=ref_id,
            voice_clone_prompt=voice_clone_prompt,
            language=state.language,
            non_streaming_mode=state.non_streaming_mode,
            instruct_id=instruct_id,
        )


def _prepare_qwen3_tts_custom_voice_request(
    *,
    state: Qwen3TTSState,
    model: Any,
    wrapper: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    input_id = wrapper._tokenize_texts([wrapper._build_assistant_text(state.text)])[0]
    instruct_id = _build_instruct_id(wrapper, state.instructions)
    with torch.no_grad():
        return model.build_custom_voice_inputs(
            input_id=input_id,
            voice=state.voice or QWEN3_TTS_DEFAULT_CUSTOM_VOICE,
            language=state.language,
            non_streaming_mode=state.non_streaming_mode,
            instruct_id=instruct_id,
        )


def _prepare_qwen3_tts_voice_design_request(
    *,
    state: Qwen3TTSState,
    model: Any,
    wrapper: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    input_id = wrapper._tokenize_texts([wrapper._build_assistant_text(state.text)])[0]
    instruct_id = _build_instruct_id(wrapper, state.instructions)
    with torch.no_grad():
        return model.build_voice_design_inputs(
            input_id=input_id,
            language=state.language,
            non_streaming_mode=state.non_streaming_mode,
            instruct_id=instruct_id,
        )


def _prepare_qwen3_tts_request(
    payload: StagePayload,
    *,
    model: Any,
    wrapper: Any,
) -> Qwen3TTSPreparedRequest:
    state = build_qwen3_tts_state(payload)

    _validate_qwen3_tts_model_task(model, state)
    gen_kwargs = wrapper._merge_generate_kwargs(**state.generation_kwargs)
    if state.task_type == QWEN3_TTS_TASK_BASE:
        (
            input_embeds,
            attention_mask,
            trailing_text_hidden,
            ref_code,
        ) = _prepare_qwen3_tts_base_request(
            state=state,
            model=model,
            wrapper=wrapper,
        )
    elif state.task_type == QWEN3_TTS_TASK_CUSTOM_VOICE:
        (
            input_embeds,
            attention_mask,
            trailing_text_hidden,
            ref_code,
        ) = _prepare_qwen3_tts_custom_voice_request(
            state=state,
            model=model,
            wrapper=wrapper,
        )
    elif state.task_type == QWEN3_TTS_TASK_VOICE_DESIGN:
        (
            input_embeds,
            attention_mask,
            trailing_text_hidden,
            ref_code,
        ) = _prepare_qwen3_tts_voice_design_request(
            state=state,
            model=model,
            wrapper=wrapper,
        )
    else:
        raise AssertionError(f"unhandled Qwen3-TTS task type: {state.task_type}")

    feedback_buffer = model.model._feedback_buffer
    prompt_input_embeds = (
        input_embeds.squeeze(0)
        .detach()
        .to(
            device=feedback_buffer.device,
            dtype=feedback_buffer.dtype,
        )
    )
    input_ids_list = build_embedding_cache_key_ids(prompt_input_embeds)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    trailing_text_hidden = (
        trailing_text_hidden.squeeze(0)
        .detach()
        .to(
            device=feedback_buffer.device,
            dtype=feedback_buffer.dtype,
        )
    )
    if ref_code is not None:
        ref_code = ref_code.detach().to(device=feedback_buffer.device)

    return Qwen3TTSPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=input_ids,
        attention_mask=attention_mask.detach(),
        trailing_text_hidden=trailing_text_hidden,
        ref_code=ref_code,
        prompt_input_embeds=prompt_input_embeds,
        tts_pad_embed=_build_qwen3_tts_pad_embed(model),
        gen_kwargs=gen_kwargs,
    )


def preprocess_qwen3_tts_payload(payload: StagePayload) -> StagePayload:
    """Run Qwen3-TTS prompt/audio preprocessing outside the AR scheduler."""

    with _PREPARED_REQUESTS_LOCK:
        context = _PREPROCESSING_CONTEXT
    if context is None:
        raise RuntimeError(
            "Qwen3-TTS preprocessing context is not initialized; "
            "create_sglang_tts_engine_executor must register it before requests run"
        )

    prepared = _prepare_qwen3_tts_request(
        payload,
        model=context.model,
        wrapper=context.wrapper,
    )
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS[payload.request_id] = prepared

    data = prepared.state.to_dict()
    data[_QWEN3_TTS_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def build_sglang_qwen3_tts_request(
    payload: StagePayload,
    *,
    model: Any,
    wrapper: Any,
) -> Qwen3TTSSGLangRequestData:
    del wrapper

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_qwen3_tts_request(payload)
    if prepared is None:
        raise RuntimeError(
            "Qwen3-TTS AR request builder requires a payload prepared by "
            "preprocess_qwen3_tts_payload"
        )

    gen_kwargs = prepared.gen_kwargs
    state = prepared.state
    do_sample = bool(gen_kwargs.get("do_sample", True))
    temperature = float(gen_kwargs.get("temperature", 0.9)) if do_sample else 0.0
    if state.seed is None:
        semantic_sampling_seed = _new_qwen3_tts_sampling_seed()
        subtalker_sampling_seed = _new_qwen3_tts_sampling_seed()
    else:
        semantic_sampling_seed, subtalker_sampling_seed = (
            derive_qwen3_tts_sampling_seeds(state.seed)
        )
    sampling_params = SamplingParams(
        max_new_tokens=int(
            gen_kwargs.get("max_new_tokens", QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS)
        ),
        temperature=temperature,
        top_p=float(gen_kwargs.get("top_p", 1.0)),
        top_k=int(gen_kwargs.get("top_k", 50)),
        repetition_penalty=float(gen_kwargs.get("repetition_penalty", 1.05)),
        stop_token_ids=[int(model.config.codec_eos_token_id)],
        sampling_seed=semantic_sampling_seed,
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(model.config.vocab_size))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={int(model.config.codec_eos_token_id)},
        vocab_size=int(model.config.vocab_size),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = tuple(
        token_id
        for token_id in range(model.config.vocab_size - 1024, model.config.vocab_size)
        if token_id != int(model.config.codec_eos_token_id)
    )

    ref_code_len = (
        int(prepared.ref_code.shape[0]) if prepared.ref_code is not None else 0
    )
    data = Qwen3TTSSGLangRequestData(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        max_new_tokens=int(
            gen_kwargs.get("max_new_tokens", QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS)
        ),
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
        ref_code=prepared.ref_code,
        ref_code_len=ref_code_len,
        prompt_input_embeds=prepared.prompt_input_embeds,
        semantic_sampling_seed=semantic_sampling_seed,
        subtalker_dosample=bool(gen_kwargs.get("subtalker_dosample", True)),
        subtalker_temperature=float(gen_kwargs.get("subtalker_temperature", 0.9)),
        subtalker_top_p=float(gen_kwargs.get("subtalker_top_p", 1.0)),
        subtalker_top_k=int(gen_kwargs.get("subtalker_top_k", 50)),
        subtalker_sampling_seed=subtalker_sampling_seed,
        engine_start_s=time.perf_counter(),
    )
    data.suppress_tokens = list(req._codec_suppress_tokens)
    data.pending_text_queue = PendingTextTensorQueue.from_tensor(
        prepared.trailing_text_hidden
    )
    data.tts_pad_embed = prepared.tts_pad_embed
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_qwen3_tts_result(
    payload: StagePayload,
    data: Qwen3TTSSGLangRequestData,
) -> StagePayload:
    code_parts: list[torch.Tensor] = []
    if data.ref_code is not None and data.ref_code_len:
        code_parts.append(data.ref_code.to(dtype=torch.long))
    if data.output_codes:
        code_parts.append(torch.stack(data.output_codes, dim=0).to(dtype=torch.long))

    if code_parts:
        device = code_parts[0].device
        codes = torch.cat(
            [part.to(device=device, dtype=torch.long) for part in code_parts],
            dim=0,
        ).cpu()
    else:
        codes = torch.empty((0, 0), dtype=torch.long)

    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={
            "audio_codes": codes,
            "ref_code_len": data.ref_code_len,
            "prompt_tokens": data.ref_code_len,
            "completion_tokens": len(data.output_codes),
            "engine_time_s": time.perf_counter() - data.engine_start_s,
            "sample_rate": 24000,
        },
    )


def make_qwen3_tts_scheduler_adapters(*, model: Any, wrapper: Any):
    """Build StagePayload <-> SGLang request adapters for Qwen3-TTS."""

    def request_builder(payload: StagePayload) -> Qwen3TTSSGLangRequestData:
        return build_sglang_qwen3_tts_request(
            payload,
            model=model,
            wrapper=wrapper,
        )

    def result_adapter(data: Qwen3TTSSGLangRequestData) -> StagePayload:
        return apply_sglang_qwen3_tts_result(data.stage_payload, data)

    return request_builder, result_adapter
