# SPDX-License-Identifier: Apache-2.0
"""Processor lowering and prepared handoff for MOSS-TTS-Realtime."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnPhase,
)
from sglang_omni.models.moss_tts_realtime.text_delta import (
    get_moss_tts_realtime_tokenizer_vocab_size,
    initialize_moss_tts_realtime_tokenizer_vocab_size,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.prepared_request_queue import (
    PreparedRequestQueue,
    QueueSnapshot,
)
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

MOSS_TTS_REALTIME_DEFAULT_MAX_NEW_TOKENS = 1000
MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY = (
    "_moss_tts_realtime_prepared_initial_token_ids"
)
_MOSS_TTS_REALTIME_PREPARED_MARKER = "_moss_tts_realtime_prepared_request"
_ASSISTANT_TURN_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
_MAX_SUPPORTED_CODEC_QUANTIZERS = 64
_STANDARD_SPEECH_GENERATION_FIELDS = frozenset(
    {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "seed",
    }
)


def _processor_model_config(processor: Any) -> Any:
    config = getattr(processor, "model_config", None)
    if config is None:
        raise ValueError("MOSS-TTS-Realtime processor must expose model_config")
    return config


@dataclass(frozen=True)
class MossTTSRealtimePreprocessingContext:
    processor: Any
    audio_encoder: Any | None = None
    reference_encoder: Any | None = None


@dataclass
class MossTTSRealtimePreparedRequest:
    """CPU preprocessing result consumed later on the scheduler thread."""

    state: MossTTSRealtimeState
    turn_prompt_rows: torch.Tensor
    turn_prompt_cache_ids: list[int]
    turn_prompt_input_ids: torch.Tensor
    initial_token_ids: tuple[int, ...]
    voice_codes: torch.Tensor | None
    user_audio_codes: torch.Tensor | None
    include_system_prompt: bool
    generation_kwargs: dict[str, Any]


_QUEUE: PreparedRequestQueue[
    MossTTSRealtimePreprocessingContext,
    MossTTSRealtimePreparedRequest,
] = PreparedRequestQueue()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _source_value(
    source: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = source.get(name)
    return default if value is None else value


def _normalize_token_ids(value: Any, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 sequence of integers")
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 sequence of integers")
        value = value.tolist()

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    return [
        _strict_int(token_id, f"{name}[{index}]", minimum=0)
        for index, token_id in enumerate(value)
    ]


def _validate_tokenizer_token_ids(
    token_ids: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    normalized = tuple(_normalize_token_ids(token_ids, name))
    vocab_size = get_moss_tts_realtime_tokenizer_vocab_size()
    for index, token_id in enumerate(normalized):
        if token_id >= vocab_size:
            raise ValueError(
                f"{name}[{index}]={token_id} exceeds tokenizer size {vocab_size}"
            )
    return normalized


def _validate_processor_contract(processor: Any) -> None:
    config = _processor_model_config(processor)
    expected = {
        "channels": int(config.rvq),
        "delay_tokens_len": int(config.delay_tokens_len),
        "audio_channel_pad": int(config.audio_pad_token),
        "audio_bos_token": int(config.audio_bos_token),
        "audio_eos_token": int(config.audio_eos_token),
        "audio_pad_token_id": int(config.reference_audio_pad),
        "text_pad_token_id": int(config.text_pad),
    }
    for name, expected_value in expected.items():
        actual = getattr(processor, name, None)
        try:
            actual_value = _strict_int(actual, f"processor {name}")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MOSS-TTS-Realtime processor {name} must match model config "
                f"{expected_value}, "
                f"got {actual!r}"
            ) from exc
        if actual_value != expected_value:
            raise ValueError(
                f"MOSS-TTS-Realtime processor {name} must match model config "
                f"{expected_value}, "
                f"got {actual!r}"
            )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("MOSS-TTS-Realtime processor must expose tokenizer")
    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("MOSS-TTS-Realtime tokenizer must implement encode()")
    for method_name in ("make_ensemble", "make_user_prompt"):
        if not callable(getattr(processor, method_name, None)):
            raise TypeError(
                f"MOSS-TTS-Realtime processor must implement {method_name}()"
            )


def set_moss_tts_realtime_preprocessing_context(
    *,
    processor: Any,
    audio_encoder: Any | None = None,
    reference_encoder: Any | None = None,
) -> None:
    """Install the process-local processor/encoder used by prepared handoff."""

    _validate_processor_contract(processor)
    initialize_moss_tts_realtime_tokenizer_vocab_size(processor.tokenizer)
    _QUEUE.set_context(
        MossTTSRealtimePreprocessingContext(
            processor=processor,
            audio_encoder=audio_encoder,
            reference_encoder=reference_encoder,
        )
    )


def clear_moss_tts_realtime_preprocessing_context() -> None:
    _QUEUE.clear_context()


def moss_tts_realtime_prepared_snapshot() -> QueueSnapshot:
    """Read-only queue state for observability and lifecycle tests."""

    return _QUEUE.snapshot()


def cleanup_prepared_moss_tts_realtime_request(request_id: str) -> None:
    _QUEUE.abort(str(request_id))


def pop_prepared_moss_tts_realtime_request(
    payload: StagePayload,
) -> MossTTSRealtimePreparedRequest | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_MOSS_TTS_REALTIME_PREPARED_MARKER)
    if marker is None:
        return None
    if str(marker) != str(payload.request_id):
        raise ValueError(
            "MOSS-TTS-Realtime prepared marker must match payload.request_id"
        )
    prepared = _QUEUE.pop(str(marker))
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS-Realtime preprocessing state is missing for prepared "
            f"payload {marker!r}; the scheduler must not rebuild it"
        )
    return prepared


def _normalize_generation_kwargs(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is not None and not isinstance(values, Mapping):
        raise TypeError("generation_kwargs must be a mapping")
    source = dict(values or {})
    max_new_tokens = _source_value(
        source,
        "max_new_tokens",
        MOSS_TTS_REALTIME_DEFAULT_MAX_NEW_TOKENS,
    )
    kwargs: dict[str, Any] = {
        "max_new_tokens": _strict_int(
            max_new_tokens,
            "max_new_tokens",
            minimum=1,
        ),
        "temperature": _strict_float(
            _source_value(source, "temperature", 0.8),
            "temperature",
        ),
        "top_p": _strict_float(
            _source_value(source, "top_p", 0.6),
            "top_p",
        ),
        "top_k": _strict_int(
            _source_value(source, "top_k", 30),
            "top_k",
            minimum=1,
        ),
        "do_sample": _strict_bool(
            _source_value(source, "do_sample", True),
            "do_sample",
        ),
        "repetition_penalty": _strict_float(
            _source_value(source, "repetition_penalty", 1.1),
            "repetition_penalty",
        ),
        "repetition_window": _strict_int(
            _source_value(source, "repetition_window", 50),
            "repetition_window",
            minimum=1,
        ),
    }
    if source.get("seed") is not None:
        kwargs["seed"] = _strict_int(source["seed"], "seed", minimum=0)

    if kwargs["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < kwargs["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if kwargs["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be positive")
    return kwargs


def _build_generation_source(
    *,
    params: Mapping[str, Any],
    options: Mapping[str, Any],
    tts_params: Mapping[str, Any],
    has_tts_params: bool,
) -> dict[str, Any]:
    """Keep model defaults unless the speech caller explicitly overrides them."""

    source = dict(options)
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()

    for key, value in params.items():
        if value is None:
            continue
        if (
            has_tts_params
            and key in _STANDARD_SPEECH_GENERATION_FIELDS
            and key not in explicit_fields
        ):
            continue
        source[key] = value
    return source


def build_moss_tts_realtime_state(
    payload: StagePayload,
    *,
    num_codebooks: int | None = None,
) -> MossTTSRealtimeState:
    """Normalize either a realtime state payload or the offline request shim."""

    data = payload.data if isinstance(payload.data, dict) else {}
    if "session_id" in data or "turn_id" in data:
        normalized_data = dict(data)
        for name in ("session_id", "turn_id"):
            value = normalized_data.get(name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if "turn_index" in normalized_data:
            normalized_data["turn_index"] = _strict_int(
                normalized_data["turn_index"],
                "turn_index",
                minimum=0,
            )
        if "initial_token_ids" in normalized_data:
            normalized_data["initial_token_ids"] = _normalize_token_ids(
                normalized_data["initial_token_ids"],
                "initial_token_ids",
            )
        if "initial_text" in normalized_data and not isinstance(
            normalized_data["initial_text"], (str, type(None))
        ):
            raise TypeError("initial_text must be a string")
        if "input_done" in normalized_data:
            normalized_data["input_done"] = _strict_bool(
                normalized_data["input_done"],
                "input_done",
            )
        normalized_data["generation_kwargs"] = _normalize_generation_kwargs(
            normalized_data.get("generation_kwargs")
        )
        state = MossTTSRealtimeState.from_dict(normalized_data)
        state.generation_kwargs = _normalize_generation_kwargs(state.generation_kwargs)
        return state

    inputs = payload.request.inputs
    params = _mapping(payload.request.params, "request.params")
    metadata = _mapping(payload.request.metadata, "request.metadata")
    realtime = _mapping(
        metadata.get("moss_tts_realtime"),
        "metadata.moss_tts_realtime",
    )
    has_tts_params = "tts_params" in metadata
    tts_params = _mapping(metadata.get("tts_params"), "metadata.tts_params")
    options = {**tts_params, **realtime}

    if isinstance(inputs, str):
        input_data: dict[str, Any] = {}
        initial_text: str | None = inputs
    elif inputs is None:
        input_data = {}
        initial_text = None
    elif isinstance(inputs, Mapping):
        input_data = dict(inputs)
        initial_text = _first_not_none(
            input_data.get("initial_text"),
            input_data.get("text"),
        )
        if initial_text is not None and not isinstance(initial_text, str):
            raise TypeError("initial_text must be a string")
    else:
        raise TypeError("request.inputs must be a string, mapping, or None")

    user = _mapping(input_data.get("user"), "inputs.user")
    initial_token_ids = _first_not_none(
        input_data.get("initial_token_ids"),
        input_data.get("token_ids"),
        options.get("initial_token_ids"),
    )
    if initial_token_ids is None:
        initial_token_ids = []
    initial_token_ids = _normalize_token_ids(
        initial_token_ids,
        "initial_token_ids",
    )

    reference = _mapping(input_data.get("reference"), "inputs.reference")
    references = input_data.get("references")
    if references is not None and not isinstance(references, list):
        raise TypeError("inputs.references must be a list")
    if not reference and references:
        reference = _mapping(references[0], "inputs.references[0]")
    reference_data_uri = audio_data_uri_from_reference(reference) if reference else None
    user_data_uri = audio_data_uri_from_reference(user) if user else None

    generation_source = _build_generation_source(
        params=params,
        options=options,
        tts_params=tts_params,
        has_tts_params=has_tts_params,
    )
    generation_kwargs = _normalize_generation_kwargs(generation_source)

    explicit_session_id = _optional_text(
        _first_not_none(
            input_data.get("session_id"),
            options.get("session_id"),
            metadata.get("session_id"),
        )
    )
    session_id = explicit_session_id or f"offline:{payload.request_id}"
    turn_id = _optional_text(
        _first_not_none(
            input_data.get("turn_id"),
            options.get("turn_id"),
        )
    ) or str(payload.request_id)
    turn_index = _strict_int(
        _first_not_none(
            input_data.get("turn_index"),
            options.get("turn_index"),
            0,
        ),
        "turn_index",
        minimum=0,
    )

    raw_input_done = _first_not_none(
        input_data.get("input_done"),
        options.get("input_done"),
        params.get("input_done"),
        True,
    )
    raw_input_done = _strict_bool(raw_input_done, "input_done")

    raw_stream = params.get("stream", True)
    if raw_stream is None:
        raw_stream = True
    stream = _strict_bool(raw_stream, "stream")

    return MossTTSRealtimeState(
        session_id=session_id,
        turn_id=turn_id,
        voice=_optional_text(
            _first_not_none(
                input_data.get("voice"),
                options.get("voice"),
                "default",
            )
        ),
        ref_audio=_first_not_none(
            input_data.get("ref_audio"),
            reference.get("audio_codes"),
            reference.get("codes"),
            reference.get("audio"),
            reference.get("audio_path"),
            reference_data_uri,
            options.get("ref_audio"),
        ),
        ref_text=_optional_text(
            _first_not_none(
                input_data.get("ref_text"),
                reference.get("text"),
                options.get("ref_text"),
            )
        ),
        language=_optional_text(
            _first_not_none(input_data.get("language"), options.get("language"))
        ),
        instructions=_optional_text(
            _first_not_none(
                input_data.get("instructions"),
                options.get("instructions"),
                options.get("instruct"),
            )
        ),
        turn_index=turn_index,
        user_text=_optional_text(user.get("text")),
        user_audio=_first_not_none(
            user.get("audio_codes"),
            user.get("codes"),
            user.get("audio"),
            user.get("audio_path"),
            user_data_uri,
        ),
        initial_text=initial_text,
        initial_token_ids=initial_token_ids,
        input_done=raw_input_done,
        keep_session=explicit_session_id is not None,
        generation_kwargs=generation_kwargs,
        stream_metadata=(
            {
                "stream": True,
                "modality": "audio_codes",
                # Session identity must ride every chunk: stream items reach the
                # vocoder long before the terminal payload, and the vocoder keys
                # its codec slot to the session, not to the per-turn request.
                "session_id": session_id,
                "turn_id": turn_id,
                **({"n_vq": int(num_codebooks)} if num_codebooks is not None else {}),
            }
            if stream
            else {}
        ),
    )


def normalize_moss_tts_realtime_audio_codes(
    value: Any,
    *,
    num_codebooks: int,
    codebook_size: int,
) -> np.ndarray:
    """Normalize codec/preencoded output to contiguous row-major audio rows."""

    if isinstance(value, Mapping):
        value = _first_not_none(
            value.get("audio_codes"),
            value.get("codes"),
            value.get("tokens"),
        )
    elif hasattr(value, "audio_codes"):
        value = value.audio_codes

    if value is None:
        raise ValueError("audio-code payload does not contain any codes")

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        codes = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("audio codes must be tensor/array-like") from exc

    if codes.dtype.kind not in {"i", "u"}:
        raise TypeError("audio codes must use an integer dtype")

    if codes.ndim == 3:
        if (
            codes.shape[1] == 1
            and num_codebooks <= codes.shape[0] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
        ):
            codes = codes[:num_codebooks, 0, :].T
        elif (
            codes.shape[0] == 1
            and num_codebooks <= codes.shape[1] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
        ):
            codes = codes[0, :num_codebooks, :].T
        else:
            raise ValueError(
                "rank-3 audio codes must have shape [Q, 1, T] or [1, Q, T]"
            )
    elif codes.ndim == 2:
        if codes.shape[0] == num_codebooks:
            codes = codes.T
        elif codes.shape[1] == num_codebooks:
            pass
        elif num_codebooks < codes.shape[0] <= _MAX_SUPPORTED_CODEC_QUANTIZERS:
            codes = codes[:num_codebooks, :].T
        elif num_codebooks < codes.shape[1] <= _MAX_SUPPORTED_CODEC_QUANTIZERS:
            codes = codes[:, :num_codebooks]
    else:
        raise ValueError("audio codes must have rank 2 or rank 3")

    if codes.ndim != 2 or codes.shape[1] != num_codebooks:
        raise ValueError(f"audio codes must normalize to shape [T, {num_codebooks}]")
    if codes.shape[0] == 0:
        raise ValueError("audio codes must contain at least one frame")
    codes = np.ascontiguousarray(codes, dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= codebook_size):
        raise ValueError(
            "reference/user audio codes must be in the codec range "
            f"[0, {codebook_size})"
        )
    return codes


def _contains_explicit_codes(value: Any, *, num_codebooks: int) -> bool:
    if isinstance(value, Mapping):
        return any(
            value.get(key) is not None for key in ("audio_codes", "codes", "tokens")
        )
    if hasattr(value, "audio_codes"):
        return value.audio_codes is not None

    if isinstance(value, torch.Tensor):
        shape = tuple(int(dim) for dim in value.shape)
        is_integer = not (
            value.dtype == torch.bool
            or torch.is_floating_point(value)
            or torch.is_complex(value)
        )
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False
        shape = tuple(int(dim) for dim in array.shape)
        is_integer = array.dtype.kind in {"i", "u"}

    if not is_integer:
        return False
    if len(shape) == 3:
        return (
            shape[1] == 1
            and num_codebooks <= shape[0] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
        ) or (
            shape[0] == 1
            and num_codebooks <= shape[1] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
        )
    if len(shape) != 2:
        return False
    return (
        shape[0] == num_codebooks
        or shape[1] == num_codebooks
        or num_codebooks < shape[0] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
        or num_codebooks < shape[1] <= _MAX_SUPPORTED_CODEC_QUANTIZERS
    )


def _run_audio_encoder(audio_encoder: Any, value: Any, *, name: str) -> Any:
    if audio_encoder is None:
        raise ValueError(
            f"{name} requires preencoded 16-codebook tokens or an audio encoder"
        )
    encode = getattr(audio_encoder, "encode", None)
    if callable(encode):
        return encode(value)
    if callable(audio_encoder):
        return audio_encoder(value)
    raise TypeError("audio_encoder must be callable or expose encode()")


def _resolve_audio_codes(
    value: Any | None,
    *,
    audio_encoder: Any | None,
    num_codebooks: int,
    codebook_size: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    encoded = (
        value
        if _contains_explicit_codes(value, num_codebooks=num_codebooks)
        else _run_audio_encoder(audio_encoder, value, name=name)
    )
    return normalize_moss_tts_realtime_audio_codes(
        encoded,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )


def _normalize_processor_rows(
    value: Any,
    *,
    processor: Any,
    name: str,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        rows = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor/array-like") from exc
    config = _processor_model_config(processor)
    row_width = int(config.rvq) + 1
    if rows.ndim != 2 or rows.shape[1] != row_width:
        raise ValueError(f"{name} must have shape [T, {row_width}]")
    if rows.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if rows.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must use an integer dtype")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if np.any(rows[:, 0] < 0):
        raise ValueError(f"{name} text token ids must be non-negative")
    audio = rows[:, 1:]
    if np.any(audio < 0) or np.any(audio >= int(config.audio_vocab_size)):
        raise ValueError(
            f"{name} audio columns must be in [0, {config.audio_vocab_size})"
        )
    return rows


def _assistant_prefix_rows(processor: Any) -> np.ndarray:
    token_ids = _validate_tokenizer_token_ids(
        processor.tokenizer.encode(_ASSISTANT_TURN_PREFIX),
        name="assistant prefix token ids",
    )
    rows = np.full(
        (len(token_ids), int(processor.model_config.rvq) + 1),
        int(processor.model_config.audio_pad_token),
        dtype=np.int64,
    )
    rows[:, 0] = np.asarray(token_ids, dtype=np.int64)
    return rows


def build_moss_tts_realtime_turn_prompt(
    *,
    processor: Any,
    voice_codes: np.ndarray | None,
    user_text: str | None,
    user_audio_codes: np.ndarray | None,
    include_system_prompt: bool,
) -> np.ndarray:
    """Build the exact HF turn suffix before assistant-text prefill rows."""

    _validate_processor_contract(processor)
    initialize_moss_tts_realtime_tokenizer_vocab_size(processor.tokenizer)
    include_system_prompt = _strict_bool(
        include_system_prompt,
        "include_system_prompt",
    )
    if user_text is not None and not isinstance(user_text, str):
        raise TypeError("user_text must be a string")
    if voice_codes is not None:
        voice_codes = normalize_moss_tts_realtime_audio_codes(
            voice_codes,
            num_codebooks=int(processor.model_config.rvq),
            codebook_size=int(processor.model_config.audio_pad_token),
        )
    if user_audio_codes is not None:
        user_audio_codes = normalize_moss_tts_realtime_audio_codes(
            user_audio_codes,
            num_codebooks=int(processor.model_config.rvq),
            codebook_size=int(processor.model_config.audio_pad_token),
        )
    has_user_text = user_text is not None
    has_user_audio = user_audio_codes is not None
    if has_user_text != has_user_audio:
        raise ValueError(
            "full-fidelity user context requires both user text and user audio"
        )

    if has_user_text:
        user_prompt = _normalize_processor_rows(
            processor.make_user_prompt(user_text, user_audio_codes),
            processor=processor,
            name="processor user prompt",
        )
        pieces = [user_prompt]
        if include_system_prompt:
            pieces.insert(
                0,
                _normalize_processor_rows(
                    processor.make_ensemble(voice_codes),
                    processor=processor,
                    name="processor system prompt",
                ),
            )
    else:
        pieces = [_assistant_prefix_rows(processor)]
        if include_system_prompt:
            pieces.insert(
                0,
                _normalize_processor_rows(
                    processor.make_ensemble(voice_codes),
                    processor=processor,
                    name="processor system prompt",
                ),
            )

    rows = np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype=np.int64)
    return _normalize_processor_rows(
        rows,
        processor=processor,
        name="MOSS-TTS-Realtime processor lowering",
    )


def _moss_tts_realtime_row_cache_key(row: np.ndarray) -> int:
    little_endian_row = np.asarray(row, dtype="<i8")
    digest = hashlib.blake2b(little_endian_row.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def build_moss_tts_realtime_row_cache_key(
    row: Sequence[int],
    *,
    model_config: Any | None = None,
) -> int:
    """Hash one complete canonical row into the signed scheduler-id range."""

    values = np.asarray([tuple(row)])
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("row cache key requires exactly one rank-1 row")
    if model_config is not None and values.shape[1] != int(model_config.rvq) + 1:
        raise ValueError(f"row cache key must have {int(model_config.rvq) + 1} columns")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError("row cache key requires integer token ids")
    return _moss_tts_realtime_row_cache_key(
        np.ascontiguousarray(values[0], dtype=np.int64)
    )


def build_moss_tts_realtime_row_cache_key_ids(
    rows: torch.Tensor,
    *,
    model_config: Any | None = None,
) -> list[int]:
    """Hash complete prompt rows; scalar text ids are not cache identities."""

    if not isinstance(rows, torch.Tensor):
        raise TypeError("row cache keys require a torch.Tensor")
    if (
        rows.dtype == torch.bool
        or torch.is_floating_point(rows)
        or torch.is_complex(rows)
    ):
        raise TypeError("row cache keys require an integer tensor")
    normalized = rows.detach().cpu().numpy()
    if normalized.ndim != 2:
        raise ValueError("row cache keys require a rank-2 tensor")
    if model_config is not None and normalized.shape[1] != int(model_config.rvq) + 1:
        raise ValueError(
            f"row cache keys must have {int(model_config.rvq) + 1} columns"
        )
    normalized = np.ascontiguousarray(normalized, dtype=np.int64)
    return [_moss_tts_realtime_row_cache_key(row) for row in normalized]


def build_moss_tts_realtime_prefill_rows(
    token_ids: Sequence[int],
    *,
    model_config: Any,
) -> torch.Tensor:
    """Build the checkpoint-defined assistant prefix used for first-frame prefill."""

    normalized = tuple(
        _strict_int(token_id, f"prefill token_ids[{index}]", minimum=0)
        for index, token_id in enumerate(token_ids)
    )
    prefill_tokens = int(model_config.delay_tokens_len)
    if not 1 <= len(normalized) <= prefill_tokens:
        raise ValueError(
            "MOSS-TTS-Realtime prefill requires between 1 and "
            f"{prefill_tokens} text tokens"
        )
    rows = torch.full(
        (len(normalized), int(model_config.rvq) + 1),
        int(model_config.audio_pad_token),
        dtype=torch.long,
    )
    rows[:, 0] = torch.tensor(normalized, dtype=torch.long)
    rows[-1, 1] = int(model_config.audio_bos_token)
    return rows


def prepare_moss_tts_realtime_state(
    state: MossTTSRealtimeState,
    *,
    processor: Any,
    audio_encoder: Any | None = None,
    reference_encoder: Any | None = None,
) -> MossTTSRealtimePreparedRequest:
    """Pure heavy-lowering step, separated for parity tests and worker use."""

    _validate_processor_contract(processor)
    model_config = _processor_model_config(processor)
    if not state.session_id or not state.turn_id:
        raise ValueError("prepared realtime state requires session_id and turn_id")
    turn_index = _strict_int(state.turn_index, "turn_index", minimum=0)
    if state.user_text is not None and not isinstance(state.user_text, str):
        raise TypeError("user_text must be a string")
    if state.initial_text is not None and not isinstance(state.initial_text, str):
        raise TypeError("initial_text must be a string")

    include_system_prompt = turn_index == 0
    voice_codes_np = (
        _resolve_audio_codes(
            state.ref_audio,
            audio_encoder=(
                reference_encoder if reference_encoder is not None else audio_encoder
            ),
            num_codebooks=int(model_config.rvq),
            codebook_size=int(model_config.audio_pad_token),
            name="voice reference",
        )
        if include_system_prompt
        else None
    )
    user_codes_np = _resolve_audio_codes(
        state.user_audio,
        audio_encoder=audio_encoder,
        num_codebooks=int(model_config.rvq),
        codebook_size=int(model_config.audio_pad_token),
        name="user audio",
    )
    prompt_rows_np = build_moss_tts_realtime_turn_prompt(
        processor=processor,
        voice_codes=voice_codes_np,
        user_text=state.user_text,
        user_audio_codes=user_codes_np,
        include_system_prompt=include_system_prompt,
    )

    if state.initial_text is not None:
        initial_token_ids = _validate_tokenizer_token_ids(
            processor.tokenizer.encode(
                state.initial_text,
                add_special_tokens=False,
            ),
            name="initial text token ids",
        )
    else:
        initial_token_ids = _validate_tokenizer_token_ids(
            state.initial_token_ids,
            name="initial_token_ids",
        )

    prompt_rows = torch.from_numpy(prompt_rows_np.copy()).to(dtype=torch.long)
    cache_ids = build_moss_tts_realtime_row_cache_key_ids(
        prompt_rows, model_config=model_config
    )
    return MossTTSRealtimePreparedRequest(
        state=state,
        turn_prompt_rows=prompt_rows,
        turn_prompt_cache_ids=cache_ids,
        turn_prompt_input_ids=torch.tensor(cache_ids, dtype=torch.long),
        initial_token_ids=initial_token_ids,
        voice_codes=(
            torch.from_numpy(voice_codes_np.copy()).to(dtype=torch.long)
            if voice_codes_np is not None
            else None
        ),
        user_audio_codes=(
            torch.from_numpy(user_codes_np.copy()).to(dtype=torch.long)
            if user_codes_np is not None
            else None
        ),
        include_system_prompt=include_system_prompt,
        generation_kwargs=_normalize_generation_kwargs(state.generation_kwargs),
    )


def prepare_moss_tts_realtime_request(
    payload: StagePayload,
    *,
    processor: Any,
    audio_encoder: Any | None = None,
    reference_encoder: Any | None = None,
) -> MossTTSRealtimePreparedRequest:
    model_config = _processor_model_config(processor)
    state = build_moss_tts_realtime_state(
        payload,
        num_codebooks=int(model_config.rvq),
    )
    return prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        audio_encoder=audio_encoder,
        reference_encoder=reference_encoder,
    )


def preprocess_moss_tts_realtime_payload(payload: StagePayload) -> StagePayload:
    """Run codec/processor lowering off the scheduler thread and publish it."""

    request_id = str(payload.request_id)
    context = _QUEUE.begin(request_id)
    if context is None:
        raise RuntimeError(
            "MOSS-TTS-Realtime preprocessing context is not initialized; "
            "the preprocessing stage must register it before serving"
        )

    try:
        prepared = prepare_moss_tts_realtime_request(
            payload,
            processor=context.processor,
            audio_encoder=context.audio_encoder,
            reference_encoder=context.reference_encoder,
        )
    except BaseException:
        _QUEUE.fail_inflight(request_id)
        raise

    published = _QUEUE.publish(request_id, prepared)
    data = prepared.state.to_dict()
    data[MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY] = list(
        prepared.initial_token_ids
    )
    if published:
        data[_MOSS_TTS_REALTIME_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def build_moss_tts_realtime_request_data(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSRealtimeRequestData:
    """Consume the CPU handoff without creating scheduler-owned live state."""

    prepared = pop_prepared_moss_tts_realtime_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS-Realtime request builder requires a payload prepared by "
            "preprocess_moss_tts_realtime_payload"
        )

    state = prepared.state
    state.generation_kwargs = dict(prepared.generation_kwargs)
    max_new_tokens = int(
        prepared.generation_kwargs.get(
            "max_new_tokens",
            MOSS_TTS_REALTIME_DEFAULT_MAX_NEW_TOKENS,
        )
    )
    seed = prepared.generation_kwargs.get("seed")
    data = MossTTSRealtimeRequestData(
        input_ids=prepared.turn_prompt_input_ids.detach().clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enforce_request_limits=True,
        state=state,
        model_config=model.config,
        prompt_rows=prepared.turn_prompt_rows.detach().clone(),
        initial_token_ids=prepared.initial_token_ids,
        provisional_output_id=int(model.config.reference_audio_pad),
        sampling_seed=int(seed) if seed is not None else None,
        engine_start_s=time.perf_counter(),
        stream_metadata=dict(state.stream_metadata) or None,
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_moss_tts_realtime_result(
    payload: StagePayload,
    data: MossTTSRealtimeRequestData,
) -> StagePayload:
    """Build the terminal AR payload after scheduler-validated audio EOS."""

    turn = data.turn_state
    if turn is None:
        raise RuntimeError("MOSS-TTS-Realtime result is missing turn state")
    if turn.phase is not MossTTSRealtimeTurnPhase.COMPLETED:
        raise RuntimeError(
            "MOSS-TTS-Realtime may emit a successful result only after audio EOS"
        )

    rows = turn.ledger.rows[data.generation_row_start :]
    if rows:
        data.state.audio_codes = torch.tensor(
            [row[1:] for row in rows],
            dtype=torch.long,
        )
    else:
        data.state.audio_codes = torch.empty(
            (0, int(data.model_config.rvq)),
            dtype=torch.long,
        )
    data.state.prompt_rows = None
    data.state.input_done = turn.pending_input.input_done
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data.state.to_dict(),
    )


def make_moss_tts_realtime_scheduler_adapters(*, model: Any):
    """Build worker-safe request and scheduler-terminal result adapters."""

    def request_builder(payload: StagePayload) -> MossTTSRealtimeRequestData:
        return build_moss_tts_realtime_request_data(payload, model=model)

    def result_adapter(data: MossTTSRealtimeRequestData) -> StagePayload:
        return apply_moss_tts_realtime_result(data.stage_payload, data)

    return request_builder, result_adapter
