# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for ZONOS2 TTS.

ZONOS2 uses a frame-shaped prompt: each token is a row of
``n_codebooks`` audio-code columns plus one text column. The text column owns
normal UTF-8 byte tokens and the conditioning-token tail of ``text_vocab``.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import logging
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.zonos2_tts.payload_types import (
    ZONOS2_AUDIO_PAD_ID,
    ZONOS2_BOS_ID,
    ZONOS2_BYTE_TEXT_VOCAB,
    ZONOS2_EOS_ID,
    ZONOS2_LEGACY_SYMBOL_VOCAB,
    ZONOS2_N_CODEBOOKS,
    ZONOS2_TEXT_VOCAB,
    Zonos2TTSState,
)
from sglang_omni.models.zonos2_tts.radix_hash import build_row_cache_key_ids
from sglang_omni.models.zonos2_tts.text_normalization import (
    normalize_zonos2_language,
    normalize_zonos2_text,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData

logger = logging.getLogger(__name__)

ZONOS2_DEFAULT_TEMPERATURE = 1.15
ZONOS2_DEFAULT_TOP_K = 106
ZONOS2_DEFAULT_TOP_P = 0.0
ZONOS2_DEFAULT_MIN_P = 0.18
ZONOS2_DEFAULT_REPETITION_PENALTY = 1.2
ZONOS2_DEFAULT_REPETITION_WINDOW = 50
ZONOS2_DEFAULT_REPETITION_CODEBOOKS = 8
ZONOS2_DURATION_SAFETY_FRAMES_PER_UTF8_BYTE = 24
ZONOS2_DURATION_SAFETY_MARGIN_FRAMES = 384
ZONOS2_DURATION_SAFETY_MIN_FRAMES = 512

_ZONOS2_PREPARED_MARKER = "_zonos2_tts_prepared_request"
_ZONOS2_PREPARED_DATA = "_zonos2_tts_prepared_data"
_DATA_URI_RE = re.compile(r"^data:[^;,]+;base64,(?P<data>.+)$", re.DOTALL)
_SPEAKING_RATE_CLOSED_BUCKET_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$"
)
_SPEAKING_RATE_OPEN_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
_QUALITY_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_CLOSED_BUCKET_RE = re.compile(
    rf"^\s*({_QUALITY_NUMBER_RE})\s*-\s*({_QUALITY_NUMBER_RE})\s*$"
)
_QUALITY_OPEN_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*\+\s*$")
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}
_DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND = 15.0
_SPEAKING_RATE_FPS = 86.0 * (44070.0 / 44000.0)
_SEED_MASK = (1 << 63) - 1
_GENERATION_FIELDS = {
    "max_new_tokens",
    "max_tokens",
    "temperature",
    "top_k",
    "topk",
    "top_p",
    "min_p",
    "repetition_penalty",
    "repetition_window",
    "repetition_codebooks",
    "ignore_eos",
    "seed",
}
_IMPLICIT_SAMPLING_DEFAULTS = {
    "temperature": {0.8, 1.0},
    "top_p": {0.8, 1.0},
    "top_k": {30, -1},
    "min_p": {0.0},
    "repetition_penalty": {1.0, 1.1},
}


# ============================================================================
# Prompt-token helpers (ported from Zyphra's prompt.py semantics)
# ============================================================================


def text_to_byte_ids(text: str) -> list[int]:
    """Convert text to ZONOS2 byte token IDs: BOS + UTF-8 bytes + EOS."""

    return [
        ZONOS2_BOS_ID,
        *(byte + ZONOS2_LEGACY_SYMBOL_VOCAB for byte in text.encode("utf-8")),
        ZONOS2_EOS_ID,
    ]


def conditioned_text_vocab_size(
    speaking_rate_num_buckets: int = 0,
    quality_num_buckets: int = 0,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = (
        int(speaking_rate_num_buckets),
        int(quality_num_buckets),
        int(speaker_background_num_buckets),
        int(accurate_mode_num_buckets),
    )
    if any(count < 0 for count in counts):
        raise ValueError("conditioning bucket counts must be non-negative")
    return ZONOS2_BYTE_TEXT_VOCAB + sum(counts)


def _normalize_quality_bucket_counts(quality_bucket_counts: Any) -> tuple[int, ...]:
    counts = tuple(int(count) for count in (quality_bucket_counts or ()))
    if any(count < 0 for count in counts):
        raise ValueError("quality_bucket_counts must be non-negative")
    return counts


def _conditioning_base_text_vocab(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Any = (),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
    *,
    context: str,
) -> int:
    if text_vocab is None:
        raise ValueError(f"text_vocab is required for {context}")
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = (
        int(text_vocab)
        - int(speaking_rate_num_buckets)
        - sum(counts)
        - int(speaker_background_num_buckets)
        - int(accurate_mode_num_buckets)
    )
    if base_text_vocab < 0:
        raise ValueError(
            "text_vocab is smaller than the configured ZONOS2 conditioning buckets"
        )
    return base_text_vocab


def speaking_rate_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    speaking_rate_bucket: int,
    quality_bucket_counts: Any = (),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    num_buckets = int(speaking_rate_num_buckets)
    if num_buckets <= 0:
        raise ValueError("Current ZONOS2 model does not define speaking-rate buckets")
    bucket = int(speaking_rate_bucket)
    if bucket < 0 or bucket >= num_buckets:
        raise ValueError(
            f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}"
        )
    return _conditioning_base_text_vocab(
        text_vocab,
        num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="speaking-rate conditioning",
    ) + bucket


def quality_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Any,
    feature_idx: int,
    quality_bucket: int,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    if not counts:
        raise ValueError("Current ZONOS2 model does not define quality buckets")
    feature = int(feature_idx)
    if feature < 0 or feature >= len(counts):
        raise ValueError(
            f"quality feature index must be in [0, {len(counts) - 1}], got {feature}"
        )
    count = counts[feature]
    bucket = int(quality_bucket)
    if bucket < 0 or bucket >= count:
        raise ValueError(
            f"quality bucket for feature {feature} must be in [0, {count - 1}], "
            f"got {bucket}"
        )
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="quality conditioning",
    )
    return base + int(speaking_rate_num_buckets) + sum(counts[:feature]) + bucket


def speaker_background_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Any,
    clean: bool,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 0,
) -> int:
    num_buckets = int(speaker_background_num_buckets)
    if num_buckets < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2")
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        num_buckets,
        accurate_mode_num_buckets,
        context="speaker-background conditioning",
    )
    return base + int(speaking_rate_num_buckets) + sum(counts) + (
        0 if bool(clean) else 1
    )


def accurate_mode_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Any,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 1,
) -> int:
    accurate_count = int(accurate_mode_num_buckets)
    background_count = int(speaker_background_num_buckets)
    if accurate_count <= 0:
        raise ValueError("accurate_mode_num_buckets must be positive")
    if background_count < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2")
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        background_count,
        accurate_count,
        context="accurate-mode conditioning",
    )
    return base + int(speaking_rate_num_buckets) + sum(counts) + background_count


def _text_row(token_id: int, *, n_codebooks: int, audio_pad_id: int) -> list[int]:
    return [int(audio_pad_id)] * int(n_codebooks) + [int(token_id)]


def build_text_prompt_rows(
    text: str,
    *,
    n_codebooks: int = ZONOS2_N_CODEBOOKS,
    audio_pad_id: int = ZONOS2_AUDIO_PAD_ID,
    text_vocab: int = ZONOS2_TEXT_VOCAB,
    speaking_rate_num_buckets: int = 0,
    speaking_rate_bucket: int | None = None,
    quality_bucket_counts: Any = (),
    quality_buckets: Any = None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> list[list[int]]:
    """Build ZONOS2 2D text-prompt rows with conditioning-token IDs."""

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    rows: list[list[int]] = []
    if speaking_rate_bucket is not None:
        rows.append(
            _text_row(
                speaking_rate_token_id(
                    text_vocab,
                    speaking_rate_num_buckets,
                    speaking_rate_bucket,
                    counts,
                    speaker_background_num_buckets,
                    accurate_mode_num_buckets,
                ),
                n_codebooks=n_codebooks,
                audio_pad_id=audio_pad_id,
            )
        )
    if quality_buckets is not None:
        for feature_idx, bucket in enumerate(quality_buckets):
            if bucket is None:
                continue
            rows.append(
                _text_row(
                    quality_token_id(
                        text_vocab,
                        speaking_rate_num_buckets,
                        counts,
                        feature_idx,
                        int(bucket),
                        speaker_background_num_buckets,
                        accurate_mode_num_buckets,
                    ),
                    n_codebooks=n_codebooks,
                    audio_pad_id=audio_pad_id,
                )
            )
    rows.extend(
        _text_row(token_id, n_codebooks=n_codebooks, audio_pad_id=audio_pad_id)
        for token_id in text_to_byte_ids(text)
    )
    return rows


# ============================================================================
# Silence prefix (0.2s at 44.1kHz, 17 frames)
# ============================================================================

_SILENCE_TOKENS_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


def shear(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply the reference delay pattern to audio codes."""

    time_steps, codebooks = x.shape
    padded = x.new_full((codebooks - 1 + time_steps, codebooks), pad)
    padded[codebooks - 1 :] = x
    row_idx = (
        codebooks
        - 1
        + torch.arange(time_steps, device=x.device).unsqueeze(1)
        - torch.arange(codebooks, device=x.device)
    )
    return padded.gather(0, row_idx)


def build_silence_prefix(
    n_codebooks: int = ZONOS2_N_CODEBOOKS,
    audio_pad_id: int = ZONOS2_AUDIO_PAD_ID,
    text_vocab: int = ZONOS2_TEXT_VOCAB,
) -> torch.Tensor:
    silence = torch.tensor(_SILENCE_TOKENS_0_2S, dtype=torch.int32)
    sheared = shear(silence[:, :n_codebooks], audio_pad_id)
    text_col = torch.full((sheared.shape[0], 1), text_vocab, dtype=torch.int32)
    return torch.cat([sheared, text_col], dim=1)


def build_speaker_slot(
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
) -> torch.Tensor:
    slot = torch.full((1, n_codebooks + 1), audio_pad_id, dtype=torch.long)
    slot[:, n_codebooks] = text_vocab
    return slot


# ============================================================================
# Request state, prepared handoff, and preprocessing context
# ============================================================================


@dataclass
class Zonos2TTSSGLangRequestData(ARRequestData):
    """Scheduler-owned request state for ZONOS2 TTS."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    state: Zonos2TTSState = field(default_factory=Zonos2TTSState)
    prompt_rows: torch.Tensor | None = None
    speaker_embedding: torch.Tensor | None = None
    speaker_token_position: int = -1
    output_rows: list[torch.Tensor] = field(default_factory=list)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    temperature: float = ZONOS2_DEFAULT_TEMPERATURE
    top_k: int = ZONOS2_DEFAULT_TOP_K
    top_p: float = ZONOS2_DEFAULT_TOP_P
    min_p: float = ZONOS2_DEFAULT_MIN_P
    repetition_penalty: float = ZONOS2_DEFAULT_REPETITION_PENALTY
    repetition_window: int = ZONOS2_DEFAULT_REPETITION_WINDOW
    repetition_codebooks: int = ZONOS2_DEFAULT_REPETITION_CODEBOOKS
    max_new_tokens: int = 0
    seed: int | None = None
    ignore_eos: bool = False
    eos_frame: int = -1
    eos_countdown: int = -1
    total_generated: int = 0
    engine_start_s: float = 0.0

    def check_eos(self, audio_codes: list[int]) -> bool:
        """Reference delayed EOS: wait n_codebooks + 1 frames after EOA."""

        if self.ignore_eos:
            self.total_generated += 1
            return False

        n_codebooks = int(self.state.n_codebooks)
        eoa_id = int(self.state.eoa_id)
        self.total_generated += 1

        if self.eos_frame < 0:
            step = self.total_generated - 1
            eos_cols = [code == eoa_id for code in audio_codes[:n_codebooks]]
            if any(eos_cols):
                max_eos_cb = max(i for i, is_eos in enumerate(eos_cols) if is_eos)
                self.eos_frame = max(0, step - max_eos_cb)
                self.eos_countdown = n_codebooks + 1

        if self.eos_countdown > 0:
            self.eos_countdown -= 1
            if self.eos_countdown == 0:
                return True

        return False


@dataclass
class Zonos2PreparedRequest:
    state: Zonos2TTSState
    prompt_rows: torch.Tensor
    input_ids_list: list[int]
    speaker_embedding: torch.Tensor | None
    speaker_token_position: int
    speaker_cache_key: str | None
    generation_kwargs: dict[str, Any]


@dataclass
class Zonos2PreprocessingContext:
    model_config: Any = None
    speaker_model: Any = None


_PREPROCESSING_CONTEXT = Zonos2PreprocessingContext()
_PREPARED_REQUESTS: dict[str, Zonos2PreparedRequest] = {}
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_zonos2_preprocessing_context(
    *,
    model_config: Any = None,
    speaker_model: Any = None,
) -> None:
    """Set the global preprocessing context (called at pipeline startup)."""

    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = Zonos2PreprocessingContext(
            model_config=model_config,
            speaker_model=speaker_model,
        )
        _PREPARED_REQUESTS.clear()


def get_zonos2_preprocessing_context() -> Zonos2PreprocessingContext:
    return _PREPROCESSING_CONTEXT


def pop_prepared_zonos2_request(payload: StagePayload) -> Zonos2PreparedRequest:
    prepared_request_id = None
    if isinstance(payload.data, dict):
        marker = payload.data.get(_ZONOS2_PREPARED_MARKER)
        if marker is not None:
            prepared_request_id = str(marker)
    if prepared_request_id is None:
        raise RuntimeError("ZONOS2 request is missing preprocessing marker")
    with _PREPARED_REQUESTS_LOCK:
        prepared = _PREPARED_REQUESTS.pop(prepared_request_id, None)
    if prepared is None and isinstance(payload.data, dict):
        prepared = _prepared_request_from_payload(payload.data)
    if prepared is None:
        raise RuntimeError(
            "ZONOS2 preprocessing state is missing for prepared payload "
            f"{prepared_request_id!r}"
        )
    return prepared


def _prepared_request_to_payload(prepared: Zonos2PreparedRequest) -> dict[str, Any]:
    data: dict[str, Any] = {
        "prompt_rows": prepared.prompt_rows.detach().cpu(),
        "input_ids_list": list(prepared.input_ids_list),
        "speaker_token_position": int(prepared.speaker_token_position),
        "speaker_cache_key": prepared.speaker_cache_key,
        "generation_kwargs": dict(prepared.generation_kwargs),
    }
    if prepared.speaker_embedding is not None:
        data["speaker_embedding"] = prepared.speaker_embedding.detach().cpu()
    return data


def _prepared_request_from_payload(data: dict[str, Any]) -> Zonos2PreparedRequest | None:
    prepared_data = data.get(_ZONOS2_PREPARED_DATA)
    if not isinstance(prepared_data, dict):
        return None

    prompt_rows = prepared_data.get("prompt_rows")
    if prompt_rows is None:
        return None
    prompt_rows = torch.as_tensor(prompt_rows, dtype=torch.long)

    input_ids = prepared_data.get("input_ids_list")
    if input_ids is None:
        input_ids = build_row_cache_key_ids(prompt_rows)

    speaker_embedding = prepared_data.get("speaker_embedding")
    if speaker_embedding is not None:
        speaker_embedding = torch.as_tensor(speaker_embedding, dtype=torch.float32)

    generation_kwargs = prepared_data.get("generation_kwargs")
    if not isinstance(generation_kwargs, dict):
        generation_kwargs = dict(Zonos2TTSState.from_dict(data).generation_kwargs)

    return Zonos2PreparedRequest(
        state=Zonos2TTSState.from_dict(data),
        prompt_rows=prompt_rows,
        input_ids_list=[int(x) for x in input_ids],
        speaker_embedding=speaker_embedding,
        speaker_token_position=int(prepared_data.get("speaker_token_position", -1)),
        speaker_cache_key=prepared_data.get("speaker_cache_key"),
        generation_kwargs=dict(generation_kwargs),
    )


def cleanup_prepared_zonos2_request(request_id: str | StagePayload) -> None:
    """Drop prepared ZONOS2 handoff state for an aborted request."""

    if isinstance(request_id, StagePayload):
        request_id = request_id.request_id
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS.pop(str(request_id), None)


# ============================================================================
# Payload normalization and conditioning controls
# ============================================================================


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _extract_text(inputs: Any, params: dict[str, Any], data: dict[str, Any]) -> str:
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, Mapping):
        text = _first_present(inputs.get("text"), inputs.get("input"), inputs.get("prompt"))
        if text is not None:
            return str(text)
    text = _first_present(data.get("text"), params.get("text"), params.get("input"))
    if text is None:
        raise ValueError("ZONOS2 TTS request requires text input")
    return str(text)


def _extract_reference_audio(
    inputs: Any,
    params: dict[str, Any],
    tts_params: dict[str, Any],
    data: dict[str, Any],
) -> Any:
    for source in (tts_params, params, data):
        ref = _first_present(
            source.get("ref_audio"),
            source.get("reference_audio"),
            source.get("speaker_audio_base64"),
            source.get("speaker_wav_base64"),
        )
        if ref is not None:
            return ref
    if isinstance(inputs, Mapping):
        for key in ("ref_audio", "reference_audio", "speaker_audio_base64", "speaker_wav_base64"):
            if inputs.get(key) is not None:
                return inputs[key]
        references = inputs.get("references")
        if isinstance(references, (list, tuple)) and references:
            first = references[0]
            if isinstance(first, Mapping):
                return _first_present(
                    first.get("audio"),
                    first.get("audio_path"),
                    first.get("path"),
                    first.get("url"),
                    first.get("data"),
                )
    return None


def _extract_direct_speaker_embedding(
    inputs: Any,
    params: dict[str, Any],
    tts_params: dict[str, Any],
    data: dict[str, Any],
    *,
    expected_dim: int | None,
) -> torch.Tensor | None:
    direct = None
    embedding_base64 = None
    embedding_name = None
    for source in (tts_params, params, data):
        if source.get("speaker_embedding") is not None:
            direct = source["speaker_embedding"]
            break
        if source.get("speaker_embedding_base64") is not None:
            embedding_base64 = source["speaker_embedding_base64"]
            embedding_name = source.get("speaker_embedding_name")
            break
    if direct is None and embedding_base64 is None and isinstance(inputs, Mapping):
        direct = inputs.get("speaker_embedding")
        embedding_base64 = inputs.get("speaker_embedding_base64")
        embedding_name = inputs.get("speaker_embedding_name")

    if direct is not None:
        return _normalize_speaker_embedding(direct, expected_dim=expected_dim)
    if embedding_base64 is not None:
        return _load_embedding_vector_from_base64(
            str(embedding_base64),
            expected_dim=expected_dim,
            file_name=_resolve_optional_text(embedding_name),
        )
    return None


def build_zonos2_state(payload: StagePayload) -> Zonos2TTSState:
    inputs = payload.request.inputs
    params = _as_dict(payload.request.params)
    metadata = _as_dict(payload.request.metadata)
    tts_params = _as_dict(metadata.get("tts_params"))
    data = _as_dict(payload.data)

    text = _extract_text(inputs, params, data)
    ref_audio = _extract_reference_audio(inputs, params, tts_params, data)
    language = normalize_zonos2_language(
        _first_present(
            tts_params.get("language"),
            params.get("language"),
            data.get("language"),
            "en_us",
        )
    )
    text_normalization = _resolve_bool(
        _first_present(
            tts_params.get("text_normalization"),
            params.get("text_normalization"),
            data.get("text_normalization"),
        ),
        True,
    )
    return Zonos2TTSState(
        text=text,
        ref_audio=ref_audio,
        ref_text=_resolve_optional_text(
            _first_present(tts_params.get("ref_text"), params.get("ref_text"), data.get("ref_text"))
        ),
        language=language,
        text_normalization=text_normalization,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def _explicit_generation_fields(tts_params: dict[str, Any]) -> set[str]:
    raw = tts_params.get("explicit_generation_params")
    if isinstance(raw, (list, tuple, set)):
        return {str(field) for field in raw}
    return set()


def _param_is_implicit_default(field: str, value: Any) -> bool:
    defaults = _IMPLICIT_SAMPLING_DEFAULTS.get(field)
    if defaults is None:
        return False
    return value in defaults


def _select_generation_value(
    name: str,
    params: dict[str, Any],
    tts_params: dict[str, Any],
    explicit_fields: set[str],
) -> Any:
    if name in tts_params and tts_params.get(name) is not None:
        return tts_params[name]
    if name == "top_k" and tts_params.get("topk") is not None:
        return tts_params["topk"]
    if name == "max_new_tokens" and tts_params.get("max_tokens") is not None:
        return tts_params["max_tokens"]
    if name not in params or params.get(name) is None:
        if name == "top_k" and params.get("topk") is not None:
            return params["topk"] if "topk" in explicit_fields else None
        if name == "max_new_tokens" and params.get("max_tokens") is not None:
            return params["max_tokens"]
        return None
    value = params[name]
    if name in _IMPLICIT_SAMPLING_DEFAULTS and name not in explicit_fields:
        if _param_is_implicit_default(name, value):
            return None
    return value


def build_generation_kwargs(
    params: dict[str, Any],
    *,
    tts_params: dict[str, Any],
) -> dict[str, Any]:
    explicit_fields = _explicit_generation_fields(tts_params)
    generation: dict[str, Any] = {
        "temperature": ZONOS2_DEFAULT_TEMPERATURE,
        "top_k": ZONOS2_DEFAULT_TOP_K,
        "top_p": ZONOS2_DEFAULT_TOP_P,
        "min_p": ZONOS2_DEFAULT_MIN_P,
        "repetition_penalty": ZONOS2_DEFAULT_REPETITION_PENALTY,
        "repetition_window": ZONOS2_DEFAULT_REPETITION_WINDOW,
        "repetition_codebooks": ZONOS2_DEFAULT_REPETITION_CODEBOOKS,
        "ignore_eos": False,
        "seed": None,
    }
    for field_name in sorted(_GENERATION_FIELDS):
        canonical = (
            "max_new_tokens"
            if field_name == "max_tokens"
            else "top_k"
            if field_name == "topk"
            else field_name
        )
        value = _select_generation_value(canonical, params, tts_params, explicit_fields)
        if value is None:
            continue
        generation[canonical] = value

    if "max_new_tokens" in generation:
        max_new_tokens = int(generation["max_new_tokens"])
        if max_new_tokens <= 0:
            raise ValueError("ZONOS2 max_new_tokens must be positive")
        generation["max_new_tokens"] = max_new_tokens
    generation["temperature"] = float(generation["temperature"])
    generation["top_k"] = int(generation["top_k"])
    generation["top_p"] = float(generation["top_p"])
    generation["min_p"] = float(generation["min_p"])
    generation["repetition_penalty"] = float(generation["repetition_penalty"])
    generation["repetition_window"] = int(generation["repetition_window"])
    generation["repetition_codebooks"] = int(generation["repetition_codebooks"])
    generation["ignore_eos"] = bool(generation["ignore_eos"])
    if generation["seed"] is not None:
        generation["seed"] = _normalize_seed(generation["seed"])
    return generation


def _model_tts_max_tokens(model: Any) -> int:
    config = getattr(model, "config", model)
    rotary_config = getattr(config, "rotary_config", None)
    candidates = (
        getattr(model, "max_seq_len", None),
        getattr(model, "max_position_embeddings", None),
        getattr(config, "max_seq_len", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "max_seqlen", None),
        getattr(rotary_config, "max_position", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return max(1, int(candidate))
    return 6144


def resolve_zonos2_max_new_tokens(
    *,
    model: Any,
    prompt_len: int,
    requested: Any,
) -> int:
    """Resolve the output budget with the same boundary as upstream ZONOS2.

    The API layer keeps an omitted max token budget as ``None``. The scheduler
    then converts it to the remaining model context after the frame prompt is
    known, while explicit user caps are still respected.
    """

    model_max = _model_tts_max_tokens(model)
    prompt_len = int(prompt_len)
    if prompt_len >= model_max:
        raise ValueError(
            f"ZONOS2 prompt length {prompt_len} exceeds model context {model_max}"
        )

    max_output_len = max(1, model_max - prompt_len)
    if requested is None:
        return max_output_len

    requested = int(requested)
    if requested <= 0:
        raise ValueError("ZONOS2 max_new_tokens must be positive")
    return min(requested, max_output_len)


def estimate_zonos2_duration_safety_frames(text: str) -> int:
    """Return a conservative frame budget for stock ZONOS2 TTS requests.

    The value is a safety limit, not a duration target. It leaves normal speech
    well below the cap, but prevents a missed EOA token from turning a short
    sentence into minutes of repeated audio until the model context is exhausted.
    """

    byte_len = max(len((text or "").encode("utf-8")), 1)
    estimated = (
        byte_len * ZONOS2_DURATION_SAFETY_FRAMES_PER_UTF8_BYTE
        + ZONOS2_DURATION_SAFETY_MARGIN_FRAMES
    )
    return max(ZONOS2_DURATION_SAFETY_MIN_FRAMES, int(estimated))


def apply_zonos2_duration_safety_limit(
    max_new_tokens: int,
    *,
    text: str,
    requested: Any,
) -> int:
    """Bound omitted max_new_tokens by an input-conditioned TTS safety limit."""

    if requested is not None:
        return int(max_new_tokens)
    return min(int(max_new_tokens), estimate_zonos2_duration_safety_frames(text))


def _normalize_seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise ValueError("ZONOS2 seed must be an integer")
    if isinstance(seed, float) and not seed.is_integer():
        raise ValueError("ZONOS2 seed must be an integer")
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("ZONOS2 seed must be an integer") from exc
    return value & _SEED_MASK


def _get_model_attr(model_config: Any, name: str, default: Any = None) -> Any:
    if model_config is None:
        return default
    return getattr(model_config, name, default)


def _model_quality_features(model_config: Any) -> list[str]:
    raw = _get_model_attr(model_config, "quality_features", None)
    if not raw:
        buckets = _get_model_attr(model_config, "quality_buckets", None) or {}
        raw = buckets.keys() if isinstance(buckets, Mapping) else ()
    if isinstance(raw, Mapping):
        return [str(feature) for feature, enabled in raw.items() if bool(enabled)]
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in (raw or ())]


def _model_quality_buckets(model_config: Any) -> dict[str, list[str]]:
    features = _model_quality_features(model_config)
    raw = _get_model_attr(model_config, "quality_buckets", None) or {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        feature: [str(item) for item in ((raw.get(feature, ()) or ()))]
        for feature in features
    }


def _model_quality_bucket_counts(model_config: Any) -> list[int]:
    buckets = _model_quality_buckets(model_config)
    return [len(buckets.get(feature, ())) for feature in _model_quality_features(model_config)]


def _model_quality_num_buckets(model_config: Any) -> int:
    configured = int(_get_model_attr(model_config, "quality_num_buckets", 0) or 0)
    return configured or sum(_model_quality_bucket_counts(model_config))


def _quality_control_to_feature_list(value: Any, features: list[str]) -> list[Any]:
    if value is None:
        return [None] * len(features)
    if isinstance(value, Mapping):
        return [value.get(feature) for feature in features]
    if isinstance(value, (list, tuple)):
        return [value[idx] if idx < len(value) else None for idx in range(len(features))]
    raise ValueError("quality_buckets and quality_values must be a list or feature-name object")


def _parse_quality_bucket(spec: str) -> tuple[str, float, float | None]:
    exact = _QUALITY_EXACT_BUCKET_RE.match(str(spec))
    if exact is not None:
        return "exact", float(exact.group(1)), None
    closed = _QUALITY_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return "range", float(closed.group(1)), float(closed.group(2))
    open_ended = _QUALITY_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return "range", float(open_ended.group(1)), None
    raise ValueError(
        f"Invalid quality bucket {spec!r}; expected exact, closed-range, or open-ended specs"
    )


def _quality_bucket_for_value(
    value: Any,
    model_config: Any,
    feature: str,
) -> int | None:
    try:
        quality_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(quality_value):
        return None
    specs = [_parse_quality_bucket(spec) for spec in _model_quality_buckets(model_config).get(feature, ())]
    if not specs:
        return None
    for idx, (kind, low, _) in enumerate(specs):
        if kind == "exact" and math.isclose(quality_value, low, rel_tol=1e-12, abs_tol=1e-9):
            return idx
    range_indexes = [idx for idx, (kind, _, _) in enumerate(specs) if kind == "range"]
    if not range_indexes:
        return None
    for idx in range_indexes:
        _, low, high = specs[idx]
        if high is None:
            if quality_value >= low:
                return idx
        elif idx == range_indexes[-1]:
            if low <= quality_value <= high:
                return idx
        elif low <= quality_value < high:
            return idx
    _, first_low, _ = specs[range_indexes[0]]
    return range_indexes[0] if quality_value < first_low else range_indexes[-1]


def _resolve_quality_buckets(
    model_config: Any,
    *,
    quality_buckets: Any = None,
    quality_values: Any = None,
    quality_enabled: bool = True,
) -> list[int | None] | None:
    if not quality_enabled:
        return None
    if quality_buckets is None and quality_values is None:
        quality_buckets = dict(_DEFAULT_QUALITY_BUCKETS)
    if quality_buckets is not None and quality_values is not None:
        raise ValueError("Provide only one of quality_buckets or quality_values")
    features = _model_quality_features(model_config)
    counts = _model_quality_bucket_counts(model_config)
    if not features or _model_quality_num_buckets(model_config) <= 0 or sum(counts) <= 0:
        return None
    if any(count <= 0 for count in counts):
        raise ValueError("Every configured quality feature must define at least one bucket")
    if quality_buckets is not None:
        raw_buckets = _quality_control_to_feature_list(quality_buckets, features)
        resolved: list[int | None] = []
        for feature, count, raw_bucket in zip(features, counts, raw_buckets, strict=True):
            if raw_bucket is None:
                resolved.append(None)
                continue
            bucket = int(raw_bucket)
            if bucket < 0 or bucket >= count:
                raise ValueError(
                    f"quality_buckets.{feature} must be in [0, {count - 1}], got {bucket}"
                )
            resolved.append(bucket)
        return resolved
    raw_values = _quality_control_to_feature_list(quality_values, features)
    return [
        _quality_bucket_for_value(raw_value, model_config, feature)
        if raw_value is not None
        else None
        for feature, raw_value in zip(features, raw_values, strict=True)
    ]


def _parse_speaking_rate_bucket(spec: str) -> tuple[float, float | None]:
    closed = _SPEAKING_RATE_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))
    open_ended = _SPEAKING_RATE_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None
    raise ValueError(f"Invalid speaking-rate bucket {spec!r}")


def _speaking_rate_bucket_ranges(model_config: Any) -> list[tuple[float, float | None]]:
    raw = _get_model_attr(model_config, "speaking_rate_buckets", None) or ()
    return [_parse_speaking_rate_bucket(str(spec)) for spec in raw]


def _neutral_speaking_rate_bytes_per_second(
    ranges: list[tuple[float, float | None]],
) -> float:
    if not ranges:
        return _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND
    low, high = ranges[len(ranges) // 2]
    if high is None:
        return max(low, _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND)
    return (low + high) / 2.0


def _speaking_rate_bucket_for_rate(
    rate_bytes_per_second: float,
    *,
    num_buckets: int,
    ranges: list[tuple[float, float | None]],
) -> int:
    if rate_bytes_per_second <= 0:
        raise ValueError("speaking_rate must be positive")
    if ranges:
        for idx, (_, high) in enumerate(ranges):
            if high is None or (
                rate_bytes_per_second < high
                and not math.isclose(rate_bytes_per_second, high, rel_tol=1e-12, abs_tol=1e-9)
            ):
                return idx
        return len(ranges) - 1
    rate_bytes_per_frame = rate_bytes_per_second / _SPEAKING_RATE_FPS
    bucket = int(rate_bytes_per_frame * num_buckets)
    return min(max(bucket, 0), num_buckets - 1)


def _resolve_speaking_rate_bucket(
    model_config: Any,
    *,
    speaking_rate_bucket: Any = None,
    speaking_rate: Any = None,
    speed: Any = None,
    speaking_rate_enabled: bool = False,
) -> int | None:
    if not speaking_rate_enabled:
        return None
    supplied = [
        speaking_rate_bucket is not None,
        speaking_rate is not None,
        speed is not None,
    ]
    if sum(supplied) == 0:
        return None
    if sum(supplied) > 1:
        raise ValueError("Provide only one of speaking_rate_bucket, speaking_rate, or speed")
    num_buckets = int(_get_model_attr(model_config, "speaking_rate_num_buckets", 0) or 0)
    if num_buckets <= 0:
        if speed is not None and speaking_rate_bucket is None and speaking_rate is None:
            return None
        raise ValueError("Current ZONOS2 model does not support speaking-rate conditioning")
    if speaking_rate_bucket is not None:
        bucket = int(speaking_rate_bucket)
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(
                f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}"
            )
        return bucket
    ranges = _speaking_rate_bucket_ranges(model_config)
    if ranges and len(ranges) != num_buckets:
        raise ValueError(
            f"Model has {num_buckets} speaking-rate buckets, but config defines {len(ranges)} ranges"
        )
    if speaking_rate is not None:
        return _speaking_rate_bucket_for_rate(
            float(speaking_rate), num_buckets=num_buckets, ranges=ranges
        )
    speed_value = float(speed)
    if speed_value <= 0:
        raise ValueError("speed must be positive")
    return _speaking_rate_bucket_for_rate(
        _neutral_speaking_rate_bytes_per_second(ranges) * speed_value,
        num_buckets=num_buckets,
        ranges=ranges,
    )


def _extract_conditioning_controls(
    payload: StagePayload,
    model_config: Any,
) -> tuple[int | None, list[int | None] | None, bool, bool]:
    params = _as_dict(payload.request.params)
    metadata = _as_dict(payload.request.metadata)
    tts_params = _as_dict(metadata.get("tts_params"))
    data = _as_dict(payload.data)
    inputs = payload.request.inputs if isinstance(payload.request.inputs, Mapping) else {}

    def control(name: str, default: Any = None) -> Any:
        return _first_present(
            tts_params.get(name),
            params.get(name),
            data.get(name),
            inputs.get(name) if isinstance(inputs, Mapping) else None,
            default,
        )

    speaking_rate_bucket = _resolve_speaking_rate_bucket(
        model_config,
        speaking_rate_bucket=control("speaking_rate_bucket"),
        speaking_rate=control("speaking_rate"),
        speed=control("speed"),
        speaking_rate_enabled=bool(control("speaking_rate_enabled", False)),
    )
    quality_enabled = bool(control("quality_enabled", True))
    quality_buckets = _resolve_quality_buckets(
        model_config,
        quality_buckets=control("quality_buckets"),
        quality_values=control("quality_values"),
        quality_enabled=quality_enabled,
    )
    clean_background = bool(control("clean_speaker_background", False))
    accurate_mode = bool(control("accurate_mode", True))
    return speaking_rate_bucket, quality_buckets, clean_background, accurate_mode


# ============================================================================
# Speaker embedding extraction
# ============================================================================


def _decode_base64_blob(data: str) -> bytes:
    payload = data.strip()
    match = _DATA_URI_RE.match(payload)
    if match:
        payload = match.group("data")
    return base64.b64decode(payload, validate=False)


def _load_audio_bytes(audio_data: Any, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio from a tensor, bytes, data URI, HTTP(S) URL, or local path."""

    import torchaudio

    if isinstance(audio_data, torch.Tensor):
        return audio_data, sample_rate
    if isinstance(audio_data, bytes):
        wav, sr = torchaudio.load(io.BytesIO(audio_data))
        return wav, int(sr)
    if isinstance(audio_data, str):
        if audio_data.startswith(("http://", "https://")):
            with urllib.request.urlopen(audio_data, timeout=30) as response:
                raw = response.read()
            wav, sr = torchaudio.load(io.BytesIO(raw))
            return wav, int(sr)
        match = _DATA_URI_RE.match(audio_data)
        if match:
            wav, sr = torchaudio.load(io.BytesIO(_decode_base64_blob(audio_data)))
            return wav, int(sr)
        if os.path.isfile(audio_data):
            wav, sr = torchaudio.load(audio_data)
            return wav, int(sr)
        try:
            raw = _decode_base64_blob(audio_data)
        except Exception:
            raw = None
        if raw:
            wav, sr = torchaudio.load(io.BytesIO(raw))
            return wav, int(sr)
    raise ValueError(f"Unsupported audio format for ZONOS2 speaker embedding: {type(audio_data)}")


def _normalize_speaker_embedding(
    value: Any,
    *,
    expected_dim: int | None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.float32, device="cpu").squeeze()
    if tensor.ndim == 2:
        if tensor.shape[0] == 0:
            raise ValueError("Speaker embedding is empty")
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor.mean(dim=0)
    if tensor.ndim != 1:
        raise ValueError(
            f"Speaker embedding must be a vector or a batch of vectors, got {tuple(tensor.shape)}"
        )
    if expected_dim is not None and tensor.numel() != int(expected_dim):
        raise ValueError(
            f"Speaker embedding dimension mismatch: expected {expected_dim}, got {tensor.numel()}"
        )
    return tensor.contiguous()


def _load_embedding_vector_from_base64(
    data: str,
    *,
    expected_dim: int | None,
    file_name: str | None = None,
) -> torch.Tensor:
    import numpy as np

    embedding_bytes = _decode_base64_blob(data)
    try:
        loaded = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
    except Exception as exc:
        desc = file_name or "speaker_embedding_base64"
        raise ValueError(f"{desc} must contain a valid .npy or .npz embedding") from exc

    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError("Embedding archive must contain 'emb' or exactly one array")
        finally:
            loaded.close()
    else:
        arr = loaded
    return _normalize_speaker_embedding(arr, expected_dim=expected_dim)


def _compute_speaker_embedding(
    *,
    speaker_model: Any,
    ref_audio: Any,
    expected_dim: int | None,
) -> torch.Tensor:
    wav, sample_rate = _load_audio_bytes(ref_audio)
    with torch.inference_mode():
        output = speaker_model(wav, sample_rate)
    if isinstance(output, tuple):
        candidates = output
    else:
        candidates = (output,)
    for candidate in candidates:
        try:
            return _normalize_speaker_embedding(candidate, expected_dim=expected_dim)
        except ValueError:
            continue
    produced = ", ".join(str(torch.as_tensor(candidate).numel()) for candidate in candidates)
    raise ValueError(
        f"Speaker encoder produced incompatible embedding dimension(s): {produced}"
    )


def _speaker_embedding_fingerprint(embedding: torch.Tensor | None) -> str | None:
    if embedding is None:
        return None
    vector = embedding.detach().to(dtype=torch.float32, device="cpu").contiguous()
    digest = hashlib.blake2b(vector.numpy().tobytes(), digest_size=16).hexdigest()
    return f"speaker:{digest}"


def _store_prepared_zonos2_request(
    payload: StagePayload,
    prepared: Zonos2PreparedRequest,
) -> StagePayload:
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS[payload.request_id] = prepared
    data = prepared.state.to_dict()
    data[_ZONOS2_PREPARED_MARKER] = payload.request_id
    data[_ZONOS2_PREPARED_DATA] = _prepared_request_to_payload(prepared)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def _speaker_prefix_rows(
    payload: StagePayload,
    state: Zonos2TTSState,
    model_config: Any,
) -> list[torch.Tensor]:
    (
        _,
        _,
        clean_background,
        accurate_mode,
    ) = _extract_conditioning_controls(payload, model_config)
    speaking_rate_num_buckets = int(
        _get_model_attr(model_config, "speaking_rate_num_buckets", 0) or 0
    )
    quality_bucket_counts = _model_quality_bucket_counts(model_config)
    speaker_background_num_buckets = (
        2
        if bool(_get_model_attr(model_config, "speaker_background_token_enabled", False))
        else 0
    )
    accurate_mode_num_buckets = (
        1
        if (
            speaker_background_num_buckets
            and bool(_get_model_attr(model_config, "accurate_mode_token_enabled", False))
        )
        else 0
    )

    rows = [
        build_speaker_slot(
            n_codebooks=state.n_codebooks,
            audio_pad_id=state.audio_pad_id,
            text_vocab=state.text_vocab,
        )
    ]
    if speaker_background_num_buckets:
        background_token = speaker_background_token_id(
            state.text_vocab,
            speaking_rate_num_buckets,
            quality_bucket_counts,
            clean_background,
            speaker_background_num_buckets,
            accurate_mode_num_buckets,
        )
        rows.append(
            torch.tensor(
                [
                    _text_row(
                        background_token,
                        n_codebooks=state.n_codebooks,
                        audio_pad_id=state.audio_pad_id,
                    )
                ],
                dtype=torch.long,
            )
        )
    if accurate_mode_num_buckets and accurate_mode:
        accurate_token = accurate_mode_token_id(
            state.text_vocab,
            speaking_rate_num_buckets,
            quality_bucket_counts,
            speaker_background_num_buckets,
            accurate_mode_num_buckets,
        )
        rows.append(
            torch.tensor(
                [
                    _text_row(
                        accurate_token,
                        n_codebooks=state.n_codebooks,
                        audio_pad_id=state.audio_pad_id,
                    )
                ],
                dtype=torch.long,
            )
        )
    return rows


# ============================================================================
# Preprocessing and scheduler adapters
# ============================================================================


def _prepare_zonos2_request(payload: StagePayload) -> Zonos2PreparedRequest:
    ctx = get_zonos2_preprocessing_context()
    model_config = ctx.model_config
    state = build_zonos2_state(payload)

    state.n_codebooks = int(_get_model_attr(model_config, "n_codebooks", state.n_codebooks))
    state.codebook_size = int(_get_model_attr(model_config, "codebook_size", state.codebook_size))
    state.eoa_id = int(_get_model_attr(model_config, "eoa_id", state.eoa_id))
    state.audio_pad_id = int(_get_model_attr(model_config, "audio_pad_id", state.audio_pad_id))
    state.text_vocab = int(_get_model_attr(model_config, "text_vocab", state.text_vocab))

    (
        speaking_rate_bucket,
        quality_buckets,
        _,
        _,
    ) = _extract_conditioning_controls(payload, model_config)

    state.speaking_rate_bucket = speaking_rate_bucket
    state.quality_buckets = quality_buckets
    state.speaker_embedding = None

    speaking_rate_num_buckets = int(
        _get_model_attr(model_config, "speaking_rate_num_buckets", 0) or 0
    )
    quality_bucket_counts = _model_quality_bucket_counts(model_config)
    speaker_background_num_buckets = (
        2 if bool(_get_model_attr(model_config, "speaker_background_token_enabled", False)) else 0
    )
    accurate_mode_num_buckets = (
        1
        if (
            speaker_background_num_buckets
            and bool(_get_model_attr(model_config, "accurate_mode_token_enabled", False))
        )
        else 0
    )

    prompt_text = normalize_zonos2_text(
        state.text,
        language=state.language,
        enabled=state.text_normalization,
    )
    state.text = prompt_text

    prompt_rows = build_text_prompt_rows(
        prompt_text,
        n_codebooks=state.n_codebooks,
        audio_pad_id=state.audio_pad_id,
        text_vocab=state.text_vocab,
        speaking_rate_num_buckets=speaking_rate_num_buckets,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_bucket_counts=quality_bucket_counts,
        quality_buckets=quality_buckets,
        speaker_background_num_buckets=speaker_background_num_buckets,
        accurate_mode_num_buckets=accurate_mode_num_buckets,
    )
    prompt_tensor = torch.tensor(prompt_rows, dtype=torch.long)

    silence = build_silence_prefix(
        state.n_codebooks,
        state.audio_pad_id,
        state.text_vocab,
    ).to(dtype=torch.long)
    all_rows = [prompt_tensor, silence]
    prompt_tensor = torch.cat(all_rows, dim=0)

    state.prompt_tokens = int(prompt_tensor.shape[0])
    input_ids_list = build_row_cache_key_ids(prompt_tensor)
    return Zonos2PreparedRequest(
        state=state,
        prompt_rows=prompt_tensor,
        input_ids_list=input_ids_list,
        speaker_embedding=None,
        speaker_token_position=-1,
        speaker_cache_key=None,
        generation_kwargs=dict(state.generation_kwargs),
    )


def preprocess_zonos2_tts_payload(payload: StagePayload) -> StagePayload:
    """Preprocess a ZONOS2 TTS request outside the AR scheduler."""

    prepared = _prepare_zonos2_request(payload)
    return _store_prepared_zonos2_request(payload, prepared)


def encode_zonos2_speaker_payload(
    payload: StagePayload,
    *,
    speaker_model: Any = None,
) -> StagePayload:
    """Attach optional direct/ref-audio speaker embedding after preprocessing."""

    ctx = get_zonos2_preprocessing_context()
    model_config = ctx.model_config
    prepared = pop_prepared_zonos2_request(payload)
    state = prepared.state

    speaker_embedding = prepared.speaker_embedding
    if speaker_embedding is None:
        params = _as_dict(payload.request.params)
        metadata = _as_dict(payload.request.metadata)
        tts_params = _as_dict(metadata.get("tts_params"))
        data = _as_dict(payload.data)
        expected_speaker_dim = (
            int(_get_model_attr(model_config, "speaker_embedding_dim", 0) or 0)
            or None
        )
        speaker_embedding = _extract_direct_speaker_embedding(
            payload.request.inputs,
            params,
            tts_params,
            data,
            expected_dim=expected_speaker_dim,
        )
        if (
            speaker_embedding is None
            and state.ref_audio is not None
            and speaker_model is not None
        ):
            try:
                speaker_embedding = _compute_speaker_embedding(
                    speaker_model=speaker_model,
                    ref_audio=state.ref_audio,
                    expected_dim=expected_speaker_dim,
                )
            except Exception as exc:
                logger.warning("Failed to extract ZONOS2 speaker embedding: %s", exc)

    if (
        speaker_embedding is not None
        and prepared.speaker_token_position < 0
        and bool(_get_model_attr(model_config, "speaker_enabled", False))
    ):
        prefix_rows = _speaker_prefix_rows(payload, state, model_config)
        prepared.prompt_rows = torch.cat([*prefix_rows, prepared.prompt_rows], dim=0)
        prepared.input_ids_list = build_row_cache_key_ids(prepared.prompt_rows)
        prepared.speaker_token_position = 0
        state.prompt_tokens = int(prepared.prompt_rows.shape[0])

    prepared.speaker_embedding = speaker_embedding
    prepared.speaker_cache_key = _speaker_embedding_fingerprint(speaker_embedding)
    prepared.state = state
    return _store_prepared_zonos2_request(payload, prepared)


def make_zonos2_scheduler_adapters(*, model: Any) -> tuple[Any, Any]:
    """Create request_builder and result_adapter for the OmniScheduler."""

    n_codebooks = int(model.n_codebooks)

    def request_builder(payload: StagePayload) -> Zonos2TTSSGLangRequestData:
        prepared = pop_prepared_zonos2_request(payload)
        gen = prepared.generation_kwargs

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang_omni.models.zonos2_tts.radix_hash import RADIX_HASH_SPACE

        prompt_len = int(prepared.prompt_rows.shape[0])
        requested_max_new_tokens = gen.get("max_new_tokens")
        max_new_tokens = resolve_zonos2_max_new_tokens(
            model=model,
            prompt_len=prompt_len,
            requested=requested_max_new_tokens,
        )
        max_new_tokens = apply_zonos2_duration_safety_limit(
            max_new_tokens,
            text=prepared.state.text,
            requested=requested_max_new_tokens,
        )
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        sampling_params.normalize(None)
        sampling_params.verify(RADIX_HASH_SPACE)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prepared.input_ids_list,
            sampling_params=sampling_params,
            eos_token_ids=set(),
            vocab_size=RADIX_HASH_SPACE,
            extra_key=prepared.speaker_cache_key,
        )
        req.tokenizer = None
        req._input_embeds_are_projected = True

        req_data = Zonos2TTSSGLangRequestData(
            stage_payload=payload,
            state=prepared.state,
            prompt_rows=prepared.prompt_rows.to(dtype=torch.long),
            speaker_embedding=prepared.speaker_embedding,
            speaker_token_position=int(prepared.speaker_token_position),
            temperature=float(gen.get("temperature", ZONOS2_DEFAULT_TEMPERATURE)),
            top_k=int(gen.get("top_k", ZONOS2_DEFAULT_TOP_K)),
            top_p=float(gen.get("top_p", ZONOS2_DEFAULT_TOP_P)),
            min_p=float(gen.get("min_p", ZONOS2_DEFAULT_MIN_P)),
            repetition_penalty=float(
                gen.get("repetition_penalty", ZONOS2_DEFAULT_REPETITION_PENALTY)
            ),
            repetition_window=int(
                gen.get("repetition_window", ZONOS2_DEFAULT_REPETITION_WINDOW)
            ),
            repetition_codebooks=int(
                gen.get("repetition_codebooks", ZONOS2_DEFAULT_REPETITION_CODEBOOKS)
            ),
            max_new_tokens=max_new_tokens,
            seed=gen.get("seed"),
            ignore_eos=bool(gen.get("ignore_eos", False)),
            engine_start_s=time.time(),
            req=req,
        )
        req_data.input_ids = torch.tensor(prepared.input_ids_list, dtype=torch.long)
        req_data.input_embeds_are_projected = True
        req_data.prompt_len = prompt_len
        req_data.max_output_len = max_new_tokens
        return req_data

    def result_adapter(req_data: Zonos2TTSSGLangRequestData) -> StagePayload:
        payload = req_data.stage_payload
        state = req_data.state
        if req_data.output_rows:
            audio_codes = torch.stack(req_data.output_rows, dim=0)[:, :n_codebooks]
            state.audio_codes = audio_codes.cpu()
        else:
            state.audio_codes = None
        if req_data.eos_frame >= 0:
            state.eos_frame = req_data.eos_frame
        state.completion_tokens = len(req_data.output_rows)
        if req_data.engine_start_s > 0:
            state.engine_time_s = time.time() - req_data.engine_start_s
        payload.data = state.to_dict()
        reset_request = getattr(model, "reset_request", None)
        if reset_request is not None and req_data.req is not None:
            reset_request(req_data.req.rid)
        return payload

    return request_builder, result_adapter
