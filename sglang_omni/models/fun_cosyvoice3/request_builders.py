# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for Fun-CosyVoice3."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.preprocessing.cache_key import hash_bytes as _hash_bytes
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import decode_audio_data_uri as _decode_audio_data_uri
from sglang_omni.utils.audio import load_audio as _shared_load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

from .sglang_model import CONTROL_TOKEN_IDS, EOS_ID, SOS_ID, TASK_ID, TOTAL_VOCAB_SIZE
from .utils import (
    CosyVoice3Tokenizer,
    SpeakerEncoder,
    SpeechTokenizerV3,
    build_llm_prompt_embeddings,
    extract_prompt_speech_feat,
)

_SAMPLE_RATE = 24000
_PROMPT_AUDIO_SR = 16000
_DEFAULT_MAX_NEW_TOKENS = 2048
_COSYVOICE3_SAMPLING_SEED_MASK = SAMPLING_SEED_MASK
_COSYVOICE3_END_OF_PROMPT = "<|endofprompt|>"
_COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_COSYVOICE3_INSTRUCT_PREFIX = "You are a helpful assistant. "

_GENERATION_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
)

_IMPLICIT_SAMPLING_DEFAULTS = {
    "temperature": {1.0, 0.8, 0.7},
    "top_p": {1.0, 0.8},
    "top_k": {-1, 20, 25, 30},
    "repetition_penalty": {1.0, 1.05, 1.1},
}

_COSYVOICE3_PREPARED_MARKER = "_cosyvoice3_prepared_request"


class _CosyVoice3NullTokenizer:
    """Tokenizer shim so SGLang's ``min_new_tokens`` stop-suppression
    penalizer can run without a real HF tokenizer attached to the request.

    CosyVoice3 speech tokens are sampled directly from the codec vocabulary
    (``TOTAL_VOCAB_SIZE``) and are never decoded to text for this request, so
    the only contract this object needs to satisfy is the attribute access
    performed by ``SamplingParams.normalize``/``verify`` and
    ``BatchedMinNewTokensPenalizer`` (``eos_token_id``,
    ``additional_stop_token_ids``).

    eos_token_id is the real EOS_ID rather than None: BatchedMinNewTokensPenalizer
    unions it into the stop-id set it suppresses until min_new_tokens is
    reached, and EOS_ID is already part of stop_token_ids and
    CONTROL_TOKEN_IDS, so the tokenizer stand-in reports the same EOS the
    request stops on.
    """

    eos_token_id: int = EOS_ID
    additional_stop_token_ids: set[int] | None = None


_COSYVOICE3_NULL_TOKENIZER = _CosyVoice3NullTokenizer()


def _cosyvoice3_model_revision(model: Any) -> str:
    """Return a stable-enough checkpoint identity for the process-local cache."""
    config = getattr(model, "config", None)
    for candidate in (
        getattr(model, "name_or_path", None),
        getattr(config, "_name_or_path", None),
        getattr(config, "name_or_path", None),
    ):
        if candidate:
            return str(candidate)
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _cosyvoice3_reference_input_key(source: Any) -> str | None:
    """Build a cache key without allowing mutable URLs to alias audio bytes."""
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, str):
        if source.startswith("data:"):
            try:
                data = _decode_audio_data_uri(source)
            except ValueError:
                # Invalid data URIs must still fail in ``encode_one``. Returning
                # no key prevents an invalid request from poisoning the cache.
                return None
            if data is None:
                return None
            return f"data:{_hash_bytes(data)}"
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            return None
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                return None
            source = unquote(parsed.path)
        return _reference_path_cache_key(source)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return f"bytes:{_hash_bytes(bytes(source))}"
    if isinstance(source, dict):
        path = source.get("audio_path") or source.get("path")
        if path is not None:
            return _cosyvoice3_reference_input_key(path)
        data_uri = audio_data_uri_from_reference(source)
        if data_uri is not None:
            return _cosyvoice3_reference_input_key(data_uri)
    if isinstance(source, np.ndarray):
        array = np.ascontiguousarray(source)
        metadata = f"numpy:{array.dtype}:{array.shape}:".encode("utf-8")
        return f"numpy:{_hash_bytes(metadata + array.tobytes())}"
    if isinstance(source, torch.Tensor):
        tensor = source.detach().cpu().contiguous()
        metadata = f"torch:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8")
        return f"torch:{_hash_bytes(metadata + tensor.numpy().tobytes())}"
    return None


def _clone_reference_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu().clone()


@dataclass
class CosyVoice3SGLangRequestData(SGLangARRequestData):
    """Fun-CosyVoice3 scheduler-owned request state."""

    output_codes: list[torch.Tensor] = None
    prompt_input_embeds: torch.Tensor | None = None
    engine_start_s: float = 0.0

    def __post_init__(self):
        if self.output_codes is None:
            self.output_codes = []


@dataclass
class CosyVoice3PreparedRequest:
    """Heavy CosyVoice3 preprocessing output consumed by the AR scheduler."""

    state: FunCosyVoice3State
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_input_embeds: torch.Tensor
    llm_prompt_speech_token: torch.Tensor
    flow_prompt_speech_token: torch.Tensor
    flow_prompt_speech_feat: torch.Tensor
    flow_embedding: torch.Tensor
    gen_kwargs: dict[str, Any]
    # Target text token count, captured before the reference prompt (ref_text
    # / instructions / cross-lingual prefix) is concatenated. CosyVoice3's
    # upstream generation-length contract is defined in terms of this count,
    # not the full prompt length.
    target_text_token_count: int = 0


@dataclass(frozen=True)
class _CosyVoice3ReferenceInput:
    """Reference audio passed to the content-addressed encoder cache."""

    ref_audio: Any


@dataclass
class _CosyVoice3ReferenceArtifact:
    """Audio-derived tensors shared by all requests using one reference clip."""

    llm_prompt_speech_token: torch.Tensor
    flow_prompt_speech_token: torch.Tensor
    flow_prompt_speech_feat: torch.Tensor
    flow_embedding: torch.Tensor


class _CosyVoice3ReferenceEncodeHook(
    KeyedReferenceEncodeHook[
        _CosyVoice3ReferenceInput,
        _CosyVoice3ReferenceArtifact,
        dict[str, Any],
    ]
):

    model_id = "fun_cosyvoice3"
    encoder_id = "speech_tokenizer_v3+campplus+matcha_mel"
    artifact_kind = "reference_conditioning"

    def __init__(self, *, model: Any, model_revision: str | None = None) -> None:
        self._speech_tokenizer = None
        self._speaker_encoder = None
        self.model_revision = model_revision or _cosyvoice3_model_revision(model)
        self.encoder_config_hash = _hash_bytes(
            json.dumps(
                {
                    "prompt_audio_sr": _PROMPT_AUDIO_SR,
                    "flow_audio_sr": _SAMPLE_RATE,
                    "mel": {
                        "n_fft": 1920,
                        "hop_size": 480,
                        "win_size": 1920,
                        "num_mels": 80,
                        "fmin": 0,
                        "fmax": None,
                        "center": False,
                    },
                    "alignment": "flow_frames=2*flow_tokens",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def bind_encoders(
        self,
        *,
        speech_tokenizer: SpeechTokenizerV3,
        speaker_encoder: SpeakerEncoder,
    ) -> None:
        """Attach the process-local ONNX encoders after hook construction."""
        self._speech_tokenizer = speech_tokenizer
        self._speaker_encoder = speaker_encoder

    def normalize_input(self, raw_input: Any) -> _CosyVoice3ReferenceInput:
        if isinstance(raw_input, _CosyVoice3ReferenceInput):
            return raw_input
        return _CosyVoice3ReferenceInput(ref_audio=raw_input)

    def input_key(self, item: _CosyVoice3ReferenceInput) -> str | None:
        return _cosyvoice3_reference_input_key(item.ref_audio)

    def options_key(self, item: _CosyVoice3ReferenceInput) -> str:
        del item
        return "prompt_16k+flow_24k+mono"

    def encode_one(
        self, item: _CosyVoice3ReferenceInput
    ) -> _CosyVoice3ReferenceArtifact:
        if self._speech_tokenizer is None or self._speaker_encoder is None:
            raise RuntimeError("CosyVoice3 reference encoders are not bound")

        prompt_audio_16k = _load_prompt_audio(item.ref_audio)
        prompt_audio_24k = _load_prompt_audio_24k(item.ref_audio)
        speaker_embedding = self._speaker_encoder.extract_embedding(
            prompt_audio_16k, _PROMPT_AUDIO_SR
        )
        prompt_speech_token = self._speech_tokenizer.extract_speech_token(
            prompt_audio_16k, _PROMPT_AUDIO_SR
        )
        prompt_speech_feat = extract_prompt_speech_feat(prompt_audio_24k, _SAMPLE_RATE)
        flow_prompt_speech_token, flow_prompt_speech_feat = _align_flow_prompt(
            prompt_speech_token, prompt_speech_feat
        )
        return _CosyVoice3ReferenceArtifact(
            llm_prompt_speech_token=prompt_speech_token,
            flow_prompt_speech_token=flow_prompt_speech_token,
            flow_prompt_speech_feat=flow_prompt_speech_feat,
            flow_embedding=speaker_embedding,
        )

    def store_artifact(self, artifact: _CosyVoice3ReferenceArtifact) -> dict[str, Any]:
        return {
            "artifact_type": "fun_cosyvoice3_reference_conditioning",
            "llm_prompt_speech_token": _clone_reference_tensor(
                artifact.llm_prompt_speech_token
            ),
            "flow_prompt_speech_token": _clone_reference_tensor(
                artifact.flow_prompt_speech_token
            ),
            "flow_prompt_speech_feat": _clone_reference_tensor(
                artifact.flow_prompt_speech_feat
            ),
            "flow_embedding": _clone_reference_tensor(artifact.flow_embedding),
        }

    def load_artifact(self, stored: dict[str, Any]) -> _CosyVoice3ReferenceArtifact:
        if stored.get("artifact_type") != "fun_cosyvoice3_reference_conditioning":
            raise RuntimeError("CosyVoice3 reference cache entry is invalid")
        fields = (
            "llm_prompt_speech_token",
            "flow_prompt_speech_token",
            "flow_prompt_speech_feat",
            "flow_embedding",
        )
        if any(field not in stored for field in fields):
            raise RuntimeError("CosyVoice3 reference cache entry is incomplete")
        return _CosyVoice3ReferenceArtifact(
            **{field: _clone_reference_tensor(stored[field]) for field in fields}
        )


@dataclass
class CosyVoice3PreprocessingContext:
    model: Any
    tokenizer: CosyVoice3Tokenizer
    speech_tokenizer: SpeechTokenizerV3
    speaker_encoder: SpeakerEncoder
    reference_service: ReferenceEncodeService


_PREPROCESSING_CONTEXT: CosyVoice3PreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, CosyVoice3PreparedRequest] = {}
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_cosyvoice3_preprocessing_context(
    *,
    model: Any,
    tokenizer: CosyVoice3Tokenizer,
    speech_tokenizer: SpeechTokenizerV3,
    speaker_encoder: SpeakerEncoder,
    model_revision: str | None = None,
) -> None:
    """Register model objects used by the preprocessing stage."""
    global _PREPROCESSING_CONTEXT
    reference_hook = _CosyVoice3ReferenceEncodeHook(
        model=model,
        model_revision=model_revision,
    )
    reference_hook.bind_encoders(
        speech_tokenizer=speech_tokenizer,
        speaker_encoder=speaker_encoder,
    )
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = CosyVoice3PreprocessingContext(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
            reference_service=ReferenceEncodeService(
                reference_hook,
                max_items=256,
                max_bytes=128 * 1024 * 1024,
                timeout_s=130.0,
                log_prefix="Fun-CosyVoice3",
            ),
        )
        _PREPARED_REQUESTS.clear()


def clear_cosyvoice3_preprocessing_context() -> None:
    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = None
        _PREPARED_REQUESTS.clear()


def cleanup_prepared_cosyvoice3_request(request_id: str) -> None:
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS.pop(str(request_id), None)


def _prepared_request_id(payload: StagePayload) -> str | None:
    data = payload.data
    if not isinstance(data, dict):
        return None
    marker = data.get(_COSYVOICE3_PREPARED_MARKER)
    return str(marker) if marker is not None else None


def pop_prepared_cosyvoice3_request(
    payload: StagePayload,
) -> CosyVoice3PreparedRequest | None:
    prepared_request_id = _prepared_request_id(payload)
    if prepared_request_id is None:
        return None
    with _PREPARED_REQUESTS_LOCK:
        prepared = _PREPARED_REQUESTS.pop(prepared_request_id, None)
    if prepared is None:
        raise RuntimeError(
            "CosyVoice3 preprocessing state is missing for prepared payload "
            f"{prepared_request_id!r}"
        )
    return prepared


def _load_prompt_audio(source: Any) -> np.ndarray:
    return _shared_load_audio(
        source,
        source_name="Fun-CosyVoice3",
        target_sample_rate=_PROMPT_AUDIO_SR,
    )


def _load_prompt_audio_24k(source: Any) -> np.ndarray:
    """Load the same reference audio at the Flow mel sample rate."""
    return _shared_load_audio(
        source,
        source_name="Fun-CosyVoice3",
        target_sample_rate=_SAMPLE_RATE,
    )


def build_cosyvoice3_state(payload: StagePayload) -> FunCosyVoice3State:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references, input_ref_audio, input_ref_text = _normalize_cosyvoice3_inputs(
        inputs
    )
    reference = references[0] if references else {}
    ref_audio = (
        input_ref_audio
        or reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or audio_data_uri_from_reference(reference)
        or tts_params.get("ref_audio")
    )
    ref_text = input_ref_text or reference.get("text") or tts_params.get("ref_text")

    language = normalize_language(tts_params.get("language") or params.get("language"))
    instructions = resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    ref_text = resolve_optional_text(ref_text)
    if instructions is not None and ref_text is not None:
        raise ValueError(
            "Fun-CosyVoice3 accepts either ref_text or instructions, not both"
        )
    stream = bool(tts_params.get("stream", params.get("stream", False)))
    speed = float(tts_params.get("speed", params.get("speed", 1.0)))
    seed_raw = tts_params.get("seed", params.get("seed"))
    seed = _normalize_cosyvoice3_seed(seed_raw) if seed_raw is not None else None

    return FunCosyVoice3State(
        text=text,
        language=language,
        instructions=instructions,
        ref_audio=ref_audio,
        ref_text=ref_text,
        stream=stream,
        speed=speed,
        seed=seed,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def _normalize_cosyvoice3_inputs(
    inputs: Any,
) -> tuple[str, list[dict[str, Any]], Any | None, str | None]:
    """Normalize flat and structured speech-reference request shapes."""
    if isinstance(inputs, str):
        return inputs, [], None, None
    if not isinstance(inputs, dict):
        return str(inputs) if inputs is not None else "", [], None, None

    raw_references = inputs.get("references") or []
    if not isinstance(raw_references, list):
        raise ValueError("Fun-CosyVoice3 references must be a list")
    if len(raw_references) > 1:
        raise ValueError("Fun-CosyVoice3 accepts exactly one reference")
    if any(not isinstance(reference, dict) for reference in raw_references):
        raise ValueError("Fun-CosyVoice3 references must be objects")
    references = [dict(reference) for reference in raw_references]
    return (
        str(inputs.get("text", inputs.get("input", ""))),
        references,
        inputs.get("ref_audio") or inputs.get("audio") or inputs.get("file"),
        inputs.get("ref_text"),
    )


def normalize_language(language: Any) -> str:
    if language is None or language == "":
        return "auto"
    return str(language)


def _normalize_cosyvoice3_seed(seed: Any) -> int:
    """Convert a public seed to SGLang's positive int32 seed domain."""
    if isinstance(seed, bool):
        raise ValueError("Fun-CosyVoice3 seed must be an integer")
    if isinstance(seed, float) and not seed.is_integer():
        raise ValueError("Fun-CosyVoice3 seed must be an integer")
    try:
        normalized = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fun-CosyVoice3 seed must be an integer") from exc
    return normalized & _COSYVOICE3_SAMPLING_SEED_MASK


def resolve_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_cosyvoice3_prompt_text(state: FunCosyVoice3State) -> str | None:
    """Format the mode-specific text that precedes the synthesis target."""
    if state.instructions is not None:
        if _COSYVOICE3_END_OF_PROMPT in state.instructions:
            return state.instructions
        return (
            _COSYVOICE3_INSTRUCT_PREFIX + state.instructions + _COSYVOICE3_END_OF_PROMPT
        )
    if state.ref_text is not None:
        if _COSYVOICE3_END_OF_PROMPT in state.ref_text:
            return state.ref_text
        return _COSYVOICE3_PROMPT_PREFIX + state.ref_text
    # note (PoTaTo): CosyVoice3 requires the end-of-prompt marker even when cross-lingual
    # mode removes reference text and reference speech tokens from the LLM.
    return _COSYVOICE3_PROMPT_PREFIX


def _align_flow_prompt(
    prompt_speech_token: torch.Tensor,
    prompt_speech_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim Flow conditioning to its required two-mel-frames-per-token ratio."""
    if prompt_speech_token.ndim != 2 or prompt_speech_feat.ndim != 3:
        raise ValueError(
            "CosyVoice3 prompt conditioning must have shapes [1, tokens] and "
            "[1, frames, 80]"
        )
    if prompt_speech_feat.shape[-1] != 80:
        raise ValueError("CosyVoice3 prompt mel conditioning must have 80 bins")
    token_count = min(
        int(prompt_speech_token.shape[1]),
        int(prompt_speech_feat.shape[1] // 2),
    )
    return (
        prompt_speech_token[:, :token_count].contiguous(),
        prompt_speech_feat[:, : 2 * token_count, :].contiguous(),
    )


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

    # note: max_new_tokens is intentionally left out of generation_kwargs
    # unless the caller explicitly passed one. CosyVoice3's own generation
    # contract derives max_new_tokens (and the minimum length before stop
    # tokens are honored) from the target text length; build_sglang_
    # cosyvoice3_request only falls back to _DEFAULT_MAX_NEW_TOKENS when
    # neither the caller nor the contract provides a bound.
    generation_kwargs: dict[str, Any] = {}
    max_new_tokens = params.get("max_new_tokens")
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = int(max_new_tokens)
    for field in _GENERATION_FIELDS:
        if field == "max_new_tokens":
            continue
        if field in selected_fields and params.get(field) is not None:
            generation_kwargs[field] = params[field]
    return generation_kwargs


def build_embedding_cache_key_ids(input_embeds: torch.Tensor) -> list[int]:
    rows = input_embeds.detach().to(dtype=torch.float32, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def _prepare_cosyvoice3_request(
    payload: StagePayload,
    *,
    model: Any,
    tokenizer: CosyVoice3Tokenizer,
    reference_service: ReferenceEncodeService,
) -> CosyVoice3PreparedRequest:
    state = build_cosyvoice3_state(payload)
    gen_kwargs = state.generation_kwargs
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    text_token = tokenizer(state.text)
    text_token = text_token.to(device=device)
    target_text_token_count = int(text_token.shape[1])
    prompt_text = _build_cosyvoice3_prompt_text(state)
    if prompt_text is not None:
        prompt_text_token = tokenizer(prompt_text).to(device=device)
        text_token = torch.cat([prompt_text_token, text_token], dim=1)
    with torch.no_grad():
        text_embed = model.text_embed_tokens(text_token)

    if state.ref_audio is not None:
        reference_artifact = reference_service.get_or_encode(
            state.ref_audio,
            desc="Fun-CosyVoice3 reference conditioning",
        )
        spk_embedding = reference_artifact.flow_embedding
        prompt_speech_token = reference_artifact.llm_prompt_speech_token
        flow_prompt_speech_token = reference_artifact.flow_prompt_speech_token
        flow_prompt_speech_feat = reference_artifact.flow_prompt_speech_feat
    else:
        spk_embedding = torch.zeros(0, 192)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        flow_prompt_speech_feat = torch.zeros(1, 0, 80)

    llm_prompt_speech_token = (
        prompt_speech_token
        if state.ref_text is not None
        else torch.zeros(1, 0, dtype=torch.int32)
    )

    flow_embedding = (
        spk_embedding.clone() if spk_embedding.numel() > 0 else spk_embedding
    )

    # build llm prompt embeddings: [sos, spk_emb, text_embed, task, prompt_speech]
    prompt_input_embeds = build_llm_prompt_embeddings(
        text_token=text_token,
        text_embed=text_embed,
        prompt_speech_token=llm_prompt_speech_token,
        speech_embed=model.speech_embedding,
        embedding=spk_embedding,
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=model.config.hidden_size,
        device=device,
        dtype=dtype,
    )

    prompt_input_embeds = (
        prompt_input_embeds.squeeze(0).detach().to(device=device, dtype=dtype)
    )
    input_ids_list = build_embedding_cache_key_ids(prompt_input_embeds)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    return CosyVoice3PreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=input_ids,
        prompt_input_embeds=prompt_input_embeds,
        llm_prompt_speech_token=llm_prompt_speech_token,
        flow_prompt_speech_token=flow_prompt_speech_token,
        flow_prompt_speech_feat=flow_prompt_speech_feat,
        flow_embedding=flow_embedding,
        gen_kwargs=gen_kwargs,
        target_text_token_count=target_text_token_count,
    )


def preprocess_cosyvoice3_payload(payload: StagePayload) -> StagePayload:
    with _PREPARED_REQUESTS_LOCK:
        context = _PREPROCESSING_CONTEXT
    if context is None:
        raise RuntimeError("CosyVoice3 preprocessing context is not initialized")

    prepared = _prepare_cosyvoice3_request(
        payload,
        model=context.model,
        tokenizer=context.tokenizer,
        reference_service=context.reference_service,
    )
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS[payload.request_id] = prepared

    prepared.state.flow_embedding = prepared.flow_embedding
    prepared.state.flow_prompt_speech_token = prepared.flow_prompt_speech_token
    prepared.state.flow_prompt_speech_feat = prepared.flow_prompt_speech_feat

    data = prepared.state.to_dict()
    data[_COSYVOICE3_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def build_sglang_cosyvoice3_request(
    payload: StagePayload,
    *,
    model: Any,
) -> CosyVoice3SGLangRequestData:
    prepared = pop_prepared_cosyvoice3_request(payload)
    if prepared is None:
        raise RuntimeError(
            "CosyVoice3 AR request builder requires a payload prepared by "
            "preprocess_cosyvoice3_payload"
        )

    gen_kwargs = prepared.gen_kwargs
    state = prepared.state
    do_sample = bool(gen_kwargs.get("do_sample", True))
    temperature = float(gen_kwargs.get("temperature", 0.7)) if do_sample else 0.0

    # CosyVoice3's own generation-length contract: stop tokens are suppressed
    # until at least 2x the target text token count has been generated, and
    # generation is capped at 20x that count (itself capped at the service
    # default). target_text_token_count is captured before the reference
    # prompt (ref_text / instructions) is concatenated, matching upstream.
    target_text_tokens = max(int(prepared.target_text_token_count), 1)
    contract_max_new_tokens = min(_DEFAULT_MAX_NEW_TOKENS, 20 * target_text_tokens)
    explicit_max_new_tokens = gen_kwargs.get("max_new_tokens")
    max_new_tokens = (
        int(explicit_max_new_tokens)
        if explicit_max_new_tokens is not None
        else contract_max_new_tokens
    )
    min_new_tokens = min(2 * target_text_tokens, max_new_tokens)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=float(gen_kwargs.get("top_p", 0.8)),
        top_k=int(gen_kwargs.get("top_k", 20)),
        repetition_penalty=float(gen_kwargs.get("repetition_penalty", 1.1)),
        # Stop on any of the 200 ids CosyVoice3 treats as terminal/non-speech
        # control ids, not only EOS_ID — the other 199 never stopped the
        # scheduler when only EOS_ID was registered, so generation could run
        # on after a non-EOS control token was emitted.
        stop_token_ids=list(CONTROL_TOKEN_IDS),
        sampling_seed=state.seed,
    )
    sampling_params.normalize(_COSYVOICE3_NULL_TOKENIZER)
    sampling_params.verify(TOTAL_VOCAB_SIZE)

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        vocab_size=TOTAL_VOCAB_SIZE,
    )
    req.tokenizer = _COSYVOICE3_NULL_TOKENIZER
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    data = CosyVoice3SGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
        prompt_input_embeds=prepared.prompt_input_embeds,
        engine_start_s=time.perf_counter(),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_cosyvoice3_result(
    payload: StagePayload,
    data: CosyVoice3SGLangRequestData,
) -> StagePayload:
    code_parts: list[torch.Tensor] = []
    if data.output_codes:
        code_parts.append(torch.stack(data.output_codes, dim=0).to(dtype=torch.long))

    if code_parts:
        device = code_parts[0].device
        codes = torch.cat(
            [part.to(device=device, dtype=torch.long) for part in code_parts],
            dim=0,
        ).cpu()
    else:
        codes = torch.empty((0,), dtype=torch.long)

    state = FunCosyVoice3State.from_dict(payload.data)
    state.audio_codes = codes
    state.prompt_tokens = (
        int(data.input_ids.numel()) if data.input_ids is not None else 0
    )
    state.completion_tokens = len(data.output_codes)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.sample_rate = _SAMPLE_RATE

    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_cosyvoice3_scheduler_adapters(*, model: Any):
    def request_builder(payload: StagePayload) -> CosyVoice3SGLangRequestData:
        return build_sglang_cosyvoice3_request(payload, model=model)

    def result_adapter(data: CosyVoice3SGLangRequestData) -> StagePayload:
        return apply_sglang_cosyvoice3_result(data.stage_payload, data)

    return request_builder, result_adapter
