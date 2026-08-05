# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for Fun-CosyVoice3."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import Req, MultimodalDataItem, Modality, MultimodalInputs
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.preprocessing.cache_key import hash_bytes as _hash_bytes
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int
from sglang_omni.utils.audio import load_audio as _shared_load_audio

from .sglang_model import EOS_ID, SOS_ID, TASK_ID, FILL_ID, TOTAL_VOCAB_SIZE
from .utils import CosyVoice3Tokenizer, SpeakerEncoder, SpeechTokenizerV3, build_llm_prompt_embeddings

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000
_PROMPT_AUDIO_SR = 16000
_MAX_PROMPT_AUDIO_S = 30.0
_DEFAULT_MAX_NEW_TOKENS = 2048

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
    prompt_speech_token: torch.Tensor
    prompt_speech_feat: torch.Tensor
    flow_embedding: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class CosyVoice3PreprocessingContext:
    model: Any
    tokenizer: CosyVoice3Tokenizer
    speech_tokenizer: SpeechTokenizerV3
    speaker_encoder: SpeakerEncoder


_PREPROCESSING_CONTEXT: CosyVoice3PreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, CosyVoice3PreparedRequest] = {}
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_cosyvoice3_preprocessing_context(
    *,
    model: Any,
    tokenizer: CosyVoice3Tokenizer,
    speech_tokenizer: SpeechTokenizerV3,
    speaker_encoder: SpeakerEncoder,
) -> None:
    """Register model objects used by the preprocessing stage."""
    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = CosyVoice3PreprocessingContext(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
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


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("ref_audio", "audio", "bytes", "file", "path"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def _load_prompt_audio(source: Any) -> np.ndarray:
    return _shared_load_audio(
        source,
        source_name="Fun-CosyVoice3",
        target_sample_rate=_PROMPT_AUDIO_SR,
    )


def build_cosyvoice3_state(payload: StagePayload) -> FunCosyVoice3State:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    if isinstance(inputs, str):
        text = inputs
        ref_audio = None
    elif isinstance(inputs, dict):
        text = inputs.get("text", inputs.get("input", ""))
        ref_audio = (
            inputs.get("ref_audio")
            or inputs.get("audio")
            or inputs.get("file")
        )
    else:
        text = str(inputs) if inputs is not None else ""
        ref_audio = None

    ref_audio = ref_audio or tts_params.get("ref_audio")
    ref_text = (
        inputs.get("ref_text") if isinstance(inputs, dict) else None
    ) or tts_params.get("ref_text")

    language = normalize_language(
        tts_params.get("language") or params.get("language")
    )
    instructions = resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    stream = bool(
        tts_params.get("stream", params.get("stream", False))
    )
    speed = float(
        tts_params.get("speed", params.get("speed", 1.0))
    )
    seed_raw = tts_params.get("seed", params.get("seed"))
    seed = int(seed_raw) if seed_raw is not None else None

    return FunCosyVoice3State(
        text=str(text),
        language=language,
        instructions=instructions,
        ref_audio=ref_audio,
        ref_text=str(ref_text) if ref_text is not None else None,
        stream=stream,
        speed=speed,
        seed=seed,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def normalize_language(language: Any) -> str:
    if language is None or language == "":
        return "auto"
    return str(language)


def resolve_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        max_new_tokens = _DEFAULT_MAX_NEW_TOKENS
    generation_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
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
    speech_tokenizer: SpeechTokenizerV3,
    speaker_encoder: SpeakerEncoder,
) -> CosyVoice3PreparedRequest:
    state = build_cosyvoice3_state(payload)
    gen_kwargs = state.generation_kwargs
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # tokenize text
    text_token = tokenizer(state.text)
    text_token = text_token.to(device=device)
    with torch.no_grad():
        text_embed = model.text_embed_tokens(text_token)

    # extract speaker embedding and prompt speech tokens from reference audio
    if state.ref_audio is not None:
        prompt_audio = _load_prompt_audio(state.ref_audio)
        spk_embedding = speaker_encoder.extract_embedding(prompt_audio, _PROMPT_AUDIO_SR)
        prompt_speech_token = speech_tokenizer.extract_speech_token(
            prompt_audio, _PROMPT_AUDIO_SR
        )
        prompt_speech_feat = torch.zeros(1, 0, 80)
    else:
        spk_embedding = torch.zeros(0, 192)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        prompt_speech_feat = torch.zeros(1, 0, 80)

    flow_embedding = spk_embedding.clone() if spk_embedding.numel() > 0 else spk_embedding

    # build llm prompt embeddings: [sos, spk_emb, text_embed, task, prompt_speech]
    prompt_input_embeds = build_llm_prompt_embeddings(
        text_token=text_token,
        text_embed=text_embed,
        prompt_speech_token=prompt_speech_token,
        speech_embed=model.speech_embedding,
        embedding=spk_embedding,
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=model.config.hidden_size,
        device=device,
        dtype=dtype,
    )

    prompt_input_embeds = prompt_input_embeds.squeeze(0).detach().to(
        device=device, dtype=dtype
    )
    input_ids_list = build_embedding_cache_key_ids(prompt_input_embeds)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    return CosyVoice3PreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=input_ids,
        prompt_input_embeds=prompt_input_embeds,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_feat=prompt_speech_feat,
        flow_embedding=flow_embedding,
        gen_kwargs=gen_kwargs,
    )


def preprocess_cosyvoice3_payload(payload: StagePayload) -> StagePayload:
    with _PREPARED_REQUESTS_LOCK:
        context = _PREPROCESSING_CONTEXT
    if context is None:
        raise RuntimeError(
            "CosyVoice3 preprocessing context is not initialized"
        )

    prepared = _prepare_cosyvoice3_request(
        payload,
        model=context.model,
        tokenizer=context.tokenizer,
        speech_tokenizer=context.speech_tokenizer,
        speaker_encoder=context.speaker_encoder,
    )
    with _PREPARED_REQUESTS_LOCK:
        _PREPARED_REQUESTS[payload.request_id] = prepared

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
    do_sample = bool(gen_kwargs.get("do_sample", True))
    temperature = float(gen_kwargs.get("temperature", 0.7)) if do_sample else 0.0
    max_new_tokens = int(gen_kwargs.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=float(gen_kwargs.get("top_p", 0.8)),
        top_k=int(gen_kwargs.get("top_k", 20)),
        repetition_penalty=float(gen_kwargs.get("repetition_penalty", 1.1)),
        stop_token_ids=[EOS_ID],
    )
    sampling_params.normalize(None)
    sampling_params.verify(TOTAL_VOCAB_SIZE)

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        vocab_size=TOTAL_VOCAB_SIZE,
    )
    req.tokenizer = None
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
        code_parts.append(
            torch.stack(data.output_codes, dim=0).to(dtype=torch.long)
        )

    if code_parts:
        device = code_parts[0].device
        codes = torch.cat(
            [part.to(device=device, dtype=torch.long) for part in code_parts],
            dim=0,
        ).cpu()
    else:
        codes = torch.empty((0,), dtype=torch.long)

    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={
            "audio_codes": codes,
            "completion_tokens": len(data.output_codes),
            "engine_time_s": time.perf_counter() - data.engine_start_s,
            "sample_rate": _SAMPLE_RATE,
        },
    )


def make_cosyvoice3_scheduler_adapters(*, model: Any):
    def request_builder(payload: StagePayload) -> CosyVoice3SGLangRequestData:
        return build_sglang_cosyvoice3_request(payload, model=model)

    def result_adapter(data: CosyVoice3SGLangRequestData) -> StagePayload:
        return apply_sglang_cosyvoice3_result(data.stage_payload, data)

    return request_builder, result_adapter
