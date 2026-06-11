# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    MOSS_TTS_DEFAULT_MAX_NEW_TOKENS,
    _new_moss_tts_sampling_seed,
    _reference_for_processor,
    _resolve_optional_text,
    _resolve_token_count,
    _validate_moss_tts_generation_kwargs,
    build_row_cache_key_ids,
    derive_moss_tts_sampling_seed,
    normalize_moss_tts_inputs,
    resolve_moss_reference,
)
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData

_MOSS_TTS_LOCAL_PREPARED_MARKER = "_moss_tts_local_prepared_request"


@dataclass
class MossTTSLocalSGLangRequestData(ARRequestData):
    """Scheduler-owned request state for MOSS-TTS Local."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    state: MossTTSLocalState = field(default_factory=MossTTSLocalState)
    model_config: Any = None
    prompt_rows: torch.Tensor | None = None
    output_rows: list[torch.Tensor] = field(default_factory=list)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    # Checkpoint generate() defaults: the binary continue/stop head samples at
    # plain temperature 1.0 while the audio channels use the model-card
    # recommendation (1.7 / 0.8 / 25, repetition penalty off).
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0
    seed: int | None = None
    sampling_seed: int = field(default_factory=_new_moss_tts_sampling_seed)
    engine_start_s: float = 0.0


@dataclass
class MossTTSLocalPreparedRequest:
    """Heavy preprocessing output consumed by the AR scheduler."""

    state: MossTTSLocalState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_rows: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class _PreprocessingContext:
    processor: Any
    reference_encoder: Any = None


_PREPROCESSING_CONTEXT: _PreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, MossTTSLocalPreparedRequest] = {}
_INFLIGHT_REQUESTS: set[str] = set()
_ABORTED_REQUESTS: set[str] = set()
_PREPARED_REQUESTS_LOCK = threading.Lock()


def set_moss_tts_local_preprocessing_context(
    *, processor: Any, reference_encoder: Any = None
) -> None:
    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = _PreprocessingContext(
            processor=processor, reference_encoder=reference_encoder
        )
        _PREPARED_REQUESTS.clear()
        _INFLIGHT_REQUESTS.clear()
        _ABORTED_REQUESTS.clear()


def clear_moss_tts_local_preprocessing_context() -> None:
    global _PREPROCESSING_CONTEXT
    with _PREPARED_REQUESTS_LOCK:
        _PREPROCESSING_CONTEXT = None
        _PREPARED_REQUESTS.clear()
        _INFLIGHT_REQUESTS.clear()
        _ABORTED_REQUESTS.clear()


def cleanup_prepared_moss_tts_local_request(request_id: str) -> None:
    """Drop any prepared handoff for an aborted request (see MOSS Delay)."""
    rid = str(request_id)
    with _PREPARED_REQUESTS_LOCK:
        if _PREPARED_REQUESTS.pop(rid, None) is not None:
            return
        if rid in _INFLIGHT_REQUESTS:
            _ABORTED_REQUESTS.add(rid)


def pop_prepared_moss_tts_local_request(
    payload: StagePayload,
) -> MossTTSLocalPreparedRequest | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_MOSS_TTS_LOCAL_PREPARED_MARKER)
    if marker is None:
        return None
    with _PREPARED_REQUESTS_LOCK:
        prepared = _PREPARED_REQUESTS.pop(str(marker), None)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS Local preprocessing state is missing for prepared payload "
            f"{marker!r}; the AR scheduler must not rebuild it"
        )
    return prepared


def build_moss_tts_local_state(payload: StagePayload) -> MossTTSLocalState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    language = _resolve_optional_text(
        tts_params.get("language") or params.get("language")
    )
    instructions = _resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    text, token_count = _resolve_token_count(text, params, tts_params)
    return MossTTSLocalState(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        instructions=instructions,
        token_count=token_count,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
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

    raw_max_new_tokens = params.get("max_new_tokens")
    if raw_max_new_tokens is None:
        max_new_tokens = MOSS_TTS_DEFAULT_MAX_NEW_TOKENS
    elif isinstance(raw_max_new_tokens, bool):
        raise ValueError(
            f"MOSS-TTS max_new_tokens must be an integer, got {raw_max_new_tokens!r}"
        )
    else:
        max_new_tokens = int(raw_max_new_tokens)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        # Checkpoint generate() / model-card defaults for v1.5.
        "text_temperature": 1.0,
        "audio_temperature": 1.7,
        "text_top_p": 1.0,
        "audio_top_p": 0.8,
        "text_top_k": 50,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
    }

    if "temperature" in explicit_fields and params.get("temperature") is not None:
        generation_kwargs["text_temperature"] = float(params["temperature"])
        generation_kwargs["audio_temperature"] = float(params["temperature"])
    if "top_p" in explicit_fields and params.get("top_p") is not None:
        generation_kwargs["text_top_p"] = float(params["top_p"])
        generation_kwargs["audio_top_p"] = float(params["top_p"])
    if "top_k" in explicit_fields and params.get("top_k") is not None:
        generation_kwargs["text_top_k"] = int(params["top_k"])
        generation_kwargs["audio_top_k"] = int(params["top_k"])
    if (
        "repetition_penalty" in explicit_fields
        and params.get("repetition_penalty") is not None
    ):
        generation_kwargs["audio_repetition_penalty"] = float(
            params["repetition_penalty"]
        )

    for source in (tts_params, params):
        for field_name in (
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
        ):
            if source.get(field_name) is not None:
                value = source[field_name]
                generation_kwargs[field_name] = (
                    int(value) if field_name.endswith("top_k") else float(value)
                )

    seed = tts_params.get("seed")
    if seed is None:
        seed = params.get("seed")
    if seed is not None:
        generation_kwargs["seed"] = seed

    _validate_moss_tts_generation_kwargs(generation_kwargs)
    return generation_kwargs


def _build_processor_message(
    processor: Any,
    state: MossTTSLocalState,
    reference_encoder: Any = None,
) -> dict[str, Any]:
    from sglang_omni.models.moss_tts.request_builders import _DATA_URI_RE

    ref_audio = state.ref_audio
    if (
        reference_encoder is not None
        and isinstance(ref_audio, str)
        and _DATA_URI_RE.match(ref_audio) is None
    ):
        # File-path references encode through the shared coalescer so
        # concurrent requests share one batched codec forward instead of
        # serializing ~0.25 GPU-seconds each.
        reference = [reference_encoder.encode(ref_audio)]
    else:
        reference = _reference_for_processor(processor, ref_audio)
    return processor.build_user_message(
        text=state.text,
        reference=reference,
        instruction=state.instructions,
        tokens=state.token_count,
        language=state.language,
    )


def _prepare_moss_tts_local_request(
    payload: StagePayload,
    *,
    processor: Any,
    reference_encoder: Any = None,
) -> MossTTSLocalPreparedRequest:
    state = build_moss_tts_local_state(payload)
    message = _build_processor_message(processor, state, reference_encoder)
    batch = processor([[message]], mode="generation")
    input_rows = batch["input_ids"]
    if input_rows.ndim != 3 or int(input_rows.shape[0]) != 1:
        raise ValueError(
            "MOSS-TTS Local processor must return input_ids with shape [1, T, C]"
        )
    prompt_rows = input_rows[0].detach().to(dtype=torch.long, device="cpu")
    input_ids_list = build_row_cache_key_ids(prompt_rows)
    return MossTTSLocalPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        prompt_rows=prompt_rows,
        gen_kwargs=state.generation_kwargs,
    )


def preprocess_moss_tts_local_payload(payload: StagePayload) -> StagePayload:
    """Run prompt/reference preprocessing outside the AR scheduler."""

    rid = str(payload.request_id)
    with _PREPARED_REQUESTS_LOCK:
        context = _PREPROCESSING_CONTEXT
        if context is not None:
            _INFLIGHT_REQUESTS.add(rid)
    if context is None:
        raise RuntimeError(
            "MOSS-TTS Local preprocessing context is not initialized; "
            "create_preprocessing_executor must register it before requests run"
        )

    try:
        prepared = _prepare_moss_tts_local_request(
            payload,
            processor=context.processor,
            reference_encoder=context.reference_encoder,
        )
    except BaseException:
        with _PREPARED_REQUESTS_LOCK:
            _INFLIGHT_REQUESTS.discard(rid)
            _ABORTED_REQUESTS.discard(rid)
        raise
    with _PREPARED_REQUESTS_LOCK:
        _INFLIGHT_REQUESTS.discard(rid)
        aborted = rid in _ABORTED_REQUESTS
        _ABORTED_REQUESTS.discard(rid)
        if not aborted:
            _PREPARED_REQUESTS[rid] = prepared

    data = prepared.state.to_dict()
    data[_MOSS_TTS_LOCAL_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=data
    )


def build_sglang_moss_tts_local_request(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSLocalSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_moss_tts_local_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS Local AR request builder requires a payload prepared by "
            "preprocess_moss_tts_local_payload"
        )

    cfg = model.config
    gen_kwargs = prepared.gen_kwargs
    max_new_tokens = int(
        gen_kwargs.get("max_new_tokens", MOSS_TTS_DEFAULT_MAX_NEW_TOKENS)
    )
    audio_end = int(cfg.audio_end_token_id)
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[audio_end],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(cfg.vocab_size_list[0]))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={audio_end},
        vocab_size=int(cfg.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    data = MossTTSLocalSGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        model_config=cfg,
        prompt_rows=prepared.prompt_rows,
        text_temperature=float(gen_kwargs.get("text_temperature", 1.0)),
        text_top_p=float(gen_kwargs.get("text_top_p", 1.0)),
        text_top_k=int(gen_kwargs.get("text_top_k", 50)),
        audio_temperature=float(gen_kwargs.get("audio_temperature", 1.7)),
        audio_top_p=float(gen_kwargs.get("audio_top_p", 0.8)),
        audio_top_k=int(gen_kwargs.get("audio_top_k", 25)),
        audio_repetition_penalty=float(gen_kwargs.get("audio_repetition_penalty", 1.0)),
        seed=gen_kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(gen_kwargs["seed"])
            if gen_kwargs.get("seed") is not None
            else _new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_moss_tts_local_result(
    payload: StagePayload,
    data: MossTTSLocalSGLangRequestData,
) -> StagePayload:
    state = data.state
    n_vq = (
        int(data.prompt_rows.shape[1] - 1)
        if data.prompt_rows is not None and data.prompt_rows.ndim == 2
        else 12
    )
    if data.output_rows:
        generated_rows = torch.stack(data.output_rows, dim=0).to(dtype=torch.long)
        state.audio_codes = generated_rows[:, 1:].detach().cpu()
    else:
        state.audio_codes = torch.empty((0, n_vq), dtype=torch.long)

    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_rows)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_moss_tts_local_scheduler_adapters(*, model: Any):
    """Build StagePayload <-> SGLang request adapters for MOSS-TTS Local."""

    def request_builder(payload: StagePayload) -> MossTTSLocalSGLangRequestData:
        return build_sglang_moss_tts_local_request(payload, model=model)

    def result_adapter(data: MossTTSLocalSGLangRequestData) -> StagePayload:
        return apply_sglang_moss_tts_local_result(data.stage_payload, data)

    return request_builder, result_adapter
