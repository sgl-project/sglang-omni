# SPDX-License-Identifier: Apache-2.0
"""Request adapters for MOSS-TTS-Realtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    MOSS_TTS_DEFAULT_MAX_NEW_TOKENS,
    _new_moss_tts_sampling_seed,
    _validate_moss_tts_generation_kwargs,
    build_row_cache_key_ids,
    derive_moss_tts_sampling_seed,
    normalize_moss_tts_inputs,
    resolve_moss_reference,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
)
from sglang_omni.models.moss_tts_realtime.payload_types import (
    N_CODEBOOKS,
    MossTTSRealtimeState,
)
from sglang_omni.models.moss_tts_realtime.processor import (
    MossTTSRealtimePromptProcessor,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

_PREPARED_MARKER = "_moss_tts_realtime_prepared_request"


@dataclass
class MossTTSRealtimeRequestData(MossTTSLocalSGLangRequestData):
    state: MossTTSRealtimeState = field(default_factory=MossTTSRealtimeState)
    text_token_ids: list[int] = field(default_factory=list)
    prefill_text_tokens: int = 0
    text_temperature: float = 0.8
    text_top_p: float = 0.6
    text_top_k: int = 30
    audio_temperature: float = 0.8
    audio_top_p: float = 0.6
    audio_top_k: int = 30
    audio_repetition_penalty: float = 1.1


@dataclass
class PreparedRequest:
    state: MossTTSRealtimeState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_rows: torch.Tensor
    text_token_ids: list[int]
    prefill_text_tokens: int
    generation_kwargs: dict[str, Any]


@dataclass
class PreprocessingContext:
    processor: MossTTSRealtimePromptProcessor
    reference_encoder: Any


_QUEUE: PreparedRequestQueue[PreprocessingContext, PreparedRequest] = (
    PreparedRequestQueue()
)


def set_preprocessing_context(
    *,
    processor: MossTTSRealtimePromptProcessor,
    reference_encoder: Any,
) -> None:
    _QUEUE.set_context(
        PreprocessingContext(
            processor=processor,
            reference_encoder=reference_encoder,
        )
    )


def clear_preprocessing_context() -> None:
    _QUEUE.clear_context()


def cleanup_prepared_request(request_id: str) -> None:
    _QUEUE.abort(str(request_id))


def _generation_kwargs(
    params: dict[str, Any], tts_params: dict[str, Any]
) -> dict[str, Any]:
    explicit = tts_params.get("explicit_generation_params")
    explicit_fields = (
        {str(value) for value in explicit}
        if isinstance(explicit, (list, tuple, set))
        else set()
    )
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(
            params.get("max_new_tokens") or MOSS_TTS_DEFAULT_MAX_NEW_TOKENS
        ),
        "text_temperature": 0.8,
        "text_top_p": 0.6,
        "text_top_k": 30,
        "audio_temperature": 0.8,
        "audio_top_p": 0.6,
        "audio_top_k": 30,
        "audio_repetition_penalty": 1.1,
        "repetition_window": 50,
    }
    generic_fields = {
        "temperature": ("text_temperature", "audio_temperature", float),
        "top_p": ("text_top_p", "audio_top_p", float),
        "top_k": ("text_top_k", "audio_top_k", int),
        "repetition_penalty": ("audio_repetition_penalty", None, float),
    }
    for public_name, (first, second, cast) in generic_fields.items():
        value = params.get(public_name)
        if public_name in explicit_fields and value is not None:
            kwargs[first] = cast(value)
            if second is not None:
                kwargs[second] = cast(value)
    for source in (tts_params, params):
        for name in (
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
            "repetition_window",
        ):
            if source.get(name) is not None:
                kwargs[name] = (
                    int(source[name])
                    if name.endswith("top_k") or name == "repetition_window"
                    else float(source[name])
                )
    seed = tts_params.get("seed", params.get("seed"))
    if seed is not None:
        kwargs["seed"] = seed
    _validate_moss_tts_generation_kwargs(kwargs)
    if int(kwargs["repetition_window"]) != 50:
        raise ValueError(
            "MOSS-TTS-Realtime currently supports repetition_window=50 only"
        )
    return kwargs


def build_state(payload: StagePayload) -> MossTTSRealtimeState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, _ = resolve_moss_reference(references, tts_params)
    if not text.strip():
        raise ValueError("MOSS-TTS-Realtime input text must not be empty")
    return MossTTSRealtimeState(
        text=text,
        ref_audio=ref_audio,
        generation_kwargs=_generation_kwargs(params, tts_params),
    )


def _prepare(payload: StagePayload, context: PreprocessingContext) -> PreparedRequest:
    state = build_state(payload)
    reference_codes = (
        context.reference_encoder(state.ref_audio)
        if state.ref_audio is not None
        else None
    )
    prompt_rows, text_token_ids, prefill_text_tokens = (
        context.processor.build_generation_prompt(state.text, reference_codes)
    )
    prompt_rows = prompt_rows.to(dtype=torch.long, device="cpu")
    input_ids_list = build_row_cache_key_ids(prompt_rows)
    return PreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        prompt_rows=prompt_rows,
        text_token_ids=text_token_ids,
        prefill_text_tokens=prefill_text_tokens,
        generation_kwargs=state.generation_kwargs,
    )


def preprocess_payload(payload: StagePayload) -> StagePayload:
    request_id = str(payload.request_id)
    context = _QUEUE.begin(request_id)
    if context is None:
        raise RuntimeError("MOSS-TTS-Realtime preprocessing is not initialized")
    try:
        prepared = _prepare(payload, context)
    except BaseException:
        _QUEUE.fail_inflight(request_id)
        raise
    published = _QUEUE.publish(request_id, prepared)
    data = prepared.state.to_dict()
    if published:
        data[_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def _pop_prepared(payload: StagePayload) -> PreparedRequest:
    marker = payload.data.get(_PREPARED_MARKER)
    prepared = _QUEUE.pop(str(marker)) if marker is not None else None
    if prepared is None:
        raise RuntimeError("MOSS-TTS-Realtime AR request requires preprocessing output")
    return prepared


def _stream_metadata(payload: StagePayload) -> dict[str, Any] | None:
    params = payload.request.params
    if not isinstance(params, dict) or not params.get("stream"):
        return None
    metadata: dict[str, Any] = {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": N_CODEBOOKS,
    }
    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = params[
            INITIAL_CODEC_CHUNK_FRAMES_PARAM
        ]
    return metadata


def build_sglang_request(
    payload: StagePayload, *, model: Any
) -> MossTTSRealtimeRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = _pop_prepared(payload)
    kwargs = prepared.generation_kwargs
    max_new_tokens = int(kwargs["max_new_tokens"])
    end_id = int(model.config.audio_end_token_id)
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[end_id],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(model.config.vocab_size_list[0]))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={end_id},
        vocab_size=int(model.config.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None
    data = MossTTSRealtimeRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        model_config=model.config,
        prompt_rows=prepared.prompt_rows,
        text_token_ids=prepared.text_token_ids,
        prefill_text_tokens=prepared.prefill_text_tokens,
        text_temperature=float(kwargs["text_temperature"]),
        text_top_p=float(kwargs["text_top_p"]),
        text_top_k=int(kwargs["text_top_k"]),
        audio_temperature=float(kwargs["audio_temperature"]),
        audio_top_p=float(kwargs["audio_top_p"]),
        audio_top_k=int(kwargs["audio_top_k"]),
        audio_repetition_penalty=float(kwargs["audio_repetition_penalty"]),
        seed=kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(kwargs["seed"])
            if kwargs.get("seed") is not None
            else _new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
        stream_metadata=_stream_metadata(payload),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_result(
    payload: StagePayload, data: MossTTSRealtimeRequestData
) -> StagePayload:
    state = data.state
    if data.output_rows:
        rows = torch.stack(data.output_rows).to(dtype=torch.long)
        state.audio_codes = rows[:, 1:].detach().cpu()
    else:
        state.audio_codes = torch.empty((0, N_CODEBOOKS), dtype=torch.long)
    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_rows)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_scheduler_adapters(*, model: Any) -> tuple[Any, Any]:
    def request_builder(payload: StagePayload) -> MossTTSRealtimeRequestData:
        return build_sglang_request(payload, model=model)

    def result_adapter(data: MossTTSRealtimeRequestData) -> StagePayload:
        try:
            return apply_result(data.stage_payload, data)
        finally:
            model.reset_request(data.stage_payload.request_id)

    return request_builder, result_adapter


__all__ = [
    "MossTTSRealtimeRequestData",
    "apply_result",
    "build_sglang_request",
    "build_state",
    "cleanup_prepared_request",
    "clear_preprocessing_context",
    "make_scheduler_adapters",
    "preprocess_payload",
    "set_preprocessing_context",
]
