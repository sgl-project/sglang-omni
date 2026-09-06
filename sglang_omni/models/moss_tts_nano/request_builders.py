# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for MOSS-TTS-Nano."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    _DATA_URI_RE,
    _new_moss_tts_sampling_seed,
    _validate_moss_tts_generation_kwargs,
    build_row_cache_key_ids,
    derive_moss_tts_sampling_seed,
    normalize_moss_tts_inputs,
    resolve_moss_reference,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MOSS_STREAM_TRANSPORT_BATCH_FRAMES,
    MossTTSLocalSGLangRequestData,
    build_moss_tts_local_stream_metadata,
)
from sglang_omni.models.moss_tts_nano.payload_types import MossTTSNanoState
from sglang_omni.models.moss_tts_nano.prompting import build_prompt_rows
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue

MOSS_TTS_NANO_DEFAULT_MAX_NEW_FRAMES = 375
_MOSS_TTS_NANO_PREPARED_MARKER = "_moss_tts_nano_prepared_request"


@dataclass
class MossTTSNanoSGLangRequestData(MossTTSLocalSGLangRequestData):
    """Scheduler-owned request state for MOSS-TTS-Nano."""

    state: MossTTSNanoState = field(default_factory=MossTTSNanoState)
    audio_temperature: float = 0.8
    audio_top_p: float = 0.95
    audio_repetition_penalty: float = 1.2


@dataclass
class MossTTSNanoPreparedRequest:
    state: MossTTSNanoState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_rows: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class _PreprocessingContext:
    tokenizer: Any
    model_config: Any
    reference_encoder: Any = None


_QUEUE: PreparedRequestQueue[_PreprocessingContext, MossTTSNanoPreparedRequest] = (
    PreparedRequestQueue()
)


def set_moss_tts_nano_preprocessing_context(
    *, tokenizer: Any, model_config: Any, reference_encoder: Any = None
) -> None:
    _QUEUE.set_context(
        _PreprocessingContext(
            tokenizer=tokenizer,
            model_config=model_config,
            reference_encoder=reference_encoder,
        )
    )


def clear_moss_tts_nano_preprocessing_context() -> None:
    _QUEUE.clear_context()


def cleanup_prepared_moss_tts_nano_request(request_id: str) -> None:
    _QUEUE.abort(str(request_id))


def pop_prepared_moss_tts_nano_request(
    payload: StagePayload,
) -> MossTTSNanoPreparedRequest | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_MOSS_TTS_NANO_PREPARED_MARKER)
    if marker is None:
        return None
    prepared = _QUEUE.pop(str(marker))
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS-Nano preprocessing state is missing for prepared payload "
            f"{marker!r}; the AR scheduler must not rebuild it"
        )
    return prepared


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
        max_new_tokens = MOSS_TTS_NANO_DEFAULT_MAX_NEW_FRAMES
    elif isinstance(raw_max_new_tokens, bool):
        raise ValueError(
            "MOSS-TTS-Nano max_new_tokens must be an integer, got "
            f"{raw_max_new_tokens!r}"
        )
    else:
        max_new_tokens = int(raw_max_new_tokens)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "text_temperature": 1.0,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_p": 0.95,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.2,
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
            if source.get(field_name) is None:
                continue
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


def build_moss_tts_nano_state(payload: StagePayload) -> MossTTSNanoState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    if len(references) > 1:
        raise ValueError("MOSS-TTS-Nano accepts at most one reference audio")
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    if isinstance(ref_text, str) and ref_text.strip():
        raise ValueError(
            "MOSS-TTS-Nano voice cloning does not accept a reference transcript"
        )
    instructions = tts_params.get("instructions") or params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        raise ValueError("MOSS-TTS-Nano does not support instructions")
    return MossTTSNanoState(
        text=text,
        ref_audio=ref_audio,
        ref_text=None,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def _encode_reference(
    ref_audio: Any | None, reference_encoder: Any
) -> torch.Tensor | None:
    if ref_audio is None:
        return None
    if isinstance(ref_audio, os.PathLike):
        ref_audio = os.fsdecode(ref_audio)
    if isinstance(ref_audio, torch.Tensor):
        return ref_audio
    if not isinstance(ref_audio, str):
        return torch.as_tensor(ref_audio, dtype=torch.long)
    if reference_encoder is None:
        raise RuntimeError(
            "MOSS-TTS-Nano reference audio requires an initialized audio encoder"
        )
    if _DATA_URI_RE.match(ref_audio) is not None:
        return reference_encoder.encode_data_uri(ref_audio)
    return reference_encoder.encode(ref_audio)


def _prepare_moss_tts_nano_request(
    payload: StagePayload,
    *,
    tokenizer: Any,
    model_config: Any,
    reference_encoder: Any = None,
) -> MossTTSNanoPreparedRequest:
    state = build_moss_tts_nano_state(payload)
    reference_codes = _encode_reference(state.ref_audio, reference_encoder)
    prompt_rows = build_prompt_rows(
        tokenizer=tokenizer,
        config=model_config,
        text=state.text,
        reference_codes=reference_codes,
    )
    input_ids_list = build_row_cache_key_ids(prompt_rows)
    return MossTTSNanoPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        prompt_rows=prompt_rows,
        gen_kwargs=state.generation_kwargs,
    )


def preprocess_moss_tts_nano_payload(payload: StagePayload) -> StagePayload:
    rid = str(payload.request_id)
    context = _QUEUE.begin(rid)
    if context is None:
        raise RuntimeError(
            "MOSS-TTS-Nano preprocessing context is not initialized; "
            "create_preprocessing_executor must register it before requests run"
        )
    try:
        prepared = _prepare_moss_tts_nano_request(
            payload,
            tokenizer=context.tokenizer,
            model_config=context.model_config,
            reference_encoder=context.reference_encoder,
        )
    except BaseException:
        _QUEUE.fail_inflight(rid)
        raise
    published = _QUEUE.publish(rid, prepared)
    data = prepared.state.to_dict()
    if published:
        data[_MOSS_TTS_NANO_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data,
    )


def build_sglang_moss_tts_nano_request(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSNanoSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_moss_tts_nano_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS-Nano AR request builder requires a payload prepared by "
            "preprocess_moss_tts_nano_payload"
        )

    cfg = model.config
    gen_kwargs = prepared.gen_kwargs
    max_new_tokens = int(
        gen_kwargs.get("max_new_tokens", MOSS_TTS_NANO_DEFAULT_MAX_NEW_FRAMES)
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
        extra_key="moss_tts_nano:prompt:v1",
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._omni_prompt_only_radix = True
    req._omni_prompt_cache_key = req.extra_key
    req._codec_suppress_tokens = None

    data = MossTTSNanoSGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        model_config=cfg,
        prompt_rows=prepared.prompt_rows,
        text_temperature=float(gen_kwargs["text_temperature"]),
        text_top_p=float(gen_kwargs["text_top_p"]),
        text_top_k=int(gen_kwargs["text_top_k"]),
        audio_temperature=float(gen_kwargs["audio_temperature"]),
        audio_top_p=float(gen_kwargs["audio_top_p"]),
        audio_top_k=int(gen_kwargs["audio_top_k"]),
        audio_repetition_penalty=float(gen_kwargs["audio_repetition_penalty"]),
        seed=gen_kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(gen_kwargs["seed"])
            if gen_kwargs.get("seed") is not None
            else _new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
        stream_metadata=build_moss_tts_local_stream_metadata(
            payload,
            n_vq=int(prepared.prompt_rows.shape[1]) - 1,
        ),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_moss_tts_nano_result(
    payload: StagePayload,
    data: MossTTSNanoSGLangRequestData,
) -> StagePayload:
    state = data.state
    n_vq = (
        int(data.prompt_rows.shape[1] - 1)
        if data.prompt_rows is not None and data.prompt_rows.ndim == 2
        else 16
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


def make_moss_tts_nano_scheduler_adapters(*, model: Any):
    def request_builder(payload: StagePayload) -> MossTTSNanoSGLangRequestData:
        return build_sglang_moss_tts_nano_request(payload, model=model)

    def result_adapter(data: MossTTSNanoSGLangRequestData) -> StagePayload:
        try:
            return apply_sglang_moss_tts_nano_result(data.stage_payload, data)
        finally:
            model.reset_request(data.stage_payload.request_id)

    return request_builder, result_adapter


__all__ = [
    "MOSS_STREAM_TRANSPORT_BATCH_FRAMES",
    "MOSS_TTS_NANO_DEFAULT_MAX_NEW_FRAMES",
    "MossTTSNanoPreparedRequest",
    "MossTTSNanoSGLangRequestData",
    "apply_sglang_moss_tts_nano_result",
    "build_generation_kwargs",
    "build_moss_tts_nano_state",
    "build_sglang_moss_tts_nano_request",
    "cleanup_prepared_moss_tts_nano_request",
    "clear_moss_tts_nano_preprocessing_context",
    "make_moss_tts_nano_scheduler_adapters",
    "pop_prepared_moss_tts_nano_request",
    "preprocess_moss_tts_nano_payload",
    "set_moss_tts_nano_preprocessing_context",
]
