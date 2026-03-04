# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from sglang_omni.engines.omni import (
    create_ar_engine,
    create_encoder_engine,
    create_sglang_ar_engine,
)
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
from sglang_omni.models.qwen3_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
    build_sglang_thinker_request,
    build_thinker_request,
)
from sglang_omni.models.qwen3_omni.pipeline.merge import decode_events
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    preprocessor = Qwen3OmniPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


def create_aggregate_executor() -> PreprocessingExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return PreprocessingExecutor(_identity)


def _create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
) -> EngineExecutor:
    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=stage_name)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=stage_name, result=result)
        return store_state(payload, state)

    engine = create_encoder_engine(model, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def create_thinker_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(model_path=model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        return build_thinker_request(state, params=payload.request.params)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_thinker_result(state, stage_name=THINKER_STAGE, result=result)
        step_counters.pop(payload.request_id, None)
        return store_state(payload, state)

    def _stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if payload is None:
            return None
        request_id = payload.request_id
        step = step_counters.get(request_id, 0) + 1
        step_counters[request_id] = step

        try:
            token_id = int(item)
        except Exception:
            return {"token_id": item, "step": step}

        state = load_state(payload)
        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        store_state(payload, state)
        if eos_token_id is not None and token_id == eos_token_id and not events:
            events = [
                OmniEvent(type="text_final", modality="text", payload={}, is_final=True)
            ]
        text_delta = ""
        for event in events:
            if event.is_final:
                continue
            t = event.payload.get("text")
            if event.modality == "text" and t:
                text_delta += t

        result: dict[str, Any] = {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": THINKER_STAGE,
        }
        if text_delta:
            result["text"] = text_delta
        return result

    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


def create_sglang_thinker_executor(
    server_args: Any,
    model_path: str,
    *,
    gpu_id: int = 0,
) -> EngineExecutor:
    """Create a thinker executor backed by SGLang's ModelWorker."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        return build_sglang_thinker_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
        )

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_thinker_result(state, stage_name=THINKER_STAGE, result=result)
        step_counters.pop(payload.request_id, None)
        return store_state(payload, state)

    def _stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if payload is None:
            return None
        request_id = payload.request_id
        step = step_counters.get(request_id, 0) + 1
        step_counters[request_id] = step

        try:
            token_id = int(item)
        except Exception:
            return {"token_id": item, "step": step}

        state = load_state(payload)
        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        store_state(payload, state)
        if eos_token_id is not None and token_id == eos_token_id and not events:
            events = [
                OmniEvent(type="text_final", modality="text", payload={}, is_final=True)
            ]

        text_to_add = ""
        for event in events:
            if event.modality == "text" and "text" in event.payload:
                if event.is_final:
                    # If a final text event is found, it contains the complete text for this step.
                    # This should override any accumulated delta.
                    text_to_add = event.payload["text"]
                    break  # No need to process further events for text accumulation
                else:
                    # Accumulate text from non-final delta events
                    text_to_add += event.payload["text"]

        result: dict[str, Any] = {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": THINKER_STAGE,
        }
        if text_to_add:
            result["text"] = text_to_add
        return result

    engine = create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=gpu_id,
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
) -> EngineExecutor:
    """Create a SGLang thinker executor from JSON-serializable config args.

    This keeps pipeline config args plain dict types while still constructing
    a typed ServerArgs object internally.
    """
    server_args_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "disable_cuda_graph": True,
        "chunked_prefill_size": 128,
        "max_prefill_tokens": 4096,
        "max_running_requests": 16,
        "mem_fraction_static": 0.7,
        "random_seed": 123,
        "context_length": thinker_max_seq_len,
    }
    if server_args_overrides:
        server_args_kwargs.update(server_args_overrides)

    server_args = ServerArgs(**server_args_kwargs)
    return create_sglang_thinker_executor(
        server_args=server_args,
        model_path=model_path,
        gpu_id=gpu_id,
    )


def create_decode_executor(model_path: str) -> PreprocessingExecutor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (
                event
                for event in reversed(events)
                if event.is_final or event.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if (
                callable(getattr(tokenizer, "decode", None))
                and isinstance(output_ids, list)
                and output_ids
            ):
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
                result.setdefault("modality", "text")

        payload.data = result
        return payload

    return PreprocessingExecutor(_decode)
