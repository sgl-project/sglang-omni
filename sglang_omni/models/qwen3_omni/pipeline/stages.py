# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.engines.omni import create_ar_engine, create_encoder_engine
from sglang_omni.executors import EngineExecutor, FrontendExecutor
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.frontend import Qwen3OmniFrontend
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
from sglang_omni.models.qwen3_omni.components.torch_audio_encoder import (
    Qwen3OmniTorchAudioEncoder,
)
from sglang_omni.models.qwen3_omni.components.torch_image_encoder import (
    VISUAL_PREFIX,
    Qwen3OmniTorchImageEncoder,
)
from sglang_omni.models.qwen3_omni.components.torch_thinker import (
    Qwen3OmniTorchThinker,
)
from sglang_omni.models.weight_loader import preload_weights, resolve_dtype
from sglang_omni.models.qwen3_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
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


def create_frontend_executor(
    model_path: str
) -> FrontendExecutor:
    frontend = Qwen3OmniFrontend(model_path=model_path)

    def _frontend(payload: StagePayload) -> StagePayload:
        return frontend(payload)

    return FrontendExecutor(_frontend)


def create_aggregate_executor() -> FrontendExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return FrontendExecutor(_identity)


def preload_torch_weights(
    *,
    model_path: str,
    image_device: str,
    audio_device: str,
    thinker_device: str,
    dtype: str | None = None,
) -> None:
    """Preload torch-native weights into cache for faster component init."""
    torch_dtype = resolve_dtype(dtype)
    device_groups: dict[str, list[tuple[str, ...]]] = {}

    device_groups.setdefault(str(image_device), []).append(VISUAL_PREFIX)
    device_groups.setdefault(str(audio_device), []).append(("thinker.audio_tower.",))
    device_groups.setdefault(str(thinker_device), []).append(("thinker.model.",))
    device_groups.setdefault(str(thinker_device), []).append(("thinker.lm_head.",))

    for device, prefix_groups in device_groups.items():
        preload_weights(
            model_path,
            prefix_groups=prefix_groups,
            device=device,
            dtype=torch_dtype,
        )


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
    model = Qwen3OmniImageEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_image_encoder_executor_torch(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniTorchImageEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def create_audio_encoder_executor_torch(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniTorchAudioEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def _create_thinker_executor(
    model: torch.nn.Module,
    *,
    model_path: str,
    device: str,
    max_seq_len: int,
) -> EngineExecutor:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
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
        return {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": THINKER_STAGE,
        }

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


def create_thinker_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_thinker_executor(
        model, model_path=model_path, device=device, max_seq_len=max_seq_len
    )


def create_thinker_executor_torch(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniTorchThinker(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )
    return _create_thinker_executor(
        model, model_path=model_path, device=device, max_seq_len=max_seq_len
    )


def create_decode_executor(model_path: str) -> FrontendExecutor:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
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

    return FrontendExecutor(_decode)
