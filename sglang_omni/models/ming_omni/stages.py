# SPDX-License-Identifier: Apache-2.0
"""（wenyao）Stage factories for Ming-Omni.

Heavy runtime imports are intentionally local to factory calls so importing
Ming's config remains usable in lightweight environments.
"""

from __future__ import annotations

from typing import Any

from sglang_omni.models.ming_omni.io import PipelineState
from sglang_omni.models.ming_omni.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage
from sglang_omni.proto import StagePayload


def project_preprocessing_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=AUDIO_STAGE)


def project_preprocessing_to_image_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)


def project_preprocessing_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    projected = PipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        mm_inputs=dict(state.mm_inputs),
        encoder_inputs=_project_encoder_input_metadata(state.encoder_inputs),
        stream_state=dict(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_encoder_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    stage_name = _single_encoder_stage_name(state)
    projected = PipelineState(
        encoder_outs={stage_name: state.encoder_outs.get(stage_name, {})}
    )
    return _payload_with_state(payload, projected)


def _project_preprocessing_to_encoder(
    payload: StagePayload,
    *,
    stage_name: str,
) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    stage_inputs = state.encoder_inputs.get(stage_name)
    projected_inputs = (
        {stage_name: dict(stage_inputs)} if isinstance(stage_inputs, dict) else {}
    )
    return _payload_with_state(payload, PipelineState(encoder_inputs=projected_inputs))


def _payload_with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _project_encoder_input_metadata(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for stage_name, stage_inputs in encoder_inputs.items():
        if not isinstance(stage_inputs, dict):
            continue
        metadata: dict[str, Any] = {}
        cache_key = stage_inputs.get("cache_key")
        if cache_key is not None:
            metadata["cache_key"] = cache_key
        if stage_inputs.get("_skip"):
            metadata["_skip"] = True
        if metadata:
            projected[stage_name] = metadata
    return projected


def _single_encoder_stage_name(state: PipelineState) -> str:
    if len(state.encoder_outs) != 1:
        raise ValueError(
            f"Expected exactly one encoder output in payload, got {sorted(state.encoder_outs)}"
        )
    return next(iter(state.encoder_outs))


def _attach_decode_final_metadata(
    result: dict[str, Any],
    state: PipelineState,
    thinker_out: dict[str, Any],
) -> None:
    finish_reason = thinker_out.get("finish_reason")
    if finish_reason is not None:
        result.setdefault("finish_reason", finish_reason)
    result.setdefault("usage", build_text_usage(state, thinker_out))


def create_preprocessing_executor(model_path: str):
    from sglang_omni.models.ming_omni.components.preprocessor import MingPreprocessor
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = MingPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.models.ming_omni.components.audio_encoder import MingAudioEncoder
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = MingAudioEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        state = PipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(AUDIO_STAGE)
        if not isinstance(inputs, dict) or not inputs:
            result = {}
        elif inputs.get("_skip"):
            skip_result = inputs.get("_result")
            result = skip_result if isinstance(skip_result, dict) else {}
        else:
            _meta = {"cache_key", "audio_placeholder_loc_lens"}
            model_inputs = {k: v for k, v in inputs.items() if k not in _meta}
            result = model(**model_inputs)
        state.encoder_outs[AUDIO_STAGE] = result if isinstance(result, dict) else {}
        state.engine_outputs[AUDIO_STAGE] = state.encoder_outs[AUDIO_STAGE]
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = MingImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        state = PipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(IMAGE_STAGE)
        if not isinstance(inputs, dict) or not inputs:
            result = {}
        elif inputs.get("_skip"):
            skip_result = inputs.get("_result")
            result = skip_result if isinstance(skip_result, dict) else {}
        else:
            model_inputs = {k: v for k, v in inputs.items() if k != "cache_key"}
            result = model(**model_inputs)
        state.encoder_outs[IMAGE_STAGE] = result if isinstance(result, dict) else {}
        state.engine_outputs[IMAGE_STAGE] = state.encoder_outs[IMAGE_STAGE]
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode)


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.ming_omni.bootstrap import create_thinker_scheduler
    from sglang_omni.models.ming_omni.registration import register_ming_hf_config
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    register_ming_hf_config()

    overrides = dict(server_args_overrides or {})
    overrides.setdefault("trust_remote_code", False)
    overrides["tp_size"] = tp_size
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    return create_thinker_scheduler(
        server_args,
        model_path=model_path,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    )


def create_talker_executor(
    model_path: str,
    *,
    talker_model_path: str | None = None,
    device: str = "cuda",
    voice: str = "DB30",
):
    from sglang_omni.models.ming_omni.components.talker_executor import (
        MingTalkerExecutor,
    )
    from sglang_omni.models.weight_loader import resolve_model_path
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    local_path = resolve_model_path(model_path)
    executor = MingTalkerExecutor(
        model_path=local_path,
        talker_model_path=talker_model_path,
        device=device,
        voice=voice,
    )
    started = False

    async def _talk(payload: StagePayload) -> StagePayload:
        nonlocal started
        if not executor.should_generate_audio(payload):
            return executor.build_empty_audio_result(payload)
        if not started:
            await executor.start()
            started = True
        await executor.add_request(payload)
        return await executor.get_result()

    return SimpleScheduler(_talk)


def create_decode_executor(model_path: str):
    from sglang_omni.models.ming_omni.components.common import load_ming_tokenizer
    from sglang_omni.models.ming_omni.io import OmniEvent
    from sglang_omni.models.ming_omni.pipeline.merge import decode_events
    from sglang_omni.models.ming_omni.pipeline.next_stage import THINKER_STAGE
    from sglang_omni.models.ming_omni.pipeline.state_io import load_state
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    tokenizer = load_ming_tokenizer(model_path)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
        return {
            "type": event.type,
            "modality": event.modality,
            "payload": dict(event.payload),
            "is_final": bool(event.is_final),
        }

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
                thinker_out=thinker_out,  # type: ignore[arg-type]
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        result: dict[str, Any] = {"events": [_event_to_dict(event) for event in events]}
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

        _attach_decode_final_metadata(result, state, thinker_out)
        payload.data = result
        return payload

    return SimpleScheduler(_decode)
