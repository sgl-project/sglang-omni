# SPDX-License-Identifier: Apache-2.0
"""SGLang request and result adapters for Cosmos3 text generation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, cast

import torch
import xxhash
from transformers import GenerationConfig, PreTrainedTokenizerBase

from sglang_omni.models.cosmos3.config import DECODE_STAGE, THINKER_STAGE
from sglang_omni.models.cosmos3.payload_types import (
    Cosmos3PipelineState,
    PromptInputs,
    TextOutput,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
VISION_STAGE = "vision_encoder"

_TRANSPORT_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}


def _payload_with_state(
    payload: StagePayload,
    state: Cosmos3PipelineState,
) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _vision_encoder_is_active(state: Cosmos3PipelineState) -> bool:
    inputs = state.encoder_inputs.get(VISION_STAGE)
    if not isinstance(inputs, dict) or inputs.get("_skip"):
        return False
    if inputs.get("_active") is not None:
        return inputs["_active"] is True
    return isinstance(inputs.get("pixel_values"), torch.Tensor) or isinstance(
        inputs.get("pixel_values_videos"), torch.Tensor
    )


def resolve_preprocessing_next_stages(
    request_id: str,
    output: StagePayload,
) -> list[str]:
    del request_id
    state = Cosmos3PipelineState.from_dict(output.data)
    stages = [THINKER_STAGE]
    if _vision_encoder_is_active(state):
        stages.insert(0, VISION_STAGE)
    return stages


def resolve_thinker_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    del request_id
    if from_stage != "preprocessing":
        return None
    state = Cosmos3PipelineState.from_dict(payload.data)
    sources = ["preprocessing"]
    if _vision_encoder_is_active(state):
        sources.append(VISION_STAGE)
    return sources


def project_preprocessing_to_vision_encoder(payload: StagePayload) -> StagePayload:
    state = Cosmos3PipelineState.from_dict(payload.data)
    vision_inputs = state.encoder_inputs.get(VISION_STAGE)
    projected = Cosmos3PipelineState(
        encoder_inputs=(
            {VISION_STAGE: dict(vision_inputs)}
            if isinstance(vision_inputs, dict)
            else {}
        )
    )
    return _payload_with_state(payload, projected)


def project_preprocessing_to_thinker(payload: StagePayload) -> StagePayload:
    state = Cosmos3PipelineState.from_dict(payload.data)
    metadata: dict[str, Any] = {}
    vision_inputs = state.encoder_inputs.get(VISION_STAGE)
    if isinstance(vision_inputs, dict):
        for key in ("image_cache_key", "video_cache_key"):
            if vision_inputs.get(key) is not None:
                metadata[key] = vision_inputs[key]
        if _vision_encoder_is_active(state):
            metadata["_active"] = True
    projected = Cosmos3PipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        mm_inputs=dict(state.mm_inputs),
        encoder_inputs={VISION_STAGE: metadata} if metadata else {},
        stream_state=dict(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_vision_encoder_to_thinker(payload: StagePayload) -> StagePayload:
    state = Cosmos3PipelineState.from_dict(payload.data)
    output = state.encoder_outs.get(VISION_STAGE, {})
    return _payload_with_state(
        payload,
        Cosmos3PipelineState(encoder_outs={VISION_STAGE: output}),
    )


def project_thinker_to_decode(payload: StagePayload) -> StagePayload:
    """Drop prefill-only visual tensors before the terminal decode hop."""

    state = Cosmos3PipelineState.from_dict(payload.data)
    state.mm_inputs = {}
    state.thinker_inputs = {}
    return _payload_with_state(payload, state)


@dataclass(slots=True)
class VisionEncoderRequestData:
    model_inputs: dict[str, Any]
    cache_key: str | None = None
    skip_result: dict[str, Any] | None = None


def build_vision_encoder_request(
    state: Cosmos3PipelineState,
) -> VisionEncoderRequestData:
    inputs = state.encoder_inputs.get(VISION_STAGE)
    if not isinstance(inputs, dict) or not inputs:
        return VisionEncoderRequestData(model_inputs={}, skip_result={})
    if inputs.get("_skip"):
        result = inputs.get("_result")
        return VisionEncoderRequestData(
            model_inputs={},
            skip_result=result if isinstance(result, dict) else {},
        )
    cache_parts = [
        str(inputs[key])
        for key in ("image_cache_key", "video_cache_key")
        if inputs.get(key) is not None
    ]
    return VisionEncoderRequestData(
        model_inputs={
            key: value
            for key, value in inputs.items()
            if key not in ("image_cache_key", "video_cache_key", "_active")
        },
        cache_key="|".join(cache_parts) or None,
    )


def apply_vision_encoder_result(
    state: Cosmos3PipelineState,
    result: Any,
) -> None:
    encoder_out = result if isinstance(result, dict) else {"result": result}
    state.encoder_outs[VISION_STAGE] = encoder_out
    state.engine_outputs[VISION_STAGE] = encoder_out


def _value_or_default(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def build_cosmos3_sampling_kwargs(
    params: dict[str, Any],
    *,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Merge request sampling overrides over Cosmos3-Nano defaults."""

    do_sample = bool(
        _value_or_default(
            generation_config.do_sample if generation_config is not None else None,
            True,
        )
    )
    default_temperature = _value_or_default(
        generation_config.temperature if generation_config is not None else None,
        0.7,
    )
    if not do_sample:
        default_temperature = 0.0
    sampling_seed = params.get("seed")
    if sampling_seed is None:
        sampling_seed = params.get("sampling_seed")
    return {
        "max_new_tokens": _value_or_default(params.get("max_new_tokens"), 2048),
        "temperature": _value_or_default(
            params.get("temperature"), default_temperature
        ),
        "top_p": _value_or_default(
            params.get("top_p"),
            _value_or_default(
                generation_config.top_p if generation_config is not None else None,
                0.8,
            ),
        ),
        "top_k": _value_or_default(
            params.get("top_k"),
            _value_or_default(
                generation_config.top_k if generation_config is not None else None,
                20,
            ),
        ),
        "min_p": _value_or_default(params.get("min_p"), 0.0),
        "repetition_penalty": _value_or_default(
            params.get("repetition_penalty"),
            _value_or_default(
                (
                    generation_config.repetition_penalty
                    if generation_config is not None
                    else None
                ),
                1.0,
            ),
        ),
        "stop": params.get("stop") or [],
        "stop_token_ids": params.get("stop_token_ids") or [],
        "sampling_seed": sampling_seed,
    }


def _effective_request_params(
    params: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Remove transport placeholders that must not shadow Nano defaults.

    OpenAI-compatible request parsing materializes generic sampling defaults.
    The explicit-field marker distinguishes those placeholders from values the
    caller actually supplied. Sparse internal requests predate that marker, so
    non-placeholder values remain valid overrides for backward compatibility.
    """

    effective = dict(params)
    raw_explicit_fields = metadata.get(EXPLICIT_GENERATION_PARAMS_KEY)
    has_explicit_marker = raw_explicit_fields is not None
    explicit_fields = set(raw_explicit_fields or ())

    for field, transport_default in _TRANSPORT_SAMPLING_DEFAULTS.items():
        if field in explicit_fields:
            continue
        value = effective.get(field)
        if has_explicit_marker or value is None or value == transport_default:
            effective.pop(field, None)
    return effective


def _eos_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    generation_config: GenerationConfig | None,
) -> set[int] | None:
    values = _value_or_default(
        generation_config.eos_token_id if generation_config is not None else None,
        tokenizer.eos_token_id,
    )
    if values is None:
        return None
    if isinstance(values, int):
        values = [values]
    result = {int(value) for value in values if value is not None and int(value) >= 0}
    return result or None


def _vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    llm_h = grid_h // spatial_merge_size
    llm_w = grid_w // spatial_merge_size
    temporal = torch.arange(grid_t, dtype=torch.long).repeat_interleave(llm_h * llm_w)
    height = torch.arange(llm_h, dtype=torch.long).repeat_interleave(llm_w)
    height = height.repeat(grid_t)
    width = torch.arange(llm_w, dtype=torch.long).repeat(llm_h * grid_t)
    return torch.stack((temporal, height, width), dim=0) + start_position


def compute_cosmos3_mrope_positions(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Nano's text-filled T/H/W positions for one prompt."""

    input_ids = input_ids.to(dtype=torch.long, device="cpu").flatten()
    token_types = mm_token_type_ids.to(dtype=torch.long, device="cpu").flatten()
    if token_types.shape != input_ids.shape:
        raise ValueError("Cosmos3 mm_token_type_ids must match input_ids")

    image_grids = (
        image_grid_thw.to(dtype=torch.long, device="cpu")
        if isinstance(image_grid_thw, torch.Tensor)
        else torch.empty((0, 3), dtype=torch.long)
    )
    if isinstance(video_grid_thw, torch.Tensor):
        video_grids = video_grid_thw.to(dtype=torch.long, device="cpu")
        video_grids = torch.repeat_interleave(
            video_grids,
            video_grids[:, 0],
            dim=0,
        )
        video_grids[:, 0] = 1
    else:
        video_grids = torch.empty((0, 3), dtype=torch.long)
    grid_iters = {1: iter(image_grids), 2: iter(video_grids)}

    current_position = 0
    parts: list[torch.Tensor] = []
    grouped = itertools.groupby(enumerate(token_types.tolist()), lambda item: item[1])
    for modality, group in grouped:
        group_len = sum(1 for _ in group)
        if modality == 0:
            positions = torch.arange(group_len, dtype=torch.long)
            parts.append(positions.unsqueeze(0).expand(3, -1) + current_position)
            current_position += group_len
            continue
        if modality not in grid_iters:
            raise ValueError(f"Unsupported Cosmos3 token type {modality}")
        try:
            grid = next(grid_iters[modality])
        except StopIteration as exc:
            raise ValueError(
                f"Cosmos3 token type {modality} has no matching vision grid"
            ) from exc
        positions = _vision_position_ids(
            current_position,
            grid,
            spatial_merge_size=spatial_merge_size,
        )
        if positions.shape[1] != group_len:
            raise ValueError(
                "Cosmos3 vision positions do not match placeholder tokens: "
                f"positions={positions.shape[1]}, tokens={group_len}"
            )
        parts.append(positions)
        current_position += int(grid[1:].max().item()) // spatial_merge_size

    positions = (
        torch.cat(parts, dim=1) if parts else torch.empty((3, 0), dtype=torch.long)
    )
    delta = positions.max() + 1 - input_ids.numel() if positions.numel() else 0
    return positions, torch.tensor([[int(delta)]], dtype=torch.long)


def build_sglang_text_request(
    state: Cosmos3PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    generation_config: GenerationConfig | None = None,
    request_id: str | None = None,
    thinker_config: Any = None,
) -> SGLangARRequestData:
    """Build one SGLang request from the canonical preprocessing state."""

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = cast(PromptInputs, state.prompt)
    original_input_ids = prompt["input_ids"]
    input_ids = original_input_ids
    thinker_inputs = state.thinker_inputs or {}
    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if thinker_config is not None:
        token_map: dict[int, int] = {}
        pad_values: dict[str, int] = {}
        for modality, token_id in (
            ("image", thinker_config.image_token_id),
            ("video", thinker_config.video_token_id),
        ):
            cache_key = thinker_inputs.get(f"{modality}_cache_key")
            if model_inputs.get(f"{modality}_embeds") is None or cache_key is None:
                continue
            digest = xxhash.xxh3_64(f"{modality}:{cache_key}".encode())
            pad_value = vocab_size + digest.intdigest() % (1 << 62)
            pad_values[modality] = pad_value
            token_map[int(token_id)] = pad_value
        if token_map:
            input_ids = original_input_ids.clone()
            for token_id, pad_value in token_map.items():
                input_ids[input_ids == token_id] = pad_value
            model_inputs["pad_values"] = pad_values

    sampling_kwargs = build_cosmos3_sampling_kwargs(
        params,
        generation_config=generation_config,
    )
    sampling_params = SamplingParams(**sampling_kwargs)
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=request_id or "cosmos3-request",
        origin_input_text=prompt["prompt_text"],
        origin_input_ids=input_ids.tolist(),
        sampling_params=sampling_params,
        eos_token_ids=_eos_token_ids(tokenizer, generation_config),
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer

    if model_inputs and thinker_config is not None:
        from sglang.srt.managers.schedule_batch import MultimodalInputs

        mrope_positions, mrope_delta = compute_cosmos3_mrope_positions(
            original_input_ids,
            prompt["mm_token_type_ids"],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            spatial_merge_size=int(thinker_config.vision_config.spatial_merge_size),
        )
        multimodal_inputs = MultimodalInputs(mm_items=[])
        multimodal_inputs.mrope_positions = mrope_positions
        multimodal_inputs.mrope_position_delta = mrope_delta
        req.multimodal_inputs = multimodal_inputs

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    return SGLangARRequestData(
        input_ids=input_ids,
        attention_mask=prompt["attention_mask"],
        max_new_tokens=int(sampling_kwargs["max_new_tokens"]),
        temperature=float(sampling_kwargs["temperature"]),
        top_p=float(sampling_kwargs["top_p"]),
        top_k=int(sampling_kwargs["top_k"]),
        repetition_penalty=float(sampling_kwargs["repetition_penalty"]),
        model_inputs=model_inputs,
        output_ids=req.output_ids,
        req=req,
    )


def apply_text_result(
    state: Cosmos3PipelineState,
    *,
    stage_name: str,
    result: SGLangARRequestData,
) -> TextOutput:
    text_out: TextOutput = {
        "output_ids": list(result.output_ids),
        "is_final": True,
    }
    finish_reason = result.finish_reason
    if finish_reason is not None:
        text_out["finish_reason"] = finish_reason
    finished_reason = getattr(result.req, "finished_reason", None)
    matched_stop = (
        finished_reason.to_json().get("matched")
        if finished_reason is not None
        else None
    )
    if matched_stop is not None:
        text_out["matched_stop"] = matched_stop
    output_token_logprobs = result.output_token_logprobs
    if output_token_logprobs:
        text_out["output_token_logprobs"] = output_token_logprobs
    weight_version = result.weight_version
    if weight_version is not None:
        text_out["weight_version"] = weight_version
    state.text_out = text_out
    state.engine_outputs[stage_name] = text_out
    return text_out


def make_text_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    generation_config: GenerationConfig | None = None,
    stage_name: str = THINKER_STAGE,
    thinker_config: Any = None,
):
    """Build StagePayload-to-SGLang adapters for the text thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = Cosmos3PipelineState.from_dict(payload.data)
        params = _effective_request_params(
            payload.request.params,
            payload.request.metadata,
        )
        req_data = build_sglang_text_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            generation_config=generation_config,
            request_id=payload.request_id,
            thinker_config=thinker_config,
        )
        req_data.stage_payload = payload
        req_data.return_logprob = bool(payload.request.params.get("return_logprob"))
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = cast(StagePayload, data.stage_payload)
        state = Cosmos3PipelineState.from_dict(payload.data)
        apply_text_result(state, stage_name=stage_name, result=data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


_PENDING_STREAM_IDS_ATTR = "_cosmos3_pending_stream_ids"


def _tail_matches_stop_str_prefix(tail_str: str, stop_strs: list[str]) -> bool:
    """Mirror upstream ``Req.check_match_stop_str_prefix`` over an explicit tail."""
    if not tail_str:
        return False
    for stop_str in stop_strs:
        if not stop_str:
            continue
        if stop_str in tail_str:
            return True
        min_len = min(len(tail_str), len(stop_str))
        for i in range(1, min_len + 1):
            if tail_str[-i:] == stop_str[:i]:
                return True
    return False


def make_text_stream_output_builder(*, decode_stage: str = DECODE_STAGE):
    """Forward generated token ids to the stream-aware detokenizer.

    Mirrors upstream SGLang's OutputStreamer stop handling: tokens whose
    decoded tail could still complete a stop string are held on the request,
    released once the text disambiguates, and otherwise shipped at finish in
    a buffer-only terminal chunk for the detokenizer's matched-stop trim.
    """

    def _token_message(request_id: str, token_id: int) -> OutgoingMessage:
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=torch.tensor([token_id], dtype=torch.long),
            target=decode_stage,
            metadata={"token_id": token_id},
        )

    def stream_output_builder(
        request_id: str,
        req_data: SGLangARRequestData,
        req_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req.inflight_middle_chunks > 0:
            return []
        if req_output.data is None:
            return []
        payload = cast(StagePayload, req_data.stage_payload)
        if not bool(payload.request.params.get("stream", False)):
            return []
        token_id = int(req_output.data)
        # Stop token ids end generation at this token; upstream SGLang trims
        # them from output text, so never forward them to the detokenizer.
        stop_token_ids = req.sampling_params.stop_token_ids
        if stop_token_ids and token_id in stop_token_ids:
            return []
        stop_strs = req.sampling_params.stop_strs
        if not stop_strs:
            return [_token_message(request_id, token_id)]
        pending = getattr(req, _PENDING_STREAM_IDS_ATTR, None)
        if pending is None:
            pending = []
            setattr(req, _PENDING_STREAM_IDS_ATTR, pending)
        pending.append(token_id)
        # This builder runs before process_batch_result appends token_id to
        # req.output_ids, so req.check_match_stop_str_prefix() would test a
        # stale tail; build the stop-check window explicitly instead.
        tail_window = req.sampling_params.stop_str_max_len + 1
        tail_ids = [*req.output_ids[-tail_window:], token_id][-tail_window:]
        if _tail_matches_stop_str_prefix(req.tokenizer.decode(tail_ids), stop_strs):
            return []
        messages = [_token_message(request_id, held_id) for held_id in pending]
        pending.clear()
        return messages

    def flush_stream_output(
        request_id: str, req_data: SGLangARRequestData
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req is None:
            return []
        pending = getattr(req, _PENDING_STREAM_IDS_ATTR, None)
        if not pending:
            return []
        message = OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=torch.tensor(pending, dtype=torch.long),
            target=decode_stage,
            metadata={"terminal_flush": True},
        )
        pending.clear()
        return [message]

    stream_output_builder.flush = flush_stream_output  # type: ignore[attr-defined]
    return stream_output_builder


__all__ = [
    "DECODE_STAGE",
    "THINKER_STAGE",
    "VISION_STAGE",
    "apply_text_result",
    "apply_vision_encoder_result",
    "build_cosmos3_sampling_kwargs",
    "build_sglang_text_request",
    "build_vision_encoder_request",
    "compute_cosmos3_mrope_positions",
    "make_text_scheduler_adapters",
    "make_text_stream_output_builder",
    "project_preprocessing_to_thinker",
    "project_preprocessing_to_vision_encoder",
    "project_thinker_to_decode",
    "project_vision_encoder_to_thinker",
    "resolve_preprocessing_next_stages",
    "resolve_thinker_wait_sources",
]
