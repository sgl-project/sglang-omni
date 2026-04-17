# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3-Omni pipelines.

Each factory returns either:
- A callable (compute_fn) for simple stages
- An OmniScheduler for AR stages
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.merge import decode_events
from sglang_omni.models.qwen3_omni.payload_types import OmniEvent
from sglang_omni.models.qwen3_omni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> PipelineState:
    return PipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def _run_single_encoder_payload(
    payload: StagePayload,
    *,
    stage_name: str,
    model: Any,
) -> StagePayload:
    state = load_state(payload)
    request = build_encoder_request(state, stage_name=stage_name)
    if request.get("_skip"):
        result = request.get("_result", {})
    else:
        with torch.no_grad():
            result = model(**request.input_dict)
    apply_encoder_result(state, stage_name=stage_name, result=result)
    return store_state(payload, state)


def _image_request_is_batchable(request: Any) -> bool:
    if request.get("_skip"):
        return False
    input_dict = request.input_dict
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        value = input_dict.get(key)
        if value is not None and not isinstance(value, torch.Tensor):
            return False
    return True


def _split_visual_features(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _split_visual_multiscale(
    tensors: list[torch.Tensor] | None,
    *,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if tensors is None:
        return None
    return [tensor[start:end] for tensor in tensors]


def _batch_image_encoder_payloads(
    payloads: list[StagePayload],
    *,
    model: Any,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[tuple[int, StagePayload, Any, Any]] = []

    for idx, payload in enumerate(payloads):
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.get("_skip") or not _image_request_is_batchable(request):
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=IMAGE_STAGE,
                model=model,
            )
        else:
            active.append((idx, payload, state, request))

    if not active:
        return [result for result in results if result is not None]

    image_pixels: list[torch.Tensor] = []
    image_grids: list[torch.Tensor] = []
    video_pixels: list[torch.Tensor] = []
    video_grids: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []
    merge = model.spatial_merge_size**2

    for idx, payload, state, request in active:
        input_dict = request.input_dict
        image_grid = input_dict.get("image_grid_thw")
        video_grid = input_dict.get("video_grid_thw")
        image_rows = (
            int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
        )
        video_rows = (
            int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
        )
        image_token_counts = (
            (image_grid.prod(-1) // merge).to(dtype=torch.long)
            if isinstance(image_grid, torch.Tensor)
            else None
        )
        video_token_counts = (
            (video_grid.prod(-1) // merge).to(dtype=torch.long)
            if isinstance(video_grid, torch.Tensor)
            else None
        )
        image_token_total = (
            int(image_token_counts.sum().item())
            if isinstance(image_token_counts, torch.Tensor)
            else 0
        )
        video_token_total = (
            int(video_token_counts.sum().item())
            if isinstance(video_token_counts, torch.Tensor)
            else 0
        )
        if isinstance(input_dict.get("pixel_values"), torch.Tensor):
            image_pixels.append(input_dict["pixel_values"])
            image_grids.append(image_grid)
        if isinstance(input_dict.get("pixel_values_videos"), torch.Tensor):
            video_pixels.append(input_dict["pixel_values_videos"])
            video_grids.append(video_grid)
        metas.append(
            {
                "idx": idx,
                "payload": payload,
                "state": state,
                "image_rows": image_rows,
                "video_rows": video_rows,
                "image_token_total": image_token_total,
                "video_token_total": video_token_total,
            }
        )

    batched_inputs: dict[str, Any] = {}
    if image_pixels:
        batched_inputs["pixel_values"] = torch.cat(image_pixels, dim=0)
        batched_inputs["image_grid_thw"] = torch.cat(image_grids, dim=0)
    if video_pixels:
        batched_inputs["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
        batched_inputs["video_grid_thw"] = torch.cat(video_grids, dim=0)

    with torch.no_grad():
        combined = model(**batched_inputs)

    image_grid_all = combined.get("image_grid_thw")
    image_counts_all = combined.get("image_token_counts")
    image_embeds_all = combined.get("image_embeds")
    image_multiscale_all = combined.get("deepstack_visual_embeds_image")
    video_grid_all = combined.get("video_grid_thw")
    video_counts_all = combined.get("video_token_counts")
    video_embeds_all = combined.get("video_embeds")
    video_multiscale_all = combined.get("deepstack_visual_embeds_video")

    image_row_cursor = 0
    image_token_cursor = 0
    video_row_cursor = 0
    video_token_cursor = 0
    for meta in metas:
        stage_result: dict[str, Any] = {}
        if meta["image_rows"] > 0:
            row_end = image_row_cursor + meta["image_rows"]
            token_end = image_token_cursor + meta["image_token_total"]
            stage_result["image_embeds"] = _split_visual_features(
                image_embeds_all, start=image_token_cursor, end=token_end
            )
            stage_result["image_grid_thw"] = image_grid_all[image_row_cursor:row_end]
            stage_result["image_token_counts"] = image_counts_all[
                image_row_cursor:row_end
            ]
            stage_result["deepstack_visual_embeds_image"] = _split_visual_multiscale(
                image_multiscale_all,
                start=image_token_cursor,
                end=token_end,
            )
            image_row_cursor = row_end
            image_token_cursor = token_end
        if meta["video_rows"] > 0:
            row_end = video_row_cursor + meta["video_rows"]
            token_end = video_token_cursor + meta["video_token_total"]
            stage_result["video_embeds"] = _split_visual_features(
                video_embeds_all, start=video_token_cursor, end=token_end
            )
            stage_result["video_grid_thw"] = video_grid_all[video_row_cursor:row_end]
            stage_result["video_token_counts"] = video_counts_all[
                video_row_cursor:row_end
            ]
            stage_result["deepstack_visual_embeds_video"] = _split_visual_multiscale(
                video_multiscale_all,
                start=video_token_cursor,
                end=token_end,
            )
            video_row_cursor = row_end
            video_token_cursor = token_end
        apply_encoder_result(meta["state"], stage_name=IMAGE_STAGE, result=stage_result)
        results[meta["idx"]] = store_state(meta["payload"], meta["state"])

    return [result for result in results if result is not None]


def _audio_request_is_batchable(request: Any) -> bool:
    if request.get("_skip"):
        return False
    input_dict = request.input_dict
    features = input_dict.get("input_features")
    if not isinstance(features, torch.Tensor):
        return False
    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    return (lengths is None or isinstance(lengths, torch.Tensor)) and (
        mask is None or isinstance(mask, torch.Tensor)
    )


def _normalize_audio_request_tensors(
    request: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dict = request.input_dict
    features = input_dict["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    elif isinstance(mask, torch.Tensor):
        lengths = mask.to(dtype=torch.long).sum(dim=1).view(-1)
    else:
        raise ValueError("audio_feature_lengths or feature_attention_mask is required")

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        steps = torch.arange(time_dim, dtype=torch.long).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)


def _batch_audio_encoder_payloads(
    payloads: list[StagePayload],
    *,
    model: Any,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[tuple[int, StagePayload, Any, Any]] = []

    for idx, payload in enumerate(payloads):
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=AUDIO_STAGE)
        if request.get("_skip") or not _audio_request_is_batchable(request):
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=AUDIO_STAGE,
                model=model,
            )
        else:
            active.append((idx, payload, state, request))

    if not active:
        return [result for result in results if result is not None]

    normalized = []
    max_time = 0
    for idx, payload, state, request in active:
        features, mask, lengths = _normalize_audio_request_tensors(request)
        max_time = max(max_time, int(features.shape[-1]))
        normalized.append(
            {
                "idx": idx,
                "payload": payload,
                "state": state,
                "features": features,
                "mask": mask,
                "lengths": lengths,
                "count": int(lengths.shape[0]),
            }
        )

    batched_features = torch.cat(
        [_pad_audio_features(item["features"], max_time) for item in normalized], dim=0
    )
    batched_mask = torch.cat(
        [_pad_audio_mask(item["mask"], max_time) for item in normalized], dim=0
    )
    batched_lengths = torch.cat([item["lengths"] for item in normalized], dim=0)

    with torch.no_grad():
        combined = model(
            input_features=batched_features,
            feature_attention_mask=batched_mask,
            audio_feature_lengths=batched_lengths,
        )

    output_lengths = combined["audio_output_lengths"]
    embeds = combined["audio_embeds"]
    row_cursor = 0
    token_cursor = 0
    for item in normalized:
        row_end = row_cursor + item["count"]
        req_output_lengths = output_lengths[row_cursor:row_end]
        token_end = token_cursor + int(req_output_lengths.sum().item())
        stage_result = {
            "audio_embeds": embeds[token_cursor:token_end],
            "audio_feature_lengths": combined["audio_feature_lengths"][
                row_cursor:row_end
            ],
            "audio_output_lengths": req_output_lengths,
        }
        apply_encoder_result(item["state"], stage_name=AUDIO_STAGE, result=stage_result)
        results[item["idx"]] = store_state(item["payload"], item["state"])
        row_cursor = row_end
        token_cursor = token_end

    return [result for result in results if result is not None]


# ---------------------------------------------------------------------------
# Simple stages — return SimpleScheduler
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_path: str):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = Qwen3OmniPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        return _run_single_encoder_payload(
            payload,
            stage_name=IMAGE_STAGE,
            model=model,
        )

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return _batch_image_encoder_payloads(payloads, model=model)

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=8,
    )


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        return _run_single_encoder_payload(
            payload,
            stage_name=AUDIO_STAGE,
            model=model,
        )

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return _batch_audio_encoder_payloads(payloads, model=model)

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=8,
    )


def create_decode_executor(model_path: str):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

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
                e
                for e in reversed(events)
                if e.is_final or e.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if isinstance(output_ids, list) and output_ids:
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
                result.setdefault("modality", "text")

        payload.data = result
        return payload

    return SimpleScheduler(_decode)


# ---------------------------------------------------------------------------
# AR stages — return OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = False,
):
    """Returns OmniScheduler for thinker."""
    from sglang_omni.scheduling.factory import create_thinker_scheduler

    overrides = {"disable_cuda_graph": True}
    if server_args_overrides:
        overrides.update(server_args_overrides)
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    return create_thinker_scheduler(
        server_args,
        gpu_id,
        speech_enabled=speech_enabled,
    )


def create_talker_ar_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    talker_max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
):
    """Returns OmniScheduler for talker."""
    from sglang_omni.scheduling.factory import create_talker_scheduler

    overrides = {"disable_cuda_graph": True}
    if server_args_overrides:
        overrides.update(server_args_overrides)
    server_args = build_sglang_server_args(
        model_path,
        context_length=talker_max_seq_len,
        **overrides,
    )
    return create_talker_scheduler(
        server_args,
        gpu_id,
        weight_prefix=weight_prefix,
        speech_enabled=speech_enabled,
        feedback_enabled=feedback_enabled,
    )
