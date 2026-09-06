# SPDX-License-Identifier: Apache-2.0
"""Vision encoder scheduling, batching, and caching for Cosmos3."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.cosmos3.components.vision_encoder import Cosmos3VisionEncoder
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.models.cosmos3.request_builders import (
    apply_vision_encoder_result,
    build_vision_encoder_request,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache

VISION_ENCODER_CACHE_MAX_BYTES = 4 * 1024**3
VISION_ENCODER_CACHE_MAX_ENTRIES = 64
VISION_ENCODER_BATCH_BUDGET_BYTES = 10 * 1024**3
VISION_ENCODER_ACTIVATION_MULTIPLIER = 5


def _store_state(
    payload: StagePayload,
    state: Cosmos3PipelineState,
) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _run_vision_request(
    payload: StagePayload,
    *,
    model: Cosmos3VisionEncoder,
    cache: StageOutputCache,
) -> StagePayload:
    state = Cosmos3PipelineState.from_dict(payload.data)
    request = build_vision_encoder_request(state)
    if request.skip_result is not None:
        result = request.skip_result
    else:
        result = cache.get(request.cache_key)
        if result is None:
            with torch.no_grad():
                result = model(**request.model_inputs)
            cache.put(request.cache_key, result)
    apply_vision_encoder_result(state, result)
    return _store_state(payload, state)


def _split_layers(
    layers: list[torch.Tensor] | None,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if layers is None:
        return None
    return [layer[start:end] for layer in layers]


def _batch_vision_requests(
    payloads: list[StagePayload],
    *,
    model: Cosmos3VisionEncoder,
    cache: StageOutputCache,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[tuple[int, StagePayload, Cosmos3PipelineState, Any]] = []
    leaders: dict[str, int] = {}
    waiters: dict[int, list[tuple[int, StagePayload, Cosmos3PipelineState]]] = {}

    for index, payload in enumerate(payloads):
        state = Cosmos3PipelineState.from_dict(payload.data)
        request = build_vision_encoder_request(state)
        if request.skip_result is not None:
            apply_vision_encoder_result(state, request.skip_result)
            results[index] = _store_state(payload, state)
            continue
        cached = cache.get(request.cache_key)
        if cached is not None:
            apply_vision_encoder_result(state, cached)
            results[index] = _store_state(payload, state)
            continue
        if request.cache_key is not None and request.cache_key in leaders:
            leader = leaders[request.cache_key]
            waiters.setdefault(leader, []).append((index, payload, state))
            continue
        leader = len(active)
        active.append((index, payload, state, request))
        if request.cache_key is not None:
            leaders[request.cache_key] = leader

    if active:
        image_pixels: list[torch.Tensor] = []
        image_grids: list[torch.Tensor] = []
        video_pixels: list[torch.Tensor] = []
        video_grids: list[torch.Tensor] = []
        metadata: list[dict[str, int]] = []
        merge = model.spatial_merge_size**2

        for _, _, _, request in active:
            values = request.model_inputs
            image_grid = values.get("image_grid_thw")
            video_grid = values.get("video_grid_thw")
            image_rows = (
                int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
            )
            video_rows = (
                int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
            )
            image_tokens = (
                int((image_grid.prod(-1) // merge).sum().item())
                if isinstance(image_grid, torch.Tensor)
                else 0
            )
            video_tokens = (
                int((video_grid.prod(-1) // merge).sum().item())
                if isinstance(video_grid, torch.Tensor)
                else 0
            )
            if isinstance(values.get("pixel_values"), torch.Tensor):
                image_pixels.append(values["pixel_values"])
                image_grids.append(image_grid)
            if isinstance(values.get("pixel_values_videos"), torch.Tensor):
                video_pixels.append(values["pixel_values_videos"])
                video_grids.append(video_grid)
            metadata.append(
                {
                    "image_rows": image_rows,
                    "video_rows": video_rows,
                    "image_tokens": image_tokens,
                    "video_tokens": video_tokens,
                }
            )

        batched: dict[str, torch.Tensor] = {}
        if image_pixels:
            batched["pixel_values"] = torch.cat(image_pixels, dim=0)
            batched["image_grid_thw"] = torch.cat(image_grids, dim=0)
        if video_pixels:
            batched["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
            batched["video_grid_thw"] = torch.cat(video_grids, dim=0)
        with torch.no_grad():
            combined = model(**batched)

        image_row = image_token = video_row = video_token = 0
        for leader, ((index, payload, state, request), meta) in enumerate(
            zip(active, metadata)
        ):
            stage_result: dict[str, Any] = {}
            if meta["image_rows"]:
                row_end = image_row + meta["image_rows"]
                token_end = image_token + meta["image_tokens"]
                stage_result.update(
                    image_embeds=combined["image_embeds"][image_token:token_end],
                    image_grid_thw=combined["image_grid_thw"][image_row:row_end],
                    image_token_counts=combined["image_token_counts"][
                        image_row:row_end
                    ],
                    deepstack_visual_embeds_image=_split_layers(
                        combined.get("deepstack_visual_embeds_image"),
                        image_token,
                        token_end,
                    ),
                )
                image_row, image_token = row_end, token_end
            if meta["video_rows"]:
                row_end = video_row + meta["video_rows"]
                token_end = video_token + meta["video_tokens"]
                stage_result.update(
                    video_embeds=combined["video_embeds"][video_token:token_end],
                    video_grid_thw=combined["video_grid_thw"][video_row:row_end],
                    video_token_counts=combined["video_token_counts"][
                        video_row:row_end
                    ],
                    deepstack_visual_embeds_video=_split_layers(
                        combined.get("deepstack_visual_embeds_video"),
                        video_token,
                        token_end,
                    ),
                )
                video_row, video_token = row_end, token_end

            cache.put(request.cache_key, stage_result)
            apply_vision_encoder_result(state, stage_result)
            results[index] = _store_state(payload, state)
            for waiter_index, waiter_payload, waiter_state in waiters.get(leader, []):
                apply_vision_encoder_result(waiter_state, stage_result)
                results[waiter_index] = _store_state(waiter_payload, waiter_state)

    if any(result is None for result in results):
        raise RuntimeError("Cosmos3 vision batch did not produce every result")
    return [result for result in results if result is not None]


def _vision_request_cost(model: Cosmos3VisionEncoder):
    merge = model.spatial_merge_size**2
    output_width = model.out_hidden_size * (1 + model.deepstack_layers)
    dtype_bytes = model.visual_dtype_bytes

    def _cost(payload: StagePayload) -> int:
        state = Cosmos3PipelineState.from_dict(payload.data)
        request = build_vision_encoder_request(state)
        if request.skip_result is not None:
            return 0
        inputs = request.model_inputs
        raw_bytes = sum(
            int(value.numel() * value.element_size())
            for key in ("pixel_values", "pixel_values_videos")
            if isinstance((value := inputs.get(key)), torch.Tensor)
        )
        visual_tokens = sum(
            int((value.prod(-1) // merge).sum().item())
            for key in ("image_grid_thw", "video_grid_thw")
            if isinstance((value := inputs.get(key)), torch.Tensor)
        )
        output_bytes = visual_tokens * output_width * dtype_bytes
        return (raw_bytes + output_bytes) * VISION_ENCODER_ACTIVATION_MULTIPLIER

    return _cost


def create_vision_encoder_scheduler(
    model_path: str,
    *,
    revision: str | None = None,
    device: str = "cuda",
    dtype: str | None = None,
) -> SimpleScheduler:
    """Create the cached, dynamically batched vision encoder scheduler."""

    model = Cosmos3VisionEncoder(
        model_path, revision=revision, device=device, dtype=dtype
    )
    cache = StageOutputCache(
        max_size=VISION_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=VISION_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )
    return SimpleScheduler(
        lambda payload: _run_vision_request(payload, model=model, cache=cache),
        batch_compute_fn=lambda payloads: _batch_vision_requests(
            payloads,
            model=model,
            cache=cache,
        ),
        max_batch_size=32,
        max_batch_wait_ms=50,
        request_cost_fn=_vision_request_cost(model),
        max_batch_cost=VISION_ENCODER_BATCH_BUDGET_BYTES,
    )


__all__ = ["create_vision_encoder_scheduler"]
