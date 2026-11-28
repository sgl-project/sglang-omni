# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlunparse

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    pass

from sglang_omni.models.llada2_uni.components.preprocessor import IMAGE_TOKEN_OFFSET
from sglang_omni.models.llada2_uni.config import THINKER_STAGE
from sglang_omni.models.llada2_uni.payload_types import (
    ImageGrid,
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
    ThinkerOutput,
)
from sglang_omni.proto import StagePayload


class LLaDA2UniEventDict(TypedDict):
    type: str
    modality: str
    payload: dict[str, str | list[str]]
    is_final: bool


def event_to_dict(event: LLaDA2UniEvent) -> LLaDA2UniEventDict:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": event.is_final,
    }


def extract_image_vq_tokens(
    state: LLaDA2UniPipelineState,
) -> tuple[list[int], int, int, dict[str, object]] | None:
    """Return decoder codebook IDs, semantic grid size, and generation options."""
    if state.task_kind not in ("t2i", "edit", "interleaved"):
        return None
    else:
        pass
    thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
    if not isinstance(thinker_out, dict):
        return None
    else:
        pass
    tokens = [
        int(tid) - IMAGE_TOKEN_OFFSET
        for tid in thinker_out.get("output_ids", [])
        if isinstance(tid, int) and tid >= IMAGE_TOKEN_OFFSET
    ]
    if not tokens:
        return None
    else:
        pass
    image_grid: ImageGrid
    if state.task_kind == "interleaved":
        image_grid = state.stream_state["interleaved"]["segments"][-1]
    else:
        image_grid = state.stream_state["image_info"][0]
    grid_height, grid_width = image_grid["grid_h"], image_grid["grid_w"]
    if grid_height * grid_width != len(tokens):
        raise ValueError(
            f"Image output grid {grid_height}x{grid_width} does not match {len(tokens)} VQ tokens"
        )
    else:
        pass
    params = state.request_metadata.get("image_generation", {})
    return tokens, grid_height, grid_width, params if isinstance(params, dict) else {}


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    tokenizer: "PreTrainedTokenizerBase",
) -> list[LLaDA2UniEvent]:
    """Convert thinker output tokens to a text_final event."""
    # TODO: add streaming support
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []
    else:
        pass

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return [
        LLaDA2UniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]


def build_image_decode_step(
    decode_png: Callable[[list[int], int, int, dict[str, object]], bytes],
    *,
    enable_serial_offload: bool = False,
) -> Callable[[StagePayload], StagePayload]:
    def decode_image(payload: StagePayload) -> StagePayload:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        if enable_serial_offload and state.task_kind == "interleaved":
            raise ValueError(
                "LLaDA2-Uni serial offload does not support interleaved generation"
            )
        else:
            pass
        result = extract_image_vq_tokens(state)
        if result is None:
            if state.task_kind in ("t2i", "edit"):
                raise ValueError(
                    f"{state.task_kind} request did not produce image VQ tokens"
                )
            else:
                pass
            payload.data = {"events": [], "modality": "image", "skipped": True}
            return payload
        else:
            pass

        vq_tokens, grid_height, grid_width, generation_params = result
        call_kwargs = {
            target: generation_params[source]
            for source, target in (
                ("decode_mode", "decode_mode"),
                ("decoder_steps", "num_steps"),
                ("seed", "seed"),
            )
            if generation_params.get(source) is not None
        }
        if (
            call_kwargs.get("decode_mode") == "decoder-turbo"
            and "num_steps" not in call_kwargs
        ):
            call_kwargs["num_steps"] = 8
        else:
            pass
        png = decode_png(vq_tokens, grid_height, grid_width, call_kwargs)
        image_b64 = base64.b64encode(png).decode("ascii")
        if state.task_kind == "interleaved":
            # One inline image segment for the UMM controller.
            payload.data = {
                "kind": "image",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "url": urlunparse(("data", "", "image/png", "base64,", "", ""))
                + image_b64,
            }
            return payload
        else:
            pass
        event = LLaDA2UniEvent(
            type="image_final",
            modality="image",
            payload={"image": image_b64, "format": "png"},
            is_final=True,
        )
        payload.data = {
            "events": [event_to_dict(event)],
            "modality": "image",
            **event.payload,
        }
        return payload

    return decode_image
