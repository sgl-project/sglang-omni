"""Small Intent-VL serializer for the live Realization adapter.

The original training repository supplies a richer frame-v5 serializer.  This
module keeps the serving checkout self-contained while preserving the contract
used by :class:`LiveIntentPlanner`: cumulative input ids, current visual patches,
and the Intent video grid.  It deliberately leaves all sampling decisions in the
planner so its KV state remains append-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, Qwen3VLProcessor

SPECIALS = ["<wait>"] + [
    f"<emo_{name}>" for name in ("neutral", "happy", "sad", "angry")
]


def video_batch(
    processor: Qwen3VLProcessor, frames: np.ndarray
) -> dict[str, torch.Tensor]:
    """Run the local processor on one RGB clip."""

    clip = np.asarray(frames)
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(
            f"expected [frames,height,width,3] RGB input, got {clip.shape}"
        )
    else:
        pass
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": clip},
                {"type": "text", "text": ""},
            ],
        }
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError):
        prompt = "Describe the video."
    batch = processor(
        text=[prompt],
        videos=[clip],
        return_tensors="pt",
    )
    return {
        "input_ids": batch.input_ids,
        "patches": batch.pixel_values_videos,
        "grid": batch.video_grid_thw,
    }


def visual_patch(
    processor: Qwen3VLProcessor, frames: np.ndarray
) -> dict[str, torch.Tensor | np.ndarray | Qwen3VLProcessor]:
    result = video_batch(processor, frames)
    return {
        **result,
        "frames": np.asarray(frames, dtype=np.uint8).copy(),
        "processor": processor,
    }


def serialize(
    record: dict[str, object],
    tokenizer: PreTrainedTokenizerBase,
    visuals: dict[tuple[str, int], dict[str, object]],
    *,
    stop_block: int,
    assistant_bodies: dict[int, list[int]],
    generation: bool,
    block_duration_ms: float = 1000 / 6,
) -> dict[str, object]:
    """Serialize a stable cumulative Intent conversation.

    Each completed block contributes one user video turn and one assistant
    response.  The current block has no assistant response, so the returned
    suffix is exactly the new visual/text prefix consumed by the KV-aware
    planner.
    """

    messages: list[dict[str, object]] = []
    videos: list[np.ndarray] = []
    blocks = record.get("blocks", [])
    text_tokens = record.get("canonical", {}).get("text_tokens", [])
    for index in range(stop_block + 1):
        visual = visuals["user", index]
        frames = np.asarray(visual["frames"], dtype=np.uint8)
        videos.append(frames)
        block_text = "".join(
            tokenizer.decode([int(item["token_id"])])
            for item in text_tokens
            if float(item["start_ms"]) >= index * block_duration_ms
            and float(item["start_ms"]) < (index + 1) * block_duration_ms
        )
        content: list[dict[str, object]] = [{"type": "video", "video": frames}]
        if block_text:
            content.append({"type": "text", "text": block_text})
        else:
            pass
        messages.append({"role": "user", "content": content})
        body = assistant_bodies.get(index)
        if body:
            messages.append(
                {
                    "role": "assistant",
                    "content": tokenizer.decode(body, skip_special_tokens=False),
                }
            )
        else:
            pass
    try:
        prompt = visuals["user", stop_block].get("processor_prompt")
        processor = visuals["user", stop_block].get("processor")
        if processor is None:
            raise ValueError
        else:
            pass
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        grid = encoded.get("video_grid_thw", visuals["user", stop_block]["grid"])
    except (AttributeError, KeyError, TypeError, ValueError):
        # The processor is attached by ``visual_patch`` below.  Keeping this
        # fallback makes malformed custom processors fail with a useful tensor
        # contract rather than an obscure attribute error.
        processor = visuals["user", stop_block]["processor"]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation
        )
        encoded = processor(text=[prompt], videos=videos, return_tensors="pt")
        input_ids = encoded.input_ids
        grid = encoded.video_grid_thw
    return {
        "input_ids": input_ids,
        "video_grid_thw": grid,
        "keep": [True] * int(input_ids.shape[1]),
    }


# Keep the nested namespace used by the original planner.
intent = SimpleNamespace(SPECIALS=SPECIALS)
