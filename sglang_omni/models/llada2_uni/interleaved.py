# SPDX-License-Identifier: Apache-2.0
"""Interleaved text and image generation state for LLaDA2-Uni."""

from __future__ import annotations

import copy
import queue
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    INTERLEAVED_COLLECT_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.serve.protocol import InterleavedGenerationParams

SYSTEM_PROMPT_INTERLEAVED = "You are a interleaved generation assistant."
UNCONDITION_TOKEN = "<uncondition>"


def project_interleaved_payload(payload: StagePayload) -> StagePayload:
    """Give the thinker and asynchronous decoder independent frame snapshots."""
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=copy.deepcopy(payload.data),
    )


def interleaved_next(request_id: str, output: StagePayload) -> str | list[str]:
    interleaved = output.data["stream_state"]["interleaved"]
    if interleaved.get("emit_frame"):
        return [
            IMAGE_DECODE_STAGE,
            INTERLEAVED_COLLECT_STAGE if interleaved.get("done") else THINKER_STAGE,
        ]
    if interleaved.get("done"):
        return INTERLEAVED_COLLECT_STAGE
    return THINKER_STAGE


@dataclass(frozen=True)
class ImageHeader:
    soi_position: int
    grid_h: int
    grid_w: int
    token_ids: tuple[int, ...]

    @property
    def image_token_count(self) -> int:
        return self.grid_h * self.grid_w


@dataclass(frozen=True)
class CFGBranchPlan:
    mode: Literal["none", "simple", "editing"]
    branches: dict[str, list[int]]
    cfg_scale: float
    cfg_text_scale: float
    cfg_image_scale: float
    cfg_rescale: float


def token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"tokenizer does not define {token}")
    return token_id


def parse_reserved_dimension(tokenizer: PreTrainedTokenizerBase, token_id: int) -> int:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    match = re.fullmatch(r"<\|reserved_token_(\d+)\|>", token or "")
    if match is None:
        raise ValueError(f"expected a reserved image dimension token, got {token!r}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError("interleaved image dimensions must be positive")
    return value


def parse_image_header(
    generated_ids: list[int], tokenizer: PreTrainedTokenizerBase
) -> ImageHeader:
    """Parse the final, exact SOI/H/W/BOI text-phase suffix."""
    boi_id = token_id(tokenizer, "<boi>")
    soi_id = token_id(tokenizer, "<|image|>")
    if not generated_ids or generated_ids[-1] != boi_id:
        raise ValueError("interleaved image header must end with <boi>")
    soi_positions = [
        index for index, token_id in enumerate(generated_ids[:-1]) if token_id == soi_id
    ]
    if not soi_positions:
        raise ValueError("interleaved image header has no <|image|> token")
    start = soi_positions[-1]
    token_ids = tuple(int(token_id) for token_id in generated_ids[start:])
    if len(token_ids) != 4:
        raise ValueError("interleaved image header must be exactly SOI/H/W/BOI")
    return ImageHeader(
        soi_position=start,
        grid_h=parse_reserved_dimension(tokenizer, token_ids[1]),
        grid_w=parse_reserved_dimension(tokenizer, token_ids[2]),
        token_ids=token_ids,
    )


def build_cfg_plan(
    *,
    full_ids: list[int],
    header: ImageHeader,
    frame_index: int,
    tokenizer: PreTrainedTokenizerBase,
    config: InterleavedGenerationParams,
) -> CFGBranchPlan:
    """Build the two- or three-row frame plan before shared group alignment."""
    header_ids = list(header.token_ids)
    unconditional = tokenizer.encode(
        f"<role>SYSTEM</role> {SYSTEM_PROMPT_INTERLEAVED} "
        f"<role>HUMAN</role>{UNCONDITION_TOKEN}<role>ASSISTANT</role>",
        add_special_tokens=False,
    )
    eoi_id = token_id(tokenizer, "<|/image|>")
    history_end = max(
        (index for index, token_id in enumerate(full_ids[:-4]) if token_id == eoi_id),
        default=-1,
    )
    use_editing_cfg = config.cfg_text_scale > 0.0 or config.cfg_image_scale > 0.0
    if use_editing_cfg and frame_index > 0 and history_end >= 0:
        history = full_ids[: history_end + 1]
        current_text = full_ids[history_end + 1 : -4]
        no_text = (
            history
            + tokenizer.encode(UNCONDITION_TOKEN, add_special_tokens=False)
            + header_ids
        )
        return CFGBranchPlan(
            "editing",
            {
                "unconditional": no_text,
                "no_image": unconditional + current_text + header_ids,
            },
            config.cfg_text_scale,
            config.cfg_text_scale,
            config.cfg_image_scale,
            config.cfg_rescale,
        )

    scale = config.cfg_scale if config.cfg_scale > 0.0 else config.cfg_text_scale
    if scale == 1.0:
        return CFGBranchPlan("none", {}, 1.0, 0.0, 0.0, config.cfg_rescale)
    return CFGBranchPlan(
        "simple",
        {"unconditional": unconditional + header_ids},
        scale,
        0.0,
        0.0,
        config.cfg_rescale,
    )


def reject_image_tokens(
    token_ids: list[int], *, image_token_offset: int, context: str
) -> None:
    if any(token_id >= image_token_offset for token_id in token_ids):
        raise ValueError(f"interleaved {context} emitted image token(s)")


def mark_done(
    state: LLaDA2UniPipelineState,
    *,
    finish_reason: str,
) -> None:
    interleaved = state.stream_state["interleaved"]
    interleaved["phase"] = "done"
    interleaved["done"] = True
    interleaved["finish_reason"] = finish_reason
    final_length = state.prompt["input_ids"].numel()
    prompt_length = int(interleaved["prompt_length"])
    completion_tokens = max(final_length - prompt_length, 0)
    interleaved["usage"] = {
        "prompt_tokens": prompt_length,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_length + completion_tokens,
    }


def advance_text(
    state: LLaDA2UniPipelineState,
    tokenizer: PreTrainedTokenizerBase,
    *,
    finish_reason: str | None,
) -> None:
    interleaved = state.stream_state["interleaved"]
    output_ids = [
        int(token_id) for token_id in (state.thinker_out or {}).get("output_ids", [])
    ]
    if not output_ids:
        raise ValueError("interleaved text phase produced no tokens")
    boi_id = token_id(tokenizer, "<boi>")
    if output_ids[-1] != boi_id:
        if boi_id in output_ids:
            raise ValueError("interleaved image header must end with <boi>")
        reject_image_tokens(
            output_ids,
            image_token_offset=state.stream_state["image_token_offset"],
            context="text phase",
        )
        prompt_ids = state.prompt["input_ids"]
        old_ids = [int(token_id) for token_id in prompt_ids.flatten().tolist()]
        full_ids = old_ids + output_ids
        segment_start = int(interleaved["segment_start"])
        interleaved["trailing_text"] = tokenizer.decode(
            full_ids[segment_start:], skip_special_tokens=True
        )
        state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
        mark_done(state, finish_reason=finish_reason or "stop")
        return
    header = parse_image_header(output_ids, tokenizer)
    reject_image_tokens(
        output_ids[: header.soi_position],
        image_token_offset=state.stream_state["image_token_offset"],
        context="text phase",
    )
    config = InterleavedGenerationParams.model_validate(
        state.request_metadata["image_generation"]
    )
    if header.image_token_count > config.max_image_tokens:
        raise ValueError(
            f"interleaved image requires {header.image_token_count} tokens; "
            f"limit is {config.max_image_tokens}"
        )
    if header.image_token_count + 1 > config.image_max_new_tokens:
        raise ValueError("interleaved image token budget cannot fit the grid and EOI")
    prompt_ids = state.prompt["input_ids"]
    old_ids = [int(token_id) for token_id in prompt_ids.flatten().tolist()]
    full_ids = old_ids + output_ids
    plan = build_cfg_plan(
        full_ids=full_ids,
        header=header,
        frame_index=int(interleaved["frame_index"]),
        tokenizer=tokenizer,
        config=config,
    )
    longest_prefix = max(len(prefix) for prefix in [full_ids, *plan.branches.values()])
    if longest_prefix + header.image_token_count + 1 > int(interleaved["max_seq_len"]):
        raise ValueError("interleaved image frame would exceed thinker context")
    segment_start = int(interleaved["segment_start"])
    header_start = len(full_ids) - 4
    interleaved["current_frame"] = {
        "index": int(interleaved["frame_index"]) + 1,
        "text": tokenizer.decode(
            full_ids[segment_start:header_start], skip_special_tokens=True
        ),
        "grid_h": header.grid_h,
        "grid_w": header.grid_w,
        "remaining_image_tokens": header.image_token_count,
        "vq_tokens": [],
    }
    interleaved["cfg_plan"] = {
        "mode": plan.mode,
        "branches": copy.deepcopy(plan.branches),
        "cfg_scale": plan.cfg_scale,
        "cfg_text_scale": plan.cfg_text_scale,
        "cfg_image_scale": plan.cfg_image_scale,
        "cfg_rescale": plan.cfg_rescale,
    }
    interleaved["phase"] = "image"
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
    state.thinker_out = None
    state.engine_outputs.pop("thinker", None)


def advance_image(
    state: LLaDA2UniPipelineState, tokenizer: PreTrainedTokenizerBase
) -> None:
    interleaved = state.stream_state["interleaved"]
    current = interleaved["current_frame"]
    output_ids = [
        int(token_id) for token_id in (state.thinker_out or {}).get("output_ids", [])
    ]
    if not output_ids:
        raise ValueError("interleaved image phase produced no tokens")
    eoi_id = token_id(tokenizer, "<|/image|>")
    eoi_positions = [
        index for index, token_id in enumerate(output_ids) if token_id == eoi_id
    ]
    if len(eoi_positions) > 1:
        raise ValueError("interleaved image phase produced multiple EOI tokens")
    if eoi_positions and eoi_positions[0] != len(output_ids) - 1:
        raise ValueError("interleaved image phase produced tokens after EOI")
    image_ids = output_ids[: eoi_positions[0]] if eoi_positions else output_ids
    remaining = int(current["remaining_image_tokens"])
    if len(image_ids) > remaining:
        raise ValueError(
            f"interleaved image phase produced {len(image_ids)} tokens with {remaining} remaining"
        )
    if any(
        token_id < state.stream_state["image_token_offset"] for token_id in image_ids
    ):
        raise ValueError(
            "interleaved image phase produced a non-image token before EOI"
        )
    accumulated = list(current["vq_tokens"]) + image_ids
    remaining -= len(image_ids)
    if not eoi_positions:
        if remaining == 0:
            raise ValueError("interleaved image phase completed VQ tokens without EOI")
        append_image_phase_tokens(state, interleaved, image_ids)
        current["vq_tokens"] = accumulated
        current["remaining_image_tokens"] = remaining
        state.thinker_out = None
        state.engine_outputs.pop("thinker", None)
        return
    if remaining != 0:
        raise ValueError(
            f"interleaved image phase emitted EOI with {remaining} VQ tokens remaining"
        )
    append_image_phase_tokens(state, interleaved, image_ids + [eoi_id])
    full_ids = [
        int(token_id) for token_id in state.prompt["input_ids"].flatten().tolist()
    ]
    state.thinker_out = {"output_ids": accumulated, "is_final": True}
    state.engine_outputs["thinker"] = dict(state.thinker_out)
    frame_index = int(current["index"])
    interleaved["frame_index"] = frame_index
    interleaved["segments"].append(
        {
            "frame_index": frame_index,
            "text": current["text"],
            "grid_h": current["grid_h"],
            "grid_w": current["grid_w"],
        }
    )
    interleaved["phase"] = "text"
    interleaved["segment_start"] = len(full_ids)
    interleaved["emit_frame"] = True
    interleaved.pop("current_frame", None)
    interleaved.pop("cfg_plan", None)
    if frame_index >= int(interleaved["max_frames"]):
        mark_done(state, finish_reason="max_frames")
    elif len(full_ids) >= interleaved["max_seq_len"]:
        mark_done(state, finish_reason="length")


def append_image_phase_tokens(
    state: LLaDA2UniPipelineState,
    interleaved: dict[str, object],
    token_ids: list[int],
) -> None:
    input_ids = state.prompt["input_ids"]
    full_ids = [int(token_id) for token_id in input_ids.flatten().tolist()]
    full_ids.extend(token_ids)
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}

    for branch in interleaved["cfg_plan"]["branches"].values():
        branch.extend(token_ids)


def advance_interleaved_state(
    state: LLaDA2UniPipelineState,
    tokenizer: PreTrainedTokenizerBase,
    *,
    completed_phase: str,
    finish_reason: str | None = None,
) -> None:
    """Advance one phase and commit only after every validation succeeds."""
    working = LLaDA2UniPipelineState.from_dict(copy.deepcopy(state.to_dict()))
    interleaved = working.stream_state.get("interleaved")
    if not isinstance(interleaved, dict):
        raise ValueError("request is missing interleaved generation state")
    if interleaved.get("phase") != completed_phase:
        raise ValueError(
            f"interleaved phase mismatch: expected {interleaved.get('phase')!r}, "
            f"got {completed_phase!r}"
        )
    if completed_phase == "text":
        advance_text(
            working,
            tokenizer,
            finish_reason=finish_reason,
        )
    elif completed_phase == "image":
        advance_image(working, tokenizer)
    else:
        raise ValueError(f"unsupported interleaved phase {completed_phase!r}")
    state.__dict__.update(working.__dict__)


class InterleavedCollectorScheduler:
    """Join independently decoded frames with the final thinker state."""

    TOMBSTONE_LIMIT = 4096

    def __init__(self) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.requires_tp_work_fanout = False
        self.allow_multiple_inflight_per_request = False
        self.running = False
        self.lock = threading.Lock()
        self.closed: OrderedDict[tuple[str, str | None], None] = OrderedDict()
        self.frames: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
        self.final_payloads: dict[tuple[str, str], StagePayload] = {}

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.type == "new_request":
                try:
                    with self.lock:
                        self.receive(message.request_id, message.data)
                except Exception as exc:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=message.request_id, type="error", data=exc
                        )
                    )

    def stop(self) -> None:
        self.running = False

    def close_request(self, key: tuple[str, str | None]) -> None:
        self.frames.pop(key, None)
        self.final_payloads.pop(key, None)
        self.closed[key] = None
        self.closed.move_to_end(key)
        while len(self.closed) > self.TOMBSTONE_LIMIT:
            self.closed.popitem(last=False)

    def abort(self, request_id: str) -> None:
        with self.lock:
            for key in self.frames.keys() | self.final_payloads.keys():
                if key[0] == request_id:
                    self.close_request(key)
            self.close_request((request_id, None))

    def receive(self, request_id: str, payload: StagePayload) -> None:
        if (request_id, None) in self.closed:
            return
        key = (request_id, payload.request.metadata["interleaved_generation_id"])
        if key in self.closed:
            return
        try:
            data = payload.data
            if data.get("kind") == "interleaved_frame":
                frame = data["frame"]
                frames = self.frames.setdefault(key, {})
                index = frame["index"]
                if index <= 0 or index in frames:
                    raise ValueError(f"invalid or duplicate decoded frame {index}")
                frames[index] = frame["image"]
            else:
                state = LLaDA2UniPipelineState.from_dict(data)
                if (
                    state.task_kind != "interleaved"
                    or not state.stream_state["interleaved"]["done"]
                ):
                    raise ValueError(
                        "collector received a non-final interleaved payload"
                    )
                if key in self.final_payloads:
                    raise ValueError("duplicate interleaved final payload")
                self.final_payloads[key] = payload
            self.maybe_finish(key)
        except Exception:
            self.close_request(key)
            raise

    def maybe_finish(self, key: tuple[str, str]) -> None:
        payload = self.final_payloads.get(key)
        if payload is None:
            return
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        interleaved = state.stream_state["interleaved"]
        expected = interleaved["frame_index"]
        frames = self.frames.get(key, {})
        expected_indexes = set(range(1, expected + 1))
        if set(frames) - expected_indexes:
            raise ValueError("collector received an unexpected interleaved frame")
        if set(frames) != expected_indexes:
            return

        content: list[dict[str, object]] = []
        images: list[dict[str, object]] = []
        for index, segment in enumerate(interleaved["segments"], 1):
            if segment["text"]:
                content.append({"type": "text", "text": segment["text"]})
            image = frames[index]
            images.append(image)
            content.append({"type": "image_ref", "image_id": image["id"]})
        trailing_text = interleaved.get("trailing_text", "")
        if trailing_text:
            content.append({"type": "text", "text": trailing_text})
        payload.data = {
            "modality": "interleaved",
            "content": content,
            "images": images,
            "finish_reason": interleaved["finish_reason"],
            "usage": interleaved["usage"],
        }
        self.outbox.put(OutgoingMessage(request_id=key[0], type="result", data=payload))
        self.close_request(key)


def create_interleaved_collector_executor(
    model_path: str,
) -> InterleavedCollectorScheduler:
    return InterleavedCollectorScheduler()
