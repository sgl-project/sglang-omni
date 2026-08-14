# SPDX-License-Identifier: Apache-2.0
"""Interleaved text and image generation state for LLaDA2-Uni."""

from __future__ import annotations

import copy
import math
import queue
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import torch

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

SYSTEM_PROMPT_INTERLEAVED = "You are an interleaved text and image assistant."
UNCONDITION_TOKEN = "<uncondition>"

_IMAGE_GENERATION_FIELDS = frozenset(
    {
        "mode",
        "max_frames",
        "text_max_new_tokens",
        "cfg_scale",
        "cfg_text_scale",
        "cfg_image_scale",
        "cfg_rescale",
        "decoder_steps",
        "seed",
        "max_image_tokens",
        "format",
        "decode_mode",
    }
)


def _positive_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"image_generation.{name} must be a positive integer")
    return value


def _non_negative_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"image_generation.{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"image_generation.{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class InterleavedGenerationConfig:
    """Validated controls for one interleaved request."""

    max_frames: int = 10
    text_max_new_tokens: int = 8192
    cfg_scale: float = 0.0
    cfg_text_scale: float = 7.5
    cfg_image_scale: float = 1.5
    cfg_rescale: float = 0.7
    decoder_steps: int = 50
    seed: int | None = None
    max_image_tokens: int = 4096
    format: str = "png"
    decode_mode: str = "normal"

    @classmethod
    def from_image_generation(
        cls, image_generation: dict[str, Any]
    ) -> "InterleavedGenerationConfig":
        if not isinstance(image_generation, dict):
            raise TypeError("image_generation must be an object")
        if image_generation.get("mode") != "interleaved":
            raise ValueError("interleaved generation requires mode='interleaved'")
        unsupported = sorted(set(image_generation) - _IMAGE_GENERATION_FIELDS)
        if unsupported:
            raise ValueError(
                "interleaved image_generation has unsupported field(s): "
                + ", ".join(unsupported)
            )
        config = cls(
            max_frames=_positive_integer(
                image_generation.get("max_frames", cls.max_frames),
                name="max_frames",
            ),
            text_max_new_tokens=_positive_integer(
                image_generation.get("text_max_new_tokens", cls.text_max_new_tokens),
                name="text_max_new_tokens",
            ),
            cfg_scale=_finite_number(
                image_generation.get("cfg_scale", cls.cfg_scale),
                name="cfg_scale",
            ),
            cfg_text_scale=_finite_number(
                image_generation.get("cfg_text_scale", cls.cfg_text_scale),
                name="cfg_text_scale",
            ),
            cfg_image_scale=_finite_number(
                image_generation.get("cfg_image_scale", cls.cfg_image_scale),
                name="cfg_image_scale",
            ),
            cfg_rescale=_finite_number(
                image_generation.get("cfg_rescale", cls.cfg_rescale),
                name="cfg_rescale",
            ),
            decoder_steps=_positive_integer(
                image_generation.get("decoder_steps", cls.decoder_steps),
                name="decoder_steps",
            ),
            seed=(
                _non_negative_integer(image_generation["seed"], name="seed")
                if image_generation.get("seed") is not None
                else None
            ),
            max_image_tokens=_positive_integer(
                image_generation.get("max_image_tokens", cls.max_image_tokens),
                name="max_image_tokens",
            ),
            format=image_generation.get("format", cls.format),
            decode_mode=image_generation.get("decode_mode", cls.decode_mode),
        )
        config.validate()
        return config

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "InterleavedGenerationConfig":
        values = {
            field: state[field] for field in cls.__dataclass_fields__ if field in state
        }
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "max_frames",
            "text_max_new_tokens",
            "decoder_steps",
            "max_image_tokens",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.seed is not None:
            _non_negative_integer(self.seed, name="seed")
        for name in ("cfg_scale", "cfg_text_scale", "cfg_image_scale"):
            value = _finite_number(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"image_generation.{name} must be non-negative")
        cfg_rescale = _finite_number(self.cfg_rescale, name="cfg_rescale")
        if not 0.0 <= cfg_rescale <= 1.0:
            raise ValueError("image_generation.cfg_rescale must be in [0, 1]")
        if self.format != "png":
            raise ValueError("interleaved image_generation.format must be 'png'")
        if self.decode_mode != "normal":
            raise ValueError(
                "interleaved image_generation.decode_mode "
                f"{self.decode_mode!r} is unsupported; "
                "only 'normal' is supported"
            )

    def to_generation_state(
        self, *, prompt_length: int, max_seq_len: int
    ) -> dict[str, Any]:
        state = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        }
        state.update(
            {
                "phase": "text",
                "frame_index": 0,
                "segment_start": prompt_length,
                "prompt_length": prompt_length,
                "max_seq_len": max_seq_len,
                "segments": [],
            }
        )
        return state


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


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"tokenizer does not define {token}")
    return token_id


def _parse_reserved_dimension(tokenizer: Any, token_id: int) -> int:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    match = re.fullmatch(r"<\|reserved_token_(\d+)\|>", token or "")
    if match is None:
        raise ValueError(f"expected a reserved image dimension token, got {token!r}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError("interleaved image dimensions must be positive")
    return value


def parse_image_header(generated_ids: list[int], tokenizer: Any) -> ImageHeader:
    """Parse the final, exact ``SOI/H/W/BOI`` text-phase suffix."""
    boi_id = _token_id(tokenizer, "<boi>")
    soi_id = _token_id(tokenizer, "<|image|>")
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
        grid_h=_parse_reserved_dimension(tokenizer, token_ids[1]),
        grid_w=_parse_reserved_dimension(tokenizer, token_ids[2]),
        token_ids=token_ids,
    )


def build_cfg_plan(
    *,
    full_ids: list[int],
    header: ImageHeader,
    frame_index: int,
    tokenizer: Any,
    config: InterleavedGenerationConfig,
) -> CFGBranchPlan:
    """Build the two- or three-row frame plan before shared group alignment."""
    if (
        config.cfg_scale == 0.0
        and config.cfg_text_scale == 0.0
        and config.cfg_image_scale == 0.0
    ):
        return CFGBranchPlan("none", {}, 0.0, 0.0, 0.0, config.cfg_rescale)

    header_ids = list(header.token_ids)
    unconditional = tokenizer.encode(
        f"<role>SYSTEM</role> {SYSTEM_PROMPT_INTERLEAVED} "
        f"<role>HUMAN</role>{UNCONDITION_TOKEN}<role>ASSISTANT</role>",
        add_special_tokens=False,
    )
    eoi_id = _token_id(tokenizer, "<|/image|>")
    history_end = max(
        (index for index, token_id in enumerate(full_ids[:-4]) if token_id == eoi_id),
        default=-1,
    )
    if frame_index > 0 and history_end >= 0:
        history = full_ids[: history_end + 1]
        current_text = full_ids[history_end + 1 : -4]
        no_text = (
            history
            + tokenizer.encode(UNCONDITION_TOKEN, add_special_tokens=False)
            + header_ids
        )
        if config.cfg_image_scale > 0.0:
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
        if config.cfg_text_scale > 0.0:
            return CFGBranchPlan(
                "simple",
                {"unconditional": no_text},
                config.cfg_text_scale,
                0.0,
                0.0,
                config.cfg_rescale,
            )

    scale = config.cfg_scale if config.cfg_scale > 0.0 else config.cfg_text_scale
    if scale <= 0.0:
        return CFGBranchPlan("none", {}, 0.0, 0.0, 0.0, config.cfg_rescale)
    return CFGBranchPlan(
        "simple",
        {"unconditional": unconditional + header_ids},
        scale,
        0.0,
        0.0,
        config.cfg_rescale,
    )


def _reject_image_tokens(
    token_ids: list[int], *, image_token_offset: int, context: str
) -> None:
    if any(token_id >= image_token_offset for token_id in token_ids):
        raise ValueError(f"interleaved {context} emitted image token(s)")


def _mark_done(
    state: LLaDA2UniPipelineState,
    *,
    finish_reason: str,
) -> None:
    interleaved = state.generation_state["interleaved"]
    interleaved["phase"] = "done"
    interleaved["done"] = True
    interleaved["finish_reason"] = finish_reason
    interleaved.pop("needs_reentry", None)
    prompt_ids = (state.prompt or {}).get("input_ids")
    final_length = (
        int(prompt_ids.numel()) if isinstance(prompt_ids, torch.Tensor) else 0
    )
    prompt_length = int(interleaved["prompt_length"])
    completion_tokens = max(final_length - prompt_length, 0)
    interleaved["usage"] = {
        "prompt_tokens": prompt_length,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_length + completion_tokens,
    }


def _advance_text(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
    *,
    finish_reason: str | None,
) -> None:
    interleaved = state.generation_state["interleaved"]
    output_ids = [
        int(token_id) for token_id in (state.thinker_out or {}).get("output_ids", [])
    ]
    if not output_ids:
        raise ValueError("interleaved text phase produced no tokens")
    boi_id = _token_id(tokenizer, "<boi>")
    if output_ids[-1] != boi_id:
        if boi_id in output_ids:
            raise ValueError("interleaved image header must end with <boi>")
        if state.image_token_offset is None:
            raise ValueError("interleaved state is missing image_token_offset")
        _reject_image_tokens(
            output_ids,
            image_token_offset=state.image_token_offset,
            context="text phase",
        )
        prompt_ids = (state.prompt or {}).get("input_ids")
        if not isinstance(prompt_ids, torch.Tensor):
            raise TypeError("interleaved prompt input_ids must be a tensor")
        old_ids = [int(token_id) for token_id in prompt_ids.flatten().tolist()]
        full_ids = old_ids + output_ids
        segment_start = int(interleaved["segment_start"])
        interleaved["trailing_text"] = tokenizer.decode(
            full_ids[segment_start:], skip_special_tokens=True
        )
        state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
        _mark_done(state, finish_reason=finish_reason or "stop")
        return
    header = parse_image_header(output_ids, tokenizer)
    if state.image_token_offset is None:
        raise ValueError("interleaved state is missing image_token_offset")
    _reject_image_tokens(
        output_ids[: header.soi_position],
        image_token_offset=state.image_token_offset,
        context="text phase",
    )
    config = InterleavedGenerationConfig.from_state(interleaved)
    if header.image_token_count > config.max_image_tokens:
        raise ValueError(
            f"interleaved image requires {header.image_token_count} tokens; "
            f"limit is {config.max_image_tokens}"
        )
    prompt = state.prompt or {}
    prompt_ids = prompt.get("input_ids")
    if not isinstance(prompt_ids, torch.Tensor):
        raise TypeError("interleaved prompt input_ids must be a tensor")
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
        "image_token_count": header.image_token_count,
        "remaining_image_tokens": header.image_token_count,
        "vq_tokens": [],
        "cfg_mode": plan.mode,
    }
    interleaved["cfg_plan"] = {
        "mode": plan.mode,
        "branches": copy.deepcopy(plan.branches),
        "cfg_scale": plan.cfg_scale,
        "cfg_text_scale": plan.cfg_text_scale,
        "cfg_image_scale": plan.cfg_image_scale,
        "cfg_rescale": plan.cfg_rescale,
    }
    interleaved["allowed_stop_token_ids"] = (_token_id(tokenizer, "<|/image|>"),)
    interleaved["phase"] = "image"
    interleaved["needs_reentry"] = True
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
    state.thinker_out = None
    state.engine_outputs.pop("thinker", None)


def _advance_image(state: LLaDA2UniPipelineState, tokenizer: Any) -> None:
    interleaved = state.generation_state["interleaved"]
    current = interleaved.get("current_frame")
    if not isinstance(current, dict):
        raise ValueError("interleaved image phase has no current frame")
    output_ids = [
        int(token_id) for token_id in (state.thinker_out or {}).get("output_ids", [])
    ]
    if not output_ids:
        raise ValueError("interleaved image phase produced no tokens")
    eoi_id = _token_id(tokenizer, "<|/image|>")
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
    if state.image_token_offset is None:
        raise ValueError("interleaved state is missing image_token_offset")
    if any(token_id < state.image_token_offset for token_id in image_ids):
        raise ValueError(
            "interleaved image phase produced a non-image token before EOI"
        )
    accumulated = list(current["vq_tokens"]) + image_ids
    remaining -= len(image_ids)
    if not eoi_positions:
        if remaining == 0:
            raise ValueError("interleaved image phase completed VQ tokens without EOI")
        _append_image_phase_tokens(state, interleaved, image_ids)
        current["vq_tokens"] = accumulated
        current["remaining_image_tokens"] = remaining
        interleaved["needs_reentry"] = True
        state.thinker_out = None
        state.engine_outputs.pop("thinker", None)
        return
    if remaining != 0:
        raise ValueError(
            f"interleaved image phase emitted EOI with {remaining} VQ tokens remaining"
        )
    _append_image_phase_tokens(state, interleaved, image_ids + [eoi_id])
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
    interleaved["needs_reentry"] = True
    interleaved.pop("current_frame", None)
    interleaved.pop("cfg_plan", None)
    interleaved.pop("allowed_stop_token_ids", None)
    if frame_index >= int(interleaved["max_frames"]):
        _mark_done(state, finish_reason="max_frames")


def _append_image_phase_tokens(
    state: LLaDA2UniPipelineState,
    interleaved: dict[str, Any],
    token_ids: list[int],
) -> None:
    prompt = state.prompt or {}
    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("interleaved prompt input_ids must be a tensor")
    full_ids = [int(token_id) for token_id in input_ids.flatten().tolist()]
    full_ids.extend(token_ids)
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}

    cfg_plan = interleaved.get("cfg_plan")
    branches = cfg_plan.get("branches") if isinstance(cfg_plan, dict) else None
    if isinstance(branches, dict):
        for branch_name, branch in branches.items():
            if not isinstance(branch, list):
                raise TypeError(
                    f"interleaved CFG branch {branch_name!r} must be a token list"
                )
            branch.extend(token_ids)


def advance_interleaved_state(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
    *,
    completed_phase: str,
    finish_reason: str | None = None,
) -> None:
    """Advance one phase and commit only after every validation succeeds."""
    working = LLaDA2UniPipelineState.from_dict(copy.deepcopy(state.to_dict()))
    interleaved = working.generation_state.get("interleaved")
    if not isinstance(interleaved, dict):
        raise ValueError("request is missing interleaved generation state")
    if interleaved.get("phase") != completed_phase:
        raise ValueError(
            f"interleaved phase mismatch: expected {interleaved.get('phase')!r}, "
            f"got {completed_phase!r}"
        )
    if completed_phase == "text":
        _advance_text(
            working,
            tokenizer,
            finish_reason=finish_reason,
        )
    elif completed_phase == "image":
        _advance_image(working, tokenizer)
    else:
        raise ValueError(f"unsupported interleaved phase {completed_phase!r}")
    state.__dict__.update(working.__dict__)


class InterleavedCollectorScheduler:
    """Join decoded frames with the final thinker state in frame order."""

    _TOMBSTONE_LIMIT = 4096

    def __init__(self) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.requires_tp_work_fanout = False
        self.allow_multiple_inflight_per_request = False
        self._running = False
        self._lock = threading.Lock()
        self._aborted: OrderedDict[str, None] = OrderedDict()
        self._completed: OrderedDict[str, None] = OrderedDict()
        self._failed: OrderedDict[str, None] = OrderedDict()
        self._frames: dict[str, dict[int, dict[str, Any]]] = {}
        self._final_payloads: dict[str, StagePayload] = {}

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.type != "new_request":
                continue
            try:
                with self._lock:
                    self._receive(message.request_id, message.data)
            except Exception as exc:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self._running = False

    @classmethod
    def _remember(cls, tombstones: OrderedDict[str, None], request_id: str) -> None:
        tombstones[request_id] = None
        tombstones.move_to_end(request_id)
        while len(tombstones) > cls._TOMBSTONE_LIMIT:
            tombstones.popitem(last=False)

    def abort(self, request_id: str) -> None:
        with self._lock:
            self._remember(self._aborted, request_id)
            self._cleanup(request_id)

    def _cleanup(self, request_id: str) -> None:
        self._frames.pop(request_id, None)
        self._final_payloads.pop(request_id, None)

    def _receive(self, request_id: str, payload: StagePayload) -> None:
        if (
            request_id in self._aborted
            or request_id in self._completed
            or request_id in self._failed
        ):
            return
        try:
            self._receive_active(request_id, payload)
        except Exception:
            self._remember(self._failed, request_id)
            self._cleanup(request_id)
            raise

    def _receive_active(self, request_id: str, payload: StagePayload) -> None:
        data = payload.data if isinstance(payload.data, dict) else {}
        if data.get("kind") == "interleaved_frame":
            frame = data.get("frame")
            if not isinstance(frame, dict):
                raise TypeError("decoded interleaved frame must be a mapping")
            frame_index = frame.get("index")
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index <= 0
            ):
                raise ValueError("decoded interleaved frame index must be positive")
            frames = self._frames.setdefault(request_id, {})
            if frame_index in frames:
                self._cleanup(request_id)
                raise ValueError(f"duplicate decoded frame {frame_index}")
            frames[frame_index] = copy.deepcopy(frame)
        else:
            state = LLaDA2UniPipelineState.from_dict(data)
            interleaved = state.generation_state.get("interleaved")
            if (
                state.task_kind != "interleaved"
                or not isinstance(interleaved, dict)
                or not interleaved.get("done")
            ):
                raise ValueError("collector received a non-final interleaved payload")
            if request_id in self._final_payloads:
                self._cleanup(request_id)
                raise ValueError("duplicate interleaved final payload")
            self._final_payloads[request_id] = payload
        self._maybe_finish(request_id)

    def _maybe_finish(self, request_id: str) -> None:
        final_payload = self._final_payloads.get(request_id)
        if final_payload is None:
            return
        state = LLaDA2UniPipelineState.from_dict(final_payload.data)
        interleaved = state.generation_state["interleaved"]
        expected = int(interleaved["frame_index"])
        frames = self._frames.get(request_id, {})
        expected_indexes = set(range(1, expected + 1))
        if set(frames) - expected_indexes:
            self._cleanup(request_id)
            raise ValueError("collector received an unexpected interleaved frame")
        if set(frames) != expected_indexes:
            return

        content: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        segments = interleaved.get("segments", [])
        seen_image_ids: set[str] = set()
        for frame_index in range(1, expected + 1):
            frame = frames[frame_index]
            segment = segments[frame_index - 1]
            text = str(segment.get("text", ""))
            if text:
                content.append({"type": "text", "text": text})
            image = frame.get("image")
            if not isinstance(image, dict):
                self._cleanup(request_id)
                raise TypeError("decoded interleaved frame is missing image data")
            image_id = image.get("id")
            if (
                not isinstance(image_id, str)
                or not image_id
                or image_id in seen_image_ids
            ):
                self._cleanup(request_id)
                raise ValueError(
                    "interleaved image ids must be unique non-empty strings"
                )
            seen_image_ids.add(image_id)
            images.append(copy.deepcopy(image))
            content.append({"type": "image_ref", "image_id": image_id})
        trailing_text = str(interleaved.get("trailing_text", ""))
        if trailing_text:
            content.append({"type": "text", "text": trailing_text})

        final_payload.data = {
            "modality": "interleaved",
            "content": content,
            "images": images,
            "finish_reason": interleaved.get("finish_reason", "stop"),
            "usage": interleaved.get("usage", {}),
        }
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=final_payload,
            )
        )
        self._remember(self._completed, request_id)
        self._cleanup(request_id)


def create_interleaved_collector_executor(model_path: str):
    del model_path
    return InterleavedCollectorScheduler()
