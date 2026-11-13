# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni interleaved generation on the shared UMM controller.

The thinker alternates text and image phases on the state the preprocessing
stage builds, and advances that state itself. This adapter only chooses the
next phases: each finished image decodes while the thinker writes the next
text, and segments leave in the order the model wrote them.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, replace

from pydantic import ValidationError

from sglang_omni.models.llada2_uni.interleaved import (
    InterleavedGenerationParams,
    validate_interleaved_inputs,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMDispatch,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.proto import OmniRequest

PREPARE = "prepare"
THINK = "think"
DECODE = "decode"


def interleaved_options(request: OmniRequest) -> dict[str, object]:
    """The client's interleaved controls from stage_params.orchestrator."""
    stage_params = request.params.get("stage_params") or {}
    options = (
        stage_params.get("orchestrator", {}) if isinstance(stage_params, dict) else None
    )
    if not isinstance(options, dict):
        raise UMMRequestError("stage_params.orchestrator must be a mapping")
    else:
        pass
    try:
        params = InterleavedGenerationParams.model_validate(
            {**options, "mode": "interleaved"}
        )
    except ValidationError as exc:
        raise UMMRequestError(f"stage_params.orchestrator: {exc}") from exc
    return params.model_dump()


def frame_state(state: dict) -> dict[str, object]:
    """The fields the image decoder reads for the frame the thinker just
    finished, so a frame waiting for the decoder keeps only its own tokens."""
    interleaved = state["stream_state"]["interleaved"]
    return {
        "task_kind": state["task_kind"],
        "thinker_out": {"output_ids": list(state["thinker_out"]["output_ids"])},
        "stream_state": {
            "interleaved": {"segments": [dict(interleaved["segments"][-1])]}
        },
        "request_metadata": {
            "image_generation": dict(state["request_metadata"]["image_generation"])
        },
    }


def phases_needed(max_frames: int) -> int:
    """Preparation, a text and an image phase per frame, the closing text
    phase, and one decode per frame."""
    return 3 * max_frames + 2


@dataclass
class InterleavedRun:
    # Outputs in model order. None marks an image still decoding.
    slots: list[UMMOutput | None] = field(default_factory=list)
    emitted: int = 0
    # Frames waiting for the decoder, which runs one frame of a request at a
    # time, with their slot and retained bytes.
    waiting: deque[tuple[int, OmniRequest, int]] = field(default_factory=deque)
    decoding: int | None = None
    thinking: str | None = None
    done: bool = False


class LLaDA2UniUMMAdapter:
    """Phases prepare, think, and decode of LLaDA2-Uni interleaved output."""

    def __init__(self, model_path: str) -> None:
        del model_path
        self.runs: dict[str, InterleavedRun] = {}

    def start(self, context: UMMContext) -> UMMStep:
        inputs = context.request.inputs
        messages = inputs.get("messages", []) if isinstance(inputs, dict) else inputs
        has_media = isinstance(inputs, dict) and any(
            inputs.get(key) for key in ("images", "audios", "videos")
        )
        try:
            validate_interleaved_inputs(messages, has_media=has_media)
        except ValueError as exc:
            raise UMMRequestError(str(exc)) from exc
        options = interleaved_options(context.request)
        needed = phases_needed(int(options["max_frames"]))
        if needed > context.remaining_dispatches:
            raise UMMRequestError(
                f"max_frames={options['max_frames']} needs {needed} phases, more "
                f"than the {context.remaining_dispatches} this deployment allows. "
                "Lower max_frames or raise stages.orchestrator.factory.max_dispatches"
            )
        else:
            pass
        self.runs[context.execution_id] = InterleavedRun()
        metadata = {**context.request.metadata, "image_generation": options}
        return UMMStep(PREPARE, replace(context.request, metadata=metadata))

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        run = self.runs[context.execution_id]
        dispatches: list[UMMDispatch] = []
        if phase == DECODE:
            run.slots[run.decoding] = UMMOutput("image", result)
            run.decoding = None
        elif phase in (PREPARE, THINK):
            dispatches.extend(self.after_thinker(context, run, phase, result))
        else:
            raise ValueError(f"LLaDA2-Uni adapter received unexpected phase {phase!r}")
        if run.decoding is None and run.waiting:
            run.decoding, request, _ = run.waiting.popleft()
            dispatches.append(UMMDispatch(DECODE, request))
        else:
            pass
        ready = run.emitted
        while ready < len(run.slots) and run.slots[ready] is not None:
            ready += 1
        outputs = tuple(run.slots[run.emitted : ready])
        run.emitted = ready
        return UMMStep(outputs=outputs, dispatches=tuple(dispatches))

    def after_thinker(
        self, context: UMMContext, run: InterleavedRun, phase: str, state: object
    ) -> list[UMMDispatch]:
        if not isinstance(state, dict):
            raise ValueError("LLaDA2-Uni thinker returned no pipeline state")
        else:
            pass
        interleaved = state["stream_state"]["interleaved"]
        finished = run.thinking
        run.thinking = None
        if finished == "text" and not interleaved.get("done"):
            # The text before this image is complete.
            self.add_text(run, interleaved["current_frame"]["text"])
        elif interleaved.get("emit_frame"):
            run.slots.append(None)
            frame = frame_state(state)
            request = OmniRequest(
                frame, context.request.params, context.request.metadata
            )
            size = len(json.dumps(frame, ensure_ascii=False).encode("utf-8"))
            run.waiting.append((len(run.slots) - 1, request, size))
        else:
            pass
        if interleaved.get("done"):
            self.add_text(run, interleaved.get("trailing_text", ""))
            run.done = True
            return []
        else:
            run.thinking = interleaved["phase"]
            request = OmniRequest(
                state, context.request.params, context.request.metadata
            )
            return [UMMDispatch(THINK, request)]

    @staticmethod
    def add_text(run: InterleavedRun, text: str) -> None:
        if text:
            run.slots.append(UMMOutput("text", text))
        else:
            pass

    def usage(self, context: UMMContext) -> int:
        """Bytes of text held behind an image that is still decoding, and of
        the frames waiting for the decoder."""
        run = self.runs.get(context.execution_id)
        if run is None:
            return 0
        else:
            held = sum(
                len(slot.data.encode("utf-8"))
                for slot in run.slots[run.emitted :]
                if slot is not None and isinstance(slot.data, str)
            )
            return held + sum(size for _, _, size in run.waiting)

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.runs.pop(context.execution_id, None)
