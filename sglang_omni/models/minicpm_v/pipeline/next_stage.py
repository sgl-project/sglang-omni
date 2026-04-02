# SPDX-License-Identifier: Apache-2.0
"""Stage routing helpers for MiniCPM-V/MiniCPM-o pipelines.

MiniCPM-V 2.6 Pipeline (vision-only):
  preprocessing -> image_encoder -> mm_aggregate -> llm -> decode

MiniCPM-o 2.6 Pipeline (vision + audio input):
  preprocessing -> image_encoder -> mm_aggregate -> llm -> decode
                -> audio_encoder -^

MiniCPM-o 2.6 Pipeline (vision + audio input/output):
  preprocessing -> image_encoder -> mm_aggregate -> llm -> decode (text)
                -> audio_encoder -^                    -> vocoder -> decode (audio)

The audio_encoder stage is only present in MiniCPM-o 2.6.
The vocoder stage is only present when audio output is requested.
"""

from __future__ import annotations

from typing import Any

from sglang_omni.models.minicpm_v.io import PipelineState
from sglang_omni.proto import StagePayload

# Stage name constants
PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"  # MiniCPM-o 2.6 only
AGGREGATE_STAGE = "mm_aggregate"
LLM_STAGE = "llm"
VOCODER_STAGE = "vocoder"  # MiniCPM-o 2.6 audio output
DECODE_STAGE = "decode"


def preprocessing_next(request_id: str, output: Any) -> list[str]:
    """Route preprocessing output to encoder stages and aggregate.

    For MiniCPM-V:
    - Routes to image_encoder (if images present) and mm_aggregate

    For MiniCPM-o:
    - Routes to image_encoder, audio_encoder (if present), and mm_aggregate
    """
    del request_id
    if not isinstance(output, StagePayload):
        return [AGGREGATE_STAGE]

    state = PipelineState.from_dict(output.data)
    encoder_inputs = state.encoder_inputs
    if not isinstance(encoder_inputs, dict):
        return [AGGREGATE_STAGE]

    # Collect all encoder stages that have inputs
    stages = []
    for stage in encoder_inputs.keys():
        if stage == AGGREGATE_STAGE:
            continue
        # Check if stage has valid inputs (not skipped)
        stage_inputs = encoder_inputs.get(stage, {})
        if isinstance(stage_inputs, dict) and not stage_inputs.get("_skip"):
            stages.append(stage)

    # Sort for deterministic ordering
    stages = sorted(stages)
    # Always end with aggregate
    stages.append(AGGREGATE_STAGE)
    return stages


def encoder_next(request_id: str, output: Any) -> str:
    """Route encoder output to aggregate stage."""
    del request_id, output
    return AGGREGATE_STAGE


def aggregate_next(request_id: str, output: Any) -> str:
    """Route aggregate output to LLM stage."""
    del request_id, output
    return LLM_STAGE


def llm_next(request_id: str, output: Any) -> str | list[str]:
    """Route LLM output to decode stage or vocoder based on output modality.

    For text-only output: route to decode
    For audio output (MiniCPM-o): route to both vocoder and decode
    """
    del request_id

    # Check if audio output is requested
    if isinstance(output, StagePayload):
        state = PipelineState.from_dict(output.data)
        # Check for audio tokens in LLM output
        llm_out = state.llm_out or state.engine_outputs.get(LLM_STAGE, {})
        if isinstance(llm_out, dict):
            # Check if output contains audio tokens (indicated by extra_model_outputs)
            extra = llm_out.get("extra_model_outputs", {})
            if isinstance(extra, dict) and extra.get("has_audio_output"):
                # Route to both vocoder (for audio) and decode (for text)
                return [VOCODER_STAGE, DECODE_STAGE]

            # Check request params for explicit audio output mode
            raw_inputs = state.raw_inputs or {}
            if isinstance(raw_inputs, dict):
                params = raw_inputs.get("params", {})
                if isinstance(params, dict) and params.get("output_modality") == "audio":
                    return [VOCODER_STAGE, DECODE_STAGE]

    return DECODE_STAGE


def vocoder_next(request_id: str, output: Any) -> str:
    """Route vocoder output to decode stage."""
    del request_id, output
    return DECODE_STAGE


def decode_next(request_id: str, output: Any) -> None:
    """Terminal stage - no next stage."""
    del request_id, output
    return None
