# SPDX-License-Identifier: Apache-2.0
"""Stage routing for Ming 3 audio requests."""

from sglang_omni.proto.request import StagePayload


def route_preprocessing(request_id: str, output: StagePayload) -> str:
    """Select TTS or understanding; request_id follows the routing interface."""
    if output.request.metadata.get("task") == "tts":
        return "talker"
    else:
        return "audio_encoder"


def tool_stream_targets(request_id: str, output: StagePayload) -> list[str]:
    return (
        ["decode"]
        if output.request.metadata.get("tools") and output.request.params.get("stream")
        else []
    )


def route_thinker(request_id: str, output: StagePayload) -> str:
    """Select the output modality; request_id follows the routing interface."""
    if "audio" in output.request.metadata.get("output_modalities", []):
        return "talker"
    else:
        return "decode"
