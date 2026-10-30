# SPDX-License-Identifier: Apache-2.0
"""Payload contracts and routing for the AURA Omni pipeline.

Qwen3-ASR transcribes the user's speech, AURA (a Qwen3-VL fine-tune) answers
from the transcript and the visual context or stays silent, and Qwen3-TTS
voices a spoken answer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

from sglang.srt.utils.common import load_video

from sglang_omni.preprocessing.text import TemplateMessage, split_content_parts
from sglang_omni.preprocessing.transcription import police_request_media
from sglang_omni.proto import OmniRequest, StagePayload

PREPROCESSING_STAGE = "preprocessing"
ASR_STAGE = "asr"
AURA_STAGE = "aura"
DECODE_STAGE = "decode"
TTS_PREPROCESSING_STAGE = "tts_preprocessing"
VOCODER_STAGE = "vocoder"

# The system prompt AURA was trained with.
DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present "
    "moment. Respond only when a response is needed based on the user's message "
    "or the visual context. Otherwise, output `<|silent|>` to signify silence."
)
# Qwen3-VL merges frames in temporal pairs, so a clip needs two of them.
MIN_VIDEO_FRAMES = 2
# An inline data URL runs to megabytes, so an error names only its start.
MAX_SOURCE_LABEL_CHARS = 256
# Chat API fields mapped to the keys the Qwen-VL video sampler reads.
VIDEO_SAMPLING_FIELDS = {
    "video_fps": "fps",
    "video_max_frames": "max_frames",
    "video_min_pixels": "min_pixels",
    "video_max_pixels": "max_pixels",
    "video_total_pixels": "total_pixels",
}
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

MediaSource = str | bytes


class HistoryMessage(TypedDict):
    role: str
    content: str


class VideoSamplingConfig(TypedDict, total=False):
    fps: float
    max_frames: int
    min_pixels: int
    max_pixels: int
    total_pixels: int


class AuraTurn(TypedDict):
    """The user turn AURA answers, with the media its prompt places."""

    system_prompt: str
    history: list[HistoryMessage]
    user_text: str
    images: list[MediaSource]
    videos: list[MediaSource]
    video_sampling: VideoSamplingConfig
    audio: MediaSource | None
    has_audio: bool
    transcript: str


class AuraAnswer(TypedDict):
    text: str
    is_silent: bool
    output_ids: list[int]
    finish_reason: str | None
    prompt_tokens: int


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class DecodeResult(TypedDict, total=False):
    text: str
    modality: str
    finish_reason: str | None
    usage: TokenUsage


def message_text(message: TemplateMessage) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    else:
        return "".join(part["text"] for part in content if part["type"] == "text")


def request_media(inputs: Mapping[str, object], key: str) -> list[str]:
    value = inputs.get(key)
    if value is None:
        return []
    elif isinstance(value, str):
        return [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    else:
        raise ValueError(f"AURA Omni expects {key} to be a URL or a list of URLs")


def request_video_sampling(inputs: Mapping[str, object]) -> VideoSamplingConfig:
    """The frame and pixel budget of one request, validated before any decode."""
    sampling: VideoSamplingConfig = {}
    fps = inputs.get("video_fps")
    if fps is None:
        pass
    elif isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0:
        sampling["fps"] = float(fps)
    else:
        raise ValueError("video_fps must be positive and finite")
    max_frames = inputs.get("video_max_frames")
    if max_frames is None:
        pass
    elif isinstance(max_frames, int) and max_frames >= MIN_VIDEO_FRAMES:
        sampling["max_frames"] = max_frames
    else:
        raise ValueError(
            f"video_max_frames must be positive and at least {MIN_VIDEO_FRAMES}"
        )
    for request_field in ("video_min_pixels", "video_max_pixels", "video_total_pixels"):
        pixels = inputs.get(request_field)
        if pixels is None:
            pass
        elif isinstance(pixels, int) and pixels > 0:
            sampling[VIDEO_SAMPLING_FIELDS[request_field]] = pixels
        else:
            raise ValueError("Video pixel budgets must be positive")
    if sampling.get("min_pixels", 0) > sampling.get("max_pixels", math.inf):
        raise ValueError("Video minimum pixel budget exceeds maximum budget")
    else:
        pass
    return sampling


def policed_media(source: str) -> MediaSource:
    policed = police_request_media(source)
    assert isinstance(policed, (str, bytes)), type(policed)
    return policed


def probed_video(source: str) -> MediaSource:
    """Police a clip and refuse one that does not decode to enough frames for Qwen3-VL."""
    policed = policed_media(source)
    source_label = (
        source
        if len(source) <= MAX_SOURCE_LABEL_CHARS
        else f"{source[:MAX_SOURCE_LABEL_CHARS]}..."
    )
    try:
        video = load_video(policed, use_gpu=False)
    except ValueError as error:
        raise ValueError(
            f"Invalid media data while decoding video path={source_label}: {error}"
        ) from error
    if len(video) < MIN_VIDEO_FRAMES or not video.avg_fps > 0:
        raise ValueError(
            f"Invalid media data while decoding video path={source_label}: "
            f"{len(video)} frames at {video.avg_fps} fps, AURA needs at least "
            f"{MIN_VIDEO_FRAMES} frames"
        )
    else:
        return policed


def preprocess_aura_payload(payload: StagePayload) -> StagePayload:
    """Split a chat request into the AURA turn and the audio Qwen3-ASR hears."""
    inputs = payload.request.inputs
    if isinstance(inputs, Mapping) and inputs.get("use_audio_in_video"):
        raise ValueError(
            "AURA Omni expects the spoken question as its own audio part, so "
            "use_audio_in_video is not supported: AURA reads a clip's frames "
            "and never its soundtrack"
        )
    elif isinstance(inputs, Mapping):
        messages = inputs.get("messages") or []
        extra_images = request_media(inputs, "images")
        extra_audios = request_media(inputs, "audios")
        extra_videos = request_media(inputs, "videos")
        video_sampling = request_video_sampling(inputs)
    else:
        messages = inputs
        extra_images, extra_audios, extra_videos = [], [], []
        video_sampling = {}
    template_messages, media = split_content_parts(messages)

    system_texts = [
        message_text(message)
        for message in template_messages
        if message["role"] == "system"
    ]
    dialog = [message for message in template_messages if message["role"] != "system"]
    if not dialog:
        user_text = ""
        history: list[HistoryMessage] = []
    elif dialog[-1]["role"] == "user":
        user_text = message_text(dialog[-1])
        history = [
            {"role": message["role"], "content": message_text(message)}
            for message in dialog[:-1]
        ]
    else:
        raise ValueError("AURA Omni expects the conversation to end with a user turn")

    audios = [*media.audios, *extra_audios]
    if len(audios) > 1:
        raise ValueError(
            f"AURA Omni transcribes one audio input per request, got {len(audios)}"
        )
    else:
        pass
    turn: AuraTurn = {
        "system_prompt": "\n".join(system_texts) or DEFAULT_AURA_SYSTEM_PROMPT,
        "history": history,
        "user_text": user_text,
        "images": [policed_media(url) for url in (*media.images, *extra_images)],
        "videos": [probed_video(url) for url in (*media.videos, *extra_videos)],
        "video_sampling": video_sampling,
        "audio": policed_media(audios[0]) if audios else None,
        "has_audio": bool(audios),
        "transcript": "",
    }
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=turn
    )


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    if output.data["has_audio"]:
        return [ASR_STAGE, AURA_STAGE]
    else:
        return [AURA_STAGE]


def project_preprocessing_to_asr(payload: StagePayload) -> StagePayload:
    """Give Qwen3-ASR the audio alone; it is not terminal, so it must not stream."""
    stage_params = payload.request.params.get("stage_params") or {}
    asr_params = dict(stage_params.get(ASR_STAGE) or {})
    return StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(
            inputs=payload.data["audio"], params={**asr_params, "stream": False}
        ),
        data=None,
    )


def project_preprocessing_to_aura(payload: StagePayload) -> StagePayload:
    request = payload.request
    return StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(
            inputs=None, params=request.params, metadata=request.metadata
        ),
        data={**payload.data, "audio": None},
    )


def resolve_aura_wait_sources(
    request_id: str, from_stage: str, payload: StagePayload
) -> list[str] | None:
    if from_stage != PREPROCESSING_STAGE:
        return None
    elif payload.data["has_audio"]:
        return [PREPROCESSING_STAGE, ASR_STAGE]
    else:
        return [PREPROCESSING_STAGE]


def merge_for_aura(payloads: dict[str, StagePayload]) -> StagePayload:
    base = payloads[PREPROCESSING_STAGE]
    asr_payload = payloads.get(ASR_STAGE)
    transcript = asr_payload.data["text"] if asr_payload is not None else ""
    return StagePayload(
        request_id=base.request_id,
        request=base.request,
        data={**base.data, "transcript": transcript},
    )


def render_aura_prompt(turn: AuraTurn) -> str:
    """ChatML prompt with the vision placeholders ahead of what the user said."""
    query = " ".join(
        text for text in (turn["transcript"].strip(), turn["user_text"].strip()) if text
    )
    vision = IMAGE_PLACEHOLDER * len(turn["images"]) + VIDEO_PLACEHOLDER * len(
        turn["videos"]
    )
    history = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in turn["history"]
    )
    return (
        f"<|im_start|>system\n{turn['system_prompt']}<|im_end|>\n"
        f"{history}"
        f"<|im_start|>user\n{vision}{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def wants_audio(request: OmniRequest) -> bool:
    modalities = request.metadata.get("output_modalities")
    return isinstance(modalities, (list, tuple)) and "audio" in modalities


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    if wants_audio(request):
        return [DECODE_STAGE, VOCODER_STAGE]
    else:
        return [DECODE_STAGE]


def resolve_aura_next_stages(request_id: str, output: StagePayload) -> list[str]:
    if not wants_audio(output.request):
        return [DECODE_STAGE]
    elif output.data["is_silent"]:
        return [DECODE_STAGE, VOCODER_STAGE]
    else:
        return [DECODE_STAGE, TTS_PREPROCESSING_STAGE]


def resolve_aura_stream_done_targets(
    request_id: str, output: StagePayload
) -> list[str]:
    if wants_audio(output.request) and output.data["is_silent"]:
        return [DECODE_STAGE, VOCODER_STAGE]
    else:
        return [DECODE_STAGE]


def project_aura_to_tts(payload: StagePayload) -> StagePayload:
    """A Qwen3-TTS request for the answer; the chat's sampling params are AURA's."""
    request = payload.request
    stage_params = request.params.get("stage_params") or {}
    tts_params = dict(stage_params.get(TTS_PREPROCESSING_STAGE) or {})
    audio_config = request.metadata.get("audio_config") or {}
    voice = audio_config.get("voice")
    if voice is not None and "voice" not in tts_params:
        tts_params["voice"] = voice
    else:
        pass
    return StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(
            inputs={"text": payload.data["text"]},
            params={
                "stream": request.params.get("stream", False),
                "stage_params": stage_params,
            },
            metadata={"tts_params": tts_params, "output_modalities": ["audio"]},
        ),
        data=None,
    )


def project_aura_to_vocoder(payload: StagePayload) -> StagePayload:
    """A silent answer has no codes: the vocoder closes it on the stream-done signal."""
    request = payload.request
    return StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(
            inputs=None,
            params={"stream": True},
            metadata=request.metadata,
        ),
        data={},
    )


def build_decode_result(payload: StagePayload, is_streaming: bool) -> DecodeResult:
    answer: AuraAnswer = payload.data
    completion_tokens = len(answer["output_ids"])
    result: DecodeResult = {
        "modality": "text",
        "finish_reason": answer["finish_reason"],
        "usage": {
            "prompt_tokens": answer["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "total_tokens": answer["prompt_tokens"] + completion_tokens,
        },
    }
    if not is_streaming:
        result["text"] = answer["text"]
    else:
        pass
    return result
