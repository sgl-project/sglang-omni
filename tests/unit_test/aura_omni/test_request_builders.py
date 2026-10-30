# SPDX-License-Identifier: Apache-2.0
"""AURA Omni splits a chat turn across Qwen3-ASR, AURA and Qwen3-TTS, checks its media and routes the answer."""

import asyncio
from pathlib import Path

import pytest
import torch
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.utils.common import load_video

from sglang_omni.models.aura_omni.config import AuraOmniPipelineConfig
from sglang_omni.models.aura_omni.request_builders import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    VIDEO_PLACEHOLDER,
    build_decode_result,
    merge_for_aura,
    preprocess_aura_payload,
    probed_video,
    project_aura_to_tts,
    project_aura_to_vocoder,
    project_preprocessing_to_asr,
    project_preprocessing_to_aura,
    render_aura_prompt,
    request_video_sampling,
    resolve_aura_next_stages,
    resolve_aura_stream_done_targets,
    resolve_aura_wait_sources,
    resolve_preprocessing_next_stages,
    resolve_terminal_stages,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error
from tests.utils.synthetic_video import Scene, Shape, write_scene_video

AUDIO = "data:audio/wav;base64,UklGRg=="
BLUE_CIRCLE = Shape(
    kind="circle",
    rgb=(0, 0, 255),
    center_x_pixels=320,
    center_y_pixels=240,
    half_size_pixels=80,
)


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A 4 s, 10 fps clip of a blue circle."""
    path = tmp_path_factory.mktemp("clips") / "blue_circle.mp4"
    return str(write_scene_video(path, [Scene(duration_s=4.0, shapes=(BLUE_CIRCLE,))]))


def require_pinnable_host_memory() -> None:
    # The Qwen-VL sampler pins its frames, which needs an accelerator unless SGLang runs its CPU engine.
    if not torch.accelerator.is_available():
        pytest.skip("pinning host memory needs an accelerator")
    else:
        pass


def chat_payload(
    content: list[dict[str, object]],
    *,
    system: str | None = None,
    extra_inputs: dict[str, object] | None = None,
) -> StagePayload:
    messages: list[dict[str, object]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    else:
        pass
    messages.append({"role": "user", "content": content})
    return StagePayload(
        "r1",
        request=OmniRequest(
            inputs={"messages": messages, **(extra_inputs or {})},
            params={"temperature": 0.2, "stream": False},
            metadata={"output_modalities": ["text", "audio"]},
        ),
        data=None,
    )


def spoken_turn(clip_path: str) -> StagePayload:
    return preprocess_aura_payload(
        chat_payload(
            [
                {"type": "video_url", "video_url": {"url": clip_path}},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklGRg==", "format": "wav"},
                },
                {"type": "text", "text": "Is it done?"},
            ]
        )
    )


def answer_payload(text: str, *, modalities: list[str]) -> StagePayload:
    return StagePayload(
        "r1",
        request=OmniRequest(
            inputs=None,
            params={
                "temperature": 0.2,
                "stream": True,
                "stage_params": {"tts_preprocessing": {"language": "English"}},
            },
            metadata={
                "output_modalities": modalities,
                "audio_config": {"voice": "ryan", "format": "wav"},
            },
        ),
        data={
            "text": text,
            "is_silent": not text,
            "output_ids": [11, 12],
            "finish_reason": "stop",
            "prompt_tokens": 40,
        },
    )


def test_spoken_turn_goes_to_asr_and_aura(clip_path: str) -> None:
    turn = spoken_turn(clip_path)

    assert turn.data["system_prompt"] == DEFAULT_AURA_SYSTEM_PROMPT
    assert turn.data["videos"] == [clip_path]
    assert turn.data["audio"] == AUDIO
    assert turn.data["user_text"] == "Is it done?"
    assert resolve_preprocessing_next_stages("r1", turn) == ["asr", "aura"]

    asr_payload = project_preprocessing_to_asr(turn)
    assert asr_payload.request.inputs == AUDIO
    assert asr_payload.request.params["stream"] is False

    aura_payload = project_preprocessing_to_aura(turn)
    assert aura_payload.data["audio"] is None
    assert aura_payload.request.inputs is None
    assert resolve_aura_wait_sources("r1", "preprocessing", aura_payload) == [
        "preprocessing",
        "asr",
    ]
    assert resolve_aura_wait_sources("r1", "asr", aura_payload) is None


def test_turn_without_speech_skips_asr(clip_path: str) -> None:
    turn = preprocess_aura_payload(
        chat_payload(
            [{"type": "video_url", "video_url": {"url": clip_path}}], system="Be brief."
        )
    )

    assert turn.data["system_prompt"] == "Be brief."
    assert resolve_preprocessing_next_stages("r1", turn) == ["aura"]
    assert resolve_aura_wait_sources(
        "r1", "preprocessing", project_preprocessing_to_aura(turn)
    ) == ["preprocessing"]


def test_merge_puts_transcript_ahead_of_typed_text(clip_path: str) -> None:
    aura_payload = project_preprocessing_to_aura(spoken_turn(clip_path))
    transcript = StagePayload(
        "r1", request=OmniRequest(inputs=None), data={"text": "What is she drawing?"}
    )

    merged = merge_for_aura({"preprocessing": aura_payload, "asr": transcript})
    prompt = render_aura_prompt(merged.data)

    assert prompt.startswith(f"<|im_start|>system\n{DEFAULT_AURA_SYSTEM_PROMPT}")
    assert (
        f"<|im_start|>user\n{VIDEO_PLACEHOLDER}What is she drawing? Is it done?"
        "<|im_end|>\n<|im_start|>assistant\n"
    ) in prompt


def test_earlier_turns_render_as_history() -> None:
    payload = chat_payload([{"type": "text", "text": "And now?"}])
    payload.request.inputs["messages"][:0] = [
        {"role": "user", "content": "Watch the desk."},
        {"role": "assistant", "content": "Watching."},
    ]

    prompt = render_aura_prompt(preprocess_aura_payload(payload).data)

    assert (
        "<|im_start|>user\nWatch the desk.<|im_end|>\n"
        "<|im_start|>assistant\nWatching.<|im_end|>\n"
        "<|im_start|>user\nAnd now?<|im_end|>\n"
    ) in prompt


@pytest.mark.parametrize(
    "content,extra_inputs,message",
    [
        (
            [{"type": "audio_url", "audio_url": {"url": AUDIO}}] * 2,
            None,
            "one audio input per request",
        ),
        ([{"type": "text", "text": "hi"}], {"video_fps": 0}, "video_fps must be"),
        ([{"type": "text", "text": "hi"}], {"video_max_frames": 1}, "video_max_frames"),
        (
            [{"type": "text", "text": "hi"}],
            {"video_min_pixels": 4096, "video_max_pixels": 1024},
            "minimum pixel budget exceeds",
        ),
    ],
)
def test_malformed_requests_are_bad_requests(
    content: list[dict[str, object]],
    extra_inputs: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        preprocess_aura_payload(chat_payload(content, extra_inputs=extra_inputs))
    assert is_bad_request_error(error.value)


def test_requests_aura_cannot_serve_are_refused() -> None:
    payload = chat_payload([{"type": "text", "text": "hi"}])
    payload.request.inputs["messages"].append({"role": "assistant", "content": "hey"})
    with pytest.raises(ValueError, match="end with a user turn"):
        preprocess_aura_payload(payload)

    with pytest.raises(ValueError, match="use_audio_in_video is not supported"):
        preprocess_aura_payload(
            chat_payload(
                [{"type": "text", "text": "hi"}],
                extra_inputs={"use_audio_in_video": True},
            )
        )


def test_unreadable_videos_are_bad_requests(tmp_path: Path) -> None:
    """A clip that does not decode to two frames fails in preprocessing, before any engine sees it."""
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x07" * 4096)
    audio_as_video = tmp_path / "speech.wav"
    audio_as_video.write_bytes(
        (Path(__file__).parents[2] / "data" / "query_to_draw.wav").read_bytes()
    )
    one_frame = write_scene_video(
        tmp_path / "one_frame.mp4", [Scene(duration_s=0.1, shapes=(BLUE_CIRCLE,))]
    )

    for source in (corrupt, audio_as_video, one_frame, tmp_path / "missing.mp4"):
        with pytest.raises(ValueError, match="Invalid media data") as error:
            probed_video(str(source))
        assert is_bad_request_error(error.value), source


@pytest.mark.parametrize(
    "request_fields,expected_frame_count",
    [({}, 8), ({"video_fps": 1}, 4), ({"video_fps": 2, "video_max_frames": 4}, 4)],
)
@pytest.mark.accelerator
def test_video_options_set_the_sampled_frames(
    clip_path: str, request_fields: dict[str, object], expected_frame_count: int
) -> None:
    """The chat API's video fields reach the Qwen-VL sampler as the keys it reads."""
    require_pinnable_host_memory()
    video, metadata = asyncio.run(
        preprocess_video(
            load_video(probed_video(clip_path), use_gpu=False),
            video_config=request_video_sampling(request_fields),
        )
    )

    frame_indices = list(metadata["frames_indices"])
    assert video.shape[0] == expected_frame_count
    assert frame_indices == sorted(set(frame_indices))
    assert metadata["duration"] == pytest.approx(4.0)


@pytest.mark.accelerator
def test_pixel_budget_caps_the_frame_size(clip_path: str) -> None:
    require_pinnable_host_memory()
    max_pixels = 128 * 32 * 32
    video, _ = asyncio.run(
        preprocess_video(
            load_video(clip_path, use_gpu=False),
            video_config=request_video_sampling(
                {"video_max_pixels": max_pixels, "video_min_pixels": 4096}
            ),
        )
    )

    assert video.shape[2] * video.shape[3] <= max_pixels


@pytest.mark.parametrize(
    "text,modalities,next_stages,stream_done,terminals",
    [
        ("Hi.", ["text"], ["decode"], ["decode"], ["decode"]),
        (
            "Hi.",
            ["text", "audio"],
            ["decode", "tts_preprocessing"],
            ["decode"],
            ["decode", "vocoder"],
        ),
        (
            "",
            ["text", "audio"],
            ["decode", "vocoder"],
            ["decode", "vocoder"],
            ["decode", "vocoder"],
        ),
    ],
)
def test_answer_routing(
    text: str,
    modalities: list[str],
    next_stages: list[str],
    stream_done: list[str],
    terminals: list[str],
) -> None:
    answer = answer_payload(text, modalities=modalities)

    assert resolve_aura_next_stages("r1", answer) == next_stages
    assert resolve_aura_stream_done_targets("r1", answer) == stream_done
    assert resolve_terminal_stages(answer.request) == terminals


def test_spoken_answer_becomes_a_tts_request() -> None:
    tts = project_aura_to_tts(answer_payload("It is a guitar.", modalities=["audio"]))

    assert tts.request.inputs == {"text": "It is a guitar."}
    assert tts.request.metadata["tts_params"] == {
        "language": "English",
        "voice": "ryan",
    }
    assert tts.request.params["stream"] is True
    assert "temperature" not in tts.request.params


def test_silent_answer_closes_the_vocoder_without_codes() -> None:
    vocoder = project_aura_to_vocoder(answer_payload("", modalities=["audio"]))

    assert vocoder.request.params == {"stream": True}
    assert vocoder.data == {}


@pytest.mark.parametrize("is_streaming", [False, True])
def test_decode_result(is_streaming: bool) -> None:
    result = build_decode_result(
        answer_payload("Hi.", modalities=["text"]), is_streaming
    )

    assert result["usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 2,
        "total_tokens": 42,
    }
    assert ("text" in result) is not is_streaming


def test_config_places_each_checkpoint_on_its_stage() -> None:
    config = AuraOmniPipelineConfig(model_path="aurateam/AURA")

    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3VLForConditionalGeneration")
        is AuraOmniPipelineConfig
    )
    assert config.stage_factory_kwargs("asr") == {"model_path": "Qwen/Qwen3-ASR-1.7B"}
    assert config.stage_factory_kwargs("tts_engine") == {
        "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    }
    assert config.stage_factory_kwargs("aura") == {}
