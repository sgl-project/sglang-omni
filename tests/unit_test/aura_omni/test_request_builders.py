# SPDX-License-Identifier: Apache-2.0
"""AURA Omni splits a chat turn across Qwen3-ASR, AURA and Qwen3-TTS and routes the answer."""

import pytest

from sglang_omni.models.aura_omni.config import AuraOmniPipelineConfig
from sglang_omni.models.aura_omni.request_builders import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    VIDEO_PLACEHOLDER,
    build_decode_result,
    merge_for_aura,
    preprocess_aura_payload,
    project_aura_to_tts,
    project_aura_to_vocoder,
    project_preprocessing_to_asr,
    project_preprocessing_to_aura,
    render_aura_prompt,
    resolve_aura_next_stages,
    resolve_aura_stream_done_targets,
    resolve_aura_wait_sources,
    resolve_preprocessing_next_stages,
    resolve_terminal_stages,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error

VIDEO = "file:///clips/desk.mp4"
AUDIO = "data:audio/wav;base64,UklGRg=="


def chat_payload(
    content: list[dict[str, object]],
    *,
    system: str | None = None,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
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
            inputs={"messages": messages},
            params=params or {"temperature": 0.2, "stream": False},
            metadata=metadata or {"output_modalities": ["text", "audio"]},
        ),
        data=None,
    )


def spoken_turn() -> StagePayload:
    return preprocess_aura_payload(
        chat_payload(
            [
                {"type": "video_url", "video_url": {"url": VIDEO}},
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


def test_spoken_turn_goes_to_asr_and_aura():
    turn = spoken_turn()

    assert turn.data["system_prompt"] == DEFAULT_AURA_SYSTEM_PROMPT
    assert turn.data["videos"] == [VIDEO]
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


def test_turn_without_speech_skips_asr():
    turn = preprocess_aura_payload(
        chat_payload(
            [{"type": "video_url", "video_url": {"url": VIDEO}}], system="Be brief."
        )
    )

    assert turn.data["system_prompt"] == "Be brief."
    assert resolve_preprocessing_next_stages("r1", turn) == ["aura"]
    assert resolve_aura_wait_sources(
        "r1", "preprocessing", project_preprocessing_to_aura(turn)
    ) == ["preprocessing"]


def test_merge_puts_transcript_ahead_of_typed_text():
    aura_payload = project_preprocessing_to_aura(spoken_turn())
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


def test_earlier_turns_render_as_history():
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


def test_second_audio_input_is_a_bad_request():
    clip = {"type": "audio_url", "audio_url": {"url": AUDIO}}

    with pytest.raises(ValueError, match="one audio input per request") as error:
        preprocess_aura_payload(chat_payload([clip, clip]))
    assert is_bad_request_error(error.value)


def test_conversation_must_end_on_a_user_turn():
    payload = chat_payload([{"type": "text", "text": "hi"}])
    payload.request.inputs["messages"].append({"role": "assistant", "content": "hey"})

    with pytest.raises(ValueError, match="end with a user turn") as error:
        preprocess_aura_payload(payload)
    assert is_bad_request_error(error.value)


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
def test_answer_routing(text, modalities, next_stages, stream_done, terminals):
    answer = answer_payload(text, modalities=modalities)

    assert resolve_aura_next_stages("r1", answer) == next_stages
    assert resolve_aura_stream_done_targets("r1", answer) == stream_done
    assert resolve_terminal_stages(answer.request) == terminals


def test_spoken_answer_becomes_a_tts_request():
    tts = project_aura_to_tts(answer_payload("It is a guitar.", modalities=["audio"]))

    assert tts.request.inputs == {"text": "It is a guitar."}
    assert tts.request.metadata["tts_params"] == {
        "language": "English",
        "voice": "ryan",
    }
    assert tts.request.params["stream"] is True
    assert "temperature" not in tts.request.params


def test_silent_answer_closes_the_vocoder_without_codes():
    vocoder = project_aura_to_vocoder(answer_payload("", modalities=["audio"]))

    assert vocoder.request.params == {"stream": True}
    assert vocoder.data == {}


@pytest.mark.parametrize("is_streaming", [False, True])
def test_decode_result(is_streaming):
    result = build_decode_result(
        answer_payload("Hi.", modalities=["text"]), is_streaming
    )

    assert result["usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 2,
        "total_tokens": 42,
    }
    assert ("text" in result) is not is_streaming


def test_config_places_each_checkpoint_on_its_stage():
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
