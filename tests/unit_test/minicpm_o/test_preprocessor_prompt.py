from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import (
    ASR_PROMPT_EN,
    ASR_PROMPT_ZH,
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    MiniCPMOPreprocessor,
)
from sglang_omni.proto import OmniRequest, StagePayload

# The generation suffix from MiniCPM-o-4_5's tokenizer template.
GENERATION_TEMPLATE = """
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
    {%- if use_tts_template is defined and use_tts_template is true %}
        {{- '<|tts_bos|>' }}
    {%- endif %}
{%- endif %}
"""


@pytest.mark.parametrize("use_tts_template", [False, True])
def test_chat_prompt_matches_checkpoint_non_thinking_default(
    use_tts_template: bool,
) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=GENERATION_TEMPLATE)

    prompt = preprocessor.render_chat_template(
        [{"role": "user", "content": "Answer the question."}],
        use_tts_template=use_tts_template,
    )

    expected = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if use_tts_template:
        expected += "<|tts_bos|>"
    assert prompt == expected


def test_raw_prompt_bypasses_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    raw_prompt = "<|im_start|>assistant\n<think>\n"

    assert preprocessor.render_chat_template(raw_prompt) == raw_prompt


def test_prompt_token_ids_bypass_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    token_ids = [151644, 151667, 198]
    payload = StagePayload(
        request_id="prompt-token-ids",
        request=OmniRequest(inputs={"messages": token_ids}),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))

    assert result.data["prompt"]["prompt_text"] == ""
    assert result.data["prompt"]["input_ids"].tolist() == token_ids
    torch.testing.assert_close(
        result.data["prompt"]["attention_mask"], torch.ones(3, dtype=torch.long)
    )


# Renders each turn ahead of the checkpoint's generation suffix.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}" + GENERATION_TEMPLATE
)


class FakeProcessor:
    def __call__(
        self,
        prompt_text: str,
        *,
        images: list[list[Image.Image]] | None,
        audios: list[list[npt.NDArray[np.float32]]] | None,
        return_tensors: str,
        **options: object,
    ) -> dict[str, object]:
        image_count = len(images[0]) if images else 0
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_bound": [[torch.tensor([0, 1])] * image_count],
            "pixel_values": [[torch.zeros(1, 2) for _ in range(image_count)]],
            "tgt_sizes": [[torch.tensor([1, 1]) for _ in range(image_count)]],
            "audio_bounds": [[]],
            "audio_feature_lens": [[]],
            "audio_features": [],
        }


async def images_as_given(raw_images: list[Image.Image] | None) -> list[Image.Image]:
    return list(raw_images or [])


async def silent_audios(
    raw_audios: list[object] | None, *, target_sr: int
) -> list[npt.NDArray[np.float32]]:
    return [np.zeros(target_sr // 10, dtype=np.float32) for _ in raw_audios or []]


@pytest.fixture
def media_preprocessor(monkeypatch: pytest.MonkeyPatch) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        FakeProcessor()  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=CHAT_TEMPLATE)
    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images_as_given)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", silent_audios)
    return preprocessor


@pytest.mark.parametrize(
    ("language", "task_prompt"), [("en", ASR_PROMPT_EN), ("zh", ASR_PROMPT_ZH)]
)
def test_transcription_prompt_precedes_audio(
    media_preprocessor: MiniCPMOPreprocessor, language: str, task_prompt: str
) -> None:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    payload = StagePayload(
        request_id="transcription",
        request=OmniRequest(
            inputs={"audio_bytes": wav_buffer.getvalue()},
            params={"language": language},
        ),
        data=None,
    )

    result = asyncio.run(media_preprocessor(payload))

    prompt_text = result.data["prompt"]["prompt_text"]
    user_turn = prompt_text.partition("<|im_start|>assistant\n")[0]
    assert user_turn == (
        f"<|im_start|>user\n{task_prompt}\n\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
    )


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        (
            "Answer the question in the audio.",
            f"{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}\nAnswer the question in the audio.",
        ),
        (
            f"Describe both.\n{AUDIO_PLACEHOLDER}",
            f"{IMAGE_PLACEHOLDER}\nDescribe both.\n{AUDIO_PLACEHOLDER}",
        ),
    ],
)
def test_chat_media_placeholders_are_added_only_when_missing(
    media_preprocessor: MiniCPMOPreprocessor, content: str, expected_content: str
) -> None:
    payload = StagePayload(
        request_id="chat",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": content}],
                "images": [Image.new("RGB", (2, 2))],
                "audios": ["question.wav"],
            }
        ),
        data=None,
    )

    result = asyncio.run(media_preprocessor(payload))

    prompt_text = result.data["prompt"]["prompt_text"]
    user_turn = prompt_text.partition("<|im_start|>assistant\n")[0]
    assert user_turn == f"<|im_start|>user\n{expected_content}<|im_end|>\n"
