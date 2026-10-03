# SPDX-License-Identifier: Apache-2.0
"""Check Qwen3-Omni transcription upload preprocessing."""

from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto.request import OmniRequest, StagePayload

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% for part in message['content'] %}"
    "{% if part['type'] == 'audio' %}"
    "{{ '<|audio_start|><|audio_pad|><|audio_end|>' }}"
    "{% else %}{{ part['text'] }}{% endif %}"
    "{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@pytest.fixture
def audio_preprocessor() -> Qwen3OmniPreprocessor:
    template_processor = object.__new__(Qwen3OmniMoeProcessor)
    template_processor.tokenizer = PreTrainedTokenizerBase()
    template_processor.chat_template = CHAT_TEMPLATE
    preprocessor = object.__new__(Qwen3OmniPreprocessor)
    preprocessor.max_seq_len = None
    preprocessor.default_video_fps = None
    preprocessor.default_video_max_frames = None
    preprocessor.default_video_min_pixels = None
    preprocessor.default_video_max_pixels = None
    preprocessor.default_video_total_pixels = None
    preprocessor.processor = Mock(
        spec=Qwen3OmniMoeProcessor,
        return_value={
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "input_features": torch.zeros(1, 128, 4),
            "feature_attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    preprocessor.processor.apply_chat_template.side_effect = (
        template_processor.apply_chat_template
    )
    return preprocessor


@pytest.mark.parametrize(
    "language,context", [(None, None), ("en", "SGLang and Qwen3-Omni")]
)
def test_transcription_upload_decodes_audio_and_sets_task(
    audio_preprocessor: Qwen3OmniPreprocessor,
    language: str | None,
    context: str | None,
) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(
            np.array([8192, -8192, 4096, -4096], dtype="<i2").tobytes()
        )
    payload = StagePayload(
        request_id="transcription-upload",
        request=OmniRequest(
            inputs={"audio_bytes": buffer.getvalue()},
            params={"language": language, "prompt": context},
        ),
        data=None,
    )

    result = asyncio.run(audio_preprocessor(payload))

    state = Qwen3OmniPipelineState.from_dict(result.data)
    assert state.prompt is not None
    prompt_text = state.prompt["prompt_text"]
    assert "transcribe" in prompt_text.lower()
    assert "original language" in prompt_text
    assert "only the transcription" in prompt_text
    assert prompt_text.count("<|audio_pad|>") == 1
    if language is not None:
        assert f"The spoken language is {language}." in prompt_text
        assert f"Transcription context: {context}" in prompt_text
    else:
        assert "The spoken language is" not in prompt_text
        assert "Transcription context:" not in prompt_text
    processor_call = audio_preprocessor.processor.call_args
    assert len(processor_call.kwargs["audio"]) == 1
    np.testing.assert_array_equal(
        processor_call.kwargs["audio"][0],
        np.array([0.25, -0.25, 0.125, -0.125], dtype=np.float32),
    )
    assert "input_features" in state.encoder_inputs["audio_encoder"]


def test_invalid_transcription_upload_reports_decode_error(
    audio_preprocessor: Qwen3OmniPreprocessor,
) -> None:
    payload = StagePayload(
        request_id="transcription-invalid",
        request=OmniRequest(inputs={"audio_bytes": b"invalid audio"}),
        data=None,
    )

    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        asyncio.run(audio_preprocessor(payload))
    audio_preprocessor.processor.assert_not_called()
