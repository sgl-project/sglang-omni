from __future__ import annotations

import asyncio
import base64
import threading
from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
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


def bare_preprocessor(*, speech_enabled: bool = False) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.speech_enabled = speech_enabled
    preprocessor.reference_service = None
    preprocessor.prompt_lock = threading.Lock()
    return preprocessor


def test_prompt_token_ids_bypass_chat_template() -> None:
    preprocessor = bare_preprocessor()
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


@pytest.mark.parametrize("output_modalities", [["text", "audio"], ["text"]])
def test_speech_requests_carry_speaker_conditioning(output_modalities) -> None:
    preprocessor = bare_preprocessor(speech_enabled=True)
    conditioning = {"speech_tokens": torch.zeros(1, 2, dtype=torch.int32)}
    preprocessor.reference_service = MagicMock()
    preprocessor.reference_service.get_or_encode.return_value = conditioning
    reference = "data:audio/wav;base64," + base64.b64encode(b"voice").decode()
    payload = StagePayload(
        request_id="speech",
        request=OmniRequest(
            inputs={"messages": [151644, 198]},
            params={"ref_audio": reference},
            metadata={"output_modalities": output_modalities},
        ),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))

    if "audio" in output_modalities:
        call = preprocessor.reference_service.get_or_encode.call_args
        assert call.args == (b"voice",)
        assert result.data["speaker_prompt"] is conditioning
    else:
        preprocessor.reference_service.get_or_encode.assert_not_called()
        assert "speaker_prompt" not in result.data
