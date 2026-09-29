from __future__ import annotations

import asyncio

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
