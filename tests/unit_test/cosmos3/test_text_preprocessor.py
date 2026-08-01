# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from sglang_omni.models.cosmos3.components.text_preprocessor import (
    Cosmos3TextPreprocessor,
)
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    chat_template = "fake-template"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.tokenize_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert text.startswith("<|im_start|>user")
        assert add_special_tokens is False
        assert return_tensors == "pt"
        self.tokenize_calls += 1
        return {
            "input_ids": torch.tensor([[101, 102, 103]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def _payload(inputs: Any, **params: Any) -> StagePayload:
    return StagePayload(
        request_id="cosmos-request",
        request=OmniRequest(inputs=inputs, params=params),
        data={},
    )


def test_preprocesses_text_messages_into_canonical_state() -> None:
    tokenizer = _FakeTokenizer()
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        max_seq_len=32,
        tokenizer=tokenizer,
    )
    payload = _payload(
        {"messages": [{"role": "user", "content": "hello"}]},
        max_new_tokens=8,
    )

    result = preprocessor(payload)
    state = Cosmos3PipelineState.from_dict(result.data)

    assert result is payload
    assert result.request.inputs is None
    assert tokenizer.messages == [{"role": "user", "content": "hello"}]
    assert state.prompt is not None
    assert state.prompt["input_ids"].tolist() == [101, 102, 103]
    assert state.prompt["input_ids"].dtype == torch.long
    assert state.prompt["attention_mask"].tolist() == [1, 1, 1]
    assert state.prompt["prompt_text"].endswith("<|im_start|>assistant\n")
    assert state.stream_state == {"token_ids": [], "text": ""}


def test_pretokenized_prompt_bypasses_chat_template() -> None:
    tokenizer = _FakeTokenizer()
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        max_seq_len=16,
        tokenizer=tokenizer,
    )

    result = preprocessor(_payload([7, 8, 9], max_new_tokens=2))
    state = Cosmos3PipelineState.from_dict(result.data)

    assert tokenizer.tokenize_calls == 0
    assert state.prompt is not None
    assert state.prompt["input_ids"].tolist() == [7, 8, 9]
    assert state.prompt["attention_mask"].tolist() == [1, 1, 1]
    assert state.prompt["prompt_text"] == ""


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {"messages": [{"role": "user", "content": "hello"}], "images": [1]},
            "does not support media inputs yet",
        ),
        (
            [{"role": "user", "content": [{"type": "image"}]}],
            "content must be a string",
        ),
        ({"prompt": "hello"}, "expects a messages field"),
    ],
)
def test_rejects_inputs_outside_pr1_text_scope(inputs: Any, expected: str) -> None:
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        tokenizer=_FakeTokenizer(),
    )

    with pytest.raises((TypeError, ValueError), match=expected):
        preprocessor(_payload(inputs, max_new_tokens=2))


def test_rejects_media_from_openai_request_metadata() -> None:
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        tokenizer=_FakeTokenizer(),
    )
    payload = _payload(
        [{"role": "user", "content": "describe this"}],
        max_new_tokens=2,
    )
    payload.request.metadata["images"] = ["data:image/png;base64,abc"]

    with pytest.raises(ValueError, match="does not support media inputs yet: images"):
        preprocessor(payload)


def test_rejects_prompt_and_completion_over_context_limit() -> None:
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        max_seq_len=8,
        tokenizer=_FakeTokenizer(),
    )

    with pytest.raises(ValueError, match="maximum context length"):
        preprocessor(_payload([1, 2, 3], max_new_tokens=5))


def test_pipeline_state_survives_stage_payload_round_trip() -> None:
    state = Cosmos3PipelineState(
        prompt={
            "input_ids": torch.tensor([1, 2], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "prompt_text": "prompt",
        },
        engine_outputs={"text": {"output_ids": [3]}},
        stream_state={"token_ids": [3], "text": "answer"},
    )
    payload = StagePayload(
        request_id="round-trip",
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )

    restored_payload = StagePayload.from_dict(payload.to_dict())
    restored_state = Cosmos3PipelineState.from_dict(restored_payload.data)

    assert restored_payload.request_id == "round-trip"
    assert restored_state.prompt is not None
    assert restored_state.prompt["input_ids"].tolist() == [1, 2]
    assert restored_state.engine_outputs == {"text": {"output_ids": [3]}}
    assert restored_state.stream_state == {"token_ids": [3], "text": "answer"}
