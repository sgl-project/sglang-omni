# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from sglang_omni.client.client import _extract_inputs
from sglang_omni.models.cosmos3.components import text_preprocessor
from sglang_omni.models.cosmos3.components.text_preprocessor import (
    Cosmos3TextPreprocessor,
    load_cosmos3_tokenizer,
)
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import ChatCompletionRequest


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


def test_tokenizer_loader_pins_local_and_remote_attempts(monkeypatch) -> None:
    tokenizer = object()
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(model_path: str, **kwargs: object):
        assert model_path == "nvidia/Cosmos3-Nano"
        calls.append(kwargs)
        if kwargs["local_files_only"]:
            raise OSError("revision is not cached")
        return tokenizer

    monkeypatch.setattr(
        text_preprocessor.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    assert (
        load_cosmos3_tokenizer(
            "nvidia/Cosmos3-Nano",
            revision="cosmos-revision",
        )
        is tokenizer
    )
    assert [call["revision"] for call in calls] == ["cosmos-revision"] * 2
    assert [call["local_files_only"] for call in calls] == [True, False]


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


def test_flattens_openai_text_content_parts() -> None:
    tokenizer = _FakeTokenizer()
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        max_seq_len=32,
        tokenizer=tokenizer,
    )
    payload = _payload(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hel"},
                    "lo",
                    {"text": " world"},
                ],
            }
        ],
        max_new_tokens=8,
    )

    result = preprocessor(payload)
    state = Cosmos3PipelineState.from_dict(result.data)

    assert tokenizer.messages == [{"role": "user", "content": "hello world"}]
    assert state.prompt is not None
    assert state.prompt["input_ids"].tolist() == [101, 102, 103]


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
            "does not support media inputs yet",
        ),
        (
            [{"role": "user", "content": [{"type": "text", "text": 7}]}],
            "text part must have a string text field",
        ),
        (
            [{"role": "user", "content": None}],
            "content must be a string or a list of text parts",
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


def test_openai_structured_text_content_reaches_inference() -> None:
    req = ChatCompletionRequest(
        model="nvidia/Cosmos3-Nano",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hel"}, {"text": "lo"}],
            }
        ],
    )
    inputs = _extract_inputs(_build_chat_generate_request(req))
    tokenizer = _FakeTokenizer()
    preprocessor = Cosmos3TextPreprocessor("unused-model", tokenizer=tokenizer)

    result = preprocessor(_payload(inputs, max_new_tokens=2))
    state = Cosmos3PipelineState.from_dict(result.data)

    assert tokenizer.messages == [{"role": "user", "content": "hello"}]
    assert state.prompt is not None
    assert state.prompt["input_ids"].tolist() == [101, 102, 103]


def test_openai_media_content_part_maps_to_bad_request() -> None:
    req = ChatCompletionRequest(
        model="nvidia/Cosmos3-Nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ],
    )
    inputs = _extract_inputs(_build_chat_generate_request(req))
    preprocessor = Cosmos3TextPreprocessor(
        "unused-model",
        tokenizer=_FakeTokenizer(),
    )

    with pytest.raises(
        ValueError, match="does not support media inputs yet"
    ) as exc_info:
        preprocessor(_payload(inputs, max_new_tokens=2))

    assert is_bad_request_error(exc_info.value)
    assert "base64" not in str(exc_info.value)


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
