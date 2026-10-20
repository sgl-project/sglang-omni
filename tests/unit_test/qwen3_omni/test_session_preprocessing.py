# SPDX-License-Identifier: Apache-2.0
"""Session preprocessing supplies new media without replaying completed utterances."""

import base64
import io
import threading
import wave
from dataclasses import dataclass, field

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.models.qwen3_omni.session_preprocessing import QwenPreprocessingHooks
from sglang_omni.preprocessing.text import ContentMedia, split_content_parts
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONFIGURE, CONTROL, INTERRUPT, SESSION_CONTROL
from sglang_omni.scheduling.session import SessionContext


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8"))


@dataclass(kw_only=True)
class RecordingProcessor:
    tokenizer: CharacterTokenizer = field(default_factory=CharacterTokenizer)
    media: list[ContentMedia] = field(default_factory=list)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        messages, media = split_content_parts(payload.request.inputs["messages"])
        self.media.append(media)
        prompt_text = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt_text, add_special_tokens=False)
        )
        payload.data = {
            "prompt": {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "prompt_text": prompt_text,
            }
        }
        return payload


def append_unit(
    hooks: QwenPreprocessingHooks,
    identity: SessionIdentity,
    *,
    turn: int = 0,
) -> StagePayload:
    chunk = TimedChunk(
        CONTROL,
        0,
        0,
        0,
        {
            "pcm": b"\x01\x00\x02\x00",
            "images": [b"encoded-image"],
        },
        format="qwen.unit",
        turn=turn,
    )
    return hooks.append(
        chunk,
        StagePayload("unit", OmniRequest(inputs=None), data={}),
        SessionContext(
            session_identity=identity,
            cancelled=threading.Event(),
            emit=lambda chunk: None,
        ),
    )


def test_two_turns_supply_prefix_for_thinker_owned_commit_and_real_media_parser_accepts_urls() -> (
    None
):
    processor = RecordingProcessor()
    hooks = QwenPreprocessingHooks(processor)
    identity = SessionIdentity("one")
    hooks.open(
        identity, OmniRequest(inputs=None, params={"instructions": "Be concise."})
    )
    first = append_unit(hooks, identity)
    second = append_unit(hooks, identity, turn=1)
    first_prompt, second_prompt = first.data["prompt"], second.data["prompt"]
    assert first_prompt["prompt_text"].startswith("<|im_start|>system\nBe concise.")
    assert second_prompt["prompt_text"] == first_prompt["prompt_text"]
    assert second_prompt["input_ids"].tolist() == processor.tokenizer.encode(
        second_prompt["prompt_text"], add_special_tokens=False
    )
    assert torch.equal(
        second_prompt["attention_mask"], torch.ones_like(second_prompt["input_ids"])
    )
    assert len(processor.media) == 2
    assert (
        base64.b64decode(processor.media[0].images[0].split(",")[1]) == b"encoded-image"
    )
    audio = base64.b64decode(processor.media[0].audios[0].split(",")[1])
    with wave.open(io.BytesIO(audio), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.readframes(2) == b"\x01\x00\x02\x00"


def test_sampling_updates_apply_to_next_unit_but_instructions_cannot_change_after_prefill() -> (
    None
):
    processor = RecordingProcessor()
    hooks = QwenPreprocessingHooks(processor)
    identity = SessionIdentity("one")
    hooks.open(
        identity,
        OmniRequest(
            inputs=None,
            params={
                "instructions": "Initial",
                "sampling": {"temperature": 0.2, "talker_top_k": 7},
            },
        ),
    )
    hooks.control(
        identity,
        TimedChunk(
            CONTROL,
            0,
            0,
            0,
            {
                "type": CONFIGURE,
                "instructions": "Configured",
                "sampling": {"temperature": 0.3},
            },
            format=SESSION_CONTROL,
        ),
    )
    first = append_unit(hooks, identity)
    assert "Configured" in first.data["prompt"]["prompt_text"]
    assert first.request.params["temperature"] == 0.3
    assert first.request.params["talker_top_k"] == 7
    hooks.control(
        identity,
        TimedChunk(
            CONTROL,
            0,
            0,
            1,
            {
                "type": CONFIGURE,
                "sampling": {"temperature": 0.4, "talker_top_k": 5},
            },
            format=SESSION_CONTROL,
        ),
    )
    with pytest.raises(ValueError, match="instructions cannot change"):
        hooks.control(
            identity,
            TimedChunk(
                CONTROL,
                0,
                0,
                2,
                {
                    "type": CONFIGURE,
                    "instructions": "Changed",
                    "sampling": {"temperature": 0.8},
                },
                format=SESSION_CONTROL,
            ),
        )
    second = append_unit(hooks, identity, turn=1)
    assert second.request.params["temperature"] == 0.4
    assert second.request.params["talker_top_k"] == 5
    with pytest.raises(ValidationError):
        hooks.control(
            identity,
            TimedChunk(
                CONTROL,
                0,
                0,
                3,
                {"type": CONFIGURE, "sampling": {"temperature": -1}},
                format=SESSION_CONTROL,
            ),
        )


def test_control_and_reopen_preserve_session_isolation() -> None:
    hooks = QwenPreprocessingHooks(RecordingProcessor())
    first, second = SessionIdentity("one"), SessionIdentity("two")
    hooks.open(first, OmniRequest(inputs=None, params={"instructions": "First"}))
    hooks.open(second, OmniRequest(inputs=None, params={"instructions": "Second"}))
    append_unit(hooks, first)
    hooks.control(
        first, TimedChunk(CONTROL, 0, 0, 0, {"type": INTERRUPT}, format=SESSION_CONTROL)
    )
    first_next = append_unit(hooks, first, turn=1)
    second_first = append_unit(hooks, second)
    assert first_next.data["prompt"]["prompt_text"].startswith(
        "<|im_start|>system\nFirst"
    )
    assert "Second" in second_first.data["prompt"]["prompt_text"]
    hooks.close(first)
    hooks.close(first)
    reopened = SessionIdentity("one", open_index=2)
    hooks.open(reopened, OmniRequest(inputs=None, params={"instructions": "Reopened"}))
    assert "Reopened" in append_unit(hooks, reopened).data["prompt"]["prompt_text"]
