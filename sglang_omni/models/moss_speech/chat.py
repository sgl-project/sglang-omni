# SPDX-License-Identifier: Apache-2.0
"""Convert text and speech chat turns into checkpoint-compatible token rows."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.fsm import AUDIO_PAD, EOSP, MODALITY_PAD, SOSP


class AudioTokenEncoder(Protocol):
    def encode_audio(self, source: str) -> list[int]: ...


class InputAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: str = Field(min_length=1)
    format: str = Field(min_length=1)


class AudioURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1)


class ChatContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "input_audio", "audio_url"]
    text: str | None = None
    input_audio: InputAudio | None = None
    audio_url: AudioURL | None = None


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[ChatContent]


def chat_rows(
    inputs: object,
    tokenizer: PreTrainedTokenizerBase,
    audio_encoder: AudioTokenEncoder | None,
    output_modality: Literal["text", "audio"],
) -> list[list[int]]:
    turns = TypeAdapter(list[ChatTurn]).validate_python(inputs)
    if not turns or turns[-1].role != "user":
        raise ValueError("Chat requires a final user turn")
    else:
        pass
    if turns[0].role != "system":
        turns.append(
            ChatTurn(
                role="system",
                content=f"You are a helpful assistant. Respond with {'spoken' if output_modality == 'audio' else 'text'} outputs.",
            ),
        )
    else:
        pass
    rows: list[list[int]] = []
    for turn in turns:
        segments: list[str | list[int]] = [f"<|im_start|>{turn.role}\n"]
        if isinstance(turn.content, str):
            if not turn.content.strip():
                raise ValueError("Chat text must not be empty")
            else:
                segments.append(turn.content)
        else:
            if not turn.content:
                raise ValueError("Chat content must not be empty")
            else:
                pass
            for part in turn.content:
                if part.type == "text" and part.text is not None and part.text.strip():
                    segments.append(part.text)
                elif part.type == "input_audio" and part.input_audio is not None:
                    if audio_encoder is None:
                        raise ValueError(
                            "Speech input requires preprocessing.factory.codec_path"
                        )
                    else:
                        segments.append(
                            audio_encoder.encode_audio(
                                f"data:audio/{part.input_audio.format};base64,{part.input_audio.data}"
                            )
                        )
                elif part.type == "audio_url" and part.audio_url is not None:
                    if audio_encoder is None:
                        raise ValueError(
                            "Speech input requires preprocessing.factory.codec_path"
                        )
                    else:
                        segments.append(audio_encoder.encode_audio(part.audio_url.url))
                else:
                    raise ValueError("Chat content does not match its declared type")
        segments.append("<|im_end|>\n")
        for segment in segments:
            if isinstance(segment, str):
                rows.extend(
                    [
                        [token, AUDIO_PAD]
                        for token in tokenizer.encode(segment, add_special_tokens=False)
                    ]
                )
            elif not segment:
                raise ValueError("Speech input produced no codec tokens")
            else:
                rows.extend(
                    [
                        [SOSP, AUDIO_PAD],
                        *[[MODALITY_PAD, token] for token in segment],
                        [MODALITY_PAD, EOSP],
                    ]
                )
    suffix = "<|im_start|>assistant\n" + (
        "<|object_ref_start|>" if output_modality == "audio" else ""
    )
    rows.extend(
        [
            [token, AUDIO_PAD]
            for token in tokenizer.encode(suffix, add_special_tokens=False)
        ]
    )
    return rows
