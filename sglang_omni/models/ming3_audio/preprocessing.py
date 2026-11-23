# SPDX-License-Identifier: Apache-2.0
"""Chat prompts and audio features for Ming 3 audio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import whisper
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.modules.ming_audio.audio_encoder import AudioEncoderConfig
from sglang_omni.modules.ming_audio.audio_media import AudioMediaIO
from sglang_omni.preprocessing.base import is_url
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto.request import StagePayload

AUDIO_PLACEHOLDER = "<|audio_pad|>"
ASR_LANGUAGE_SEPARATOR = "\t"
ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it with appropriate "
    "punctuation marks, placing a `\t` between the identified language and the transcribe."
)


def separate_asr_language_prefix(transcript: str) -> str:
    """Drop the language tag the ASR prompt asks the model to emit."""
    separator_index = transcript.find(ASR_LANGUAGE_SEPARATOR)
    if separator_index >= 0:
        return transcript[separator_index + 1 :]
    else:
        return transcript


class InputAudio(BaseModel):
    data: str
    format: str


class AudioURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: Literal["text", "audio", "input_audio", "audio_url"]
    text: str = ""
    audio: str | None = None
    input_audio: InputAudio | None = None
    audio_url: AudioURL | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class AudioChatInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage] = Field(default_factory=list)
    audio: list[str | npt.NDArray[np.float32]] = Field(
        default_factory=list, validation_alias=AliasChoices("audio", "audios")
    )
    audio_bytes: bytes | None = None


class AudioPipelineState(TypedDict):
    input_ids: torch.Tensor
    audio_positions: torch.Tensor
    features: list[torch.Tensor]
    audio_embeddings: torch.Tensor


class Ming3AudioPreprocessor:
    def __init__(self, model_path: str) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        configuration = json.loads((Path(model_path) / "config.json").read_text())
        self.audio_config = AudioEncoderConfig.model_validate(
            configuration["audio_config"]
        )
        self.hidden_size: int = configuration["text_config"]["hidden_size"]
        self.audio_token_id: int = configuration["audio_token_id"]

    async def __call__(self, payload: StagePayload) -> StagePayload:
        raw_inputs = payload.request.inputs
        if payload.request.metadata.get("task") == "tts":
            if isinstance(raw_inputs, str):
                payload.data = {"text": raw_inputs}
            else:
                payload.data = {"text": raw_inputs["text"]}
            return payload
        elif isinstance(raw_inputs, str):
            inputs = AudioChatInputs(
                messages=[ChatMessage(role="user", content=raw_inputs)]
            )
        elif isinstance(raw_inputs, list):
            inputs = AudioChatInputs(messages=raw_inputs)
        else:
            inputs = AudioChatInputs.model_validate(raw_inputs)
        if inputs.audio_bytes is not None:
            waveforms = [AudioMediaIO().load_bytes(inputs.audio_bytes)]
            messages = [
                ChatMessage(
                    role="user",
                    content=[
                        ContentPart(type="text", text=ASR_PROMPT),
                        ContentPart(type="audio"),
                    ],
                )
            ]
        else:
            messages = inputs.messages
            sources = list(inputs.audio)
            for message in messages:
                if isinstance(message.content, list):
                    for part in message.content:
                        if part.type == "input_audio":
                            assert part.input_audio is not None
                            audio = part.input_audio
                            sources.append(
                                f"data:audio/{audio.format};base64,{audio.data}"
                            )
                            part.type = "audio"
                        elif part.type == "audio_url":
                            assert part.audio_url is not None
                            sources.append(part.audio_url.url)
                            part.type = "audio"
                        elif part.type == "audio" and part.audio is not None:
                            sources.append(part.audio)
                        else:
                            pass
                else:
                    pass
            waveforms: list[torch.Tensor] = []
            connector = get_global_resource_connector()
            media_io = AudioMediaIO()
            for source in sources:
                if isinstance(source, np.ndarray):
                    waveform = torch.from_numpy(source)
                elif is_url(source):
                    waveform = await connector.load_resource_async(source, media_io)
                else:
                    waveform = media_io.load_file(Path(source))
                waveforms.append(waveform)

        features = [
            whisper.log_mel_spectrogram(
                torch.as_tensor(waveform), n_mels=self.audio_config.n_mels
            ).T.contiguous()
            for waveform in waveforms
        ]
        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump(exclude_none=True) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if features and AUDIO_PLACEHOLDER not in prompt:
            # note (Codex): the HTTP adapter separates audio from message content.
            content = messages[-1].content
            if isinstance(content, str):
                content = [ContentPart(type="text", text=content)]
            else:
                pass
            messages[-1].content = content + [
                ContentPart(type="audio") for _ in features
            ]
            prompt = self.tokenizer.apply_chat_template(
                [message.model_dump(exclude_none=True) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            pass
        sections = prompt.split(AUDIO_PLACEHOLDER)
        if len(sections) != len(features) + 1:
            raise ValueError("Audio placeholders must match the number of audio inputs")
        else:
            pass
        expanded = sections[0]
        for mel, section in zip(features, sections[1:], strict=True):
            expanded += (
                AUDIO_PLACEHOLDER * self.audio_config.output_length(mel.shape[0])
                + section
            )
        input_ids = torch.tensor(self.tokenizer.encode(expanded), dtype=torch.long)
        payload.data = AudioPipelineState(
            input_ids=input_ids,
            audio_positions=torch.where(input_ids == self.audio_token_id)[0],
            features=features,
            audio_embeddings=torch.empty((0, self.hidden_size)),
        )
        return payload
