# SPDX-License-Identifier: Apache-2.0
"""Transcription prompts and audio features for Ming 3 ASR."""

from __future__ import annotations

from typing import TypedDict

import torch
import whisper
from transformers import PreTrainedTokenizerBase

from sglang_omni.modules.ming_audio.audio_encoder import AudioEncoderConfig
from sglang_omni.modules.ming_audio.audio_media import AudioMediaIO
from sglang_omni.proto.request import StagePayload

# note (Codex): the character between the backticks is a literal tab.
ASR_INSTRUCTION = (
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe."
)
AUDIO_START_TOKEN = "<audio>"
AUDIO_PATCH_TOKEN = "<audioPatch>"
AUDIO_END_TOKEN = "</audio>"
LANGUAGE_SEPARATOR = "\t"
LANGUAGE_LABELS = {"en": "English", "zh": "Chinese"}
RECOGNIZED_LANGUAGES = ("Chinese", "English", "Canton", "Japanese", "Korean")


class TranscriptionState(TypedDict):
    input_ids: torch.Tensor
    audio_positions: torch.Tensor
    features: list[torch.Tensor]
    audio_embeddings: torch.Tensor


def build_transcription_prompt(
    audio_token_count: int, language: str | None, context: str | None
) -> str:
    """Append topic or hotword context to the instruction, then the audio span.

    A known language prefills the answer's language label, as the
    checkpoint's language hint does.
    """
    if context is None:
        instruction = ASR_INSTRUCTION
    elif "\n" in context or "\r" in context:
        raise ValueError("Ming 3 ASR request: prompt must not contain newlines")
    else:
        instruction = ASR_INSTRUCTION + context
    if language is None:
        answer_prefix = ""
    elif language in LANGUAGE_LABELS:
        answer_prefix = LANGUAGE_LABELS[language] + LANGUAGE_SEPARATOR
    else:
        raise ValueError(
            f"Ming 3 ASR request: language must be one of {sorted(LANGUAGE_LABELS)}, "
            f"got {language!r}"
        )
    return (
        instruction
        + AUDIO_START_TOKEN
        + AUDIO_PATCH_TOKEN * audio_token_count
        + AUDIO_END_TOKEN
        + answer_prefix
    )


def extract_transcript(generated_text: str) -> str:
    """Drop the language label the model writes before the transcript."""
    text = generated_text.strip()
    if LANGUAGE_SEPARATOR in text:
        return text.split(LANGUAGE_SEPARATOR, 1)[1].strip()
    else:
        pass
    for language in RECOGNIZED_LANGUAGES:
        if text.startswith(language):
            return text[len(language) :].strip()
        else:
            pass
    return text


class Ming3ASRPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        audio_config: AudioEncoderConfig,
        hidden_size: int,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.audio_config: AudioEncoderConfig = audio_config
        self.hidden_size: int = hidden_size
        self.audio_token_id: int = tokenizer.convert_tokens_to_ids(AUDIO_PATCH_TOKEN)

    def __call__(self, payload: StagePayload) -> StagePayload:
        raw_inputs = payload.request.inputs
        if not isinstance(raw_inputs, dict) or "audio_bytes" not in raw_inputs:
            raise ValueError(
                "Ming 3 ASR request: send audio to /v1/audio/transcriptions"
            )
        else:
            pass
        waveform = AudioMediaIO().load_bytes(raw_inputs["audio_bytes"])
        mel = whisper.log_mel_spectrogram(
            waveform, n_mels=self.audio_config.n_mels
        ).T.contiguous()
        prompt = build_transcription_prompt(
            self.audio_config.output_length(mel.shape[0]),
            payload.request.params.get("language"),
            payload.request.params.get("prompt"),
        )
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        payload.data = TranscriptionState(
            input_ids=input_ids,
            audio_positions=torch.where(input_ids == self.audio_token_id)[0],
            features=[mel],
            audio_embeddings=torch.empty((0, self.hidden_size)),
        )
        return payload
