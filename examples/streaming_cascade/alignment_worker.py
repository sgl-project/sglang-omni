# SPDX-License-Identifier: Apache-2.0
"""A forced-alignment worker for the `forced` method of the cascade's alignment node: word ends of known text in its audio.

Each request is aligned in its `language`, a language name such as `English`
or `Chinese`, or in the worker's `--language` when it names none.
"""

import argparse
import base64
import binascii
import logging
import threading
import unicodedata

import numpy as np
import regex
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from qwen_asr import Qwen3ForcedAligner

from sglang_omni.models.streaming_cascade.alignment import (
    MAX_ALIGNMENT_AUDIO_BYTES,
    AlignmentBoundary,
    AlignmentResult,
)
from sglang_omni.models.streaming_cascade.numerals import spoken_text

logger = logging.getLogger("alignment_worker")


class AlignmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=16000)
    audio: str = Field(max_length=MAX_ALIGNMENT_AUDIO_BYTES * 4 // 3 + 4)
    sample_rate_hz: int = Field(gt=0, le=192000)
    audio_format: str
    language: str | None = Field(default=None, min_length=1, max_length=64)


def create_app(model: str, device: str, default_language: str) -> FastAPI:
    torch.set_num_threads(4)
    aligner = Qwen3ForcedAligner.from_pretrained(
        model, device_map=device, dtype=torch.bfloat16, attn_implementation="eager"
    )
    aligner.align(
        audio=(np.zeros(16000, dtype=np.float32), 16000),
        text="hello",
        language="English",
    )
    inference_lock = threading.Lock()
    application = FastAPI()

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.post("/align", response_model=AlignmentResult)
    def align(request: AlignmentRequest) -> AlignmentResult:
        try:
            pcm16 = base64.b64decode(request.audio, validate=True)
        except binascii.Error as exc:
            raise HTTPException(422, "Invalid base64 audio") from exc
        if (
            request.audio_format != "pcm16"
            or not pcm16
            or len(pcm16) % 2
            or len(pcm16) > MAX_ALIGNMENT_AUDIO_BYTES
        ):
            logger.warning(
                "rejected %s audio of %d bytes", request.audio_format, len(pcm16)
            )
            raise HTTPException(422, "Expected bounded, complete mono PCM16")
        else:
            pass
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        if request.language is None:
            language = default_language
        else:
            language = request.language
        text, origin = spoken_text(request.text, language)
        normalized_characters: list[str] = []
        original_ends_chars: list[int] = []
        for cluster in regex.finditer(r"\X", text):
            characters = (
                unicodedata.normalize("NFKC", cluster.group())
                if language.lower() == "japanese"
                else cluster.group()
            )
            for character in characters:
                if aligner.aligner_processor.is_kept_char(character):
                    normalized_characters.append(character)
                    original_ends_chars.append(origin[cluster.end() - 1] + 1)
                else:
                    pass
        normalized_text = "".join(normalized_characters)
        with inference_lock:
            result = aligner.align(
                audio=(samples, request.sample_rate_hz),
                text=text,
                language=language,
            )[0]
        boundaries: list[AlignmentBoundary] = []
        normalized_end_chars = 0
        for token in result:
            if not token.text or not normalized_text.startswith(
                token.text, normalized_end_chars
            ):
                logger.warning(
                    f"aligner token {token.text!r} does not match normalized text "
                    f"at character {normalized_end_chars}"
                )
                raise HTTPException(
                    422,
                    "Aligner tokens do not match the normalized source text",
                )
            else:
                pass
            normalized_end_chars += len(token.text)
            # Note (Junnan Li): The aligner works in frames of its own, so a token may end a few milliseconds past the audio; that is the audio's end.
            sample_end = min(
                round(token.end_time * request.sample_rate_hz), len(samples)
            )
            if sample_end < 0:
                logger.warning(
                    "aligner token %r ends before the audio starts", token.text
                )
                raise HTTPException(422, "Alignment precedes source audio")
            else:
                pass
            text_end_chars = original_ends_chars[normalized_end_chars - 1]
            if (
                normalized_end_chars < len(original_ends_chars)
                and original_ends_chars[normalized_end_chars] == text_end_chars
            ):
                pass
            else:
                boundaries.append(
                    AlignmentBoundary(
                        text_end_chars=text_end_chars, audio_end_samples=sample_end
                    )
                )
        if normalized_end_chars != len(normalized_text):
            raise HTTPException(422, "Aligner tokens omit the end of the source text")
        else:
            pass
        return AlignmentResult(boundaries=boundaries)

    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--language",
        default="Chinese",
        help="the language of a request that names none",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    uvicorn.run(
        create_app(arguments.model, arguments.device, arguments.language),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
